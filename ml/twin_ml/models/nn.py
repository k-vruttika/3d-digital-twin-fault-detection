"""Sequence models on raw normalised windows: LSTM, GRU and a small Transformer.

One shared encoder with four heads: a health classifier per horizon, time to fault (in units
of the cap, sigmoid output, censoring-aware loss) and the fault type (auxiliary). Training is
deterministic for a seed (torch deterministic algorithms, seeded batch order). Early stopping
watches an exponential moving average of the validation macro F1 (operating windows, mean of
the horizons) with min_epochs, patience and a max-epoch cap; the checkpoint kept is the epoch
with the best raw validation macro F1. Optional reduce-on-plateau on the validation loss.
This is the only module of twin_ml that imports torch.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from twin_common.enums import MachineState
from twin_ml.config import NnConfig
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import HEALTH, STATES
from twin_ml.evaluate.evaluator import Predictions
from twin_ml.evaluate.metrics import confusion, scores_from_confusion
from twin_ml.features.inputs import SEQUENCE_CHANNELS
from twin_ml.models.base import class_weights
from twin_ml.train.early_stopping import EarlyStopper
from twin_ml.train.losses import censored_ttf_loss, weighted_cross_entropy


class SequenceNet(nn.Module):
    def __init__(
        self, cfg: NnConfig, n_horizons: int, n_fault_types: int, window: int, n_in: int
    ) -> None:
        super().__init__()
        a = cfg.arch
        self.kind = cfg.type
        d = a.hidden
        self.rnn: nn.LSTM | nn.GRU | None = None
        if cfg.type in ("lstm", "gru"):
            cls = nn.LSTM if cfg.type == "lstm" else nn.GRU
            self.rnn = cls(
                n_in,
                d,
                num_layers=a.layers,
                batch_first=True,
                dropout=a.dropout if a.layers > 1 else 0.0,
            )
        else:
            self.proj = nn.Linear(n_in, d)
            self.pos = nn.Parameter(torch.zeros(1, window, d))
            layer = nn.TransformerEncoderLayer(
                d,
                a.heads,
                dim_feedforward=a.ff_mult * d,
                dropout=a.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, a.layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(a.dropout)
        self.cls_heads = nn.ModuleList([nn.Linear(d, len(HEALTH)) for _ in range(n_horizons)])
        self.ttf_head = nn.Linear(d, 1)
        self.ft_head = nn.Linear(d, n_fault_types)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.rnn is not None:
            out, _ = self.rnn(x)
            return out[:, -1]  # type: ignore[no-any-return]
        h = self.encoder(self.proj(x) + self.pos)
        return self.norm(h).mean(dim=1)  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        z = self.drop(self.encode(x))
        logits = [head(z) for head in self.cls_heads]
        ttf = torch.sigmoid(self.ttf_head(z)).squeeze(-1)
        return logits, ttf, self.ft_head(z)


@dataclass(frozen=True)
class _Targets:
    """Per-window targets and sample weights of one split."""

    sw: npt.NDArray[np.float32]
    ys: list[npt.NDArray[np.int64]]
    y_ft: npt.NDArray[np.int64]
    ttf: npt.NDArray[np.float32]  # in units of the cap
    cens: npt.NDArray[np.bool_]


class SequenceModel:
    def __init__(
        self,
        cfg: NnConfig,
        n_fault_types: int,
        ttf_cap_s: float,
        operating: Sequence[MachineState] = (MachineState.RUNNING, MachineState.WARNING),
    ) -> None:
        self.cfg = cfg
        self.name: str = cfg.type
        self.n_fault_types = n_fault_types
        self.ttf_cap_s = ttf_cap_s
        self.operating_codes = [STATES.index(s) for s in operating]
        self.net: SequenceNet | None = None
        self.horizons: tuple[int, ...] = ()

    # ------------------------------------------------------------------ helpers
    def _x(self, split: SplitData, idx: npt.NDArray[np.int64]) -> torch.Tensor:
        return torch.from_numpy(split.sequence(idx))

    def _forward_probs(
        self, split: SplitData, batch: int
    ) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        assert self.net is not None
        self.net.eval()
        probs: list[list[npt.NDArray[np.float32]]] = [[] for _ in self.horizons]
        ttf: list[npt.NDArray[np.float32]] = []
        ft: list[npt.NDArray[np.float32]] = []
        with torch.no_grad():
            for idx in split.batches(batch):
                logits, t, f = self.net(self._x(split, idx))
                for i, lg in enumerate(logits):
                    probs[i].append(torch.softmax(lg, dim=1).numpy())
                ttf.append(t.numpy())
                ft.append(torch.softmax(f, dim=1).numpy())
        n_ft = self.n_fault_types
        return (
            [np.concatenate(p) if p else np.zeros((0, len(HEALTH)), np.float32) for p in probs],
            np.concatenate(ttf) if ttf else np.zeros(0, np.float32),
            np.concatenate(ft) if ft else np.zeros((0, n_ft), np.float32),
        )

    def _targets(self, split: SplitData, sw: npt.NDArray[np.float32]) -> _Targets:
        return _Targets(
            sw=sw,
            ys=[split.target(h).astype(np.int64) for h in self.horizons],
            y_ft=split.at_end("fault_type").astype(np.int64),
            ttf=(split.at_end("ttf") / self.ttf_cap_s).astype(np.float32),
            cens=split.at_end("ttf_censored").astype(bool),
        )

    def _loss(
        self,
        out: tuple[list[torch.Tensor], torch.Tensor, torch.Tensor],
        tg: _Targets,
        idx: npt.NDArray[np.int64],
        cws: list[npt.NDArray[np.float64]],
        cw_ft: npt.NDArray[np.float64],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Total weighted loss of one batch and its components (for the learning curves)."""
        t = self.cfg.training
        lw = t.loss_weights
        logits, ttf_pred, ft_logits = out
        w = torch.from_numpy(tg.sw[idx])
        parts: dict[str, torch.Tensor] = {}
        for h, lg, y, cw in zip(self.horizons, logits, tg.ys, cws, strict=True):
            wb = w * torch.from_numpy(cw[y[idx]].astype(np.float32))
            parts[f"cls_h{h}"] = weighted_cross_entropy(
                lg, torch.from_numpy(y[idx]), wb, t.focal_gamma
            )
        parts["ttf"] = censored_ttf_loss(
            ttf_pred,
            torch.from_numpy(tg.ttf[idx]),
            torch.from_numpy(tg.cens[idx]),
            w,
            t.huber_delta,
        )
        wft = w * torch.from_numpy(cw_ft[tg.y_ft[idx]].astype(np.float32))
        parts["fault_type"] = weighted_cross_entropy(
            ft_logits, torch.from_numpy(tg.y_ft[idx]), wft, t.focal_gamma
        )
        cls_sum = torch.zeros(())
        for k, v in parts.items():
            if k.startswith("cls_"):
                cls_sum = cls_sum + v
        total = lw.cls * cls_sum + lw.ttf * parts["ttf"] + lw.fault_type * parts["fault_type"]
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def _validate(
        self,
        val: SplitData,
        tg: _Targets,
        cws: list[npt.NDArray[np.float64]],
        cw_ft: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Validation loss (training weighting, batch means averaged by batch size) and the
        macro F1 per horizon on operating windows."""
        assert self.net is not None
        self.net.eval()
        sums: dict[str, float] = {}
        n = 0
        probs: list[list[npt.NDArray[np.float32]]] = [[] for _ in self.horizons]
        with torch.no_grad():
            for idx in val.batches(self.cfg.training.eval_batch_size):
                out = self.net(self._x(val, idx))
                total, parts = self._loss(out, tg, idx, cws, cw_ft)
                for k, v in {"loss": float(total), **parts}.items():
                    sums[k] = sums.get(k, 0.0) + v * idx.shape[0]
                n += idx.shape[0]
                for i, lg in enumerate(out[0]):
                    probs[i].append(torch.softmax(lg, dim=1).numpy())
        res = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
        op = np.isin(val.at_end("state"), self.operating_codes)
        scores = []
        for h, p in zip(self.horizons, probs, strict=True):
            pr = np.concatenate(p) if p else np.zeros((0, len(HEALTH)), np.float32)
            cm = confusion(val.target(h)[op], np.argmax(pr[op], axis=1), len(HEALTH))
            f1 = scores_from_confusion(cm)["macro_f1"]
            res[f"val_macro_f1_h{h}"] = float(f1) if np.isfinite(f1) else 0.0
            scores.append(res[f"val_macro_f1_h{h}"])
        res["val_macro_f1"] = float(np.mean(scores)) if scores else 0.0
        return res

    # ------------------------------------------------------------------ training
    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        t = self.cfg.training
        es = t.early_stopping
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if t.num_threads is not None:
            torch.set_num_threads(t.num_threads)
        rng = np.random.default_rng(seed)
        self.horizons = train.horizons
        self.net = SequenceNet(
            self.cfg, len(self.horizons), self.n_fault_types, train.window, len(SEQUENCE_CHANNELS)
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        sched = None
        if t.lr_schedule.type == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=t.lr_schedule.factor,
                patience=t.lr_schedule.patience,
                min_lr=t.lr_schedule.min_lr,
            )
        tg = self._targets(train, train.weight.astype(np.float32))
        cws = [class_weights(y, tg.sw, len(HEALTH), self.cfg.class_weight) for y in tg.ys]
        cw_ft = class_weights(tg.y_ft, tg.sw, self.n_fault_types, self.cfg.class_weight)
        # validation is never subsampled: sample weight 1, class weights from training
        tg_val = self._targets(val, np.ones(val.n_windows, dtype=np.float32))

        stopper = EarlyStopper(es)
        best_state = None
        history: list[dict[str, float]] = []
        for epoch in range(1, es.max_epochs + 1):
            t0 = time.perf_counter()
            lr = float(opt.param_groups[0]["lr"])
            self.net.train()
            sums: dict[str, float] = {}
            steps = 0
            for idx in train.batches(t.batch_size, rng):
                loss, parts = self._loss(self.net(self._x(train, idx)), tg, idx, cws, cw_ft)
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(self.net.parameters(), t.grad_clip)
                opt.step()
                for k, v in {"loss": float(loss.detach()), **parts}.items():
                    sums[k] = sums.get(k, 0.0) + v
                steps += 1
            entry: dict[str, float] = {"epoch": float(epoch), "lr": lr}
            entry |= {f"train_{k}": v / max(steps, 1) for k, v in sums.items()}
            entry |= self._validate(val, tg_val, cws, cw_ft)
            is_best, stop = stopper.update(entry["val_macro_f1"])
            entry["val_macro_f1_ema"] = float(stopper.ema or 0.0)
            entry["seconds"] = time.perf_counter() - t0
            history.append(entry)
            if is_best:  # checkpoint: best raw validation macro F1
                best_state = copy.deepcopy(self.net.state_dict())
            if sched is not None:
                sched.step(entry["val_loss"])
            if stop:  # stopping signal: the smoothed score (or the cap)
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {
            "epochs": len(history),
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best_score,
            "stop_reason": stopper.stop_reason,
            "history": history,
            "class_weights": [c.tolist() for c in cws],
            "parameters": sum(p.numel() for p in self.net.parameters()),
            "threads": torch.get_num_threads(),
        }

    # ------------------------------------------------------------------ inference
    def predict(self, split: SplitData) -> Predictions:
        probs, ttf, ft = self._forward_probs(split, self.cfg.training.eval_batch_size)
        return Predictions(
            probs=dict(zip(self.horizons, probs, strict=True)),
            ttf=(ttf * self.ttf_cap_s).astype(np.float32),
            fault_type=ft,
        )

    def predict_one(self, split: SplitData, i: int) -> None:
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            self.net(self._x(split, np.array([i])))

    def save(self, out_dir: Path) -> list[Path]:
        assert self.net is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "model.pt"
        torch.save(self.net.state_dict(), path)
        return [path]
