"""Small MLP autoencoder on raw normalised windows (Phase 9, ADR 0024, 0025). Imports torch.

Input: the 6 normalised value channels of a 30 s window, flattened, plus the machine kind as a
one-hot condition on the encoder and the decoder. Trained on normal-operation windows only
(``models.anomaly.normal_windows``) with Adam for a fixed number of epochs; the epoch with
the lowest validation normal-window loss is kept. Score: mean squared reconstruction error
over observed values (a value channel is unobserved where its sensor's missing flag is set;
``vib_norm`` follows ``vib``). Deterministic for a seed (torch deterministic algorithms,
seeded batch order).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from twin_common.enums import MachineState
from twin_ml.config import AutoencoderConfig
from twin_ml.data.dataset import SplitData
from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema
from twin_ml.models.anomaly import normal_windows


def channel_sensors(schema: ChannelSchema) -> tuple[str, ...]:
    """Value channel -> sensor whose missing flag / attribution it belongs to."""
    return tuple(schema.value_sensor(c) for c in schema.value_channels)


# simulator schema (kept for callers of the Phase 9 name)
CHANNEL_SENSOR = channel_sensors(SIM_SCHEMA)


class AutoencoderNet(nn.Module):
    def __init__(self, n_in: int, n_cond: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in + n_cond, hidden), nn.ReLU(), nn.Linear(hidden, latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent + n_cond, hidden), nn.ReLU(), nn.Linear(hidden, n_in)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.enc(torch.cat([x, cond], dim=1))
        out: torch.Tensor = self.dec(torch.cat([z, cond], dim=1))
        return out


def _observed(
    mask: npt.NDArray[Any], schema: ChannelSchema = SIM_SCHEMA
) -> npt.NDArray[np.float32]:
    """(B, W, sensors) missing flags -> (B, W, value channels) observed weights."""
    cols = [schema.sensors.index(s) for s in channel_sensors(schema)]
    return (1.0 - mask[:, :, cols].astype(np.float32)).astype(np.float32)


class AutoencoderDetector:
    name = "autoencoder"

    def __init__(
        self,
        cfg: AutoencoderConfig,
        no_fault_code: int,
        horizon: int,
        states: Sequence[MachineState],
    ) -> None:
        self.cfg = cfg
        self.no_fault_code = no_fault_code
        self.horizon = horizon
        self.states = tuple(states)
        self.net: AutoencoderNet | None = None
        self.window = 0
        self.schema = SIM_SCHEMA

    def _tensors(
        self, split: SplitData, idx: npt.NDArray[np.int64]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, mask, kind = split.gather(idx)
        w = _observed(mask, split.schema)
        x = torch.from_numpy(np.ascontiguousarray(values.reshape(values.shape[0], -1)))
        return x, torch.from_numpy(kind), torch.from_numpy(w.reshape(w.shape[0], -1))

    @staticmethod
    def _loss(recon: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        per = ((recon - x) ** 2 * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return per

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        c = self.cfg
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if c.num_threads is not None:
            torch.set_num_threads(c.num_threads)
        rng = np.random.default_rng(seed)
        self.window = train.window
        self.schema = train.schema
        n_in = train.window * len(self.schema.value_channels)
        self.net = AutoencoderNet(n_in, len(self.schema.kinds), c.hidden, c.latent)
        opt = torch.optim.Adam(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        tr = normal_windows(train, self.no_fault_code, self.horizon, self.states)
        va = normal_windows(val, self.no_fault_code, self.horizon, self.states)
        best, best_state, history = np.inf, copy.deepcopy(self.net.state_dict()), []
        for epoch in range(c.epochs):
            t0 = time.perf_counter()
            self.net.train()
            order = tr[rng.permutation(tr.shape[0])]
            total = 0.0
            for i in range(0, order.shape[0], c.batch_size):
                x, cond, w = self._tensors(train, order[i : i + c.batch_size])
                loss = self._loss(self.net(x, cond), x, w).mean()
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
                total += float(loss.detach()) * x.shape[0]
            val_loss = float(self._scores(val, va).mean()) if va.size else float("nan")
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / max(1, order.shape[0]),
                    "val_loss": val_loss,
                    "seconds": time.perf_counter() - t0,
                }
            )
            if val_loss < best:
                best, best_state = val_loss, copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best_state)
        best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
        return {
            "n_train_windows": int(tr.shape[0]),
            "n_val_windows": int(va.shape[0]),
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "history": history,
        }

    def _scores(self, split: SplitData, idx: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        return self._errors(split, idx, per_channel=False)

    def _errors(
        self, split: SplitData, idx: npt.NDArray[np.int64], per_channel: bool
    ) -> npt.NDArray[np.float64]:
        if self.net is None:
            raise RuntimeError("fit the detector first")
        self.net.eval()
        out = []
        n_ch = len(split.schema.value_channels)
        with torch.no_grad():
            for i in range(0, idx.shape[0], self.cfg.eval_batch_size):
                x, cond, w = self._tensors(split, idx[i : i + self.cfg.eval_batch_size])
                recon = self.net(x, cond)
                if per_channel:
                    sq = ((recon - x) ** 2 * w).reshape(x.shape[0], -1, n_ch).sum(dim=1)
                    out.append(sq.numpy().astype(np.float64))
                else:
                    out.append(self._loss(recon, x, w).numpy().astype(np.float64))
        if not out:
            return np.zeros((0, n_ch) if per_channel else 0)
        return np.concatenate(out)

    def score(self, split: SplitData, idx: npt.NDArray[np.int64] | None = None) -> Any:
        all_idx = np.arange(split.n_windows) if idx is None else idx
        return self._scores(split, all_idx)

    def save(self, path: Path) -> Path:
        """The fitted network's state dict (Phase 11 deploy bundle)."""
        if self.net is None:
            raise RuntimeError("fit the detector first")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        return path

    def load(self, path: Path, window: int, schema: ChannelSchema = SIM_SCHEMA) -> None:
        """Restore a saved state dict for scoring (same config as the fit)."""
        self.window = window
        self.schema = schema
        n_in = window * len(schema.value_channels)
        self.net = AutoencoderNet(n_in, len(schema.kinds), self.cfg.hidden, self.cfg.latent)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    def sensor_errors(
        self, split: SplitData, idx: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """(B, sensors) squared reconstruction error per sensor (``vib_norm`` -> vib)."""
        ch = self._errors(split, idx, per_channel=True)
        sensors = split.schema.sensors
        out = np.zeros((ch.shape[0], len(sensors)))
        for j, s in enumerate(channel_sensors(split.schema)):
            out[:, sensors.index(s)] += ch[:, j]
        return out
