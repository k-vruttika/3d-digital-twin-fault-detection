"""Predictions of one deploy bundle for a batch of live windows (ADR 0030).

Windows go through the exact training code path: they are packed into a ``SplitData`` (rows
``values`` / ``mask`` / ``kind``, one window end per window) and passed to the model's own
``predict`` (``SequenceModel.predict`` / ``TreeModel.predict``) and to the autoencoder's
``score``. Calibration, Mondrian sets and time-to-fault intervals use the exported maps
(``serve.calib``). Attributions (``SupervisedLive.explain``, the batched form of
``twin_ml.explain.explain()``) are requested by the worker only for windows whose
uncalibrated +30 s argmax is WARNING or FAULT.

Library loading (ADR 0020): a supervised XGBoost bundle loads XGBoost only; a sequence-model
bundle and the autoencoder load torch only. :func:`require_alone` is checked before loading.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import HEALTH, WARNING
from twin_ml.features.normalize import Normalizer
from twin_ml.features.schema import SIM_SCHEMA
from twin_ml.serve.calib import LiveCalibration

CLASSES = tuple(h.value for h in HEALTH)  # NORMAL, WARNING, FAULT


def pack(
    values: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8], kinds: Sequence[MachineKind]
) -> SplitData:
    """(B, W, 6) values, (B, W, 5) masks and B kinds -> a SplitData with B windows."""
    b, w = values.shape[:2]
    kind_codes = np.array([SIM_SCHEMA.kind_index(str(k)) for k in kinds], dtype=np.int8)
    arrays: dict[str, npt.NDArray[Any]] = {
        "values": np.ascontiguousarray(values.reshape(b * w, values.shape[2])),
        "mask": np.ascontiguousarray(mask.reshape(b * w, mask.shape[2])),
        "kind": np.repeat(kind_codes, w),
        "ends": np.arange(w - 1, b * w, w, dtype=np.int64),
    }
    return SplitData(name="live", window=w, horizons=(), stride=1, arrays=arrays)


def kind_onehot(kinds: Sequence[MachineKind]) -> npt.NDArray[np.float32]:
    out = np.zeros((len(kinds), len(SIM_SCHEMA.kinds)), dtype=np.float32)
    for i, k in enumerate(kinds):
        out[i, SIM_SCHEMA.kind_index(str(k))] = 1.0
    return out


@dataclass
class Timer:
    seconds: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, t0: float) -> float:
        now = time.perf_counter()
        self.seconds[stage] = self.seconds.get(stage, 0.0) + (now - t0)
        return now


class SupervisedLive:
    """A supervised bundle: probabilities, calibration, sets, TTF, attributions."""

    def __init__(self, meta: dict[str, Any], path: Path, ig_steps: int) -> None:
        from twin_ml.config import parse_model_config
        from twin_ml.explain import require_alone

        self.meta = meta
        m, d = meta["model"], meta["dataset"]
        self.family = str(m["family"])
        require_alone("xgboost" if self.family == "xgb" else "torch")
        self.horizons = tuple(int(h) for h in d["horizons"])
        self.window = int(d["window"])
        self.primary = 30 if 30 in self.horizons else self.horizons[-1]
        self.normalizer = Normalizer.from_dict(d["normalizer"])
        self.calibration = LiveCalibration.from_dict(meta["calibration"])
        self.ig_steps = ig_steps
        cfg = parse_model_config(m["config"])
        n_ft = len(d["fault_types"])
        cap = float(d["ttf_cap_s"])
        if self.family == "xgb":
            from twin_ml.models.trees import TreeModel

            tree = TreeModel(cfg, n_ft, cap)  # type: ignore[arg-type]
            tree.load_heads(path / "model", self.horizons, m["tree_classes"])
            tree.set_threads(1)
            self.model: Any = tree
        else:
            import torch

            from twin_ml.models.nn import SequenceModel

            torch.set_num_threads(1)
            seq = SequenceModel(cfg, n_ft, cap)  # type: ignore[arg-type]
            seq.load(path / "model" / "model.pt", self.horizons, self.window)
            self.model = seq

    def run(
        self,
        values: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        kinds: Sequence[MachineKind],
        timer: Timer,
    ) -> list[dict[str, Any]]:
        """Probabilities (raw and calibrated), Mondrian sets and TTF of B windows."""
        t = time.perf_counter()
        split = pack(values, mask, kinds)
        split.horizons = self.horizons
        preds = self.model.predict(split)
        t = timer.add("model", t)
        logits = preds.extra.get("logits", {})
        cal = {
            h: self.calibration.calibrated(h, preds.probs[h], logits.get(h)) for h in self.horizons
        }
        sets = {h: self.calibration.sets(h, cal[h]) for h in self.horizons}
        ttf = preds.ttf
        iv = None if ttf is None else self.calibration.ttf_interval(ttf)
        timer.add("calibration", t)
        top = np.argmax(preds.probs[self.primary], axis=1)
        out = []
        for i in range(values.shape[0]):
            out.append(
                {
                    "probs": {
                        str(h): preds.probs[h][i].astype(float).tolist() for h in self.horizons
                    },
                    "cal": {str(h): cal[h][i].tolist() for h in self.horizons},
                    "set": {
                        str(h): [c for c, inside in zip(CLASSES, sets[h][i], strict=True) if inside]
                        for h in self.horizons
                    },
                    "ttf": None if ttf is None else float(ttf[i]),
                    "ttfInterval": None if iv is None else [float(iv[0][i]), float(iv[1][i])],
                    "alarmClass": bool(top[i] >= WARNING),
                }
            )
        return out

    def explain(
        self,
        values: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        kinds: Sequence[MachineKind],
        sensor_parts: Sequence[Mapping[str, str]],
    ) -> list[list[dict[str, Any]]]:
        """``explain()`` of B windows in one batch: ``[{sensor, part, weight}]`` each."""
        from twin_ml.explain import attribute
        from twin_ml.explain.aggregate import normalise, to_parts

        att = attribute(self.model, values, mask, kind_onehot(kinds), self.primary, self.ig_steps)
        w = normalise(att.sensors)
        return [to_parts(w[j], sensor_parts[j]) for j in range(values.shape[0])]


class AnomalyLive:
    """The autoencoder bundle: reconstruction-error score and per-kind threshold flag."""

    def __init__(self, meta: dict[str, Any], path: Path) -> None:
        import torch

        from twin_common.enums import MachineState
        from twin_ml.config import parse_anomaly_config
        from twin_ml.explain import require_alone
        from twin_ml.models.autoencoder import AutoencoderDetector

        require_alone("torch")
        torch.set_num_threads(1)
        self.meta = meta
        a = meta["anomaly"]
        self.window = int(meta["dataset"]["window"])
        self.normalizer = Normalizer.from_dict(meta["dataset"]["normalizer"])
        self.kinds = [str(k) for k in a["kinds"]]
        # training reference of the drift monitor (None for bundles built before Phase 11 Part C)
        self.drift_reference: dict[str, Any] | None = meta.get("drift")
        self.per_kind = bool(a["per_kind"])
        p = float(a["primary_percentile"])
        self.percentile = p
        self.threshold = np.asarray(a["thresholds"][str(p)], dtype=np.float64)
        cfg = parse_anomaly_config(meta["model"]["config"])
        self.det = AutoencoderDetector(
            cfg,  # type: ignore[arg-type]
            -1,
            int(a["horizon"]),
            [MachineState(s) for s in a["normal_states"]],
        )
        self.det.load(path / "model" / "ae.pt", self.window)

    def thresholds_for(self, kinds: Sequence[MachineKind]) -> npt.NDArray[np.float64]:
        if not self.per_kind:
            return np.full(len(kinds), float(self.threshold[0]))
        return np.array([self.threshold[self.kinds.index(str(k))] for k in kinds], dtype=float)

    def run(
        self,
        values: npt.NDArray[np.float32],
        mask: npt.NDArray[np.uint8],
        kinds: Sequence[MachineKind],
        timer: Timer,
    ) -> list[dict[str, Any]]:
        t = time.perf_counter()
        scores = self.det.score(pack(values, mask, kinds))
        thr = self.thresholds_for(kinds)
        timer.add("anomaly", t)
        return [
            {
                "anomalyScore": float(scores[i]),
                "anomalyThreshold": float(thr[i]),
                "anomalyFlag": bool(scores[i] > thr[i]),
            }
            for i in range(values.shape[0])
        ]
