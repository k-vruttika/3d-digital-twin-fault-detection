"""Anomaly detectors trained on normal operation only (Phase 9, ADR 0024, 0025).

* :func:`normal_windows`: the training / threshold windows, i.e. the window ends in a normal
  state (RUNNING by default), no fault is active on any of its rows, and the health target at
  the evaluation horizon is NORMAL;
* :class:`AnomalyThresholds`: per-kind percentiles of validation normal-operation scores;
* :class:`IForestDetector`: Isolation Forest on window summary features (scikit-learn,
  imported lazily so this module stays importable without it);
* the autoencoder lives in ``models.autoencoder`` (the only other torch module).

Scores grow with abnormality. An alarm is ``score > threshold`` for the window's kind; it is
made sustained exactly like a supervised alarm by the early-warning code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineState
from twin_ml.config import AnomalyModelConfig, IForestConfig
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import NORMAL, STATES
from twin_ml.features.schema import SIM_SCHEMA


class Detector(Protocol):
    name: str

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]: ...

    def score(self, split: SplitData) -> npt.NDArray[np.float64]: ...


def normal_windows(
    split: SplitData,
    no_fault_code: int,
    horizon: int,
    states: Sequence[MachineState] = (MachineState.RUNNING,),
) -> npt.NDArray[np.int64]:
    """Indices of normal-operation windows (ADR 0024 § Anomaly detectors)."""
    a = split.arrays
    ends = a["ends"]
    fault_rows = (a["fault_type"] != no_fault_code).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(fault_rows)])
    # rows ends - window + 1 .. ends; windows never cross series
    in_window = c[ends + 1] - c[ends + 1 - split.window]
    codes = [STATES.index(s) for s in states]
    ok = np.isin(a["state"][ends], codes) & (in_window == 0) & (split.target(horizon) == NORMAL)
    return np.flatnonzero(ok).astype(np.int64)


@dataclass
class AnomalyThresholds:
    """Per-kind (or global) percentiles of validation normal-operation scores."""

    percentiles: list[float]
    per_kind: bool = True
    kinds: tuple[str, ...] = SIM_SCHEMA.kinds
    values: dict[float, npt.NDArray[np.float64]] = field(default_factory=dict)  # p -> (kinds,)
    n_cal: list[int] = field(default_factory=list)

    def fit(self, scores: npt.NDArray[np.float64], kinds: npt.NDArray[Any]) -> AnomalyThresholds:
        k = np.asarray(kinds, dtype=np.int64)
        self.n_cal = [int((k == i).sum()) for i in range(len(self.kinds))]
        for p in self.percentiles:
            if self.per_kind:
                v = np.array(
                    [
                        np.percentile(scores[k == i], p) if (k == i).any() else np.inf
                        for i in range(len(self.kinds))
                    ]
                )
            else:
                v = np.full(len(self.kinds), np.percentile(scores, p))
            self.values[p] = v.astype(np.float64)
        return self

    def alarm(
        self, scores: npt.NDArray[np.float64], kinds: npt.NDArray[Any], p: float
    ) -> npt.NDArray[np.bool_]:
        thr = self.values[p][np.asarray(kinds, dtype=np.int64)]
        out: npt.NDArray[np.bool_] = scores > thr
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "per_kind": self.per_kind,
            "n_cal": self.n_cal,
            "values": {str(p): v.tolist() for p, v in self.values.items()},
            "kinds": list(self.kinds),
        }


class IForestDetector:
    """Isolation Forest on the 50 window summary features; score = -score_samples."""

    name = "iforest"

    def __init__(
        self, cfg: IForestConfig, no_fault_code: int, horizon: int, states: Sequence[MachineState]
    ) -> None:
        self.cfg = cfg
        self.no_fault_code = no_fault_code
        self.horizon = horizon
        self.states = tuple(states)
        self.model: Any = None

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        from sklearn.ensemble import IsolationForest

        idx = normal_windows(train, self.no_fault_code, self.horizon, self.states)
        x = train.summary(idx)
        self.model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            max_samples=min(self.cfg.max_samples, x.shape[0]),
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            random_state=seed,
        ).fit(x)
        return {"n_train_windows": int(idx.shape[0])}

    def score(
        self, split: SplitData, idx: npt.NDArray[np.int64] | None = None, chunk: int = 200_000
    ) -> npt.NDArray[np.float64]:
        if self.model is None:
            raise RuntimeError("fit the detector first")
        all_idx = np.arange(split.n_windows) if idx is None else idx
        parts = [
            -self.model.score_samples(split.summary(all_idx[i : i + chunk]))
            for i in range(0, all_idx.shape[0], chunk)
        ]
        return np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)


def make_detector(
    cfg: AnomalyModelConfig, no_fault_code: int, horizon: int, states: Sequence[MachineState]
) -> Any:
    if isinstance(cfg, IForestConfig):
        return IForestDetector(cfg, no_fault_code, horizon, states)
    from twin_ml.models.autoencoder import AutoencoderDetector

    return AutoencoderDetector(cfg, no_fault_code, horizon, states)
