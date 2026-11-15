"""Static threshold baseline (imports twin_sim for the nominal thresholds).

FAULT when the last observed vib, temp or current reaches its nominal (unperturbed) per-kind
fault threshold from ``config/simulator.yaml``, WARNING when one reaches its warning
threshold, NORMAL otherwise. The same class is predicted for every horizon. The simulator
itself classifies on noise-free signals with per-run perturbed thresholds, so this rule shows
what a fixed alarm limit on the observed data achieves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import FAULT, HEALTH, NORMAL, WARNING
from twin_ml.evaluate.evaluator import Predictions
from twin_ml.features.inputs import KINDS, SENSORS
from twin_ml.features.normalize import Normalizer
from twin_sim.config import SimConfig, load_sim_config

RULE_SENSORS = ("vib", "temp", "current")


@dataclass(frozen=True)
class KindLimits:
    warning: npt.NDArray[np.float64]  # per RULE_SENSORS
    fault: npt.NDArray[np.float64]


def nominal_limits(sim: SimConfig) -> dict[MachineKind, KindLimits]:
    out: dict[MachineKind, KindLimits] = {}
    for kind, phys in sim.kinds.items():
        t = phys.thresholds
        rated = phys.current_rated_a
        out[kind] = KindLimits(
            warning=np.array([t.vib_warning, t.temp_warning, t.current_warning_x * rated]),
            fault=np.array([t.vib_fault, t.temp_fault, t.current_fault_x * rated]),
        )
    return out


def classify(
    values: npt.NDArray[np.float64],
    kinds: npt.NDArray[np.int64],
    limits: dict[MachineKind, KindLimits],
) -> npt.NDArray[np.int64]:
    """values: (N, len(RULE_SENSORS)) observed values; kinds: kind code per row."""
    warn = np.stack([limits[KINDS[k]].warning for k in range(len(KINDS))])[kinds]
    fault = np.stack([limits[KINDS[k]].fault for k in range(len(KINDS))])[kinds]
    out = np.full(values.shape[0], NORMAL, dtype=np.int64)
    out[(values >= warn).any(axis=1)] = WARNING
    out[(values >= fault).any(axis=1)] = FAULT
    return out


class ThresholdBaseline:
    name = "threshold"

    def __init__(self, sim_config: Path, normalizer: dict[str, Any]) -> None:
        self.limits = nominal_limits(load_sim_config(sim_config))
        self.norm = Normalizer.from_dict(normalizer)
        self._cols = [SENSORS.index(s) for s in RULE_SENSORS]

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        return {}

    def _observed(self, split: SplitData, idx: npt.NDArray[np.int64] | None) -> Any:
        # Last observed values (forward-filled), recovered from the normalised channels.
        v = split.at_end("values", idx)[:, self._cols].astype(np.float64)
        mean, std = self.norm.mean[self._cols], self.norm.std[self._cols]
        return v * std + mean

    def predict(self, split: SplitData) -> Predictions:
        labels = classify(
            self._observed(split, None), split.at_end("kind").astype(np.int64), self.limits
        )
        onehot = np.eye(len(HEALTH), dtype=np.float32)[labels]
        return Predictions(probs=dict.fromkeys(split.horizons, onehot))

    def predict_one(self, split: SplitData, i: int) -> None:
        idx = np.array([i])
        classify(
            self._observed(split, idx), split.at_end("kind", idx).astype(np.int64), self.limits
        )
