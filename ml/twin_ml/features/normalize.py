"""Normalisation fitted on training rows only, plus the series transform used everywhere."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from twin_common.enums import MachineKind
from twin_ml.features.inputs import (
    SENSORS,
    VALUE_CHANNELS,
    FloatArray,
    MaskArray,
    forward_fill,
    vib_norm,
)

_SPEED = SENSORS.index("speed")


@dataclass(frozen=True)
class Normalizer:
    """Per-channel mean/std and per-kind reference speed, all from training data."""

    speed_ref: dict[MachineKind, float]
    mean: FloatArray  # (len(VALUE_CHANNELS),)
    std: FloatArray
    min_speed_ratio: float

    @classmethod
    def fit(
        cls,
        series: Sequence[tuple[FloatArray, MachineKind]],
        min_speed_ratio: float,
    ) -> Normalizer:
        """Fit on raw training series ``(rows x 5 sensors, kind)``; NaNs are ignored."""
        if not series:
            raise ValueError("no training series to fit the normalizer on")
        speed_ref: dict[MachineKind, float] = {}
        for kind in sorted({k for _, k in series}):
            speeds = np.concatenate([raw[:, _SPEED] for raw, k in series if k == kind])
            moving = speeds[np.isfinite(speeds) & (speeds > 0)]
            if moving.size == 0:
                raise ValueError(f"no nonzero speed for kind {kind} in the training data")
            speed_ref[kind] = float(np.median(moving))
        cols = [
            np.concatenate([raw, vib_norm(raw, speed_ref[k], min_speed_ratio)[:, None]], axis=1)
            for raw, k in series
        ]
        stacked = np.concatenate(cols, axis=0).astype(np.float64)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        std = np.where(std > 1e-9, std, 1.0)
        return cls(
            speed_ref=speed_ref,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            min_speed_ratio=min_speed_ratio,
        )

    def transform(self, raw: FloatArray, kind: MachineKind) -> tuple[FloatArray, MaskArray]:
        """One time-ordered raw series -> (normalised value channels, missing mask).

        Missing values are forward-filled; leading gaps take the training mean.
        """
        filled, mask = forward_fill(raw, self.mean[: len(SENSORS)])
        values = np.concatenate(
            [filled, vib_norm(filled, self.ref(kind), self.min_speed_ratio)[:, None]], axis=1
        )
        return ((values - self.mean) / self.std).astype(np.float32), mask

    def ref(self, kind: MachineKind) -> float:
        if kind not in self.speed_ref:
            raise KeyError(f"no reference speed for kind {kind} (not in the training data)")
        return self.speed_ref[kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": list(VALUE_CHANNELS),
            "speed_ref": {k.value: v for k, v in sorted(self.speed_ref.items())},
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
            "min_speed_ratio": self.min_speed_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Normalizer:
        if list(data["channels"]) != list(VALUE_CHANNELS):
            raise ValueError("normalizer channels do not match this code version")
        return cls(
            speed_ref={MachineKind(k): float(v) for k, v in data["speed_ref"].items()},
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            min_speed_ratio=float(data["min_speed_ratio"]),
        )
