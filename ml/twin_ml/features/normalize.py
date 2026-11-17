"""Normalisation fitted on training rows only, plus the series transform used everywhere."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twin_common.enums import MachineKind
from twin_ml.features.inputs import FloatArray, MaskArray, forward_fill, vib_norm
from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema


@dataclass(frozen=True)
class Normalizer:
    """Per-channel mean/std and per-kind reference speed, all from training data.

    The reference speed exists only when the schema has the ``vib_norm`` channel (simulator
    datasets); real-data schemas without it normalise their sensor channels only (ADR 0028).
    """

    speed_ref: dict[MachineKind, float]
    mean: FloatArray  # (len(schema.value_channels),)
    std: FloatArray
    min_speed_ratio: float
    schema: ChannelSchema = field(default=SIM_SCHEMA)

    @classmethod
    def fit(
        cls,
        series: Sequence[tuple[FloatArray, Any]],
        min_speed_ratio: float,
        schema: ChannelSchema = SIM_SCHEMA,
    ) -> Normalizer:
        """Fit on raw training series ``(rows x sensors, kind)``; NaNs are ignored."""
        if not series:
            raise ValueError("no training series to fit the normalizer on")
        speed_ref: dict[MachineKind, float] = {}
        if schema.vib_norm:
            i_speed = schema.sensors.index("speed")
            for kind in sorted({k for _, k in series}):
                speeds = np.concatenate([raw[:, i_speed] for raw, k in series if k == kind])
                moving = speeds[np.isfinite(speeds) & (speeds > 0)]
                if moving.size == 0:
                    raise ValueError(f"no nonzero speed for kind {kind} in the training data")
                speed_ref[kind] = float(np.median(moving))
            cols = [
                np.concatenate([raw, vib_norm(raw, speed_ref[k], min_speed_ratio)[:, None]], axis=1)
                for raw, k in series
            ]
        else:
            cols = [raw for raw, _ in series]
        stacked = np.concatenate(cols, axis=0).astype(np.float64)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        std = np.where(std > 1e-9, std, 1.0)
        return cls(
            speed_ref=speed_ref,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            min_speed_ratio=min_speed_ratio,
            schema=schema,
        )

    def transform(self, raw: FloatArray, kind: Any) -> tuple[FloatArray, MaskArray]:
        """One time-ordered raw series -> (normalised value channels, missing mask).

        Missing values are forward-filled; leading gaps take the training mean.
        """
        filled, mask = forward_fill(raw, self.mean[: len(self.schema.sensors)])
        if self.schema.vib_norm:
            values = np.concatenate(
                [filled, vib_norm(filled, self.ref(kind), self.min_speed_ratio)[:, None]], axis=1
            )
        else:
            values = filled
        return ((values - self.mean) / self.std).astype(np.float32), mask

    def ref(self, kind: MachineKind) -> float:
        if kind not in self.speed_ref:
            raise KeyError(f"no reference speed for kind {kind} (not in the training data)")
        return self.speed_ref[kind]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "channels": list(self.schema.value_channels),
            "speed_ref": {str(k): v for k, v in sorted(self.speed_ref.items())},
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
            "min_speed_ratio": self.min_speed_ratio,
        }
        if self.schema != SIM_SCHEMA:  # simulator builds keep their exact pre-Phase-10 dict
            out["schema"] = self.schema.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Normalizer:
        schema = ChannelSchema.from_dict(data["schema"]) if "schema" in data else SIM_SCHEMA
        if list(data["channels"]) != list(schema.value_channels):
            raise ValueError("normalizer channels do not match this code version")
        ref_key: Any = MachineKind if schema == SIM_SCHEMA else str
        return cls(
            speed_ref={ref_key(k): float(v) for k, v in data["speed_ref"].items()},
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            min_speed_ratio=float(data["min_speed_ratio"]),
            schema=schema,
        )
