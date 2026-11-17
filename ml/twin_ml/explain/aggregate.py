"""Feature / channel attributions -> per-sensor weights -> machine parts (pure numpy).

Phase 9 (ADR 0026). Every model input belongs to one sensor or to the machine kind:

* tree summary features ``<channel>_<stat>``: the channel's sensor (``vib_norm`` -> vib),
  ``<sensor>_missing_frac`` -> that sensor, ``kind_<KIND>`` -> dropped;
* sequence channels: value channel -> sensor (``vib_norm`` -> vib), ``<sensor>_missing`` ->
  that sensor, ``kind_<KIND>`` -> dropped.

Attributions are summed per sensor (signed: positive = evidence for the explained class) and,
for display, normalised by the sum of their absolute values. Parts come from the line
config's ``sensor_parts`` (``config/line.yaml``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema
from twin_ml.features.summary import summary_names


def summary_feature_sensors(schema: ChannelSchema = SIM_SCHEMA) -> list[str | None]:
    """Sensor of every tree summary feature (None = kind feature, dropped)."""
    out: list[str | None] = []
    by_len = sorted(schema.value_channels, key=len, reverse=True)
    for name in summary_names(schema):
        if name in schema.kind_channels:
            out.append(None)
        elif name.endswith("_missing_frac"):
            out.append(name[: -len("_missing_frac")])
        else:
            ch = next(c for c in by_len if name.startswith(f"{c}_"))
            out.append(schema.value_sensor(ch))
    return out


def sequence_channel_sensors(schema: ChannelSchema = SIM_SCHEMA) -> list[str | None]:
    """Sensor of every sequence-model input channel (None = kind channel, dropped)."""
    out: list[str | None] = []
    for ch in schema.sequence_channels:
        if ch in schema.kind_channels:
            out.append(None)
        elif ch in schema.mask_channels:
            out.append(ch[: -len("_missing")])
        else:
            out.append(schema.value_sensor(ch))
    return out


def sensor_matrix(
    feature_sensors: Sequence[str | None], schema: ChannelSchema = SIM_SCHEMA
) -> npt.NDArray[np.float64]:
    """(features, sensors) 0/1 matrix; kind features have an all-zero row."""
    m = np.zeros((len(feature_sensors), len(schema.sensors)))
    for i, s in enumerate(feature_sensors):
        if s is not None:
            m[i, schema.sensors.index(s)] = 1.0
    return m


def to_sensors(
    attr: npt.NDArray[np.floating[Any]],
    feature_sensors: Sequence[str | None],
    schema: ChannelSchema = SIM_SCHEMA,
) -> npt.NDArray[np.float64]:
    """(B, features) attributions -> (B, sensors) signed sums."""
    out: npt.NDArray[np.float64] = np.asarray(attr, dtype=np.float64) @ sensor_matrix(
        feature_sensors, schema
    )
    return out


def normalise(w: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Divide by the sum of absolute weights (rows); an all-zero row stays zero."""
    a = np.asarray(w, dtype=np.float64)
    s = np.abs(a).sum(axis=-1, keepdims=True)
    out: npt.NDArray[np.float64] = np.divide(a, s, out=np.zeros_like(a), where=s > 0)
    return out


def to_parts(
    weights: npt.NDArray[np.floating[Any]] | Sequence[float],
    sensor_parts: Mapping[str, str],
    schema: ChannelSchema = SIM_SCHEMA,
) -> list[dict[str, Any]]:
    """One machine's sensor weights -> ``[{sensor, part, weight}]``, largest |weight| first.

    ``sensor_parts`` maps each sensor to a 3D part (``MachineConfig.sensor_parts``).
    """
    w = normalise(np.asarray(weights, dtype=np.float64)[None, :])[0]
    items = [
        {"sensor": s, "part": sensor_parts[s], "weight": float(w[i])}
        for i, s in enumerate(schema.sensors)
        if s in sensor_parts
    ]
    return sorted(items, key=lambda d: -abs(d["weight"]))
