"""The only door from raw readings into model inputs (ADR 0019).

Models see the five sensor values (with forward fill and missing masks), the engineered
``vib_norm`` channel and the machine kind. Nothing else: not the state, not any label column,
not ``run_time``. :func:`select_inputs` reads only :data:`ALLOWED_INPUT_COLUMNS`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind

# Raw columns a model input may be computed from. The machine kind comes from the line config.
ALLOWED_INPUT_COLUMNS: tuple[str, ...] = ("speed", "temp", "vib", "load", "current")
SENSORS = ALLOWED_INPUT_COLUMNS
VALUE_CHANNELS: tuple[str, ...] = (*SENSORS, "vib_norm")
MASK_CHANNELS: tuple[str, ...] = tuple(f"{s}_missing" for s in SENSORS)
KINDS: tuple[MachineKind, ...] = tuple(MachineKind)
KIND_CHANNELS: tuple[str, ...] = tuple(f"kind_{k.value}" for k in KINDS)
# Channels of a raw window as the sequence models see it (time x channel).
SEQUENCE_CHANNELS: tuple[str, ...] = (*VALUE_CHANNELS, *MASK_CHANNELS, *KIND_CHANNELS)

_SPEED, _VIB = SENSORS.index("speed"), SENSORS.index("vib")

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.uint8]


class ForbiddenInputError(ValueError):
    """A column outside ALLOWED_INPUT_COLUMNS was requested as a model input."""


def select_inputs(columns: Mapping[str, npt.ArrayLike]) -> FloatArray:
    """Stack the allowed sensor columns (rows x 5, float32, NaN = missing).

    Only the allowed names are looked up; any other column in ``columns`` is never read.
    """
    missing = [c for c in ALLOWED_INPUT_COLUMNS if c not in columns]
    if missing:
        raise ForbiddenInputError(f"input columns missing: {missing}")
    return np.stack(
        [np.asarray(columns[c], dtype=np.float32) for c in ALLOWED_INPUT_COLUMNS], axis=1
    )


def check_input_names(names: list[str] | tuple[str, ...]) -> None:
    bad = [n for n in names if n not in ALLOWED_INPUT_COLUMNS]
    if bad:
        raise ForbiddenInputError(f"not allowed as model inputs: {bad}")


def forward_fill(raw: FloatArray, fallback: FloatArray) -> tuple[FloatArray, MaskArray]:
    """Forward-fill NaNs per column of one time-ordered series.

    Leading NaNs (nothing to carry forward) take ``fallback`` (the training mean). Returns the
    filled values and the missing mask (1 = the raw value was missing).
    """
    if raw.ndim != 2 or raw.shape[1] != fallback.shape[0]:
        raise ValueError("raw must be (T, C) with C == len(fallback)")
    missing = np.isnan(raw)
    n = raw.shape[0]
    idx = np.where(missing, 0, np.arange(n)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = raw[idx, np.arange(raw.shape[1])[None, :]]
    # Rows before the first observed value of a column still hold NaN.
    still = np.isnan(filled)
    if still.any():
        filled = np.where(still, fallback[None, :], filled)
    return filled.astype(np.float32), missing.astype(np.uint8)


def vib_norm(sensors: FloatArray, speed_ref: float, min_speed_ratio: float) -> FloatArray:
    """Vibration normalised by the speed ratio: vib / max(speed / speed_ref, min_ratio).

    Bearing-wear vibration scales with speed and imbalance with speed squared, so this ratio
    separates them when the CNC changes speed between jobs (docs/simulator.md).
    """
    ratio = np.maximum(sensors[:, _SPEED] / np.float32(speed_ref), np.float32(min_speed_ratio))
    out: FloatArray = (sensors[:, _VIB] / ratio).astype(np.float32)
    return out


def kind_onehot(kind: MachineKind) -> FloatArray:
    out = np.zeros(len(KINDS), dtype=np.float32)
    out[KINDS.index(kind)] = 1.0
    return out
