"""Sliding windows over one machine series, and the training-only subsample (ADR 0019)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from twin_ml.data.labels import NORMAL

IntArray = npt.NDArray[np.int64]


def window_ends(n_rows: int, window: int, max_horizon: int, stride: int) -> IntArray:
    """Row indices (within one series) where a window ends.

    A window covers rows ``end - window + 1 .. end``; its targets look ``h`` rows ahead, so the
    last end leaves ``max_horizon`` rows. Ends are ``window - 1, window - 1 + stride, ...``.
    """
    if window < 1 or stride < 1 or max_horizon < 0:
        raise ValueError("window and stride must be >= 1, max_horizon >= 0")
    first, last = window - 1, n_rows - 1 - max_horizon
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


def far_normal(
    health_now: npt.NDArray[np.int8],
    health_ahead: list[npt.NDArray[np.int8]],
    fault_active: npt.NDArray[np.bool_],
    fault_within_cap: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Windows that are NORMAL now and at every horizon, far from any fault.

    Far = no active (hidden) fault and no FAULT onset within the time-to-fault cap.
    """
    ok = (health_now == NORMAL) & ~fault_active & ~fault_within_cap
    for h in health_ahead:
        ok &= h == NORMAL
    result: npt.NDArray[np.bool_] = ok
    return result


def subsample_far_normal(
    far: npt.NDArray[np.bool_], keep_frac: float, rng: np.random.Generator
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """Keep every window that is not far-normal, and each far-normal one with ``keep_frac``.

    Returns the keep mask and the sample weight of every window: ``1 / keep_frac`` for kept
    far-normal windows, 1 otherwise. The weighted class distribution of the kept windows then
    equals the original one in expectation, so class weights are not applied twice (ADR 0019).
    """
    if not 0 < keep_frac <= 1:
        raise ValueError("keep_frac must be in (0, 1]")
    if keep_frac == 1:
        return np.ones(far.shape, dtype=bool), np.ones(far.shape, dtype=np.float32)
    draw = rng.random(far.shape[0])
    keep = ~far | (draw < keep_frac)
    weight = np.where(far, np.float32(1.0 / keep_frac), np.float32(1.0)).astype(np.float32)
    return keep, weight
