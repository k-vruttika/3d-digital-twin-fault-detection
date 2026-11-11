"""Label computation (vectorised, per machine)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def time_to_fault(
    t_s: FloatArray, is_fault: BoolArray, cap_s: float
) -> tuple[FloatArray, BoolArray]:
    """Time until the next transition into FAULT for one machine's time-ordered rows.

    * rows in FAULT: 0, not censored;
    * rows with a FAULT onset ahead: ``min(t_onset - t, cap)``, not censored;
    * rows with no FAULT onset ahead in the run: ``min(t_end - t, cap)`` (observed lower
      bound, survival-analysis convention) and ``censored = True``.
    """
    if t_s.shape != is_fault.shape:
        raise ValueError("t_s and is_fault must have the same shape")
    n = t_s.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    prev = np.concatenate(([False], is_fault[:-1]))
    onset_t = t_s[is_fault & ~prev]
    idx = np.searchsorted(onset_t, t_s, side="left")
    has_next = idx < onset_t.shape[0]
    if onset_t.shape[0]:
        next_t = onset_t[np.minimum(idx, onset_t.shape[0] - 1)]
    else:
        next_t = np.full(n, np.inf)
    ahead = np.where(has_next, next_t - t_s, t_s[-1] - t_s)
    ttf = np.where(is_fault, 0.0, np.minimum(ahead, cap_s))
    censored = ~is_fault & ~has_next
    return ttf.astype(np.float64), censored
