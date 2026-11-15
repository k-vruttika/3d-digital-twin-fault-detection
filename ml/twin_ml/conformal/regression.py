"""Split conformal intervals for the time to fault (pure numpy; ADR 0022).

Score: absolute residual ``|y - y_hat|`` on clean validation windows whose target is *known*
and below the cap (a FAULT onset lies within the cap: the rows where a time to fault exists).
The interval is ``[y_hat - q, y_hat + q]`` clipped to ``[0, cap]``.

Censoring: on a right-censored row only ``true >= y`` is known, so coverage cannot be
determined. Such rows never enter the calibration. On test data they are reported separately:
the *certain-miss rate* (the interval ends below the known lower bound, ``hi < y``) and the
interval width. Rows with a known target equal to the cap (no fault within the cap) are left
out of both, because the regression target is capped there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.conformal.classification import conformal_quantile

# fields of the per-run time-to-fault interval sums
TTF_FIELDS = ("n_known", "covered", "width_known", "n_censored", "certain_miss", "width_censored")


def calibration_rows(
    ttf: npt.NDArray[np.floating[Any]], censored: npt.NDArray[np.bool_], cap_s: float
) -> npt.NDArray[np.bool_]:
    out: npt.NDArray[np.bool_] = ~censored & (ttf < cap_s)
    return out


def fit_ttf_quantile(
    pred: npt.NDArray[np.floating[Any]],
    ttf: npt.NDArray[np.floating[Any]],
    censored: npt.NDArray[np.bool_],
    cap_s: float,
    alpha: float,
) -> tuple[float, int]:
    rows = calibration_rows(ttf, censored, cap_s)
    resid = np.abs(
        np.asarray(ttf, dtype=np.float64)[rows] - np.asarray(pred, dtype=np.float64)[rows]
    )
    return conformal_quantile(resid, alpha), int(rows.sum())


def interval(
    pred: npt.NDArray[np.floating[Any]], q: float, cap_s: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p = np.asarray(pred, dtype=np.float64)
    return np.clip(p - q, 0.0, cap_s), np.clip(p + q, 0.0, cap_s)


def interval_sums(
    pred: npt.NDArray[np.floating[Any]],
    ttf: npt.NDArray[np.floating[Any]],
    censored: npt.NDArray[np.bool_],
    q: float,
    cap_s: float,
) -> npt.NDArray[np.float64]:
    """Sums over rows in TTF_FIELDS order."""
    lo, hi = interval(pred, q, cap_s)
    y = np.asarray(ttf, dtype=np.float64)
    known = calibration_rows(ttf, censored, cap_s)
    cens = np.asarray(censored, dtype=bool)
    width = hi - lo
    return np.array(
        [
            known.sum(),
            ((y >= lo) & (y <= hi) & known).sum(),
            width[known].sum(),
            cens.sum(),
            ((hi < y) & cens).sum(),
            width[cens].sum(),
        ],
        dtype=np.float64,
    )
