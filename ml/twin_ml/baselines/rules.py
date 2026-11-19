"""Rule baselines for the real-data studies (Phase 10, ADR 0027). Pure numpy, no training.

Every rule returns one boolean alarm per window of a split; the caller sustains it exactly like
a model's alarm. Window means use the *observed* raw values only (the missing mask and NaN raw
values are skipped), so a forward-filled stretch never looks like a constant reading.

* :class:`ThresholdRule`: any sensor's window mean outside ``[p_low, p_high]`` of the training
  normal-operation window means, per machine kind and sensor;
* :func:`zscore_alarm`: ``z = (window mean - trailing mean) / trailing std`` over the
  ``trailing`` rows before the window, alarm if ``max |z|`` over sensors exceeds ``z``;
* :func:`persistence_alarm`: the machine is in FAULT now (the only observable state; the
  WARNING band is a label, not something a machine reports);
* :func:`error_alarm`: an error-log entry (any code, or one code) inside the window
  (exploratory, Azure error-count channels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import FAULT_STATE

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ANY_CODE = "any"


def _observed_values(
    split: SplitData, cols: list[int] | None = None
) -> tuple[FloatArray, BoolArray]:
    """Raw values (NaN where missing) and the observed flags of the chosen sensor columns."""
    a = split.arrays
    raw = a["raw"].astype(np.float64)
    obs = ~a["mask"].astype(bool) & np.isfinite(raw)
    if cols is not None:
        raw, obs = raw[:, cols], obs[:, cols]
    return np.where(obs, raw, 0.0), obs


def _cumsum0(x: npt.NDArray[Any]) -> FloatArray:
    out = np.zeros((x.shape[0] + 1, *x.shape[1:]), dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def window_means(split: SplitData) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """(N, sensors) mean of the observed raw values in each window, and their counts.

    A sensor without an observed value in a window has mean NaN and count 0.
    """
    x, obs = _observed_values(split)
    ends = split.arrays["ends"]
    cx, cn = _cumsum0(x), _cumsum0(obs.astype(np.float64))
    lo, hi = ends + 1 - split.window, ends + 1  # windows never cross series
    s, n = cx[hi] - cx[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return mean, n.astype(np.int64)


@dataclass
class ThresholdRule:
    """Per-kind, per-sensor bounds from the training normal-operation window means."""

    low_pct: float
    high_pct: float
    kinds: tuple[str, ...] = ()
    low: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    high: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    n_fit: list[int] = field(default_factory=list)

    def fit(self, train: SplitData, normal_idx: npt.NDArray[np.int64]) -> ThresholdRule:
        mean, _ = window_means(train)
        kind = train.at_end("kind")
        self.kinds = train.schema.kinds
        n_s = len(train.schema.sensors)
        self.low = np.full((len(self.kinds), n_s), -np.inf)
        self.high = np.full((len(self.kinds), n_s), np.inf)
        self.n_fit = []
        for k in range(len(self.kinds)):
            sel = normal_idx[kind[normal_idx] == k]
            self.n_fit.append(int(sel.size))
            if not sel.size:
                continue
            m = mean[sel]
            for j in range(n_s):
                col = m[:, j][np.isfinite(m[:, j])]
                if col.size:
                    self.low[k, j] = np.percentile(col, self.low_pct)
                    self.high[k, j] = np.percentile(col, self.high_pct)
        return self

    def alarm(self, split: SplitData) -> BoolArray:
        mean, n = window_means(split)
        k = split.at_end("kind").astype(np.int64)
        lo, hi = self.low[k], self.high[k]
        with np.errstate(invalid="ignore"):
            out = np.asarray((n > 0) & ((mean < lo) | (mean > hi)), dtype=bool)
        result: BoolArray = np.asarray(out.any(axis=1), dtype=bool)
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "low_pct": self.low_pct,
            "high_pct": self.high_pct,
            "kinds": list(self.kinds),
            "n_fit": self.n_fit,
            "low": [[None if not np.isfinite(v) else float(v) for v in r] for r in self.low],
            "high": [[None if not np.isfinite(v) else float(v) for v in r] for r in self.high],
        }


def series_first_row(split: SplitData) -> npt.NDArray[np.int64]:
    """First row index of the series of each window (rows are ordered by series)."""
    series = split.arrays["series"]
    starts = np.flatnonzero(np.concatenate([[True], series[1:] != series[:-1]]))
    first = np.zeros(int(series.max()) + 1 if series.size else 0, dtype=np.int64)
    first[series[starts]] = starts
    out: npt.NDArray[np.int64] = first[split.at_end("series")]
    return out


def zscores(
    split: SplitData, trailing: int, min_std: float, min_trailing: int | None = None
) -> FloatArray:
    """(N, sensors) z of each window mean against the trailing rows before the window.

    The trailing stretch is the ``trailing`` rows before the window's first row, clipped at the
    start of the window's series (a series never borrows rows across a gap or a split cut).
    A sensor needs at least ``min_trailing`` observed trailing rows (default: one window) and
    one observed value in the window; otherwise its z is NaN (no alarm from that sensor).
    """
    need = split.window if min_trailing is None else min_trailing
    x, obs = _observed_values(split)
    ends = split.arrays["ends"]
    cx, cx2, cn = _cumsum0(x), _cumsum0(x * x), _cumsum0(obs.astype(np.float64))
    w_lo, w_hi = ends + 1 - split.window, ends + 1
    t_hi = w_lo
    t_lo = np.maximum(series_first_row(split), t_hi - trailing)
    wn = cn[w_hi] - cn[w_lo]
    tn = cn[t_hi] - cn[t_lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        w_mean = (cx[w_hi] - cx[w_lo]) / np.maximum(wn, 1)
        t_mean = (cx[t_hi] - cx[t_lo]) / np.maximum(tn, 1)
        t_var = (cx2[t_hi] - cx2[t_lo]) / np.maximum(tn, 1) - t_mean**2
        std = np.maximum(np.sqrt(np.maximum(t_var, 0.0)), min_std)
        z = (w_mean - t_mean) / std
    ok = (wn > 0) & (tn >= need)
    out: FloatArray = np.where(ok, z, np.nan)
    return out


def zscore_alarm(
    split: SplitData, trailing: int, z: float, min_std: float, min_trailing: int | None = None
) -> BoolArray:
    zs = np.abs(zscores(split, trailing, min_std, min_trailing))
    with np.errstate(invalid="ignore"):
        out: BoolArray = np.nanmax(np.where(np.isfinite(zs), zs, -np.inf), axis=1) > z
    return out


def persistence_alarm(split: SplitData) -> BoolArray:
    """FAULT now -> FAULT (observable state only)."""
    out: BoolArray = split.at_end("state") == FAULT_STATE
    return out


def error_alarm(split: SplitData, code: str, error_codes: list[str]) -> BoolArray:
    """An error-log entry inside the window: ``code`` is ``any`` or one of ``error_codes``
    (which must be sensor channels of the split, i.e. a dataset built with error channels)."""
    sensors = list(split.schema.sensors)
    missing = [c for c in error_codes if c not in sensors]
    if missing:
        raise ValueError(f"the split has no error-count channels {missing}")
    cols = [sensors.index(c) for c in (error_codes if code == ANY_CODE else [code])]
    x, _ = _observed_values(split, cols)
    c = _cumsum0(x.sum(axis=1))
    ends = split.arrays["ends"]
    out: BoolArray = (c[ends + 1] - c[ends + 1 - split.window]) > 0
    return out
