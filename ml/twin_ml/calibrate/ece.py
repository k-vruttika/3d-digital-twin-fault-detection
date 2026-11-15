"""Expected calibration error from per-bin sufficient statistics (pure numpy).

A reliability bin holds (count, sum of confidences, sum of outcomes). Sums of several runs add
up, so the bootstrap can resample runs and recompute the ECE (``evaluate.bootstrap``).

* top-label: confidence = max probability, outcome = (argmax == true class);
* class-wise (e.g. FAULT): confidence = p(class), outcome = (true class == class).

``ECE = sum_b n_b / N * |mean outcome_b - mean confidence_b|`` over equal-width bins of [0, 1];
the top bin includes 1.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

COUNT, CONF, HIT = 0, 1, 2  # last axis of a bin-sums array


def bin_sums(
    confidence: npt.NDArray[np.floating[Any]], outcome: npt.NDArray[Any], n_bins: int
) -> npt.NDArray[np.float64]:
    """(n_bins, 3) array of (count, confidence sum, outcome sum)."""
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(outcome, dtype=np.float64)
    idx = np.clip((conf * n_bins).astype(np.int64), 0, n_bins - 1)
    out = np.zeros((n_bins, 3), dtype=np.float64)
    out[:, COUNT] = np.bincount(idx, minlength=n_bins)
    out[:, CONF] = np.bincount(idx, weights=conf, minlength=n_bins)
    out[:, HIT] = np.bincount(idx, weights=hit, minlength=n_bins)
    return out


def top_label_sums(
    probs: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any], n_bins: int
) -> npt.NDArray[np.float64]:
    pred = np.argmax(probs, axis=1)
    return bin_sums(probs.max(axis=1), pred == y, n_bins)


def class_sums(
    probs: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any], cls: int, n_bins: int
) -> npt.NDArray[np.float64]:
    return bin_sums(probs[:, cls], y == cls, n_bins)


def ece_from_sums(sums: npt.NDArray[np.float64]) -> float:
    """ECE of a (n_bins, 3) sums array; NaN without samples."""
    n = float(sums[:, COUNT].sum())
    if n == 0:
        return float("nan")
    return float(np.abs(sums[:, HIT] - sums[:, CONF]).sum() / n)


def reliability(sums: npt.NDArray[np.float64]) -> dict[str, list[float | None]]:
    """Per-bin mean confidence, mean outcome and count (None for empty bins), for plots."""
    count = sums[:, COUNT]
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(count > 0, sums[:, CONF] / count, np.nan)
        acc = np.where(count > 0, sums[:, HIT] / count, np.nan)
    return {
        "confidence": [None if not np.isfinite(v) else float(v) for v in conf],
        "accuracy": [None if not np.isfinite(v) else float(v) for v in acc],
        "count": [float(v) for v in count],
    }
