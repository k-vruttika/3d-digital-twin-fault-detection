"""One-vs-rest isotonic and Platt calibration of class probabilities (tree models).

Each class column is mapped by its own monotone function fitted on clean validation data
(class k vs rest), then the row is renormalised to sum to 1. Unlike temperature scaling this
can change the argmax; alarms therefore keep using the uncalibrated argmax (ADR 0022).

* isotonic: ``IsotonicRegression`` (clipped to [0, 1] outside the fitted range);
* Platt: logistic regression on ``logit(p_k)`` (effectively unregularised).

The method per model and horizon is chosen by the mean NLL over run-grouped folds of the
validation split (:func:`select_method`), so the choice uses validation data only.
scikit-learn is imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

EPS = 1e-6  # probability clip for logits and the NLL
PLATT_C = 1e6  # inverse regularisation strength: effectively unregularised Platt scaling

Probs = npt.NDArray[np.float64]


def _logit(p: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    out: npt.NDArray[np.float64] = np.log(q / (1.0 - q))
    return out


def mean_nll(probs: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any]) -> float:
    if y.shape[0] == 0:
        return float("nan")
    p = np.clip(
        np.asarray(probs, dtype=np.float64)[np.arange(y.shape[0]), y.astype(np.int64)], EPS, 1
    )
    return float(-np.log(p).mean())


class OvRCalibrator:
    def __init__(self, method: str) -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.maps: dict[int, Any] = {}

    def fit(self, probs: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any]) -> OvRCalibrator:
        y = y.astype(np.int64)
        self.maps = {}
        for k in range(probs.shape[1]):
            target = (y == k).astype(np.float64)
            if target.min() == target.max():  # class absent (or only class): keep the column
                continue
            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                m: Any = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                m.fit(np.asarray(probs[:, k], dtype=np.float64), target)
            else:
                from sklearn.linear_model import LogisticRegression

                m = LogisticRegression(C=PLATT_C)
                m.fit(_logit(probs[:, k])[:, None], target.astype(np.int64))
            self.maps[k] = m
        return self

    def transform(self, probs: npt.NDArray[np.floating[Any]]) -> Probs:
        p = np.asarray(probs, dtype=np.float64)
        out = p.copy()
        for k, m in self.maps.items():
            if self.method == "isotonic":
                out[:, k] = m.predict(p[:, k])
            else:
                out[:, k] = m.predict_proba(_logit(p[:, k])[:, None])[:, 1]
        total = out.sum(axis=1, keepdims=True)
        bad = ~(total[:, 0] > 0)
        out = np.where(total > 0, out / np.where(total > 0, total, 1.0), p)
        if bad.any():  # every column mapped to 0: keep the uncalibrated row
            out[bad] = p[bad]
        return out


def group_folds(groups: npt.NDArray[Any], n_folds: int) -> list[npt.NDArray[np.int64]]:
    """Deterministic run-grouped folds: sorted unique groups dealt round-robin."""
    uniq = np.unique(groups)
    k = max(2, min(n_folds, uniq.shape[0]))
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    assign = np.array([fold_of[g] for g in groups], dtype=np.int64)
    return [np.flatnonzero(assign == i) for i in range(k)]


def select_method(
    probs: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    methods: Sequence[str],
    n_folds: int,
) -> tuple[str, dict[str, float]]:
    """The method with the lowest mean held-out NLL over run-grouped folds (ties: first)."""
    folds = group_folds(groups, n_folds)
    scores: dict[str, float] = {}
    for method in methods:
        losses = []
        for held in folds:
            train = np.setdiff1d(np.arange(y.shape[0]), held)
            if held.size == 0 or train.size == 0:
                continue
            cal = OvRCalibrator(method).fit(probs[train], y[train])
            losses.append(mean_nll(cal.transform(probs[held]), y[held]))
        scores[method] = float(np.mean(losses)) if losses else float("inf")
    best = min(methods, key=lambda m: (scores[m], list(methods).index(m)))
    return best, scores
