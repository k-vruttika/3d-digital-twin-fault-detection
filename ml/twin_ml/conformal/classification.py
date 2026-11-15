"""Split conformal prediction sets for the health classes (pure numpy; ADR 0022).

Calibrated on clean validation windows with the *calibrated* probabilities, at level alpha:

* LAC (least ambiguous set-valued classifier, Sadinle et al. 2019): score ``1 - p_y``; the set
  holds every class with ``1 - p_k <= q``.
* APS (adaptive prediction sets, Romano et al. 2020), randomised: classes sorted by
  probability, score = mass ranked strictly above the true class + ``u * p_y`` with
  ``u ~ U(0, 1)``; the set holds every class whose (mass above + ``u * p_k``) is ``<= q``.
  Randomised APS can return an empty set; it counts as size 0 and not covered.
* Mondrian LAC (class-conditional): one LAC threshold per true class, from the validation
  windows of that class, so coverage holds per class rather than only on average.

``q`` is the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the calibration scores
(infinite when n is too small, so the set is every class). Validity assumes exchangeable
calibration and test data; windows of one run are dependent, so the guarantee is approximate
(runs are the exchangeable unit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

Probs = npt.NDArray[np.floating[Any]]
Sets = npt.NDArray[np.bool_]

# columns of a per-class set-sums array (classes x SUM_FIELDS)
SUM_FIELDS = ("n", "covered", "size", "singleton", "empty")


def conformal_quantile(scores: npt.NDArray[np.floating[Any]], alpha: float) -> float:
    n = int(scores.shape[0])
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(np.sort(np.asarray(scores, dtype=np.float64))[k - 1])


def lac_scores(probs: Probs, y: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    p = np.asarray(probs, dtype=np.float64)
    out: npt.NDArray[np.float64] = 1.0 - p[np.arange(y.shape[0]), y.astype(np.int64)]
    return out


def lac_sets(probs: Probs, q: float | npt.NDArray[np.float64]) -> Sets:
    """``q`` scalar (LAC) or per class (Mondrian LAC)."""
    thr = np.asarray(q, dtype=np.float64)
    out: Sets = (1.0 - np.asarray(probs, dtype=np.float64)) <= (
        thr if thr.ndim == 0 else thr[None, :]
    )
    return out


def _mass_above(probs: Probs) -> npt.NDArray[np.float64]:
    """For every (row, class): total probability of the classes ranked strictly before it
    (descending probability, ties broken by class index)."""
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=1)
    before_sorted = np.cumsum(sorted_p, axis=1) - sorted_p
    out = np.empty_like(p)
    np.put_along_axis(out, order, before_sorted, axis=1)
    return out


def aps_scores(
    probs: Probs, y: npt.NDArray[Any], u: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    p = np.asarray(probs, dtype=np.float64)
    rows = np.arange(y.shape[0])
    yi = y.astype(np.int64)
    out: npt.NDArray[np.float64] = _mass_above(p)[rows, yi] + u * p[rows, yi]
    return out


def aps_sets(probs: Probs, q: float, u: npt.NDArray[np.floating[Any]]) -> Sets:
    p = np.asarray(probs, dtype=np.float64)
    out: Sets = (_mass_above(p) + u[:, None] * p) <= q
    return out


def mondrian_quantiles(
    scores: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any], n_classes: int, alpha: float
) -> npt.NDArray[np.float64]:
    return np.array(
        [conformal_quantile(scores[y == k], alpha) for k in range(n_classes)], dtype=np.float64
    )


@dataclass
class ConformalClassifier:
    alpha: float
    methods: list[str]
    n_classes: int
    q: dict[str, Any] = field(default_factory=dict)  # method -> float or per-class array
    n_cal: int = 0

    def fit(
        self, probs: Probs, y: npt.NDArray[Any], u: npt.NDArray[np.floating[Any]]
    ) -> ConformalClassifier:
        y = y.astype(np.int64)
        self.n_cal = int(y.shape[0])
        lac = lac_scores(probs, y)
        for m in self.methods:
            if m == "lac":
                self.q[m] = conformal_quantile(lac, self.alpha)
            elif m == "aps":
                self.q[m] = conformal_quantile(aps_scores(probs, y, u), self.alpha)
            elif m == "mondrian_lac":
                self.q[m] = mondrian_quantiles(lac, y, self.n_classes, self.alpha)
            else:
                raise ValueError(f"unknown conformal method {m!r}")
        return self

    def sets(self, probs: Probs, u: npt.NDArray[np.floating[Any]]) -> dict[str, Sets]:
        out: dict[str, Sets] = {}
        for m in self.methods:
            out[m] = aps_sets(probs, self.q[m], u) if m == "aps" else lac_sets(probs, self.q[m])
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n_cal": self.n_cal,
            "q": {
                m: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for m, v in self.q.items()
            },
        }


def set_sums(sets: Sets, y: npt.NDArray[Any], n_classes: int) -> npt.NDArray[np.float64]:
    """(n_classes, len(SUM_FIELDS)) sums per *true* class: n, covered, size, singleton, empty."""
    yi = y.astype(np.int64)
    size = sets.sum(axis=1)
    covered = sets[np.arange(yi.shape[0]), yi] if yi.size else np.zeros(0, dtype=bool)
    out = np.zeros((n_classes, len(SUM_FIELDS)), dtype=np.float64)
    out[:, 0] = np.bincount(yi, minlength=n_classes)
    out[:, 1] = np.bincount(yi, weights=covered.astype(np.float64), minlength=n_classes)
    out[:, 2] = np.bincount(yi, weights=size.astype(np.float64), minlength=n_classes)
    out[:, 3] = np.bincount(yi, weights=(size == 1).astype(np.float64), minlength=n_classes)
    out[:, 4] = np.bincount(yi, weights=(size == 0).astype(np.float64), minlength=n_classes)
    return out
