"""Attributions for the tree models on summary features (Phase 9, ADR 0026).

* XGBoost: native TreeSHAP, ``Booster.predict(..., pred_contribs=True)``, on the margin
  (pre-softmax score) of the explained class. Contributions plus the bias equal that margin.
* Random Forest: Saabas-style decision-path contributions. Along each tree's path, the change
  in the explained class's leaf-value fraction from a node to its child is credited to the
  node's split feature; the root value is the bias. Averaged over trees, bias plus
  contributions equal the forest's predicted probability. These are **not** Shapley values:
  path attributions are known to over-credit features split near the root (ADR 0026), so the
  RF explanations are exploratory.

xgboost / scikit-learn are only touched through the fitted model objects (no imports here).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _present_index(clf: Any, cls: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Map health classes to the wrapped model's column (classes absent in training -> -1)."""
    enc = np.full(clf.k, -1, dtype=np.int64)
    enc[clf.present] = np.arange(clf.present.size)
    out: npt.NDArray[np.int64] = enc[cls]
    return out


def xgb_contributions(
    clf: Any, x: npt.NDArray[np.float32], cls: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """TreeSHAP of one XGBoost head (a ``trees._Classifier``) for class ``cls`` per row.

    Returns (contributions (B, F), bias (B,)); contributions.sum(1) + bias = class margin.
    """
    import xgboost as xgb

    model = clf.model
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    rng = (0, int(best) + 1) if best is not None else (0, 0)
    raw = np.asarray(
        booster.predict(xgb.DMatrix(x), pred_contribs=True, iteration_range=rng),
        dtype=np.float64,
    )
    col = _present_index(clf, cls)
    if (col < 0).any():
        raise ValueError("explained class was absent in training")
    rows = np.arange(x.shape[0])
    if raw.ndim == 2:  # binary head: one margin for the second present class
        sign = np.where(col == 1, 1.0, -1.0)[:, None]
        full = raw * sign
    else:
        full = raw[rows, col, :]
    return full[:, :-1], full[:, -1]


def xgb_margin(clf: Any, x: npt.NDArray[np.float32], cls: npt.NDArray[np.int64]) -> Any:
    """The explained class's margin (for the additivity check)."""
    import xgboost as xgb

    model = clf.model
    best = getattr(model, "best_iteration", None)
    rng = (0, int(best) + 1) if best is not None else (0, 0)
    m = np.asarray(
        model.get_booster().predict(xgb.DMatrix(x), output_margin=True, iteration_range=rng),
        dtype=np.float64,
    )
    col = _present_index(clf, cls)
    if m.ndim == 1:
        return np.where(col == 1, m, -m)
    return m[np.arange(x.shape[0]), col]


def saabas_contributions(
    clf: Any, x: npt.NDArray[np.float32], cls: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decision-path contributions of a random forest head for class ``cls`` per row.

    Returns (contributions (B, F), bias (B,)); contributions.sum(1) + bias = predicted
    probability of that class (sklearn averages the trees' leaf class fractions).
    """
    forest = clf.model
    col = _present_index(clf, cls)
    if (col < 0).any():
        raise ValueError("explained class was absent in training")
    n, f = x.shape
    contrib = np.zeros((n, f))
    bias = np.zeros(n)
    xx = np.asarray(x, dtype=np.float32)
    for est in forest.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-300)  # class fractions
        parent = np.full(t.node_count, -1, dtype=np.int64)
        for side in (t.children_left, t.children_right):
            ok = side >= 0
            parent[side[ok]] = np.flatnonzero(ok)
        has_parent = parent >= 0
        # edge into node j (j != root): delta of every class, credited to the parent's feature
        delta = np.zeros_like(v)
        delta[has_parent] = v[has_parent] - v[parent[has_parent]]
        feat = np.where(has_parent, t.feature[np.maximum(parent, 0)], 0)
        path = est.decision_path(xx).tocsr()  # (n, nodes) indicator
        # per class: sum of deltas along the path, scattered to features
        for c in np.unique(col):
            sel = col == c
            m = np.zeros((t.node_count, f))
            m[np.flatnonzero(has_parent), feat[has_parent]] = delta[has_parent, c]
            contrib[sel] += np.asarray(path[sel] @ m)
            bias[sel] += v[0, c]
    k = len(forest.estimators_)
    return contrib / k, bias / k
