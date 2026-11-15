"""Classification and regression metrics (numpy only, tested against hand-computed values).

Macro F1 averages the per-class F1 over the classes present in the ground truth of the
evaluated subset (a class with no support has no meaningful recall). Classes that are
predicted but absent still lower the precision of nothing and the recall of the true classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.integer[Any]]
FloatArray = npt.NDArray[np.floating[Any]]


def confusion(y_true: IntArray, y_pred: IntArray, n_classes: int) -> npt.NDArray[np.int64]:
    """Rows: true class, columns: predicted class."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _safe_div(num: npt.NDArray[Any], den: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    out = np.zeros(num.shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def scores_from_confusion(cm: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Accuracy, per-class precision / recall / F1 / support and macro F1."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    present = support > 0
    total = cm.sum()
    return {
        "n": int(total),
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
    }


def average_precision(y_true: npt.NDArray[np.bool_], score: FloatArray) -> float:
    """Area under the precision-recall curve as a step sum (scikit-learn's definition).

    AP = sum_n (R_n - R_{n-1}) P_n over the distinct score thresholds, high to low. NaN when
    there are no positives.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y_true[order].astype(np.float64)
    # last index of each run of equal scores = one threshold
    distinct = np.flatnonzero(np.diff(s)) if s.size > 1 else np.zeros(0, dtype=np.int64)
    last = np.concatenate([distinct, [s.size - 1]])
    tps = np.cumsum(y)[last]
    fps = (last + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def pr_auc(y_true: IntArray, probs: FloatArray, classes: list[int]) -> dict[int, float]:
    """One-vs-rest average precision for each class in ``classes``."""
    return {c: average_precision(y_true == c, probs[:, c]) for c in classes}


def mae_uncensored(
    pred: FloatArray, ttf: FloatArray, censored: npt.NDArray[np.bool_], cap_s: float
) -> tuple[float, int]:
    """MAE of time-to-fault on rows with a known target below the cap. Returns (mae, n).

    Censored rows (only a lower bound is known) and rows at the cap (no fault within the cap)
    are excluded, so the score measures how well the time to an actual fault is predicted.
    """
    keep = ~censored & (ttf < cap_s)
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(pred[keep] - ttf[keep]))), n


def percentiles_ms(seconds: list[float]) -> dict[str, float]:
    arr = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }
