"""Seeded bootstrap over test runs (whole runs are resampled with replacement).

Each run contributes sufficient statistics (confusion matrices, episode lead sums, false alarm
counts, negative time); a resample adds up the statistics of the drawn runs and recomputes the
metrics, so windows and episodes of one run always stay together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.data.labels import FAULT
from twin_ml.evaluate.metrics import scores_from_confusion


@dataclass
class RunStats:
    """Per-run statistics of one horizon; arrays are indexed by test run."""

    confusion: npt.NDArray[np.int64]  # (runs, 3, 3), operating subset
    episodes: npt.NDArray[np.int64]  # (runs,)
    detected: npt.NDArray[np.int64]
    lead_sum: npt.NDArray[np.float64]
    rel_sum: npt.NDArray[np.float64]
    rel_n: npt.NDArray[np.int64]
    false_alarms: npt.NDArray[np.int64]
    negative_s: npt.NDArray[np.float64]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, n_runs: int, n_classes: int) -> RunStats:
        return cls(
            confusion=np.zeros((n_runs, n_classes, n_classes), dtype=np.int64),
            episodes=np.zeros(n_runs, dtype=np.int64),
            detected=np.zeros(n_runs, dtype=np.int64),
            lead_sum=np.zeros(n_runs),
            rel_sum=np.zeros(n_runs),
            rel_n=np.zeros(n_runs, dtype=np.int64),
            false_alarms=np.zeros(n_runs, dtype=np.int64),
            negative_s=np.zeros(n_runs),
        )


def _metrics(s: RunStats, counts: npt.NDArray[np.int64]) -> dict[str, float]:
    """Metrics of a resample where run i is drawn counts[i] times."""
    cm = np.tensordot(counts, s.confusion, axes=1).astype(np.int64)
    scores = scores_from_confusion(cm)
    ep = float(counts @ s.episodes)
    det = float(counts @ s.detected)
    rel_n = float(counts @ s.rel_n)
    neg_h = float(counts @ s.negative_s) / 3600.0
    support_fault = scores["support"][FAULT]
    return {
        "macro_f1": scores["macro_f1"],
        "fault_recall": scores["recall"][FAULT] if support_fault else float("nan"),
        "miss_rate": (ep - det) / ep if ep else float("nan"),
        "lead_mean_s": float(counts @ s.lead_sum) / det if det else float("nan"),
        "rel_lead_mean_s": float(counts @ s.rel_sum) / rel_n if rel_n else float("nan"),
        "false_alarms_per_h": float(counts @ s.false_alarms) / neg_h if neg_h else float("nan"),
    }


def bootstrap_ci(
    stats: RunStats, n_resamples: int, seed: int, ci: float
) -> dict[str, dict[str, float]]:
    """Point estimate (all runs once) and percentile CI per metric."""
    n_runs = stats.episodes.shape[0]
    point = _metrics(stats, np.ones(n_runs, dtype=np.int64))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_resamples):
        counts = np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs)
        for k, v in _metrics(stats, counts).items():
            draws[k].append(v)
    alpha = (1.0 - ci) / 2.0
    out: dict[str, dict[str, float]] = {}
    for k, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        lo, hi = (
            (float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha)))
            if finite.size
            else (float("nan"), float("nan"))
        )
        out[k] = {"point": point[k], "lo": lo, "hi": hi}
    return out


PAIRED_METRICS = ("lead_mean_s", "rel_lead_mean_s", "macro_f1", "false_alarms_per_h")


def paired_bootstrap(
    a: RunStats,
    b: RunStats,
    n_resamples: int,
    seed: int,
    ci: float,
    metrics: tuple[str, ...] = PAIRED_METRICS,
) -> dict[str, dict[str, Any]]:
    """Difference a - b per metric, with both sides evaluated on the *same* resampled runs.

    Returns the point difference (all runs once), the percentile CI of the resampled
    differences and whether that CI excludes zero.
    """
    n_runs = a.episodes.shape[0]
    if b.episodes.shape[0] != n_runs:
        raise ValueError("paired comparison needs the same test runs on both sides")
    ones = np.ones(n_runs, dtype=np.int64)
    pa, pb = _metrics(a, ones), _metrics(b, ones)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        counts = np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs)
        ma, mb = _metrics(a, counts), _metrics(b, counts)
        for m in metrics:
            diffs[m].append(ma[m] - mb[m])
    alpha = (1.0 - ci) / 2.0
    out: dict[str, dict[str, Any]] = {}
    for m in metrics:
        arr = np.asarray(diffs[m], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo, hi = float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))
        else:
            lo = hi = float("nan")
        out[m] = {
            "point": pa[m] - pb[m],
            "lo": lo,
            "hi": hi,
            "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
        }
    return out
