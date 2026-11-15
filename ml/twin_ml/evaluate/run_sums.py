"""Per-run sufficient statistics and run-level bootstraps for Phase 8 (ADR 0021, 0022).

A ``Sums`` dict maps a field name to an array whose first axis is the test run. Every metric
of the robustness report is a function of the *sums over runs* (confusion matrix, episode
lead sums, false alarms, reliability-bin sums, conformal set sums, interval sums), so a
bootstrap resample is a weighted sum of the runs' rows, as in ``evaluate.bootstrap``:

* :func:`bootstrap` percentile CI of every metric;
* :func:`paired` difference a - b on the *same* resampled runs (same test runs on both sides,
  e.g. clean vs perturbed, or clean-trained vs augmentation-trained);
* :func:`unpaired` difference with independent resampling of each side (different run sets,
  e.g. the E4 high-load runs vs the low + medium test runs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.calibrate.ece import ece_from_sums
from twin_ml.conformal.classification import SUM_FIELDS
from twin_ml.conformal.regression import TTF_FIELDS
from twin_ml.data.labels import FAULT, HEALTH
from twin_ml.evaluate.bootstrap import RunStats
from twin_ml.evaluate.metrics import scores_from_confusion

Sums = dict[str, npt.NDArray[np.float64]]
MetricFn = Callable[[dict[str, npt.NDArray[np.float64]]], dict[str, float]]

RUNSTATS_FIELDS = (
    "confusion",
    "episodes",
    "detected",
    "lead_sum",
    "rel_sum",
    "rel_n",
    "false_alarms",
    "negative_s",
)


def from_runstats(rs: RunStats) -> Sums:
    return {f: np.asarray(getattr(rs, f), dtype=np.float64) for f in RUNSTATS_FIELDS}


def n_runs(sums: Sums) -> int:
    return int(next(iter(sums.values())).shape[0])


def select(sums: Sums, idx: npt.NDArray[np.int64] | Sequence[int]) -> Sums:
    """Restrict to some runs (e.g. the test runs shared with another split)."""
    ix = np.asarray(idx, dtype=np.int64)
    return {k: v[ix] for k, v in sums.items()}


def _nan_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def robust_metrics(agg: dict[str, npt.NDArray[np.float64]]) -> dict[str, float]:
    """Every report metric from run-summed statistics (missing fields are skipped)."""
    out: dict[str, float] = {}
    if "confusion" in agg:
        scores = scores_from_confusion(np.rint(agg["confusion"]).astype(np.int64))
        ep, det = float(agg["episodes"]), float(agg["detected"])
        out |= {
            "macro_f1": scores["macro_f1"],
            "fault_recall": scores["recall"][FAULT] if scores["support"][FAULT] else float("nan"),
            "miss_rate": _nan_div(ep - det, ep),
            "lead_mean_s": _nan_div(float(agg["lead_sum"]), det),
            "rel_lead_mean_s": _nan_div(float(agg["rel_sum"]), float(agg["rel_n"])),
            "false_alarms_per_h": _nan_div(
                float(agg["false_alarms"]), float(agg["negative_s"]) / 3600.0
            ),
        }
    for key in ("ece_uncal", "ece_cal", "ece_fault_uncal", "ece_fault_cal"):
        if key in agg:
            out[key] = ece_from_sums(agg[key])
    n_i, cov_i, size_i, single_i, empty_i = (SUM_FIELDS.index(f) for f in SUM_FIELDS)
    for key, arr in agg.items():
        if not key.startswith("conf_"):
            continue
        m = key[len("conf_") :]
        n = float(arr[:, n_i].sum())
        out[f"cov_{m}"] = _nan_div(float(arr[:, cov_i].sum()), n)
        out[f"size_{m}"] = _nan_div(float(arr[:, size_i].sum()), n)
        out[f"singleton_{m}"] = _nan_div(float(arr[:, single_i].sum()), n)
        out[f"empty_{m}"] = _nan_div(float(arr[:, empty_i].sum()), n)
        for c, cls in enumerate(HEALTH):
            out[f"cov_{m}_{cls.value}"] = _nan_div(float(arr[c, cov_i]), float(arr[c, n_i]))
            out[f"size_{m}_{cls.value}"] = _nan_div(float(arr[c, size_i]), float(arr[c, n_i]))
    if "ttf" in agg:
        t = {f: float(agg["ttf"][i]) for i, f in enumerate(TTF_FIELDS)}
        out["ttf_cov"] = _nan_div(t["covered"], t["n_known"])
        out["ttf_width_s"] = _nan_div(t["width_known"], t["n_known"])
        out["ttf_certain_miss"] = _nan_div(t["certain_miss"], t["n_censored"])
        out["ttf_width_censored_s"] = _nan_div(t["width_censored"], t["n_censored"])
    if "sf_false_alarms" in agg:
        out["sf_false_alarms_per_h"] = _nan_div(
            float(agg["sf_false_alarms"]), float(agg["sf_negative_s"]) / 3600.0
        )
        out["sf_negative_h"] = float(agg["sf_negative_s"]) / 3600.0
    return out


def _aggregate_many(
    sums: Sums, counts: npt.NDArray[np.int64]
) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Resampled run sums for every row of ``counts`` (resamples x runs)."""
    c = counts.astype(np.float64)
    flat = {k: c @ v.reshape(v.shape[0], -1) for k, v in sums.items()}
    shapes = {k: v.shape[1:] for k, v in sums.items()}
    return [{k: flat[k][i].reshape(shapes[k]) for k in sums} for i in range(counts.shape[0])]


def _draws(n: int, n_resamples: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    out = np.zeros((n_resamples, n), dtype=np.int64)
    for i in range(n_resamples):
        out[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    return out


def _ci(values: list[float], ci: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))


def point(sums: Sums, metric_fn: MetricFn = robust_metrics) -> dict[str, float]:
    return metric_fn({k: v.sum(axis=0) for k, v in sums.items()})


def bootstrap(
    sums: Sums,
    n_resamples: int,
    seed: int,
    ci: float,
    metric_fn: MetricFn = robust_metrics,
) -> dict[str, dict[str, float]]:
    pt = point(sums, metric_fn)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in pt}
    for agg in _aggregate_many(sums, _draws(n_runs(sums), n_resamples, rng)):
        res = metric_fn(agg)
        for k in pt:
            draws[k].append(res.get(k, float("nan")))
    out = {}
    for k, v in pt.items():
        lo, hi = _ci(draws[k], ci)
        out[k] = {"point": v, "lo": lo, "hi": hi}
    return out


def _diff_result(
    pa: dict[str, float], pb: dict[str, float], diffs: dict[str, list[float]], ci: float
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m, values in diffs.items():
        lo, hi = _ci(values, ci)
        out[m] = {
            "point": pa.get(m, float("nan")) - pb.get(m, float("nan")),
            "lo": lo,
            "hi": hi,
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        }
    return out


def paired(
    a: Sums,
    b: Sums,
    n_resamples: int,
    seed: int,
    ci: float,
    metrics: Sequence[str] | None = None,
    metric_fn: MetricFn = robust_metrics,
) -> dict[str, dict[str, Any]]:
    """a - b with both sides on the same resampled runs (same run order required)."""
    if n_runs(a) != n_runs(b):
        raise ValueError("paired comparison needs the same test runs on both sides")
    pa, pb = point(a, metric_fn), point(b, metric_fn)
    names = list(metrics) if metrics is not None else sorted(set(pa) & set(pb))
    counts = _draws(n_runs(a), n_resamples, np.random.default_rng(seed))
    diffs: dict[str, list[float]] = {m: [] for m in names}
    for ga, gb in zip(_aggregate_many(a, counts), _aggregate_many(b, counts), strict=True):
        ra, rb = metric_fn(ga), metric_fn(gb)
        for m in names:
            diffs[m].append(ra.get(m, float("nan")) - rb.get(m, float("nan")))
    return _diff_result(pa, pb, diffs, ci)


def unpaired(
    a: Sums,
    b: Sums,
    n_resamples: int,
    seed: int,
    ci: float,
    metrics: Sequence[str] | None = None,
    metric_fn: MetricFn = robust_metrics,
) -> dict[str, dict[str, Any]]:
    """a - b with each side's runs resampled independently (different run sets)."""
    pa, pb = point(a, metric_fn), point(b, metric_fn)
    names = list(metrics) if metrics is not None else sorted(set(pa) & set(pb))
    rng = np.random.default_rng(seed)
    ca = _draws(n_runs(a), n_resamples, rng)
    cb = _draws(n_runs(b), n_resamples, rng)
    diffs: dict[str, list[float]] = {m: [] for m in names}
    for ga, gb in zip(_aggregate_many(a, ca), _aggregate_many(b, cb), strict=True):
        ra, rb = metric_fn(ga), metric_fn(gb)
        for m in names:
            diffs[m].append(ra.get(m, float("nan")) - rb.get(m, float("nan")))
    return _diff_result(pa, pb, diffs, ci)


def save_npz_dict(groups: dict[str, Sums]) -> dict[str, npt.NDArray[Any]]:
    """Flatten {group: sums} into npz keys ``group|field``."""
    return {f"{g}|{k}": v for g, sums in groups.items() for k, v in sums.items()}


def load_npz_dict(arrays: dict[str, npt.NDArray[Any]]) -> dict[str, Sums]:
    out: dict[str, Sums] = {}
    for key, v in arrays.items():
        if "|" not in key:
            continue
        g, f = key.rsplit("|", 1)
        out.setdefault(g, {})[f] = np.asarray(v, dtype=np.float64)
    return out
