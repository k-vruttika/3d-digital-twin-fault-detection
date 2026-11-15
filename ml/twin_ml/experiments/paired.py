"""Paired bootstrap comparisons for the E1 report (from the runs' stored test predictions).

Both sides of a comparison are evaluated on the *same* resampled test runs, so run-to-run
variation that affects both models cancels out. A model with several seeds is compared seed by
seed (seed i vs seed i, or each seed vs the deterministic baseline). The report shows the mean
over seeds of the point difference and of the CI bounds, whether that averaged CI excludes zero,
and in how many seeds the individual CI excludes zero.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import EvalConfig
from twin_ml.data.dataset import Dataset, SplitData
from twin_ml.evaluate.bootstrap import PAIRED_METRICS, RunStats, paired_bootstrap
from twin_ml.evaluate.evaluator import horizon_run_stats, subset_masks

REFERENCE = "threshold"


def _load_labels(artifact_uri: str, horizons: list[str]) -> dict[str, np.ndarray]:
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{artifact_uri}/test_predictions.npz", dst_path=tmp
        )
        with np.load(Path(path)) as npz:
            return {h: np.argmax(npz[f"probs_h{h}"], axis=1) for h in horizons}


def run_stats(
    runs: dict[str, dict[int, dict[str, Any]]],
    test: SplitData,
    fault_types: list[str],
    cfg: EvalConfig,
) -> dict[str, dict[int, dict[str, RunStats]]]:
    """Per model, seed and horizon: the per-run statistics rebuilt from stored predictions."""
    horizons = [str(h) for h in test.horizons]
    operating = subset_masks(test, cfg)["operating"]
    rw = test.window_run()
    out: dict[str, dict[int, dict[str, RunStats]]] = {}
    for model, seeds in runs.items():
        out[model] = {}
        for seed, entry in seeds.items():
            labels = _load_labels(entry["artifact_uri"], horizons)
            out[model][seed] = {
                h: horizon_run_stats(test, labels[h], int(h), operating, fault_types, cfg, rw)[1]
                for h in horizons
            }
    return out


def _compare(
    a: dict[int, RunStats], b: dict[int, RunStats], cfg: EvalConfig
) -> dict[str, dict[str, Any]]:
    """Seed-wise paired differences a - b (a deterministic side is reused for every seed)."""
    pairs: list[tuple[RunStats, RunStats]]
    if len(b) == 1:
        (sb,) = b.values()
        pairs = [(sa, sb) for sa in a.values()]
    elif len(a) == 1:
        (sa,) = a.values()
        pairs = [(sa, sb) for sb in b.values()]
    else:
        common = sorted(set(a) & set(b))
        pairs = [(a[s], b[s]) for s in common]
    per_seed = [
        paired_bootstrap(x, y, cfg.bootstrap.n_resamples, cfg.bootstrap.seed, cfg.bootstrap.ci)
        for x, y in pairs
    ]
    out: dict[str, dict[str, Any]] = {}
    for m in PAIRED_METRICS:
        vals = {k: [r[m][k] for r in per_seed] for k in ("point", "lo", "hi")}
        mean = {k: _finite_mean(v) for k, v in vals.items()}
        lo, hi = mean["lo"], mean["hi"]
        out[m] = {
            **mean,
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
            "seeds_excluding_zero": int(sum(r[m]["excludes_zero"] for r in per_seed)),
            "seeds": len(per_seed),
        }
    return out


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def best_of(summaries: list[dict[str, Any]], family: str, horizon: str, key: str) -> str | None:
    """Model of ``family`` with the highest mean of metric ``key`` at ``horizon``."""
    best, best_v = None, -np.inf
    for s in summaries:
        if s["family"] != family:
            continue
        st = s["metrics"].get(key.format(h=horizon))
        if st and st["n"] and np.isfinite(st["mean"]) and st["mean"] > best_v:
            best, best_v = s["model"], st["mean"]
    return best


SELECTION_METRIC = "h{h}.ew.lead_mean_s"  # primary result: mean lead time
FALLBACK_METRIC = "h{h}.operating.macro_f1"  # when no model of a family detected an episode


def paired_comparisons(
    runs: dict[str, dict[int, dict[str, Any]]],
    summaries: list[dict[str, Any]],
    ds: Dataset,
    cfg: EvalConfig,
) -> dict[str, Any]:
    test = ds.split("test")
    horizons = [str(h) for h in test.horizons]
    stats = run_stats(runs, test, ds.fault_types, cfg)
    out: dict[str, Any] = {
        "metrics": list(PAIRED_METRICS),
        "reference": REFERENCE,
        "selection_metric": SELECTION_METRIC,
        "vs_reference": {},
        "tree_vs_sequence": {},
    }
    for h in horizons:
        if REFERENCE in stats:
            ref = {s: v[h] for s, v in stats[REFERENCE].items()}
            out["vs_reference"][h] = {
                m: _compare({s: v[h] for s, v in stats[m].items()}, ref, cfg)
                for m in [x["model"] for x in summaries]
                if m != REFERENCE
            }
        selected_by = SELECTION_METRIC
        tree = best_of(summaries, "tree", h, SELECTION_METRIC)
        seq = best_of(summaries, "nn", h, SELECTION_METRIC)
        if not (tree and seq):
            selected_by = FALLBACK_METRIC
            tree = best_of(summaries, "tree", h, FALLBACK_METRIC)
            seq = best_of(summaries, "nn", h, FALLBACK_METRIC)
        if tree and seq:
            out["tree_vs_sequence"][h] = {
                "tree": tree,
                "sequence": seq,
                "selected_by": selected_by.format(h=h),
                "diff": _compare(
                    {s: v[h] for s, v in stats[seq].items()},
                    {s: v[h] for s, v in stats[tree].items()},
                    cfg,
                ),
            }
    return out
