"""Evaluate test-split predictions: E1 metrics per horizon and subset, early warning,
time-to-fault, fault type, and the run-level bootstrap. No simulator import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.config import EvalConfig
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import (
    EPISODE_GRADUAL,
    EPISODE_JAM,
    FAULT,
    HEALTH,
    NORMAL,
    STATES,
    WARNING,
)
from twin_ml.evaluate.bootstrap import RunStats, bootstrap_ci
from twin_ml.evaluate.early_warning import (
    EpisodeResult,
    episode_lead,
    false_alarms,
    summarize,
    sustain_windows,
    sustained,
)
from twin_ml.evaluate.metrics import confusion, mae_uncensored, pr_auc, scores_from_confusion

N_CLASSES = len(HEALTH)


@dataclass
class Predictions:
    """Model outputs for every window of a split (in window order)."""

    probs: dict[int, npt.NDArray[np.float32]]  # horizon -> (N, 3)
    ttf: npt.NDArray[np.float32] | None = None  # (N,) seconds, None if not predicted
    fault_type: npt.NDArray[np.float32] | None = None  # (N, K) probabilities
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self, horizon: int) -> npt.NDArray[np.int64]:
        out: npt.NDArray[np.int64] = np.argmax(self.probs[horizon], axis=1)
        return out

    def to_arrays(self) -> dict[str, npt.NDArray[Any]]:
        out: dict[str, npt.NDArray[Any]] = {f"probs_h{h}": p for h, p in self.probs.items()}
        if self.ttf is not None:
            out["ttf"] = self.ttf
        if self.fault_type is not None:
            out["fault_type"] = self.fault_type
        return out


def _state_mask(split: SplitData, states: list[Any]) -> npt.NDArray[np.bool_]:
    codes = [STATES.index(s) for s in states]
    return np.isin(split.at_end("state"), codes)


def subset_masks(split: SplitData, cfg: EvalConfig) -> dict[str, npt.NDArray[np.bool_]]:
    """Evaluation subsets of the windows (docs/experiments.md).

    gradual / jam: operating windows whose next fault onset within the ttf cap is a gradual
    fault / a belt jam, so they measure warning *before* the fault (jams have no precursors).
    """
    episode = split.at_end("episode_type")
    operating = _state_mask(split, list(cfg.subsets.operating))
    return {
        "all": np.ones(split.n_windows, dtype=bool),
        "operating": operating,
        "stopped": _state_mask(split, list(cfg.subsets.stopped)),
        "gradual": operating & (episode == EPISODE_GRADUAL),
        "jam": operating & (episode == EPISODE_JAM),
    }


def classification_block(
    y: npt.NDArray[Any], probs: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    yt, pr = y[mask], probs[mask]
    scores = scores_from_confusion(confusion(yt, np.argmax(pr, axis=1), N_CLASSES))
    ap = pr_auc(yt, pr, [WARNING, FAULT])
    finite = [v for v in ap.values() if np.isfinite(v)]
    scores["fault_recall"] = scores["recall"][FAULT] if scores["support"][FAULT] else float("nan")
    scores["pr_auc"] = {HEALTH[c].value: v for c, v in ap.items()}
    scores["pr_auc_macro"] = float(np.mean(finite)) if finite else float("nan")
    return scores


def early_warning(
    split: SplitData,
    labels: npt.NDArray[np.int64],
    horizon: int,
    operating: npt.NDArray[np.bool_],
    no_fault_code: int,
    fault_types: list[str],
    sustain_s: int,
    n_runs: int,
    run_of_window: npt.NDArray[np.int64],
) -> tuple[list[EpisodeResult], RunStats]:
    """Per-episode lead times and per-run statistics for one horizon head."""
    k = sustain_windows(sustain_s, split.stride * split.step_s)
    alarm = labels >= WARNING
    ends = split.arrays["ends"]
    t_all = split.arrays["t_s"]
    series_at_end = split.arrays["series"][ends]
    negative = (
        operating
        & (split.at_end("fault_type") == no_fault_code)
        & (split.target(horizon) == NORMAL)
    )
    stats = RunStats.empty(n_runs, N_CLASSES)
    results: list[EpisodeResult] = []
    # windows are ordered by series, then time
    bounds = np.flatnonzero(np.diff(series_at_end)) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [series_at_end.shape[0]]])
    ep_series = split.arrays["ep_series"]
    for lo, hi in zip(starts, stops, strict=True):
        if hi <= lo:
            continue
        sid = int(series_at_end[lo])
        run = int(run_of_window[lo])
        t = t_all[ends[lo:hi]]
        sus = sustained(alarm[lo:hi], k)
        fa, neg_s = false_alarms(sus, negative[lo:hi], split.stride * split.step_s)
        stats.false_alarms[run] += fa
        stats.negative_s[run] += neg_s
        for e in np.flatnonzero(ep_series == sid):
            i_start, i_fault = int(split.arrays["ep_start"][e]), int(split.arrays["ep_fault"][e])
            i_warn = int(split.arrays["ep_warn"][e])
            t_warn = float(t_all[i_warn]) if i_warn >= 0 else None
            detected, lead, rel = episode_lead(
                t, sus, float(t_all[i_start]), float(t_all[i_fault]), t_warn
            )
            results.append(
                EpisodeResult(
                    run=run,
                    machine=str(split.arrays["series_machine"][sid]),
                    kind=split.schema.kinds[int(split.arrays["kind"][i_fault])],
                    fault_type=fault_types[int(split.arrays["ep_fault_type"][e])],
                    t_start=float(t_all[i_start]),
                    t_warn=t_warn,
                    t_fault=float(t_all[i_fault]),
                    detected=detected,
                    lead_s=lead,
                    rel_lead_s=rel,
                )
            )
            stats.episodes[run] += 1
            if detected and lead is not None:
                stats.detected[run] += 1
                stats.lead_sum[run] += lead
                if rel is not None:
                    stats.rel_sum[run] += rel
                    stats.rel_n[run] += 1
    return results, stats


def horizon_run_stats(
    split: SplitData,
    labels: npt.NDArray[np.int64],
    horizon: int,
    operating: npt.NDArray[np.bool_],
    fault_types: list[str],
    cfg: EvalConfig,
    run_of_window: npt.NDArray[np.int64] | None = None,
) -> tuple[list[EpisodeResult], RunStats]:
    """Episodes and per-run statistics (operating confusion, early warning) of one horizon.

    The per-run statistics feed the bootstrap and the paired comparisons.
    """
    rw = split.window_run() if run_of_window is None else run_of_window
    n_runs = len(split.run_ids)
    no_fault = fault_types.index("none") if "none" in fault_types else -1
    episodes, stats = early_warning(
        split,
        labels,
        horizon,
        operating,
        no_fault,
        fault_types,
        cfg.early_warning.sustain_s,
        n_runs,
        rw,
    )
    y = split.target(horizon)
    for r in range(n_runs):
        sel = operating & (rw == r)
        stats.confusion[r] = confusion(y[sel], labels[sel], N_CLASSES)
    return episodes, stats


def evaluate(
    split: SplitData,
    preds: Predictions,
    cfg: EvalConfig,
    fault_types: list[str],
    ttf_cap_s: float,
) -> dict[str, Any]:
    """All E1 metrics of one model on one split (nested dict, JSON-serialisable)."""
    masks = subset_masks(split, cfg)
    run_of_window = split.window_run()
    n_runs = len(split.run_ids)
    out: dict[str, Any] = {"n_windows": split.n_windows, "n_runs": n_runs, "horizons": {}}
    for h in split.horizons:
        y = split.target(h)
        labels = preds.label(h)
        block: dict[str, Any] = {
            name: classification_block(y, preds.probs[h], m) for name, m in masks.items()
        }
        episodes, stats = horizon_run_stats(
            split, labels, h, masks["operating"], fault_types, cfg, run_of_window
        )
        ew = summarize(episodes, int(stats.false_alarms.sum()), float(stats.negative_s.sum()))
        block["early_warning"] = ew
        block["episodes"] = [e.to_dict() for e in episodes]
        block["bootstrap"] = bootstrap_ci(
            stats, cfg.bootstrap.n_resamples, cfg.bootstrap.seed, cfg.bootstrap.ci
        )
        out["horizons"][str(h)] = block
    if preds.ttf is not None:
        ttf, cens = split.at_end("ttf"), split.at_end("ttf_censored")
        out["ttf"] = {}
        for name in ("all", "operating"):
            m = masks[name]
            mae, n = mae_uncensored(preds.ttf[m], ttf[m], cens[m], ttf_cap_s)
            out["ttf"][name] = {"mae_s": mae, "n": n}
    if preds.fault_type is not None:
        ft_true = split.at_end("fault_type").astype(np.int64)
        ft_pred = np.argmax(preds.fault_type, axis=1)
        out["fault_type"] = {}
        for name in ("all", "operating"):
            m = masks[name]
            sc = scores_from_confusion(confusion(ft_true[m], ft_pred[m], len(fault_types)))
            out["fault_type"][name] = {"macro_f1": sc["macro_f1"], "accuracy": sc["accuracy"]}
    return out


def flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    """MLflow metric keys, e.g. ``h30.operating.macro_f1``; NaN values are skipped."""
    out: dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        if isinstance(value, int | float) and np.isfinite(value):
            out[key] = float(value)

    for h, block in result["horizons"].items():
        for subset in ("all", "operating", "stopped", "gradual", "jam"):
            sc = block[subset]
            for m in ("accuracy", "macro_f1", "fault_recall", "pr_auc_macro"):
                put(f"h{h}.{subset}.{m}", sc[m])
            for c, name in enumerate(HEALTH):
                put(f"h{h}.{subset}.precision.{name.value}", sc["precision"][c])
                put(f"h{h}.{subset}.recall.{name.value}", sc["recall"][c])
                put(f"h{h}.{subset}.f1.{name.value}", sc["f1"][c])
        for m, v in block["early_warning"].items():
            if not isinstance(v, dict):
                put(f"h{h}.ew.{m}", v)
        for m, ci in block["bootstrap"].items():
            put(f"h{h}.ci.{m}.lo", ci["lo"])
            put(f"h{h}.ci.{m}.hi", ci["hi"])
    for name, v in result.get("ttf", {}).items():
        put(f"ttf.{name}.mae_s", v["mae_s"])
    for name, v in result.get("fault_type", {}).items():
        put(f"fault_type.{name}.macro_f1", v["macro_f1"])
    return out
