"""Phase 9 evaluation pieces (no MLflow, no simulator, no ML library; ADR 0024, 0025).

* :func:`sustained_alarm`: the per-window sustained alarm, exactly as the early-warning code
  forms it (per series, ``sustain_s`` of consecutive alarmed windows);
* :func:`alarm_run_sums`: early-warning per-run sums of a *precomputed* sustained alarm, so
  that the combined policy (supervised sustained alarm OR detector sustained alarm) uses the
  same lead / miss / false-alarm code without sustaining a second time;
* :func:`detector_condition`: every metric of an anomaly detector on one test condition, per
  threshold percentile;
* :func:`supervised_extras`: the exploratory confidence-of-wrong-predictions sums and the
  conformal set sums on precursor windows, added to a supervised model's Phase 8 sums;
* :func:`drift_flag_sums`: per run, sensor-fault segments, those flagged by a sustained alarm
  on a negative window inside the segment, and the time to the first such alarm;
* :func:`pack_alarm` / :func:`unpack_alarm`: compact alarm artifacts for the combined policy.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.config import EvalConfig
from twin_ml.conformal.classification import set_sums
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import EPISODE_GRADUAL, FAULT, HEALTH, NORMAL, WARNING
from twin_ml.evaluate import run_sums
from twin_ml.evaluate.early_warning import sustain_windows, sustained
from twin_ml.evaluate.evaluator import horizon_run_stats, subset_masks
from twin_ml.evaluate.metrics import average_precision
from twin_ml.evaluate.run_sums import CONFW_FIELDS, DRIFT_FIELDS


def _series_bounds(split: SplitData) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    s = split.at_end("series")
    bounds = np.flatnonzero(np.diff(s)) + 1
    return np.concatenate([[0], bounds]).astype(np.int64), np.concatenate(
        [bounds, [s.shape[0]]]
    ).astype(np.int64)


def sustained_alarm(
    split: SplitData, alarm: npt.NDArray[np.bool_], sustain_s: int
) -> npt.NDArray[np.bool_]:
    """Per-window sustained alarm (windows are ordered by series, then time)."""
    k = sustain_windows(sustain_s, split.stride)
    out = np.zeros(alarm.shape[0], dtype=bool)
    for lo, hi in zip(*_series_bounds(split), strict=True):
        if hi > lo:
            out[lo:hi] = sustained(alarm[lo:hi], k)
    return out


def alarm_labels(alarm: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    return np.where(alarm, WARNING, NORMAL).astype(np.int64)


def alarm_run_sums(
    split: SplitData,
    sustained_alarm_: npt.NDArray[np.bool_],
    horizon: int,
    fault_types: list[str],
    eval_cfg: EvalConfig,
    operating: npt.NDArray[np.bool_] | None = None,
) -> run_sums.Sums:
    """Early-warning per-run sums of an already sustained alarm (sustain of one window)."""
    once = eval_cfg.model_copy(
        update={"early_warning": eval_cfg.early_warning.model_copy(update={"sustain_s": 0})}
    )
    op = subset_masks(split, eval_cfg)["operating"] if operating is None else operating
    _, rs = horizon_run_stats(split, alarm_labels(sustained_alarm_), horizon, op, fault_types, once)
    return run_sums.from_runstats(rs)


def negative_windows(
    split: SplitData, eval_cfg: EvalConfig, fault_types: list[str], horizon: int
) -> npt.NDArray[np.bool_]:
    """The false-alarm windows of the early-warning code: operating, no fault, NORMAL ahead."""
    no_fault = fault_types.index("none")
    out: npt.NDArray[np.bool_] = (
        subset_masks(split, eval_cfg)["operating"]
        & (split.at_end("fault_type") == no_fault)
        & (split.target(horizon) == NORMAL)
    )
    return out


def drift_flag_sums(
    split: SplitData,
    sus: npt.NDArray[np.bool_],
    negative: npt.NDArray[np.bool_],
    sf_rows: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """(runs, 3): sensor-fault segments, flagged segments, summed time to the first flag (s)."""
    a = split.arrays
    runs = split.run_ids
    lookup = {r: i for i, r in enumerate(runs)}
    out = np.zeros((len(runs), len(DRIFT_FIELDS)))
    ends = a["ends"]
    win_series = split.at_end("series")
    t = a["t_s"]
    rows_series = a["series"]
    # contiguous sensor-fault row segments within a series
    edge = np.diff(np.concatenate([[0], sf_rows.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1
    flag = sus & negative
    for r0, r1 in zip(starts, stops, strict=True):
        sid = int(rows_series[r0])
        while int(rows_series[r1]) != sid:  # a segment never crosses series; guard anyway
            r1 -= 1
        run = lookup[str(a["series_run"][sid])]
        out[run, 0] += 1
        inside = (win_series == sid) & (ends >= r0) & (ends <= r1) & flag
        if inside.any():
            first = int(ends[np.flatnonzero(inside)[0]])
            out[run, 1] += 1
            out[run, 2] += float(t[first] - t[r0])
    return out


def conf_sums(
    p_cal: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any], high: float
) -> npt.NDArray[np.float64]:
    """CONFW_FIELDS sums of calibrated predictions (argmax of the calibrated probabilities)."""
    pred = np.argmax(p_cal, axis=1)
    conf = p_cal.max(axis=1).astype(np.float64)
    wrong = pred != y
    missed = (y >= WARNING) & (pred == NORMAL)  # a precursor or fault predicted NORMAL
    return np.array(
        [
            float(y.shape[0]),
            float(wrong.sum()),
            float(conf[wrong].sum()),
            float((wrong & (conf >= high)).sum()),
            float(conf[~wrong].sum()),
            float(missed.sum()),
            float(conf[missed].sum()),
            float((missed & (conf >= high)).sum()),
        ]
    )


def supervised_extras(
    split: SplitData,
    p_cal: npt.NDArray[np.floating[Any]],
    sets: dict[str, npt.NDArray[np.bool_]],
    horizon: int,
    sub: npt.NDArray[np.bool_],
    high: float,
) -> run_sums.Sums:
    """Per-run ``confw`` (CONFW_FIELDS) and ``conf_pre_<method>`` set sums on precursor
    windows (operating windows before a gradual fault, the ``gradual`` subset)."""
    y = split.target(horizon).astype(np.int64)
    rw = split.window_run()
    n_runs = len(split.run_ids)
    episode = split.at_end("episode_type")
    pre = sub & (episode == EPISODE_GRADUAL)
    out: run_sums.Sums = {"confw": np.zeros((n_runs, len(CONFW_FIELDS)))}
    for m in sets:
        out[f"conf_pre_{m}"] = np.zeros((n_runs, len(HEALTH), 5))
    for r in range(n_runs):
        sel = sub & (rw == r)
        out["confw"][r] = conf_sums(p_cal[sel], y[sel], high)
        psel = pre & (rw == r)
        for m, st in sets.items():
            out[f"conf_pre_{m}"][r] = set_sums(st[psel], y[psel], len(HEALTH))
    return out


def detector_condition(
    split: SplitData,
    scores: npt.NDArray[np.float64],
    alarms: dict[float, npt.NDArray[np.bool_]],
    horizon: int,
    fault_types: list[str],
    eval_cfg: EvalConfig,
    sf_rows: npt.NDArray[np.bool_] | None = None,
) -> tuple[dict[str, Any], dict[str, run_sums.Sums], dict[float, npt.NDArray[np.bool_]]]:
    """Detail, per-percentile sums (group ``p<percentile>``) and sustained alarms."""
    masks = subset_masks(split, eval_cfg)
    operating = masks["operating"]
    neg = negative_windows(split, eval_cfg, fault_types, horizon)
    sf_win = None if sf_rows is None else sf_rows[split.arrays["ends"]]
    y = split.target(horizon)
    pos = operating & (y >= WARNING)
    detail: dict[str, Any] = {
        "n_windows": split.n_windows,
        "n_runs": len(split.run_ids),
        "auprc_precursor": average_precision(pos[operating], scores[operating])
        if pos.any()
        else float("nan"),
        "auprc_fault_ahead": average_precision(y[operating] == FAULT, scores[operating])
        if (y[operating] == FAULT).any()
        else float("nan"),
        "score_quantiles": {
            "normal": np.quantile(scores[neg], [0.5, 0.9, 0.99]).tolist() if neg.any() else None,
            "precursor": np.quantile(scores[pos], [0.5, 0.9, 0.99]).tolist() if pos.any() else None,
        },
        "percentiles": {},
    }
    groups: dict[str, run_sums.Sums] = {}
    sustained_out: dict[float, npt.NDArray[np.bool_]] = {}
    for p, alarm in alarms.items():
        sus = sustained_alarm(split, alarm, eval_cfg.early_warning.sustain_s)
        sustained_out[p] = sus
        s = alarm_run_sums(split, sus, horizon, fault_types, eval_cfg, operating)
        if sf_win is not None and sf_rows is not None:
            once = alarm_run_sums(split, sus, horizon, fault_types, eval_cfg, operating & sf_win)
            s["sf_false_alarms"] = once["false_alarms"]
            s["sf_negative_s"] = once["negative_s"]
            s["drift"] = drift_flag_sums(split, sus, neg, sf_rows)
        groups[f"p{p:g}"] = s
        detail["percentiles"][f"{p:g}"] = {
            "alarm_frac": float(alarm.mean()) if alarm.size else float("nan"),
            "point": run_sums.point(s),
        }
    return detail, groups, sustained_out


def ends_digest(split: SplitData) -> str:
    """Identity of a split's windows (series, run, end time), to align alarm artifacts."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(split.at_end("t_s")).tobytes())
    h.update(np.ascontiguousarray(split.at_end("series")).tobytes())
    h.update("|".join(str(r) for r in split.arrays["series_run"]).encode())
    return h.hexdigest()[:16]


def pack_alarm(alarm: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    return np.packbits(alarm.astype(bool))


def unpack_alarm(packed: npt.NDArray[np.uint8], n: int) -> npt.NDArray[np.bool_]:
    out: npt.NDArray[np.bool_] = np.unpackbits(packed, count=n).astype(bool)
    return out
