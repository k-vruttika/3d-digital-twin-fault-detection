"""Event-based early-warning evaluation on the real-data splits (Phase 10, ADR 0027).

Pure numpy: no MLflow, no ML library, no simulator. Every method (model, detector, rule) is
reduced to one per-window *sustained* alarm (``e5_eval.sustained_alarm``); everything below is a
function of that alarm and of the failure events recorded in the dataset's ``meta.json``.

* **Warned** for lookback L: a sustained alarm on a window of the failed machine whose end lies
  in ``[start - L, start)``. If an earlier failure of the same machine ends inside that range,
  the range is clipped to start after it (the event is counted as ``clipped``). Lead time =
  start - the first such window end.
* **Evaluable**: the whole (unclipped) lookback lies inside the split, and at least
  ``min_coverage`` of its L grid rows hold data. Others are *not evaluable*: listed, never a miss.
* **Detected during the failure**: a sustained alarm on a window ending in ``[start, end]``
  (reported separately, never counted as a warning).
* **Negative windows**: state RUNNING and no fault type at the window end (i.e. further than the
  fault context before, and the post-failure exclusion after, any failure). **False alarms**:
  rising edges of the sustained alarm on negative windows, per day of negative windows.
* **Chance check**: with the onset rate lambda (false alarms per day) and the alarm-on fraction
  f on negative windows, P(chance hit within L) = 1 - (1 - f) exp(-lambda L); the expected number
  of evaluable failures warned by chance alone is their count times that probability.

Per-machine sums (``Sums``, first axis = machine, in ``split.run_ids`` order) feed the machine
bootstrap and the paired comparisons (``evaluate.run_sums`` with :func:`metric_fn`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineState
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import STATES
from twin_ml.experiments.e5_eval import sustained_alarm

BoolArray = npt.NDArray[np.bool_]
Sums = dict[str, npt.NDArray[np.float64]]

EV_FIELDS = (
    "events",  # events of the machine in the split
    "evaluable",
    "warned",  # among the evaluable events
    "lead_sum_s",
    "clipped",
    "during_events",  # events with at least one window ending inside the failure
    "during_detected",
    "during_delay_sum_s",
)
FA_FIELDS = ("false_alarms", "negative_s", "alarm_on_negative", "negative_windows")
DAY_S = 86_400.0
RUNNING_STATE = STATES.index(MachineState.RUNNING)


# ---------------------------------------------------------------------------- events


@dataclass(frozen=True)
class RealEvent:
    id: str
    machine_id: str
    start_s: float
    end_s: float
    fault_type: str


def split_events(meta: dict[str, Any], split: str) -> list[RealEvent]:
    return [
        RealEvent(
            str(e["id"]),
            str(e["machine_id"]),
            float(e["start_s"]),
            float(e["end_s"]),
            str(e["fault_type"]),
        )
        for e in meta["events"][split]
    ]


def _seconds(meta: dict[str, Any], text: str) -> float:
    t0 = np.datetime64(str(meta["t0"]).replace(" ", "T"), "s")
    return float((np.datetime64(str(text).replace(" ", "T"), "s") - t0).astype(np.int64))


def split_bounds(meta: dict[str, Any], split: str) -> tuple[float, float]:
    """[lo, hi) of a split in seconds since the dataset origin ``t0``."""
    a, b = meta["config"]["split"][split]
    return _seconds(meta, a), _seconds(meta, b)


def month_of(meta: dict[str, Any], t_s: npt.NDArray[np.float64]) -> npt.NDArray[np.str_]:
    t0 = np.datetime64(str(meta["t0"]).replace(" ", "T"), "s")
    out: npt.NDArray[np.str_] = (
        (t0 + t_s.astype("timedelta64[s]")).astype("datetime64[M]").astype(str)
    )
    return out


# ---------------------------------------------------------------------------- split index


@dataclass
class SplitIndex:
    """Per-machine window ends and observed rows of one split (built once per split)."""

    machines: list[str]  # == split.run_ids (bootstrap order)
    win: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]  # idx, end time
    rows: dict[str, tuple[npt.NDArray[np.float64], BoolArray]]  # row time, observed
    step_s: float
    negative: BoolArray
    window_machine: npt.NDArray[np.int64]

    @classmethod
    def of(cls, split: SplitData, fault_types: list[str]) -> SplitIndex:
        a = split.arrays
        machines = split.run_ids
        lookup = {m: i for i, m in enumerate(machines)}
        series_machine = np.array([lookup[str(m)] for m in a["series_run"]], dtype=np.int64)
        wm = series_machine[split.at_end("series")]
        t_end = split.at_end("t_s").astype(np.float64)
        rm = series_machine[a["series"]]
        observed = np.asarray(~a["mask"].astype(bool).all(axis=1), dtype=bool)
        win, rows = {}, {}
        for i, m in enumerate(machines):
            w = np.flatnonzero(wm == i)
            w = w[np.argsort(t_end[w], kind="stable")]
            win[m] = (w.astype(np.int64), t_end[w])
            r = np.flatnonzero(rm == i)
            r = r[np.argsort(a["t_s"][r], kind="stable")]
            rows[m] = (a["t_s"][r].astype(np.float64), observed[r])
        neg = np.asarray(
            (split.at_end("state") == RUNNING_STATE)
            & (split.at_end("fault_type") == fault_types.index("none")),
            dtype=bool,
        )
        return cls(machines, win, rows, split.step_s, neg, wm)


# ---------------------------------------------------------------------------- per event


@dataclass(frozen=True)
class EventResult:
    id: str
    machine_id: str
    fault_type: str
    lookback: int  # rows
    inside_split: bool
    coverage: float  # observed grid rows in the lookback / L
    window_coverage: float  # window ends in the (clipped) lookback / L
    evaluable: bool
    clipped: bool
    warned: bool
    lead_s: float | None
    during_windows: int
    detected_during: bool
    delay_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def event_results(
    index: SplitIndex,
    sus: BoolArray,
    events: list[RealEvent],
    lookback: int,
    split_lo: float,
    min_coverage: float,
) -> list[EventResult]:
    step = index.step_s
    by_machine: dict[str, list[RealEvent]] = {}
    for e in events:
        by_machine.setdefault(e.machine_id, []).append(e)
    out: list[EventResult] = []
    for e in events:
        w_idx, w_t = index.win.get(e.machine_id, (np.zeros(0, np.int64), np.zeros(0)))
        r_t, r_obs = index.rows.get(e.machine_id, (np.zeros(0), np.zeros(0, bool)))
        lo_full = e.start_s - lookback * step
        prev = [p.end_s for p in by_machine[e.machine_id] if p.end_s < e.start_s]
        lo = lo_full
        clipped = bool(prev) and max(prev) >= lo_full
        if clipped:
            lo = max(prev) + step
        inside = lo_full >= split_lo
        coverage = float(r_obs[(r_t >= lo) & (r_t < e.start_s)].sum() / lookback)
        in_lb = (w_t >= lo) & (w_t < e.start_s)
        hits = np.flatnonzero(sus[w_idx] & in_lb)
        warned = bool(hits.size)
        lead = float(e.start_s - w_t[hits[0]]) if warned else None
        in_f = (w_t >= e.start_s) & (w_t <= e.end_s)
        hits_f = np.flatnonzero(sus[w_idx] & in_f)
        out.append(
            EventResult(
                id=e.id,
                machine_id=e.machine_id,
                fault_type=e.fault_type,
                lookback=lookback,
                inside_split=bool(inside),
                coverage=coverage,
                window_coverage=float(in_lb.sum() / lookback),
                evaluable=bool(inside and coverage >= min_coverage),
                clipped=clipped,
                warned=warned,
                lead_s=lead,
                during_windows=int(in_f.sum()),
                detected_during=bool(hits_f.size),
                delay_s=float(w_t[hits_f[0]] - e.start_s) if hits_f.size else None,
            )
        )
    return out


# ---------------------------------------------------------------------------- sums


def false_alarm_sums(
    index: SplitIndex, sus: BoolArray, split: SplitData
) -> npt.NDArray[np.float64]:
    """(machines, FA_FIELDS): rising edges of the sustained alarm on negative windows, per
    series (an alarm carried over a gap or a split cut starts a new edge)."""
    series = split.at_end("series")
    prev_same = np.concatenate([[False], series[1:] == series[:-1]])
    edges = sus & ~(np.concatenate([[False], sus[:-1]]) & prev_same)
    neg = index.negative
    n_m = len(index.machines)
    stride_s = split.stride * split.step_s
    out = np.zeros((n_m, len(FA_FIELDS)))
    wm = index.window_machine
    out[:, 0] = np.bincount(wm, weights=(edges & neg).astype(np.float64), minlength=n_m)
    out[:, 1] = np.bincount(wm, weights=neg.astype(np.float64), minlength=n_m) * stride_s
    out[:, 2] = np.bincount(wm, weights=(sus & neg).astype(np.float64), minlength=n_m)
    out[:, 3] = np.bincount(wm, weights=neg.astype(np.float64), minlength=n_m)
    return out


def event_sums(index: SplitIndex, results: list[EventResult]) -> npt.NDArray[np.float64]:
    lookup = {m: i for i, m in enumerate(index.machines)}
    out = np.zeros((len(index.machines), len(EV_FIELDS)))
    for r in results:
        i = lookup[r.machine_id]
        out[i, 0] += 1
        out[i, 4] += r.clipped
        if r.evaluable:
            out[i, 1] += 1
            if r.warned and r.lead_s is not None:
                out[i, 2] += 1
                out[i, 3] += r.lead_s
        if r.during_windows:
            out[i, 5] += 1
            if r.detected_during and r.delay_s is not None:
                out[i, 6] += 1
                out[i, 7] += r.delay_s
    return out


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def chance_probability(rate_per_day: float, on_frac: float, lookback_s: float) -> float:
    """P(a sustained alarm in a lookback of ``lookback_s`` by chance alone)."""
    if not (math.isfinite(rate_per_day) and math.isfinite(on_frac)):
        return float("nan")
    return 1.0 - (1.0 - on_frac) * math.exp(-rate_per_day * lookback_s / DAY_S)


def metric_fn(
    lookback_s: float,
) -> Callable[[dict[str, npt.NDArray[np.float64]]], dict[str, float]]:
    """Metric function of summed per-machine ``ev`` / ``fa`` arrays (for ``run_sums``)."""

    def fn(agg: dict[str, npt.NDArray[np.float64]]) -> dict[str, float]:
        ev = dict(zip(EV_FIELDS, (float(v) for v in agg["ev"]), strict=True))
        fa = dict(zip(FA_FIELDS, (float(v) for v in agg["fa"]), strict=True))
        rate = _div(fa["false_alarms"], fa["negative_s"] / DAY_S)
        on = _div(fa["alarm_on_negative"], fa["negative_windows"])
        p = chance_probability(rate, on, lookback_s)
        lead = _div(ev["lead_sum_s"], ev["warned"])
        return {
            "events": ev["events"],
            "evaluable": ev["evaluable"],
            "warned_count": ev["warned"],
            "detection_rate": _div(ev["warned"], ev["evaluable"]),
            "lead_mean_h": lead / 3600.0,
            "lead_min": lead / 60.0,
            "clipped": ev["clipped"],
            "during_rate": _div(ev["during_detected"], ev["during_events"]),
            "during_delay_min": _div(ev["during_delay_sum_s"], ev["during_detected"]) / 60.0,
            "false_alarms": fa["false_alarms"],
            "negative_days": fa["negative_s"] / DAY_S,
            "false_alarms_per_day": rate,
            "alarm_frac": on,
            "chance_p": p,
            "chance_expected": ev["evaluable"] * p if math.isfinite(p) else float("nan"),
        }

    return fn


def group_lookback_s(group: str, step_s: float) -> float:
    """Lookback of a sums group key ``...|L<rows>`` in seconds."""
    return float(group.rsplit("|L", 1)[1]) * step_s


# ---------------------------------------------------------------------------- one alarm


@dataclass
class AlarmEval:
    """Everything one sustained alarm yields on one split."""

    sums: dict[str, Sums]  # "L<rows>" -> {"ev", "fa"}
    events: dict[str, list[dict[str, Any]]]  # "L<rows>" -> per-event records
    points: dict[str, dict[str, float]]  # "L<rows>" -> point metrics


def evaluate_alarm(
    split: SplitData,
    index: SplitIndex,
    sus: BoolArray,
    events: list[RealEvent],
    lookbacks: list[int],
    split_lo: float,
    min_coverage: float,
) -> AlarmEval:
    fa = false_alarm_sums(index, sus, split)
    sums: dict[str, Sums] = {}
    recs: dict[str, list[dict[str, Any]]] = {}
    points: dict[str, dict[str, float]] = {}
    for lb in lookbacks:
        res = event_results(index, sus, events, lb, split_lo, min_coverage)
        key = f"L{lb}"
        sums[key] = {"ev": event_sums(index, res), "fa": fa}
        recs[key] = [r.to_dict() for r in res]
        fn = metric_fn(lb * index.step_s)
        points[key] = fn({k: v.sum(axis=0) for k, v in sums[key].items()})
    return AlarmEval(sums, recs, points)


def sustain(split: SplitData, alarm: BoolArray, sustain_rows: int) -> BoolArray:
    """Sustained alarm over ``sustain_rows`` consecutive windows of a series."""
    return sustained_alarm(split, alarm, round(sustain_rows * split.step_s * split.stride))


def fa_by_month(
    meta: dict[str, Any], split: SplitData, index: SplitIndex, sus: BoolArray
) -> dict[str, dict[str, float]]:
    """Exploratory: false alarms and negative days per calendar month of the window end."""
    series = split.at_end("series")
    prev_same = np.concatenate([[False], series[1:] == series[:-1]])
    edges = sus & ~(np.concatenate([[False], sus[:-1]]) & prev_same)
    months = month_of(meta, split.at_end("t_s").astype(np.float64))
    stride_s = split.stride * split.step_s
    out: dict[str, dict[str, float]] = {}
    for m in sorted(set(months.tolist())):
        sel = (months == m) & index.negative
        days = float(sel.sum()) * stride_s / DAY_S
        n = float((edges & sel).sum())
        out[m] = {
            "false_alarms": n,
            "negative_days": days,
            "false_alarms_per_day": _div(n, days),
        }
    return out


# ---------------------------------------------------------------------------- robustness


def normal_sigma(train: SplitData, fault_types: list[str], normal_states: list[Any]) -> Any:
    """(kinds, sensors) normal-operation std of the training rows (Phase 8 ``normal_std``)."""
    from twin_ml.data.perturb import normal_std

    a = train.arrays
    return normal_std(
        a["raw"],
        a["state"],
        a["kind"],
        a["fault_type"],
        fault_types.index("none"),
        normal_states,
        n_kinds=len(train.schema.kinds),
    )


def perturbed_split(
    split: SplitData,
    normalizer: Any,
    condition: Any,
    perturbations: Any,
    sigma: Any,
    seed: int,
) -> SplitData:
    """A copy of ``split`` whose raw series are perturbed (Phase 8 functions) and re-normalised.

    Labels, windows and events are untouched. Each series draws from
    ``SeedSequence(seed, (condition, machine, machine#series))``, so a condition is identical for
    every method (paired comparisons against clean are valid).
    """
    from twin_ml.data.perturb import perturb_array, series_rng

    a = split.arrays
    raw = a["raw"].copy()
    values = a["values"].copy()
    mask = a["mask"].copy()
    series = a["series"]
    kinds = split.schema.kinds
    for sid in np.unique(series):
        rows = np.flatnonzero(series == sid)
        machine = str(a["series_machine"][sid])
        k = int(a["kind"][rows[0]])
        rng = series_rng(seed, condition.name, machine, f"{machine}#{int(sid)}")
        r = perturb_array(a["raw"][rows], condition, perturbations, sigma[k], rng)
        raw[rows] = r
        values[rows], mask[rows] = normalizer.transform(r, kinds[k])
    arrays = dict(a)
    arrays |= {"raw": raw, "values": values, "mask": mask}
    return SplitData(
        name=f"{split.name}:{condition.name}",
        window=split.window,
        horizons=split.horizons,
        stride=split.stride,
        arrays=arrays,
        schema=split.schema,
    )
