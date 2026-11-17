"""Early-warning metrics on dense test windows (docs/experiments.md).

* alarm: the model predicts WARNING or FAULT for ``k`` consecutive windows of one machine
  series (``k = ceil(sustain_s / stride)``), raised at the window that completes the run;
* lead time of a gradual fault episode: ``t_fault - t_alarm`` for the first sustained alarm
  with ``t_start <= t_alarm < t_fault`` (``t_start`` = the degradation start); no such alarm =
  missed;
* relative lead: ``t_warn - t_alarm``, where ``t_warn`` is the simulator's first WARNING of the
  episode (threshold onset); positive = the model warned earlier;
* false alarms per hour: rising edges of the sustained alarm in negative windows (machine
  operating, no active fault, NORMAL at the horizon), per hour of negative windows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def sustained(alarm: BoolArray, k: int) -> BoolArray:
    """True where the last ``k`` entries (this one included) are all True."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return alarm.copy()
    c = np.concatenate([[0], np.cumsum(alarm.astype(np.int64))])
    n = alarm.shape[0]
    out = np.zeros(n, dtype=bool)
    if n >= k:
        out[k - 1 :] = (c[k:] - c[: n - k + 1]) == k
    return out


def rising_edges(x: BoolArray) -> BoolArray:
    prev = np.concatenate([[False], x[:-1]])
    out: BoolArray = x & ~prev
    return out


def sustain_windows(sustain_s: float, stride: float) -> int:
    """Consecutive windows that span ``sustain_s`` seconds, windows ``stride`` seconds apart."""
    return max(1, math.ceil(sustain_s / stride))


@dataclass(frozen=True)
class EpisodeResult:
    run: int
    machine: str
    kind: str
    fault_type: str
    t_start: float
    t_warn: float | None
    t_fault: float
    detected: bool
    lead_s: float | None
    rel_lead_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_lead(
    t: npt.NDArray[np.floating[Any]],
    alarm_sustained: BoolArray,
    t_start: float,
    t_fault: float,
    t_warn: float | None,
) -> tuple[bool, float | None, float | None]:
    """(detected, lead, relative lead) of one episode from one series' alarm trace."""
    hits = np.flatnonzero(alarm_sustained & (t >= t_start) & (t < t_fault))
    if hits.size == 0:
        return False, None, None
    t_alarm = float(t[hits[0]])
    rel = None if t_warn is None else float(t_warn - t_alarm)
    return True, float(t_fault - t_alarm), rel


def false_alarms(
    alarm_sustained: BoolArray, negative: BoolArray, stride: float
) -> tuple[int, float]:
    """(false alarm count, negative seconds) of one series; windows ``stride`` seconds apart."""
    edges = rising_edges(alarm_sustained) & negative
    return int(edges.sum()), float(negative.sum() * stride)


def summarize(episodes: list[EpisodeResult], fa_count: int, negative_s: float) -> dict[str, Any]:
    leads = [e.lead_s for e in episodes if e.detected and e.lead_s is not None]
    rels = [e.rel_lead_s for e in episodes if e.detected and e.rel_lead_s is not None]
    n = len(episodes)
    per_kind: dict[str, int] = {}
    for e in episodes:
        per_kind[e.kind] = per_kind.get(e.kind, 0) + 1
    return {
        "episodes": n,
        "episodes_per_kind": dict(sorted(per_kind.items())),
        "detected": len(leads),
        "miss_rate": (n - len(leads)) / n if n else float("nan"),
        "lead_mean_s": float(np.mean(leads)) if leads else float("nan"),
        "lead_median_s": float(np.median(leads)) if leads else float("nan"),
        "rel_lead_mean_s": float(np.mean(rels)) if rels else float("nan"),
        "rel_lead_median_s": float(np.median(rels)) if rels else float("nan"),
        "rel_lead_n": len(rels),
        "false_alarms": fa_count,
        "negative_h": negative_s / 3600.0,
        "false_alarms_per_h": fa_count / (negative_s / 3600.0) if negative_s > 0 else float("nan"),
    }
