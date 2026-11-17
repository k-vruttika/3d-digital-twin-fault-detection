"""Shared helpers of the public-dataset loaders (Phase 10, ADR 0027). Pure numpy + pyarrow.

* :func:`require_files`: every expected file exists, or :class:`MissingDatasetError` says what
  to download and where (datasets are never downloaded automatically);
* :func:`resample_mean`: irregular samples -> a regular grid of bin means, empty bins NaN;
* :func:`segments`: contiguous stretches of a grid, split where a gap is longer than allowed;
* :func:`label_rows`: failure events -> per-row pseudo-state and fault type, so the existing
  label code (health classes, subsets, normal windows) works unchanged;
* :func:`time_to_fault`: time to the next FAULT onset within one series.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineState
from twin_ml.datasets.config import DatasetFacts, LabelBands

NO_FAULT = "none"  # == twin_sim.config.NO_FAULT_LABEL (checked by a test)
EPOCH = np.datetime64("1970-01-01T00:00:00", "s")

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class MissingDatasetError(FileNotFoundError):
    """An expected public-dataset file is missing (with download instructions)."""


def require_files(facts: DatasetFacts, root: Path = Path()) -> dict[str, Path]:
    """Paths of every required file of ``facts``; raises with download instructions."""
    base = root / facts.dir
    paths = {key: base / name for key, name in facts.files.items()}
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        names = ", ".join(sorted(p.name for p in missing))
        raise MissingDatasetError(
            f"{facts.title}: missing file(s) {names} in {base.as_posix()}/. "
            f"{facts.download.strip()} Datasets are never downloaded automatically."
        )
    return paths


def files_sha256(paths: Sequence[Path], chunk: int = 1 << 22) -> str:
    """One sha256 over the bytes of ``paths`` (in the given order, names included)."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        with p.open("rb") as f:
            while block := f.read(chunk):
                h.update(block)
    return h.hexdigest()


def to_seconds(t: datetime) -> int:
    """Naive datetime -> integer seconds since 1970 (no time-zone conversion)."""
    return int((np.datetime64(t, "s") - EPOCH).astype(np.int64))


# ---------------------------------------------------------------------- grid


@dataclass(frozen=True)
class Grid:
    """One machine on a regular grid. ``t_s``: seconds since the dataset origin ``t0``."""

    machine_id: str
    kind: str
    t_s: FloatArray  # (T,) regular, step_s apart
    values: npt.NDArray[np.float32]  # (T, sensors), NaN = no data
    observed: BoolArray  # (T,) the row holds at least one sample


def resample_mean(
    t: IntArray, values: npt.NDArray[np.floating], step_s: int, t0: int
) -> tuple[IntArray, npt.NDArray[np.float64], IntArray]:
    """Bin samples at integer times ``t`` (s) into ``[t0 + k step, t0 + (k + 1) step)``.

    Returns (bin indices k of the non-empty bins, per-bin mean of each column ignoring NaN,
    samples per bin). ``t`` must be sorted.
    """
    if t.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, values.shape[1])), np.zeros(0, np.int64)
    if np.any(np.diff(t) < 0):
        raise ValueError("timestamps must be sorted")
    k = (t - t0) // step_s
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    counts = np.diff(np.concatenate((starts, [k.size])))
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    sums = np.add.reduceat(np.where(finite, v, 0.0), starts, axis=0)
    n = np.add.reduceat(finite.astype(np.int64), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n > 0, sums / np.maximum(n, 1), np.nan)
    return k[starts].astype(np.int64), means, counts.astype(np.int64)


def dense(
    bins: IntArray, means: npt.NDArray[np.float64], n_bins: int
) -> tuple[npt.NDArray[np.float32], BoolArray]:
    """Scatter non-empty bins onto a dense grid of ``n_bins`` rows (empty rows NaN)."""
    out = np.full((n_bins, means.shape[1]), np.nan, dtype=np.float32)
    obs = np.zeros(n_bins, dtype=bool)
    out[bins] = means.astype(np.float32)
    obs[bins] = True
    return out, obs


def segments(observed: BoolArray, max_gap_rows: int) -> list[tuple[int, int]]:
    """Half-open row ranges of contiguous data: a run of more than ``max_gap_rows`` empty rows
    ends a segment; each segment starts and ends on an observed row."""
    idx = np.flatnonzero(observed)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap_rows + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def gap_runs(t: IntArray, threshold_s: int) -> tuple[IntArray, IntArray]:
    """(start time, length in s) of every gap between consecutive samples > ``threshold_s``."""
    d = np.diff(t)
    i = np.flatnonzero(d > threshold_s)
    return t[i], d[i]


# ---------------------------------------------------------------------- labels


@dataclass(frozen=True)
class Event:
    """One failure event of one machine; times in seconds since the dataset origin."""

    id: str
    machine_id: str
    start_s: float
    end_s: float  # inclusive (== start_s for an instantaneous failure)
    fault_type: str
    components: tuple[str, ...] = ()


def label_rows(
    t_s: FloatArray, events: Sequence[Event], bands: LabelBands, step_s: int
) -> tuple[list[str], list[str]]:
    """Pseudo-state and fault type of every row of one machine's grid (ADR 0027).

    In priority order (later wins): MAINTENANCE for ``post_failure`` rows after a failure end,
    WARNING for ``warning_band`` rows before a start, FAULT from start to end (inclusive);
    RUNNING elsewhere. The fault type is the event's from ``fault_context`` rows before its
    start to its end, events in time order (a later event overrides an earlier context).
    """
    n = t_s.shape[0]
    state = np.full(n, MachineState.RUNNING.value, dtype=object)
    ftype = np.full(n, NO_FAULT, dtype=object)
    ordered = sorted(events, key=lambda e: e.start_s)
    for e in ordered:
        post = (t_s > e.end_s) & (t_s <= e.end_s + bands.post_failure * step_s)
        state[post & (state == MachineState.RUNNING.value)] = MachineState.MAINTENANCE.value
    for e in ordered:
        warn = (t_s >= e.start_s - bands.warning_band * step_s) & (t_s < e.start_s)
        state[warn & (state != MachineState.FAULT.value)] = MachineState.WARNING.value
    for e in ordered:
        state[(t_s >= e.start_s) & (t_s <= e.end_s)] = MachineState.FAULT.value
    for e in ordered:
        ctx = (t_s >= e.start_s - bands.fault_context * step_s) & (t_s <= e.end_s)
        ftype[ctx] = e.fault_type
    return [str(s) for s in state], [str(f) for f in ftype]


def time_to_fault(t_s: FloatArray, is_fault: BoolArray) -> tuple[FloatArray, BoolArray]:
    """Seconds from each row to the next FAULT onset of the series (0 in FAULT).

    Rows with no onset ahead get the time to the series end and ``no_fault_ahead`` True, as
    the simulator's ``time_to_fault`` (then capped and censored with ``labels.censor_ttf``).
    """
    n = t_s.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    prev = np.concatenate(([False], is_fault[:-1]))
    onsets = np.flatnonzero(is_fault & ~prev)
    if onsets.size == 0:
        return (t_s[-1] - t_s).astype(np.float64), ~is_fault
    nxt = np.searchsorted(onsets, np.arange(n), side="left")
    has = nxt < onsets.size
    t_next = np.where(has, t_s[onsets[np.minimum(nxt, onsets.size - 1)]], t_s[-1])
    ttf = np.where(is_fault, 0.0, t_next - t_s)
    no_ahead = ~has & ~is_fault
    return ttf.astype(np.float64), no_ahead
