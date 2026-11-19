"""Per-machine 1 Hz reading buffers and the prediction gate (pure numpy; ADR 0030).

**Same inputs as training.** The dataset build forward-fills each machine series over its
whole length (``Normalizer.transform``), so a value missing at the start of a window is filled
from rows before it. A buffer therefore keeps, besides its rows, the last observed value of
every sensor before its first row (``carry``) and transforms ``[carry] + rows``: the result is
bit-identical to the series-wide transform for every kept row. Values are cast to float32
exactly as ``features.inputs.select_inputs`` does.

**Rows.** One reading per second (``seq`` + 1 per row). A gap of up to ``max_gap_rows`` missing
readings is filled with all-missing rows (masks set, as a quarantined reading would leave
them), each carrying the previous row's state. A longer gap, or a sequence restart, empties
the rows (the carry is kept, as series-wide forward fill would) and the warm-up starts again.
Readings with ``seq`` at or below the last one are ignored (already applied).

**Gate** (Phase 11 decision with the user): a window is predicted only when every one of its
``window`` rows is in an operating state (RUNNING / WARNING), so live windows never mix stopped
and running data. The evaluated "operating" subset of Phases 7-10 checked only the window's
last row; the gate is stricter (a subset of the evaluated windows). Otherwise the status is
``stopped`` (IDLE, FAULT, MAINTENANCE, ...), ``waiting`` (STARVED / BLOCKED) or ``warming_up``
(operating, but fewer than ``window`` operating rows in a row).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind
from twin_ml.features.inputs import FloatArray, MaskArray
from twin_ml.features.normalize import Normalizer

PREDICT = "predict"
WARMING_UP = "warming_up"
WAITING = "waiting"
STOPPED = "stopped"
GATE_STATUSES = (PREDICT, WARMING_UP, WAITING, STOPPED)


@dataclass(frozen=True)
class GateConfig:
    window: int  # rows per window (the training window)
    operating: frozenset[str]  # states a window's rows must all be in
    waiting: frozenset[str]  # not operating because a neighbour stops the flow
    max_gap_rows: int  # missing readings filled with all-missing rows; longer gap: reset

    @classmethod
    def of(
        cls, window: int, operating: Iterable[str], waiting: Iterable[str], max_gap_rows: int
    ) -> GateConfig:
        return cls(window, frozenset(operating), frozenset(waiting), max_gap_rows)


def gate_status(state: str, run_len: int, cfg: GateConfig) -> str:
    """Gate status at a row: ``run_len`` consecutive operating rows end there."""
    if state in cfg.operating:
        return PREDICT if run_len >= cfg.window else WARMING_UP
    if state in cfg.waiting:
        return WAITING
    return STOPPED


def gate_over_series(states: Sequence[str], cfg: GateConfig) -> tuple[list[str], list[int]]:
    """Gate status and operating run length at every row of one complete series (offline)."""
    out, runs, run = [], [], 0
    for s in states:
        run = run + 1 if s in cfg.operating else 0
        out.append(gate_status(s, run, cfg))
        runs.append(run)
    return out, runs


@dataclass(frozen=True)
class NewRow:
    """A reading appended to a buffer (gap-filler rows are never reported)."""

    seq: int
    ts: float  # epoch seconds
    state: str
    status: str  # gate status at this row
    run_len: int  # consecutive operating rows ending here (capped at the window)
    row: int  # absolute row number in this buffer (gap fillers included)


class MachineBuffer:
    """Rows of one machine. ``capacity`` must hold a window plus one batch of new rows."""

    def __init__(self, machine_id: str, kind: MachineKind, cfg: GateConfig, capacity: int) -> None:
        if capacity < cfg.window:
            raise ValueError("capacity must be at least one window")
        self.machine_id = machine_id
        self.kind = kind
        self.cfg = cfg
        self.capacity = capacity
        self.raw: deque[npt.NDArray[np.float32]] = deque()
        self.states: deque[str] = deque()
        self.carry = np.full(5, np.nan, dtype=np.float32)  # last observed before raw[0]
        self.total = 0  # rows ever pushed; raw[0] is absolute row ``total - len(raw)``
        self.last_seq: int | None = None
        self.last_ts: float | None = None
        self.run_len = 0
        self.ignored = 0  # readings at or below the last seq
        self.resets = 0

    @property
    def base(self) -> int:
        """Absolute row number of the first buffered row."""
        return self.total - len(self.raw)

    def index(self, row: int) -> int | None:
        """Buffer index of an absolute row number, None when it is no longer buffered."""
        i = row - self.base
        return i if 0 <= i < len(self.raw) else None

    # -------------------------------------------------------------------- rows
    def _push(self, values: npt.NDArray[np.float32], state: str) -> None:
        self.raw.append(values)
        self.states.append(state)
        self.total += 1
        self.run_len = min(self.run_len + 1, self.cfg.window) if state in self.cfg.operating else 0
        while len(self.raw) > self.capacity:
            old = self.raw.popleft()
            self.states.popleft()
            seen = ~np.isnan(old)
            self.carry[seen] = old[seen]

    def _reset(self) -> None:
        while self.raw:
            old = self.raw.popleft()
            seen = ~np.isnan(old)
            self.carry[seen] = old[seen]
        self.states.clear()
        self.run_len = 0
        self.resets += 1

    def append(
        self, seq: int, ts: float, state: str, values: Sequence[float | None]
    ) -> NewRow | None:
        """Append one reading; None when it is ignored (seq at or below the last one)."""
        if self.last_seq is not None and seq <= self.last_seq:
            if self.last_ts is not None and ts > self.last_ts:  # the source restarted
                self._reset()
                self.carry[:] = np.nan
            else:
                self.ignored += 1
                return None
        elif self.last_seq is not None:
            gap = seq - self.last_seq - 1
            if gap > self.cfg.max_gap_rows:
                self._reset()
            elif gap > 0:
                prev = self.states[-1] if self.states else state
                for _ in range(gap):
                    self._push(np.full(5, np.nan, dtype=np.float32), prev)
        row = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
        self._push(row, state)
        self.last_seq, self.last_ts = seq, ts
        return NewRow(
            seq, ts, state, gate_status(state, self.run_len, self.cfg), self.run_len, self.total - 1
        )

    def extend(self, rows: Iterable[Sequence[Any]]) -> list[NewRow]:
        """Append ``[seq, ts, state, speed, temp, vib, load, current]`` rows in order."""
        out = []
        for r in rows:
            new = self.append(int(r[0]), float(r[1]), str(r[2]), r[3:8])
            if new is not None:
                out.append(new)
        return out

    # -------------------------------------------------------------------- windows
    def transform(self, norm: Normalizer) -> tuple[FloatArray, MaskArray]:
        """Normalised values and masks of every buffered row (series-wide equivalent)."""
        raw = np.vstack([self.carry[None, :], *[r[None, :] for r in self.raw]])
        values, mask = norm.transform(raw, self.kind)
        return values[1:], mask[1:]

    def has_window(self, row: int) -> bool:
        """Is the whole window ending at absolute row ``row`` still buffered?"""
        i = self.index(row)
        return i is not None and i >= self.cfg.window - 1

    def windows(self, norm: Normalizer, rows: Sequence[int]) -> tuple[FloatArray, MaskArray]:
        """(B, W, 6) values and (B, W, 5) masks of the windows ending at absolute ``rows``."""
        w = self.cfg.window
        values, mask = self.transform(norm)
        idx = np.asarray([r - self.base for r in rows], dtype=np.int64)
        if ((idx < w - 1) | (idx >= len(self.raw))).any():
            raise ValueError("a requested window is not (fully) buffered")
        take = idx[:, None] + np.arange(-w + 1, 1)[None, :]
        return values[take], mask[take]
