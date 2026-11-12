"""Duplicate detection on (machineId, seq) with a bounded memory per machine.

A seq far below the highest seen (more than ``reset_threshold``) means the source restarted
(simulator restart, new recording): that machine's history is cleared and the message is new.
The config guarantees ``reset_threshold <= window``, so every seq is either remembered or
classified as a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DedupResult(Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    RESET = "reset"  # new, and the source restarted


@dataclass
class _MachineSeqs:
    max_seq: int
    seen: set[int] = field(default_factory=set)


class Deduplicator:
    def __init__(self, window: int, reset_threshold: int) -> None:
        if reset_threshold > window:
            raise ValueError("reset_threshold must be <= window")
        self.window = window
        self.reset_threshold = reset_threshold
        self._machines: dict[str, _MachineSeqs] = {}

    def check(self, machine_id: str, seq: int) -> DedupResult:
        state = self._machines.get(machine_id)
        if state is None:
            self._machines[machine_id] = _MachineSeqs(max_seq=seq, seen={seq})
            return DedupResult.NEW
        if seq < state.max_seq - self.reset_threshold:
            self._machines[machine_id] = _MachineSeqs(max_seq=seq, seen={seq})
            return DedupResult.RESET
        if seq in state.seen:
            return DedupResult.DUPLICATE
        state.seen.add(seq)
        if seq > state.max_seq:
            state.max_seq = seq
        if len(state.seen) > 2 * self.window:
            floor = state.max_seq - self.window
            state.seen = {s for s in state.seen if s > floor}
        return DedupResult.NEW

    def forget(self, machine_id: str) -> None:
        self._machines.pop(machine_id, None)
