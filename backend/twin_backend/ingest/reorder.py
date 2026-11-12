"""Per-machine reorder buffer with a lateness window (arrival time, injectable clock).

Rules, per machine (``next`` = the seq expected next):

- ``seq == next``: released at once, together with any held messages that are now contiguous.
- ``seq > next``: held until the gap fills or the message has waited ``window_s``; then the
  gap is given up (counted) and held messages are released in seq order.
- ``seq < next``: already passed, so it is released immediately but flagged late
  (``quality=LATE`` downstream: stored, never applied to live state).

Before a machine's first release the expected seq is unknown, so its first messages are held
for the window and released sorted; this way early out-of-order messages are not marked late.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Released(Generic[T]):
    item: T
    seq: int
    late: bool = False
    reordered: bool = False  # arrived before a predecessor and was put back in order
    gap_before: int = 0  # seqs given up just before this one


@dataclass
class _MachineBuffer(Generic[T]):
    next_seq: int | None = None
    heap: list[tuple[int, float, int]] = field(default_factory=list)  # (seq, arrival, id)
    items: dict[int, T] = field(default_factory=dict)  # id -> item
    counter: int = 0


class ReorderBuffer(Generic[T]):
    def __init__(self, window_s: float) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.window_s = window_s
        self._machines: dict[str, _MachineBuffer[T]] = {}

    def _buf(self, machine_id: str) -> _MachineBuffer[T]:
        buf = self._machines.get(machine_id)
        if buf is None:
            buf = self._machines[machine_id] = _MachineBuffer()
        return buf

    def is_late(self, machine_id: str, seq: int) -> bool:
        buf = self._machines.get(machine_id)
        return buf is not None and buf.next_seq is not None and seq < buf.next_seq

    def held(self, machine_id: str) -> int:
        buf = self._machines.get(machine_id)
        return len(buf.heap) if buf else 0

    def push(self, machine_id: str, seq: int, item: T, now: float) -> list[Released[T]]:
        buf = self._buf(machine_id)
        if buf.next_seq is not None and seq < buf.next_seq:
            return [Released(item, seq, late=True)]
        if buf.next_seq is not None and seq == buf.next_seq and not buf.heap:
            buf.next_seq = seq + 1
            return [Released(item, seq)]
        buf.counter += 1
        heapq.heappush(buf.heap, (seq, now, buf.counter))
        buf.items[buf.counter] = item
        return self._drain_contiguous(buf, mark_reordered=True)

    def _pop(self, buf: _MachineBuffer[T]) -> tuple[int, T]:
        seq, _arrival, ident = heapq.heappop(buf.heap)
        return seq, buf.items.pop(ident)

    def _drain_contiguous(
        self, buf: _MachineBuffer[T], *, mark_reordered: bool, gap: int = 0
    ) -> list[Released[T]]:
        """Pop held messages while they continue the expected sequence.

        With ``mark_reordered`` (a missing predecessor just arrived) every message after the
        first was held only because it arrived early: it is counted as reordered.
        """
        out: list[Released[T]] = []
        while buf.heap and buf.next_seq is not None and buf.heap[0][0] == buf.next_seq:
            seq, item = self._pop(buf)
            out.append(
                Released(
                    item,
                    seq,
                    reordered=mark_reordered and bool(out),
                    gap_before=0 if out else gap,
                )
            )
            buf.next_seq = seq + 1
        return out

    def tick(self, now: float) -> list[tuple[str, Released[T]]]:
        """Release messages whose hold time reached the window, giving up the gaps before them."""
        out: list[tuple[str, Released[T]]] = []
        for machine_id, buf in self._machines.items():
            while buf.heap and min(a for _, a, _ in buf.heap) + self.window_s <= now:
                head_seq = buf.heap[0][0]
                gap = 0 if buf.next_seq is None else head_seq - buf.next_seq
                buf.next_seq = head_seq
                released = self._drain_contiguous(buf, mark_reordered=False, gap=gap)
                out.extend((machine_id, r) for r in released)
        return out

    def flush(self) -> list[tuple[str, Released[T]]]:
        """Release everything held (end of stream), in seq order per machine."""
        out: list[tuple[str, Released[T]]] = []
        for machine_id, buf in self._machines.items():
            while buf.heap:
                head_seq = buf.heap[0][0]
                gap = 0 if buf.next_seq is None else head_seq - buf.next_seq
                buf.next_seq = head_seq
                released = self._drain_contiguous(buf, mark_reordered=False, gap=gap)
                out.extend((machine_id, r) for r in released)
        return out

    def reset(self, machine_id: str) -> list[Released[T]]:
        """Forget the expected seq (source restarted); held messages are released first."""
        buf = self._machines.pop(machine_id, None)
        if buf is None:
            return []
        out: list[Released[T]] = []
        while buf.heap:
            seq, item = self._pop(buf)
            out.append(Released(item, seq))
        return out
