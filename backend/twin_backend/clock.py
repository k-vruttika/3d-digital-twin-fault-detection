"""Injectable clocks so time-dependent logic (reorder windows, staleness, pacing) is testable.

``monotonic()`` measures intervals, ``now()`` stamps wall-clock times and ``sleep()`` waits.
Tests use :class:`FakeClock`, which only moves when told to (or on ``sleep`` if auto-advance
is on), so nothing has to really sleep.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock."""

    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Manually driven clock. ``sleep`` advances time instantly when ``auto_advance`` is set."""

    def __init__(
        self,
        start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
        *,
        auto_advance: bool = False,
    ) -> None:
        self._mono = 0.0
        self._start = start
        self.auto_advance = auto_advance
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self._mono

    def now(self) -> datetime:
        return self._start + timedelta(seconds=self._mono)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot move a clock backwards")
        self._mono += seconds

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if self.auto_advance and seconds > 0:
            self.advance(seconds)
        await asyncio.sleep(0)
