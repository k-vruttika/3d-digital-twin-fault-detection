"""Real-time runner: ticks the simulator at 1 Hz x speed multiplier and emits Readings."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from twin_common.contracts import Command, CommandAck, Reading
from twin_common.logging import get_logger
from twin_sim.line import LineSimulator

ReadingCallback = Callable[[Reading], Awaitable[None] | None]
T = TypeVar("T")
TICK_S = 1.0

log = get_logger(__name__)


class RealtimeRunner:
    """Drives a LineSimulator in (scaled) real time.

    Ticks are scheduled on absolute ``loop.time()`` deadlines so the period does not drift.
    Commands submitted while running are applied at the start of the next tick.
    """

    def __init__(self, sim: LineSimulator, on_reading: ReadingCallback, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.sim = sim
        self.on_reading = on_reading
        self.speed = speed
        self.ticks = 0
        self._stop = asyncio.Event()
        # Calls applied to the simulator at the next tick boundary (commands, scenario requests).
        self._pending: list[tuple[Callable[[LineSimulator], Any], asyncio.Future[Any]]] = []

    @property
    def period_s(self) -> float:
        return TICK_S / self.speed

    def stop(self) -> None:
        self._stop.set()

    def submit(self, cmd: Command) -> asyncio.Future[CommandAck]:
        """Queue a command; the future resolves with the ack at the next tick."""
        return self.submit_call(lambda sim: sim.apply_command(cmd))

    def submit_call(self, fn: Callable[[LineSimulator], T]) -> asyncio.Future[T]:
        """Run ``fn(sim)`` at the next tick boundary; the future resolves with its result."""
        fut: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        self._pending.append((fn, fut))
        return fut

    async def run(self, max_ticks: int | None = None) -> int:
        """Run until ``stop()`` or ``max_ticks``; returns the number of ticks executed."""
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        log.info("runner.start", speed=self.speed, max_ticks=max_ticks)
        while not self._stop.is_set() and (max_ticks is None or self.ticks < max_ticks):
            next_deadline += self.period_s
            pending, self._pending = self._pending, []
            for fn, fut in pending:
                if not fut.done():
                    try:
                        fut.set_result(fn(self.sim))
                    except Exception as exc:  # a bad request must not stop the simulation
                        fut.set_exception(exc)
            frame = self.sim.step(TICK_S)
            for reading in self.sim.readings(frame):
                result = self.on_reading(reading)
                if inspect.isawaitable(result):
                    await result
            self.ticks += 1
            delay = next_deadline - loop.time()
            if delay > 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
            else:
                await asyncio.sleep(0)
        log.info("runner.stop", ticks=self.ticks)
        return self.ticks
