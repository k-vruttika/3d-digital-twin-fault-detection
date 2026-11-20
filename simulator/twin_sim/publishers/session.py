"""SimSession: one simulated line in real time, shared by every publisher.

It wraps a :class:`~twin_sim.runner.RealtimeRunner`, keeps a bounded per-machine history
(the mock server's ``GET /history`` and gap backfill) and fans each Reading out to listeners.
Commands go through ``RealtimeRunner.submit`` and are applied at the next tick, as are
scenario requests (simulation only, ADR 0015) through ``submit_scenario``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from twin_common.config import load_line_config
from twin_common.config.line import LineConfig
from twin_common.contracts import Command, CommandAck, Reading, ScenarioAck, ScenarioRequest
from twin_common.logging import get_logger
from twin_common.whatif import WhatIfSimRequest, WhatIfSimResult
from twin_sim.config import (
    ScenarioConfig,
    SimConfig,
    check_scenario_compat,
    load_scenario,
    load_sim_config,
)
from twin_sim.line import LineSimulator
from twin_sim.publishers.scenario import apply_scenario
from twin_sim.runner import RealtimeRunner
from twin_sim.scenario import make_run_plan
from twin_sim.whatif import run_fork, take_fork

log = get_logger(__name__)

# Scenario files LOAD_SCENARIO may name (overridden by the session's --scenario directory).
DEFAULT_SCENARIO_DIR = Path("config/scenarios")

Listener = Callable[[Reading], Awaitable[None] | None]


@dataclass(frozen=True)
class SessionOptions:
    """What to simulate and how fast (CLI arguments of the publisher subcommands)."""

    line_path: Path
    scenario_path: Path
    sim_config_path: Path
    seed: int | None = None
    run_index: int = 0
    speed: float = 1.0
    duration_s: float | None = None
    history_size: int = 3600


def build_simulator(
    line: LineConfig,
    sim_cfg: SimConfig,
    scenario: ScenarioConfig,
    seed: int | None = None,
    run_index: int = 0,
) -> LineSimulator:
    check_scenario_compat(scenario, sim_cfg, line)
    plan = make_run_plan(
        scenario, sim_cfg, line, seed if seed is not None else scenario.seed, run_index
    )
    return LineSimulator(line, sim_cfg, plan)


class ReadingHistory:
    """Last ``size`` Readings per machine."""

    def __init__(self, machine_ids: list[str], size: int) -> None:
        self._by_machine: dict[str, deque[Reading]] = {m: deque(maxlen=size) for m in machine_ids}

    def add(self, reading: Reading) -> None:
        buf = self._by_machine.get(reading.machine_id)
        if buf is not None:
            buf.append(reading)

    def recent(self, n: int) -> list[Reading]:
        """The last ``n`` Readings of every machine, oldest first (what-if history)."""
        out = [r for buf in self._by_machine.values() for r in list(buf)[-n:]]
        out.sort(key=lambda r: (r.ts, r.machine_id))
        return out

    def latest(self) -> list[Reading]:
        return [buf[-1] for buf in self._by_machine.values() if buf]

    def since(self, machine_id: str | None, since_seq: int, limit: int) -> list[Reading]:
        """Readings with ``seq > since_seq`` (oldest first), for one machine or all."""
        machines = [machine_id] if machine_id is not None else list(self._by_machine)
        out: list[Reading] = []
        for m in machines:
            out.extend(r for r in self._by_machine.get(m, ()) if r.seq > since_seq)
        out.sort(key=lambda r: (r.ts, r.machine_id))
        return out[:limit]


class SimSession:
    def __init__(self, sim: LineSimulator, speed: float, history_size: int) -> None:
        self.sim = sim
        self.line = sim.line
        self.machine_ids = [m.id for m in sim.line.machines]
        self.history = ReadingHistory(self.machine_ids, history_size)
        self.runner = RealtimeRunner(sim, self._on_reading, speed=speed)
        self._listeners: list[Listener] = []
        self._task: asyncio.Task[int] | None = None
        self.max_ticks: int | None = None
        self.scenario_dir = DEFAULT_SCENARIO_DIR
        self.scenario_requests = 0

    @classmethod
    def from_options(cls, opts: SessionOptions) -> SimSession:
        line = load_line_config(opts.line_path)
        sim_cfg = load_sim_config(opts.sim_config_path)
        scenario = load_scenario(opts.scenario_path)
        sim = build_simulator(line, sim_cfg, scenario, opts.seed, opts.run_index)
        session = cls(sim, opts.speed, opts.history_size)
        session.scenario_dir = opts.scenario_path.parent
        duration = opts.duration_s if opts.duration_s is not None else sim.plan.duration_s
        session.max_ticks = round(duration)
        return session

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    async def _on_reading(self, reading: Reading) -> None:
        self.history.add(reading)
        for listener in list(self._listeners):
            result = listener(reading)
            if inspect.isawaitable(result):
                await result

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, max_ticks: int | None = None) -> asyncio.Task[int]:
        if self._task is None:
            ticks = max_ticks if max_ticks is not None else self.max_ticks
            self._task = asyncio.create_task(self.runner.run(max_ticks=ticks), name="sim-runner")
        return self._task

    async def wait(self) -> int:
        if self._task is None:
            return 0
        return await self._task

    async def stop(self) -> None:
        self.runner.stop()
        if self._task is not None:
            await self._task

    async def submit(self, cmd: Command, timeout_s: float) -> CommandAck | None:
        """Apply a command at the next tick; None if the runner did not take it in time."""
        if not self.running:
            return None
        fut = self.runner.submit(cmd)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except TimeoutError:
            fut.cancel()
            return None

    async def submit_scenario(self, req: ScenarioRequest, timeout_s: float) -> ScenarioAck | None:
        """Apply a scenario request at the next tick; None if the runner did not take it."""
        if not self.running:
            return None
        self.scenario_requests += 1
        scenario_dir = self.scenario_dir
        fut = self.runner.submit_call(lambda sim: apply_scenario(sim, req, scenario_dir))
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except TimeoutError:
            fut.cancel()
            return None

    async def whatif(self, req: WhatIfSimRequest, timeout_s: float) -> WhatIfSimResult | None:
        """Fork the running simulation (ADR 0031): None if it is not running or too slow.

        The state is copied at the next tick boundary; the two copies then run in a thread,
        so the live simulation keeps its pace.
        """
        if not self.running:
            return None
        # the history is taken in the same tick-boundary call as the snapshot, so it ends at
        # the fork point (the simulation goes on while the copies run)
        fut = self.runner.submit_call(
            lambda sim: (take_fork(sim), self.history.recent(req.history_s))
        )
        try:
            fork, history = await asyncio.wait_for(fut, timeout=timeout_s)
            return await asyncio.wait_for(
                asyncio.to_thread(run_fork, self.sim, fork, req, history), timeout=timeout_s
            )
        except TimeoutError:
            fut.cancel()
            return None
