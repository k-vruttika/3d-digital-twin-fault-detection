from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from twin_common.contracts import Command, Reading
from twin_common.enums import AckStatus, CommandType, MachineState
from twin_sim.config import ScenarioConfig
from twin_sim.line import LineSimulator
from twin_sim.runner import RealtimeRunner

SimFactory = Callable[..., LineSimulator]
SC = ScenarioConfig(name="rt", duration_s=600, load_ranges=["medium"])


async def test_runner_emits_readings_in_order(make_sim: SimFactory) -> None:
    got: list[Reading] = []
    runner = RealtimeRunner(make_sim(SC), got.append, speed=1000.0)
    started = time.perf_counter()
    ticks = await runner.run(max_ticks=20)
    elapsed = time.perf_counter() - started
    assert ticks == 20
    assert len(got) == 80
    for mid in ("CNV-01", "CNC-01", "INSP-01", "CNV-02"):
        seqs = [r.seq for r in got if r.machine_id == mid]
        assert seqs == list(range(1, 21))
    ts = sorted({r.ts for r in got})
    assert len(ts) == 20
    assert (ts[1] - ts[0]).total_seconds() == 1.0
    assert elapsed < 5.0


async def test_runner_paces_by_speed(make_sim: SimFactory) -> None:
    runner = RealtimeRunner(make_sim(SC), lambda r: None, speed=50.0)  # 20 ms per tick
    started = time.perf_counter()
    await runner.run(max_ticks=10)
    assert time.perf_counter() - started >= 0.15


async def test_runner_async_callback_and_commands(make_sim: SimFactory) -> None:
    states: list[MachineState] = []

    async def on_reading(r: Reading) -> None:
        if r.machine_id == "CNC-01":
            states.append(r.state)

    sim = make_sim(SC)
    runner = RealtimeRunner(sim, on_reading, speed=1000.0)
    fut = runner.submit(
        Command(
            command_id="c1",
            idempotency_key="k1",
            machine_id="CNC-01",
            type=CommandType.STOP,
            issued_by="t",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await runner.run(max_ticks=3)
    assert fut.result().status is AckStatus.EXECUTED
    assert states == [MachineState.IDLE] * 3


async def test_runner_stop(make_sim: SimFactory) -> None:
    runner = RealtimeRunner(make_sim(SC), lambda r: runner.stop(), speed=1.0)
    assert await runner.run() == 1


def test_runner_rejects_bad_speed(make_sim: SimFactory) -> None:
    with pytest.raises(ValueError, match="speed"):
        RealtimeRunner(make_sim(SC), lambda r: None, speed=0.0)
