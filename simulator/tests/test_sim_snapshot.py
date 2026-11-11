from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from twin_common.config.line import LineConfig
from twin_common.contracts import Command
from twin_common.enums import CommandType
from twin_sim.config import FaultSchedule, FaultType, ScenarioConfig, ScheduledFault, SimConfig
from twin_sim.line import LineSimulator, SimSnapshot, StepFrame

SimFactory = Callable[..., LineSimulator]

SCENARIO = ScenarioConfig(
    name="snap",
    duration_s=3600,
    load_ranges=["high"],
    faults=FaultSchedule(
        scheduled=[
            ScheduledFault(type=FaultType.BEARING_WEAR, machine="CNC-01", onset_s=50),
            ScheduledFault(type=FaultType.BELT_JAM, machine="CNV-01", onset_s=400),
            ScheduledFault(type=FaultType.LUBRICATION_LOSS, machine="INSP-01", onset_s=100),
        ]
    ),
)


def _key(frames: list[StepFrame]) -> list[object]:
    return [(f.t_s, f.samples) for f in frames]


@pytest.mark.parametrize("split_at", [200, 401, 402, 700])
def test_snapshot_json_restore_is_bit_identical(
    make_sim: SimFactory, line_cfg: LineConfig, sim_cfg: SimConfig, split_at: int
) -> None:
    original = make_sim(SCENARIO, seed=9)
    original.fast_forward(split_at)
    snap_json = original.snapshot().model_dump_json()
    expected = original.fast_forward(400, record=True)

    restored = LineSimulator.from_snapshot(
        line_cfg, sim_cfg, SimSnapshot.model_validate_json(snap_json)
    )
    assert _key(restored.fast_forward(400, record=True)) == _key(expected)


def test_restore_in_place_rewinds(make_sim: SimFactory) -> None:
    sim = make_sim(SCENARIO, seed=9)
    sim.fast_forward(100)
    snap = sim.snapshot()
    a = sim.fast_forward(50, record=True)
    sim.apply_command(
        Command(
            command_id="c",
            idempotency_key="k",
            machine_id="CNC-01",
            type=CommandType.STOP,
            issued_by="t",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    sim.fast_forward(50)
    sim.restore(snap)
    assert sim.t == 100.0
    assert _key(sim.fast_forward(50, record=True)) == _key(a)


def test_snapshot_is_isolated_from_later_steps(make_sim: SimFactory) -> None:
    sim = make_sim(SCENARIO, seed=9)
    sim.fast_forward(10)
    snap = sim.snapshot()
    temp = snap.machines[1].temp
    sim.fast_forward(100)
    assert snap.machines[1].temp == temp
    assert snap.t == 10.0


def test_fast_forward_equals_repeated_steps(make_sim: SimFactory) -> None:
    a, b = make_sim(SCENARIO, seed=4), make_sim(SCENARIO, seed=4)
    steps = [a.step(1.0) for _ in range(300)]
    assert _key(b.fast_forward(300, record=True)) == _key(steps)
    assert a.snapshot().model_dump_json() == b.snapshot().model_dump_json()


def test_restore_rejects_other_plan(make_sim: SimFactory) -> None:
    a, b = make_sim(SCENARIO, seed=1), make_sim(SCENARIO, seed=2)
    with pytest.raises(ValueError, match="different run plan"):
        a.restore(b.snapshot())


def test_large_dt_is_split_into_substeps(make_sim: SimFactory) -> None:
    sim = make_sim(SCENARIO, seed=1)
    frame = sim.step(5.0)
    assert frame.t_s == 5.0
    with pytest.raises(ValueError, match="dt"):
        sim.step(0.0)
