"""Line coupling (STARVED/BLOCKED propagation, root cause) and commands."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin_common.contracts import Command, ParamValue
from twin_common.enums import AckStatus, CommandType, MachineState
from twin_sim.config import (
    FaultSchedule,
    FaultType,
    RecoveryParams,
    ScenarioConfig,
    ScheduledFault,
    SourceParams,
    load_scenario,
)
from twin_sim.line import SOURCE_ID, LineSimulator

SimFactory = Callable[..., LineSimulator]
TS = datetime(2026, 1, 1, tzinfo=UTC)
_counter = iter(range(1_000_000))


def cmd(machine: str, ctype: CommandType, key: str | None = None, **params: float) -> Command:
    n = next(_counter)
    wire: dict[str, ParamValue] = dict(params)
    return Command(
        command_id=f"c{n}",
        idempotency_key=key or f"k{n}",
        machine_id=machine,
        type=ctype,
        params=wire,
        issued_by="test",
        ts=TS,
    )


def _healthy(duration: float = 3600) -> ScenarioConfig:
    return ScenarioConfig(name="t", duration_s=duration, load_ranges=["medium"])


def test_stopping_cnc_blocks_upstream_and_starves_downstream(
    make_sim: SimFactory, config_dir: Path
) -> None:
    sim = make_sim(load_scenario(config_dir / "scenarios" / "cnc_stop.yaml"), seed=7)
    frames = sim.run()
    by_t = {round(f.t_s): {s.machine_id: s for s in f.samples} for f in frames}

    # Before the stop: everything runs normally (no flow states in steady operation).
    assert all(s.state is MachineState.RUNNING for t in range(100, 600) for s in by_t[t].values())
    # After the stop (600 s): eventually CNV-01 BLOCKED, INSP-01 and CNV-02 STARVED.
    late = by_t[1100]
    assert late["CNC-01"].state is MachineState.IDLE
    assert late["CNV-01"].state is MachineState.BLOCKED
    assert late["INSP-01"].state is MachineState.STARVED
    assert late["CNV-02"].state is MachineState.STARVED
    for mid in ("CNV-01", "INSP-01", "CNV-02", "CNC-01"):
        assert late[mid].root_cause == "CNC-01"
    assert late["CNV-01"].buffer_out == 10  # B1 full
    first_blocked = min(t for t in by_t if by_t[t]["CNV-01"].state is MachineState.BLOCKED)
    assert 600 < first_blocked < 800
    # After the restart (1200 s): recovered.
    assert all(s.state is MachineState.RUNNING for s in by_t[1500].values())


def test_direct_stop_command_propagates(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    sim.fast_forward(60)
    assert sim.apply_command(cmd("CNC-01", CommandType.STOP)).status is AckStatus.EXECUTED
    sim.fast_forward(300)
    assert sim.machine("CNV-01").state is MachineState.BLOCKED
    assert sim.machine("INSP-01").state is MachineState.STARVED


def test_fault_propagates_with_root_cause(make_sim: SimFactory) -> None:
    sc = _healthy(800).model_copy(
        update={
            "recovery": RecoveryParams(auto=False),
            "faults": FaultSchedule(
                scheduled=[ScheduledFault(type=FaultType.BELT_JAM, machine="INSP-01", onset_s=100)]
            ),
        }
    )
    sim = make_sim(sc)
    sim.fast_forward(400)
    assert sim.machine("INSP-01").state is MachineState.FAULT
    assert sim.machine("CNV-02").state is MachineState.STARVED
    assert sim.machine("CNV-02").root_cause == "INSP-01"
    assert sim.machine("CNC-01").state is MachineState.BLOCKED
    assert sim.machine("CNC-01").root_cause == "INSP-01"
    # CNV-01 blocked behind a blocked CNC: root cause follows the chain to INSP-01.
    assert sim.machine("CNV-01").state is MachineState.BLOCKED
    assert sim.machine("CNV-01").root_cause == "INSP-01"


def test_source_starvation_root_cause(make_sim: SimFactory) -> None:
    sc = _healthy(300).model_copy(
        update={"source": SourceParams(arrival_interval_s=200.0, jitter_frac=0.0)}
    )
    sim = make_sim(sc)
    sim.fast_forward(150)
    assert sim.machine("CNV-01").state is MachineState.STARVED
    assert sim.machine("CNV-01").root_cause == SOURCE_ID


# --------------------------------------------------------------------------- commands


def test_start_stop_and_rejections(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    sim.step()
    assert sim.apply_command(cmd("CNC-01", CommandType.STOP)).status is AckStatus.EXECUTED
    assert sim.machine("CNC-01").state is MachineState.IDLE
    again = sim.apply_command(cmd("CNC-01", CommandType.STOP))
    assert again.status is AckStatus.REJECTED
    assert again.reason
    assert sim.apply_command(cmd("CNC-01", CommandType.START)).status is AckStatus.EXECUTED
    sim.step()
    assert sim.machine("CNC-01").state is MachineState.RUNNING
    unknown = sim.apply_command(cmd("XYZ", CommandType.START))
    assert unknown.status is AckStatus.REJECTED
    assert sim.apply_command(cmd("CNC-01", CommandType.RESET)).status is AckStatus.REJECTED


def test_set_speed_and_load(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    sim.fast_forward(30)
    ack = sim.apply_command(cmd("CNC-01", CommandType.SET_SPEED, speed=6000.0))
    assert ack.status is AckStatus.EXECUTED
    too_fast = sim.apply_command(cmd("CNC-01", CommandType.SET_SPEED, speed=50000.0))
    assert too_fast.status is AckStatus.REJECTED
    sim.fast_forward(30)
    assert 5700 < sim.machine("CNC-01").speed < 6000
    assert sim.apply_command(cmd("CNC-01", CommandType.SET_LOAD, load=90.0)).status is (
        AckStatus.EXECUTED
    )
    assert sim.apply_command(cmd("CNC-01", CommandType.SET_LOAD, load=150.0)).status is (
        AckStatus.REJECTED
    )
    sim.fast_forward(600)
    assert sim.machine("CNC-01").load_eff > 75.0


def test_idempotency_returns_cached_ack(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    first = sim.apply_command(cmd("CNC-01", CommandType.STOP, key="same"))
    second = sim.apply_command(cmd("CNC-01", CommandType.STOP, key="same"))
    assert first == second
    assert first.status is AckStatus.EXECUTED


def test_maintenance_resets_wear_and_reset_requires_fault(make_sim: SimFactory) -> None:
    sc = _healthy(3000).model_copy(
        update={
            "recovery": RecoveryParams(auto=False),
            "faults": FaultSchedule(
                scheduled=[
                    ScheduledFault(
                        type=FaultType.BEARING_WEAR,
                        machine="CNC-01",
                        onset_s=10,
                        progression_s=900.0,
                    )
                ]
            ),
        }
    )
    sim = make_sim(sc)
    while sim.machine("CNC-01").state is not MachineState.FAULT:
        sim.step()
        assert sim.t < 3000
    m = sim.machine("CNC-01")

    def state() -> MachineState:  # re-read each time (avoids stale type narrowing)
        return sim.machine("CNC-01").state

    assert m.fault_w > 0.5
    rejected = sim.apply_command(cmd("CNC-01", CommandType.START))
    assert rejected.status is AckStatus.REJECTED
    # RESET keeps the wear: after START the machine trips again quickly.
    assert sim.apply_command(cmd("CNC-01", CommandType.RESET)).status is AckStatus.EXECUTED
    assert state() is MachineState.IDLE
    assert m.fault_w > 0.5
    sim.apply_command(cmd("CNC-01", CommandType.START))
    sim.fast_forward(5)
    assert sim.machine("CNC-01").state is MachineState.FAULT
    # Maintenance clears everything and returns to RUNNING afterwards.
    ack = sim.apply_command(cmd("CNC-01", CommandType.SCHEDULE_MAINTENANCE, durationS=120.0))
    assert ack.status is AckStatus.EXECUTED
    assert state() is MachineState.MAINTENANCE
    assert m.fault_w == 0.0
    assert m.fault is None
    assert sim.apply_command(cmd("CNC-01", CommandType.START)).status is AckStatus.REJECTED
    sim.fast_forward(130)
    assert state() is MachineState.RUNNING
    sim.fast_forward(300)
    assert state() is MachineState.RUNNING


def test_auto_recovery_creates_multiple_fault_cycles(make_sim: SimFactory) -> None:
    sc = _healthy(1500).model_copy(
        update={
            "faults": FaultSchedule(
                scheduled=[
                    ScheduledFault(type=FaultType.BELT_JAM, machine="CNV-01", onset_s=100),
                    ScheduledFault(type=FaultType.BELT_JAM, machine="CNV-01", onset_s=600),
                ]
            )
        }
    )
    sim = make_sim(sc)
    frames = sim.run()
    states = [f.samples[0].state for f in frames]
    onsets = sum(
        1
        for a, b in itertools.pairwise(states)
        if a is not MachineState.FAULT and b is MachineState.FAULT
    )
    assert onsets == 2
    assert states[-1] is MachineState.RUNNING
    assert [e.fault_s for e in sim.fault_log] == [pytest.approx(100.0), pytest.approx(600.0)]
