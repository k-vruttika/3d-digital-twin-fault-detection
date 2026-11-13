"""Scenario injection (simulation only, ADR 0015): inject_fault, schedule_plan, apply_scenario,
the runner's tick-boundary calls, and SimSession.submit_scenario."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin_common.config.line import LineConfig
from twin_common.contracts import ScenarioRequest
from twin_common.enums import AckStatus, MachineState, ScenarioAction
from twin_sim.config import (
    FaultType,
    RecoveryParams,
    ScenarioConfig,
    SimConfig,
    load_scenario,
)
from twin_sim.line import LineSimulator
from twin_sim.publishers.scenario import apply_scenario
from twin_sim.publishers.session import SimSession
from twin_sim.scenario import make_run_plan
from twin_sim.testing import SimFactory

TS = datetime(2026, 1, 1, tzinfo=UTC)


def _healthy() -> ScenarioConfig:
    return ScenarioConfig(
        name="t", duration_s=3600, load_ranges=["medium"], recovery=RecoveryParams(auto=False)
    )


def _req(**kw: object) -> ScenarioRequest:
    data: dict[str, object] = {
        "request_id": "r1",
        "action": ScenarioAction.INJECT_FAULT,
        "requested_by": "test",
        "source": "mock",
        "ts": TS,
    }
    return ScenarioRequest.model_validate(data | kw)


def _states(sim: LineSimulator, seconds: int) -> list[dict[str, MachineState]]:
    out = []
    for _ in range(seconds):
        frame = sim.step(1.0)
        out.append({s.machine_id: s.state for s in frame.samples})
    return out


def test_belt_jam_trips_now_and_is_logged(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    _states(sim, 30)
    assert sim.inject_fault("CNV-01", FaultType.BELT_JAM) == (AckStatus.EXECUTED, None)
    states = _states(sim, 10)
    assert any(s["CNV-01"] is MachineState.FAULT for s in states)
    entry = sim.fault_log[-1]
    assert (entry.machine, entry.type, entry.planned_onset_s) == ("CNV-01", "belt_jam", 30.0)
    status, reason = sim.inject_fault("CNV-01", FaultType.BELT_JAM)
    assert status is AckStatus.REJECTED
    assert reason == "another fault is active on this machine"


def test_rejections(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    _states(sim, 5)
    assert sim.inject_fault("NOPE", FaultType.BEARING_WEAR)[0] is AckStatus.REJECTED
    status, reason = sim.inject_fault("CNV-01", FaultType.SPINDLE_IMBALANCE)
    assert status is AckStatus.REJECTED
    assert reason is not None
    assert "does not apply to a CONVEYOR" in reason


def test_gradual_fault_reaches_fault_with_short_progression(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    _states(sim, 10)
    assert sim.inject_fault("CNC-01", FaultType.OVERHEATING, 120.0)[0] is AckStatus.EXECUTED
    states = _states(sim, 900)
    seen = [s["CNC-01"] for s in states]
    assert MachineState.FAULT in seen
    assert seen.index(MachineState.FAULT) > 0
    assert sim.fault_log[-1].start_s == 10.0


def test_injection_survives_snapshot_restore(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy())
    _states(sim, 20)
    sim.inject_fault("CNC-01", FaultType.BEARING_WEAR, 300.0)
    snap = sim.snapshot()
    a = _states(sim, 60)
    sim.restore(snap)
    assert _states(sim, 60) == a


def test_schedule_plan_offsets_events_to_now(
    make_sim: SimFactory, config_dir: Path, sim_cfg: SimConfig, line_cfg: LineConfig
) -> None:
    sim = make_sim(_healthy())
    _states(sim, 100)
    sc = load_scenario(config_dir / "scenarios" / "cnc_stop.yaml")
    n = sim.schedule_plan(make_run_plan(sc, sim_cfg, line_cfg, 7, 0))
    assert n == 2  # STOP at +600 s, START at +1200 s
    states = _states(sim, 650)
    assert states[598]["CNC-01"] is not MachineState.IDLE
    assert states[601]["CNC-01"] is MachineState.IDLE


def test_apply_scenario_validates_names_and_lines(
    make_sim: SimFactory, config_dir: Path, sim_cfg: SimConfig, single_line_cfg: LineConfig
) -> None:
    sim = make_sim(_healthy())
    scen = config_dir / "scenarios"
    bad = apply_scenario(sim, _req(machine_id="CNC-01", fault_type="melted"), scen)
    assert bad.status is AckStatus.REJECTED
    assert "unknown fault 'melted'" in (bad.reason or "")
    load = {"action": ScenarioAction.LOAD_SCENARIO}
    missing = apply_scenario(sim, _req(**load, scenario="nope"), scen)
    assert missing.status is AckStatus.REJECTED
    assert "no scenario 'nope'" in (missing.reason or "")
    ok = apply_scenario(sim, _req(**load, scenario="cnc_stop"), scen)
    assert ok.status is AckStatus.EXECUTED
    assert ok.ts == sim.timestamp(sim.t)  # simulation time, like command acks
    single = LineSimulator(
        single_line_cfg,
        sim_cfg,
        make_run_plan(load_scenario(scen / "quick_single.yaml"), sim_cfg, single_line_cfg, 1, 0),
    )
    other = apply_scenario(single, _req(**load, scenario="cnc_stop"), scen)
    assert other.status is AckStatus.REJECTED
    assert "cannot run here" in (other.reason or "")


def test_request_contract_requires_action_fields() -> None:
    with pytest.raises(ValueError, match="requires machineId and faultType"):
        _req(machine_id="CNC-01")
    with pytest.raises(ValueError, match="requires scenario"):
        _req(action=ScenarioAction.LOAD_SCENARIO)
    with pytest.raises(ValueError, match="pattern"):
        _req(action=ScenarioAction.LOAD_SCENARIO, scenario="../secrets")


async def test_session_submit_scenario_and_failing_calls(make_sim: SimFactory) -> None:
    session = SimSession(make_sim(_healthy()), speed=50.0, history_size=100)
    assert await session.submit_scenario(_req(machine_id="CNC-01", fault_type="x"), 1.0) is None
    session.start()
    try:
        ack = await session.submit_scenario(_req(machine_id="CNV-01", fault_type="belt_jam"), 2.0)
        assert ack is not None
        assert ack.status is AckStatus.EXECUTED
        # an exception in a tick-boundary call reaches its caller and the runner keeps going
        fut = session.runner.submit_call(lambda _sim: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            await asyncio.wait_for(fut, 2.0)
        ticks = session.runner.ticks
        await asyncio.sleep(0.2)
        assert session.runner.ticks > ticks
    finally:
        await session.stop()
