"""What-if forks (ADR 0031): the copies, the live simulator untouched, the mock server endpoint."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twin_common.contracts import Reading
from twin_common.enums import AckStatus, CommandType, MachineState
from twin_common.whatif import WhatIfSimRequest, WhatIfSimResult
from twin_sim.config import RecoveryParams, ScenarioConfig
from twin_sim.line import LineSimulator
from twin_sim.publishers.mock_server import create_app
from twin_sim.publishers.session import SessionOptions, SimSession
from twin_sim.testing import SimFactory
from twin_sim.whatif import run_fork, take_fork

HORIZON_S = 60


def _healthy() -> ScenarioConfig:
    return ScenarioConfig(
        name="t", duration_s=3600, load_ranges=["medium"], recovery=RecoveryParams(auto=False)
    )


def _req(
    ctype: CommandType = CommandType.STOP, machine: str = "CNC-01", **params: float
) -> WhatIfSimRequest:
    return WhatIfSimRequest(
        machine_id=machine, type=ctype, params=dict(params), horizon_s=HORIZON_S, history_s=20
    )


def _states(readings: list[Reading], machine: str) -> list[MachineState]:
    return [r.state for r in readings if r.machine_id == machine]


def _run_to_steady(sim: LineSimulator, seconds: int = 60) -> None:
    sim.fast_forward(seconds)


def test_the_baseline_is_exactly_what_the_live_simulator_does_next(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy(), seed=5)
    _run_to_steady(sim)
    fork = take_fork(sim)
    t_before = sim.t
    result = run_fork(sim, fork, _req(), [])
    assert sim.t == t_before  # the live simulator was not advanced or changed
    live: list[Reading] = []
    for frame in sim.fast_forward(HORIZON_S, record=True):
        live.extend(sim.readings(frame))
    assert [r.model_dump() for r in result.baseline] == [r.model_dump() for r in live]
    assert result.as_of == sim.timestamp(t_before)


def test_stop_changes_the_projection_and_propagates_down_the_line(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy(), seed=5)
    _run_to_steady(sim)
    result = run_fork(sim, take_fork(sim), _req(), [])
    assert result.action_status is AckStatus.EXECUTED
    assert MachineState.IDLE not in _states(result.baseline, "CNC-01")
    assert MachineState.IDLE in _states(result.scenario, "CNC-01")
    # the buffers fill or drain: neighbours leave RUNNING with the stop
    assert MachineState.RUNNING in _states(result.baseline, "CNV-01")
    assert _states(result.scenario, "INSP-01")[-1] is MachineState.STARVED
    # same times and sequence numbers in both projections, continuing after the fork
    assert [r.ts for r in result.baseline] == [r.ts for r in result.scenario]
    assert result.baseline[0].ts > result.as_of


def test_a_rejected_change_leaves_the_projection_equal_to_the_baseline(
    make_sim: SimFactory,
) -> None:
    sim = make_sim(_healthy(), seed=5)
    _run_to_steady(sim)
    result = run_fork(sim, take_fork(sim), _req(CommandType.SET_SPEED, "NOPE", speed=100.0), [])
    assert result.action_status is AckStatus.REJECTED
    assert result.action_reason is not None
    assert result.scenario == result.baseline


def test_set_speed_changes_the_speed_of_the_projection_only(make_sim: SimFactory) -> None:
    sim = make_sim(_healthy(), seed=5)
    _run_to_steady(sim)
    result = run_fork(sim, take_fork(sim), _req(CommandType.SET_SPEED, speed=3000.0), [])
    assert result.action_status is AckStatus.EXECUTED
    base = [r.speed for r in result.baseline if r.machine_id == "CNC-01" and r.speed is not None]
    scen = [r.speed for r in result.scenario if r.machine_id == "CNC-01" and r.speed is not None]
    assert scen[-1] == pytest.approx(3000.0, rel=0.1)
    assert abs(base[-1] - 3000.0) > 500.0


# --------------------------------------------------------------------------- session + endpoint
def _session(config_dir: Path, duration: float = 100_000) -> SimSession:
    return SimSession.from_options(
        SessionOptions(
            line_path=config_dir / "line.yaml",
            scenario_path=config_dir / "scenarios" / "quick.yaml",
            sim_config_path=config_dir / "simulator.yaml",
            seed=3,
            speed=200.0,
            duration_s=duration,
            history_size=500,
        )
    )


@pytest.fixture
def client(config_dir: Path) -> Iterator[tuple[TestClient, SimSession]]:
    session = _session(config_dir)
    with TestClient(create_app(session, own_session=True)) as c:
        yield c, session


def test_endpoint_forks_the_running_simulation(client: tuple[TestClient, SimSession]) -> None:
    c, session = client
    deadline = time.monotonic() + 10
    while len(c.get("/history", params={"machineId": "CNC-01"}).json()) < 30:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    body = _req().model_dump(by_alias=True, mode="json")
    resp = c.post("/whatif", json=body)
    assert resp.status_code == 200, resp.text
    result = WhatIfSimResult.model_validate(resp.json())
    assert result.action_status is AckStatus.EXECUTED
    assert len(result.history) == 4 * 20  # history_s readings of every machine
    assert len(result.baseline) == len(result.scenario) == 4 * HORIZON_S
    assert all(r.ts <= result.as_of for r in result.history)
    assert all(r.ts > result.as_of for r in result.baseline)
    assert session.running  # the live simulation kept going


def test_endpoint_is_unavailable_when_the_simulation_is_not_running(config_dir: Path) -> None:
    session = _session(config_dir)  # never started
    with TestClient(create_app(session)) as c:
        resp = c.post("/whatif", json=_req().model_dump(by_alias=True, mode="json"))
        assert resp.status_code == 503
