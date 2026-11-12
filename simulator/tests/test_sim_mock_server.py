"""Mock machine server (FastAPI) endpoints, in-process through TestClient."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twin_common.contracts import CommandAck, Reading
from twin_common.enums import AckStatus, CommandType
from twin_common.testing import sample_command
from twin_sim.publishers.mock_server import create_app
from twin_sim.publishers.session import SessionOptions, SimSession

SPEED = 200.0  # 5 ms per simulated second
WAIT_S = 10.0


def make_session(config_dir: Path, line: str, scenario: str, duration: float) -> SimSession:
    return SimSession.from_options(
        SessionOptions(
            line_path=config_dir / line,
            scenario_path=config_dir / "scenarios" / scenario,
            sim_config_path=config_dir / "simulator.yaml",
            seed=3,
            speed=SPEED,
            duration_s=duration,
            history_size=500,
        )
    )


def wait_for(cond: object, timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while not cond():  # type: ignore[operator]
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


@pytest.fixture
def client(config_dir: Path) -> Iterator[tuple[TestClient, SimSession]]:
    session = make_session(config_dir, "line.yaml", "quick.yaml", duration=100_000)
    with TestClient(create_app(session, own_session=True)) as c:
        yield c, session


def test_info_state_history(client: tuple[TestClient, SimSession]) -> None:
    c, _ = client
    info = c.get("/info").json()
    assert info == {"lineId": "line1", "machines": ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]}
    wait_for(lambda: len(c.get("/history", params={"machineId": "CNC-01"}).json()) >= 10)
    state = [Reading.model_validate(r) for r in c.get("/state").json()]
    assert {r.machine_id for r in state} == set(info["machines"])
    hist = [
        Reading.model_validate(r)
        for r in c.get("/history", params={"machineId": "CNC-01", "sinceSeq": 3, "limit": 5}).json()
    ]
    assert [r.seq for r in hist] == [4, 5, 6, 7, 8]
    assert c.get("/history", params={"sinceSeq": -1}).status_code == 422


def test_command_round_trip(client: tuple[TestClient, SimSession]) -> None:
    c, session = client
    cmd = sample_command(machine_id="CNV-01", type=CommandType.STOP, params={})
    ack = CommandAck.model_validate(
        c.post(
            "/command", content=cmd.model_dump_json(), headers={"content-type": "application/json"}
        ).json()
    )
    assert ack.command_id == cmd.command_id
    assert ack.status is AckStatus.EXECUTED
    assert session.sim.machine("CNV-01").commanded_run is False
    again = sample_command(
        machine_id="CNV-01",
        type=CommandType.STOP,
        params={},
        command_id="cmd-2",
        idempotency_key="idem-2",
    )
    ack2 = c.post("/command", json=again.model_dump(mode="json")).json()
    assert ack2["status"] == "REJECTED"
    assert "already stopped" in ack2["reason"]


def test_command_validation(client: tuple[TestClient, SimSession]) -> None:
    c, _ = client
    bad = sample_command().model_dump(mode="json")
    bad["type"] = "SELF_DESTRUCT"
    assert c.post("/command", json=bad).status_code == 422


def test_websocket_stream_envelope(client: tuple[TestClient, SimSession]) -> None:
    c, _ = client
    with c.websocket_connect("/stream") as ws:
        envelopes = [json.loads(ws.receive_text()) for _ in range(8)]
    for env in envelopes:
        Reading.model_validate(env["reading"])
        assert env["sentAt"].endswith("+00:00")


def test_single_station_server(config_dir: Path) -> None:
    session = make_session(config_dir, "line_single.yaml", "quick_single.yaml", duration=100_000)
    with TestClient(create_app(session, own_session=True)) as c:
        assert c.get("/info").json()["machines"] == ["CNV-01"]
        wait_for(lambda: len(c.get("/state").json()) == 1)


def test_command_times_out_when_simulation_stopped(config_dir: Path) -> None:
    session = make_session(config_dir, "line.yaml", "quick.yaml", duration=1)
    with TestClient(create_app(session, own_session=True, command_timeout_s=0.5)) as c:
        wait_for(lambda: not session.running)
        cmd = sample_command(machine_id="CNV-01", type=CommandType.STOP, params={})
        ack = c.post("/command", json=cmd.model_dump(mode="json")).json()
        assert ack["status"] == "TIMEOUT"
