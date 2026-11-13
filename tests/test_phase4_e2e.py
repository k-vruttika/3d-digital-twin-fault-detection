"""Phase 4 end to end: simulator publisher → adapter → backend API → command → machine → back.

For each live source (MQTT, OPC UA, mock server) a simulated full line runs at 20x speed
behind its publisher, and the backend (real uvicorn, SQLite store, API keys) ingests it:

1. STOP CNC-01 via ``POST /commands`` → ack EXECUTED; CNC-01 changes to IDLE; CNV-01 becomes
   BLOCKED and INSP-01 / CNV-02 STARVED, all with root cause CNC-01; the command and its acks
   are in the audit trail (API and store); a ``line_stoppage`` alert is raised on CNC-01.
2. START → the line runs again; the alert clears (still listed until acknowledged), is
   acknowledged and resolved; the store holds raise, clear and ack.
3. A repeated idempotency key returns the original ack without sending again.
4. Scenario injection (simulation only): a belt jam on CNV-01 is acknowledged, CNV-01 goes to
   FAULT, the request precedes the FAULT in the store; SET_SPEED on the faulted machine is
   rejected by a safety rule while STOP passes them (fail-safe).

Further tests (mock server only, no services): STOP while the data link is STALE / OFFLINE
passes the safety rules, rejections from the file source, viewer role and no keys.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from twin_backend.api.app import create_app
from twin_backend.api.server import ServerHandle
from twin_backend.runtime import Runtime, build_runtime
from twin_backend.storage.base import Stream
from twin_backend.testing import (
    REPO_ROOT,
    SimExport,
    file_runtime,
    free_port,
    settings_for,
    storage_config,
)
from twin_common.config import load_line_config
from twin_common.config.sources import SourcesConfig
from twin_common.contracts import Alert, Command, ScenarioRequest, StateChange
from twin_common.enums import AdapterKind, MachineState
from twin_sim.config import load_scenario, load_sim_config
from twin_sim.publishers.http import ServerHandle as SimServerHandle
from twin_sim.publishers.mock_server import create_app as create_mock_app
from twin_sim.publishers.mqtt import MqttPublisher, MqttPublisherOptions
from twin_sim.publishers.opcua import OpcUaServer
from twin_sim.publishers.session import SimSession, build_simulator

SPEED = 20.0
WAIT_S = 60.0
OP_KEY = "e2e-operator-key-0123456789"
VIEW_KEY = "e2e-viewer-key-0123456789"
KEYS: dict[str, Any] = {
    "api_keys_operator": SecretStr(f"olga:{OP_KEY}"),
    "api_keys_viewer": SecretStr(f"victor:{VIEW_KEY}"),
}
FAST_RECONNECT = {"initial_s": 0.1, "max_s": 0.5, "jitter": 0.0}
CONFIG = REPO_ROOT / "config"
ALL = ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]


@dataclass
class Stack:
    rt: Runtime
    api: ServerHandle
    session: SimSession
    op: httpx.AsyncClient
    viewer: httpx.AsyncClient


def _session() -> SimSession:
    """The full line without scripted commands, faults or auto-recovery: only the test acts."""
    line = load_line_config(CONFIG / "line.yaml")
    scenario = load_scenario(CONFIG / "scenarios" / "cnc_stop.yaml").model_copy(
        update={"commands": [], "duration_s": 36000.0}
    )
    scenario = scenario.model_copy(
        update={"recovery": scenario.recovery.model_copy(update={"auto": False})}
    )
    sim = build_simulator(line, load_sim_config(CONFIG / "simulator.yaml"), scenario, 7)
    session = SimSession(sim, SPEED, 20_000)
    session.scenario_dir = CONFIG / "scenarios"
    return session


@asynccontextmanager
async def live_stack(
    kind: AdapterKind, sources: SourcesConfig, tmp: Path, broker: Any = None
) -> AsyncIterator[Stack]:
    session = _session()
    srcs = sources.model_copy(
        update={
            "active": kind,
            "reconnect": sources.reconnect.model_copy(update=FAST_RECONNECT),
        }
    )
    stop_publisher: Callable[[], Awaitable[None]]
    if kind is AdapterKind.MOCK:
        assert srcs.mock is not None
        sim_server = SimServerHandle(create_mock_app(session))
        await sim_server.start()
        srcs = srcs.model_copy(
            update={"mock": srcs.mock.model_copy(update={"base_url": sim_server.base_url})}
        )
        stop_publisher = sim_server.stop
    elif kind is AdapterKind.OPCUA:
        assert srcs.opcua is not None
        endpoint = f"opc.tcp://127.0.0.1:{free_port()}/linetwin/"
        opc = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
        await opc.start()
        srcs = srcs.model_copy(
            update={"opcua": srcs.opcua.model_copy(update={"endpoint": endpoint})}
        )
        stop_publisher = opc.stop
    else:
        assert srcs.mqtt is not None
        prefix = f"test/{uuid.uuid4().hex[:12]}/line1"
        mqtt_cfg = srcs.mqtt.model_copy(
            update={
                "host": broker.host,
                "port": broker.port,
                "topic_prefix": prefix,
                "client_id": f"linetwin-e2e-{uuid.uuid4().hex[:8]}",
                "session_expiry_s": 10,
            }
        )
        srcs = srcs.model_copy(update={"mqtt": mqtt_cfg})
        publisher = MqttPublisher(
            session,
            MqttPublisherOptions(
                host=broker.host,
                port=broker.port,
                topic_prefix=prefix,
                qos=mqtt_cfg.qos,
                keepalive_s=mqtt_cfg.keepalive_s,
                client_id=f"linetwin-e2e-sim-{uuid.uuid4().hex[:8]}",
                reconnect=srcs.reconnect,
                command_timeout_s=srcs.command_timeout_s,
                buffer_size=100_000,
            ),
        )
        task = asyncio.create_task(publisher.run())
        await asyncio.wait_for(publisher.connected.wait(), WAIT_S)

        async def stop_mqtt() -> None:
            publisher.stop()
            await asyncio.wait_for(task, WAIT_S)

        stop_publisher = stop_mqtt
    session.start()
    base = storage_config()
    st = base.model_copy(
        update={"sqlite": base.sqlite.model_copy(update={"path": str(tmp / "e2e.sqlite")})}
    )
    rt = build_runtime(settings_for(**KEYS), root=REPO_ROOT, sources=srcs, storage_cfg=st)
    api = ServerHandle(create_app(rt))
    await api.start()
    op = httpx.AsyncClient(base_url=api.base_url, headers={"X-API-Key": OP_KEY}, timeout=30)
    viewer = httpx.AsyncClient(base_url=api.base_url, headers={"X-API-Key": VIEW_KEY}, timeout=30)
    try:
        yield Stack(rt, api, session, op, viewer)
    finally:
        await op.aclose()
        await viewer.aclose()
        await api.stop()
        await stop_publisher()
        await session.stop()


async def poll(fn: Callable[[], Awaitable[Any]], what: str, timeout: float = WAIT_S) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = await fn()
        if last:
            return last
        await asyncio.sleep(0.1)
    pytest.fail(f"timed out waiting for {what} (last: {last!r})")


async def machines(c: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    body = (await c.get("/machines")).json()
    return {m["machineId"]: m for m in body["machines"]}


async def send(
    c: httpx.AsyncClient, ctype: str, machine: str, key: str | None = None, **params: float
) -> httpx.Response:
    return await c.post(
        "/commands",
        json={
            "machineId": machine,
            "type": ctype,
            "params": params,
            "idempotencyKey": key or uuid.uuid4().hex,
        },
    )


async def stored(rt: Runtime, stream: Stream) -> list[Any]:
    assert rt.storage is not None
    await rt.storage.flush()
    return [s.item for s in await rt.storage.store.fetch(stream)]


async def _command_flow(s: Stack) -> None:
    async def all_running() -> bool:
        ms = await machines(s.viewer)
        return all(ms[m]["state"] == "RUNNING" for m in ALL)

    await poll(all_running, "the whole line RUNNING")

    # 1. STOP CNC-01
    stop_key = uuid.uuid4().hex
    r = await send(s.op, "STOP", "CNC-01", stop_key)
    assert r.status_code == 200, r.text
    ack = r.json()
    assert ack["status"] == "EXECUTED", ack
    assert r.headers["Idempotent-Replay"] == "false"
    command_id = ack["commandId"]

    async def propagated() -> dict[str, dict[str, Any]] | None:
        ms = await machines(s.viewer)
        want = {"CNC-01": "IDLE", "CNV-01": "BLOCKED", "INSP-01": "STARVED", "CNV-02": "STARVED"}
        return ms if all(ms[m]["state"] == st for m, st in want.items()) else None

    ms = await poll(propagated, "BLOCKED/STARVED propagation")
    for m in ("CNV-01", "INSP-01", "CNV-02"):
        assert ms[m]["rootCause"] == {"cause": "CNC-01", "unknown": False}, m
    body = (await s.viewer.get("/machines")).json()
    assert body["line"]["impacted"]["CNC-01"] == ["CNV-01", "INSP-01", "CNV-02"]

    changes = [c for c in await stored(s.rt, Stream.STATE_CHANGES) if isinstance(c, StateChange)]
    assert any(c.machine_id == "CNC-01" and c.state is MachineState.IDLE for c in changes)
    assert any(c.machine_id == "CNV-01" and c.reason == "BLOCKED by CNC-01" for c in changes)
    assert any(c.machine_id == "INSP-01" and c.reason == "STARVED by CNC-01" for c in changes)

    audit = (await s.viewer.get(f"/commands/{command_id}")).json()
    assert audit["command"]["type"] == "STOP"
    assert audit["command"]["issuedBy"] == "olga"
    assert [a["status"] for a in audit["acks"]] == ["ACCEPTED", "EXECUTED"]
    cmds = [c for c in await stored(s.rt, Stream.COMMANDS) if isinstance(c, Command)]
    assert [c.command_id for c in cmds] == [command_id]
    acks = [a for a in await stored(s.rt, Stream.COMMAND_ACKS) if a.command_id == command_id]
    assert sorted(a.status.value for a in acks) == ["ACCEPTED", "EXECUTED"]

    async def stoppage_alert() -> dict[str, Any] | None:
        alerts = (await s.viewer.get("/alerts", params={"machine": "CNC-01"})).json()["alerts"]
        return next((a for a in alerts if a["kind"] == "line_stoppage"), None)

    alert = await poll(stoppage_alert, "line_stoppage alert")
    assert alert["active"] is True
    assert alert["acknowledged"] is False

    # 2. START: the line recovers and the alert clears
    r = await send(s.op, "START", "CNC-01")
    assert r.json()["status"] == "EXECUTED", r.text
    await poll(all_running, "the line RUNNING again")

    async def cleared() -> dict[str, Any] | None:
        a = await stoppage_alert()
        return a if a is not None and not a["active"] else None

    a = await poll(cleared, "the alert to clear")
    assert a["alertId"] == alert["alertId"]
    assert a["acknowledged"] is False  # cleared, still listed until acknowledged
    r = await s.op.post(f"/alerts/{alert['alertId']}/ack")
    assert r.status_code == 200
    assert r.json()["ackedBy"] == "olga"
    assert await stoppage_alert() is None  # resolved
    await s.rt.engine.drain()
    events = [
        e
        for e in await stored(s.rt, Stream.ALERTS)
        if isinstance(e, Alert) and e.alert_id == alert["alertId"]
    ]
    assert [(e.active, e.acknowledged) for e in sorted(events, key=lambda e: e.rev)] == [
        (True, False),
        (False, False),
        (False, True),
    ]

    # 3. the same idempotency key returns the original ack and sends nothing
    again = await send(s.op, "STOP", "CNC-01", stop_key)
    assert again.headers["Idempotent-Replay"] == "true"
    assert again.json() == ack
    assert len(await stored(s.rt, Stream.COMMANDS)) == 2  # STOP + START only
    clash = await send(s.op, "START", "CNC-01", stop_key)
    assert clash.status_code == 409


async def _scenario_flow(s: Stack) -> None:
    r = await s.op.post(
        "/scenario",
        json={"action": "INJECT_FAULT", "machineId": "CNV-01", "faultType": "belt_jam"},
    )
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["acks"][0]["status"] == "EXECUTED", record
    request_id = record["request"]["requestId"]

    async def faulted() -> bool:
        return bool((await machines(s.viewer))["CNV-01"]["state"] == "FAULT")

    await poll(faulted, "CNV-01 FAULT after the injected jam")
    fault_alert = await poll(
        lambda: _alert(s, "CNV-01", "machine_fault"), "machine_fault alert on CNV-01"
    )
    assert fault_alert["severity"] == "CRITICAL"

    # the request is stored and precedes the FAULT on the event-time axis
    rows = await stored(s.rt, Stream.SCENARIO_REQUESTS)
    req = next(x for x in rows if isinstance(x, ScenarioRequest) and x.request_id == request_id)
    assert req.requested_by == "olga"
    first_fault = min(
        c.ts
        for c in await stored(s.rt, Stream.STATE_CHANGES)
        if isinstance(c, StateChange) and c.machine_id == "CNV-01" and c.state is MachineState.FAULT
    )
    assert req.ts <= first_fault
    listed = (await s.viewer.get("/scenario")).json()["requests"]
    assert [x["request"]["requestId"] for x in listed] == [request_id]
    assert (await s.viewer.get(f"/scenario/{request_id}")).json()["acks"][0]["status"] == "EXECUTED"

    unknown = await s.op.post(
        "/scenario",
        json={"action": "INJECT_FAULT", "machineId": "CNV-01", "faultType": "melted"},
    )
    assert unknown.json()["acks"][0]["status"] == "REJECTED"
    loaded = await s.op.post("/scenario", json={"action": "LOAD_SCENARIO", "scenario": "cnc_stop"})
    assert loaded.json()["acks"][0]["status"] == "EXECUTED", loaded.text

    # safety: no setpoint change on a faulted machine; STOP always passes (fail-safe)
    r = await send(s.op, "SET_SPEED", "CNV-01", speed=1200.0)
    assert r.json()["status"] == "REJECTED"
    assert "faulted" in r.json()["reason"]
    rid = r.json()["commandId"]
    assert [a["status"] for a in (await s.viewer.get(f"/commands/{rid}")).json()["acks"]] == [
        "REJECTED"
    ]
    stop = await send(s.op, "STOP", "CNV-01")
    stop_audit = (await s.viewer.get(f"/commands/{stop.json()['commandId']}")).json()
    assert stop_audit["acks"][0]["status"] == "ACCEPTED"  # passed the safety rules
    assert stop.json()["status"] in ("EXECUTED", "REJECTED")  # the machine decides
    mnt = await send(s.op, "SCHEDULE_MAINTENANCE", "CNC-01")  # CNC-01 is RUNNING
    assert mnt.json()["status"] == "REJECTED"
    assert "STOP it first" in mnt.json()["reason"]


async def _alert(s: Stack, machine: str, kind: str) -> dict[str, Any] | None:
    alerts = (await s.viewer.get("/alerts", params={"machine": machine})).json()["alerts"]
    return next((a for a in alerts if a["kind"] == kind and a["active"]), None)


async def test_command_flow_over_mock(sources_cfg: SourcesConfig, tmp_path: Path) -> None:
    async with live_stack(AdapterKind.MOCK, sources_cfg, tmp_path) as s:
        await _command_flow(s)
        await _scenario_flow(s)


async def test_command_flow_over_opcua(sources_cfg: SourcesConfig, tmp_path: Path) -> None:
    async with live_stack(AdapterKind.OPCUA, sources_cfg, tmp_path) as s:
        await _command_flow(s)
        await _scenario_flow(s)


@pytest.mark.services
async def test_command_flow_over_mqtt(
    mqtt_broker: Any, sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    async with live_stack(AdapterKind.MQTT, sources_cfg, tmp_path, mqtt_broker) as s:
        await _command_flow(s)
        await _scenario_flow(s)


async def test_stop_passes_safety_while_the_link_is_down(
    sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    """Publisher paused: CNC-01 goes STALE, then OFFLINE. STOP is still sent (fail-safe),
    setpoints are refused by the data-link rule."""
    async with live_stack(AdapterKind.MOCK, sources_cfg, tmp_path) as s:
        await poll(lambda: _state_is(s, "CNC-01", "RUNNING"), "CNC-01 RUNNING")
        await s.session.stop()  # no more data and no more command handling
        for state in ("STALE", "OFFLINE"):
            await poll(lambda st=state: _state_is(s, "CNC-01", st), f"CNC-01 {state}")  # type: ignore[misc]
            r = await send(s.op, "STOP", "CNC-01")
            audit = (await s.viewer.get(f"/commands/{r.json()['commandId']}")).json()
            assert audit["acks"][0]["status"] == "ACCEPTED", audit  # not a safety rejection
            assert r.json()["status"] == "TIMEOUT"  # the paused simulator never applies it
            r = await send(s.op, "SET_SPEED", "CNC-01", speed=8000.0)
            assert r.json()["status"] == "REJECTED"
            assert "data link is down" in r.json()["reason"]
        offline = await poll(
            lambda: _alert(s, "CNC-01", "machine_offline"), "machine_offline alert"
        )
        assert offline["severity"] == "CRITICAL"


async def _state_is(s: Stack, machine: str, state: str) -> bool:
    return bool((await machines(s.viewer))[machine]["state"] == state)


# ---------------------------------------------------------------- rejections without a live source
async def test_file_source_viewer_and_no_keys_rejections(
    sim_export: SimExport, tmp_path: Path
) -> None:
    rt = file_runtime(sim_export, tmp_path, settings=settings_for(**KEYS))
    api = ServerHandle(create_app(rt))
    await api.start()
    try:
        await rt.manager.wait_idle()
        await rt.engine.drain()
        async with (
            httpx.AsyncClient(base_url=api.base_url, headers={"X-API-Key": OP_KEY}) as op,
            httpx.AsyncClient(base_url=api.base_url, headers={"X-API-Key": VIEW_KEY}) as viewer,
        ):
            r = await send(op, "STOP", "CNC-01")
            assert r.status_code == 200
            assert r.json()["status"] == "REJECTED"
            assert "file adapter) does not accept commands" in r.json()["reason"]
            audit = (await viewer.get(f"/commands/{r.json()['commandId']}")).json()
            assert [a["status"] for a in audit["acks"]] == ["REJECTED"]
            assert (await send(viewer, "STOP", "CNC-01")).status_code == 403
            assert (await send(op, "STOP", "NOPE")).status_code == 404
            bad = await op.post(
                "/commands",
                json={"machineId": "CNC-01", "type": "SET_SPEED", "idempotencyKey": "z"},
            )
            assert bad.status_code == 422  # SET_SPEED needs a speed
            sc = await op.post(
                "/scenario",
                json={"action": "INJECT_FAULT", "machineId": "CNC-01", "faultType": "belt_jam"},
            )
            assert sc.status_code == 409
            assert "does not support scenario injection" in sc.json()["detail"]
            assert (
                await viewer.post("/scenario", json={"action": "LOAD_SCENARIO", "scenario": "x"})
            ).status_code == 403
            incomplete = await op.post("/scenario", json={"action": "INJECT_FAULT"})
            assert incomplete.status_code == 422
            listed = (await viewer.get("/scenario")).json()["requests"]
            assert len(listed) == 1  # the rejected request is stored too
            assert listed[0]["acks"][0]["status"] == "REJECTED"
            history = (await viewer.get("/commands", params={"machines": "CNC-01"})).json()
            assert [c["command"]["commandId"] for c in history["commands"]] == [
                r.json()["commandId"]
            ]
    finally:
        await api.stop()
    rt2 = file_runtime(sim_export, tmp_path / "nokeys", settings=settings_for())
    api2 = ServerHandle(create_app(rt2))
    await api2.start()
    try:
        async with httpx.AsyncClient(base_url=api2.base_url) as anon:
            r = await send(anon, "STOP", "CNC-01")
            assert r.status_code == 403
            assert "LINETWIN_API_KEYS_OPERATOR" in r.json()["detail"]
    finally:
        await api2.stop()
