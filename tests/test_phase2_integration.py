"""Phase 2 integration: one simulated run through every adapter.

Key test: the same simulated run fed through the File (JSONL and CSV), Mock, MQTT and OPC UA
adapters produces identical canonical Reading sequences. Commands sent through the MQTT,
OPC UA and Mock adapters reach the simulator and return an ack. OPC UA and mock run
in-process; MQTT needs the Docker Compose broker (``services`` marker).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from twin_backend.adapters.file import FileReplayAdapter
from twin_backend.adapters.mock import MockServerAdapter
from twin_backend.adapters.mqtt import MqttAdapter
from twin_backend.adapters.opcua import OpcUaAdapter
from twin_backend.clock import SystemClock
from twin_backend.testing import (
    MqttService,
    SimExport,
    StreamCollector,
    canonical_order,
    free_port,
    load_jsonl_readings,
    matching_session,
)
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping, SourcesConfig
from twin_common.contracts import CommandAck, Reading
from twin_common.enums import AckStatus, AdapterStatus, CommandType, Quality
from twin_common.testing import sample_command
from twin_sim.publishers.http import ServerHandle
from twin_sim.publishers.mock_server import create_app
from twin_sim.publishers.mqtt import MqttPublisher, MqttPublisherOptions
from twin_sim.publishers.opcua import OpcUaServer
from twin_sim.publishers.session import SimSession

SPEED = 50.0  # the 120 s run takes ~2.4 s
TIMEOUT_S = 30.0
FAST_RECONNECT = {"initial_s": 0.1, "max_s": 0.5, "jitter": 0.0}


def fast(sources: SourcesConfig) -> SourcesConfig:
    return sources.model_copy(
        update={"reconnect": sources.reconnect.model_copy(update=FAST_RECONNECT)}
    )


def expected_run(export: SimExport, line: LineConfig) -> list[Reading]:
    return canonical_order(load_jsonl_readings(export.jsonl), [m.id for m in line.machines])


async def collect_live(
    collector: StreamCollector, session: SimSession, n_expected: int
) -> list[Reading]:
    """Connect first, then run the simulation, then collect every reading."""
    await collector.start()
    try:
        await collector.wait_connected()
        session.start()
        await collector.until(lambda: len(collector.out) >= n_expected, TIMEOUT_S)
        await session.wait()
    finally:
        await collector.stop()
    return collector.out


# -- adapter factories ----------------------------------------------------------------------
def file_adapter(sources: SourcesConfig, line: LineConfig, path: Path, fmt: str) -> Any:
    assert sources.file is not None
    cfg = sources.file.model_copy(
        update={"path": str(path), "format": fmt, "loop": False, "replay_speed": SPEED}
    )
    if fmt != "csv":
        cfg = cfg.model_copy(update={"mapping": FieldMapping()})
    return FileReplayAdapter(cfg, line, root=Path("."), clock=SystemClock()), cfg


def opcua_sources(sources: SourcesConfig, endpoint: str) -> SourcesConfig:
    assert sources.opcua is not None
    return fast(sources).model_copy(
        update={"opcua": sources.opcua.model_copy(update={"endpoint": endpoint})}
    )


def opcua_adapter(sources: SourcesConfig, line: LineConfig) -> OpcUaAdapter:
    assert sources.opcua is not None
    return OpcUaAdapter(
        sources.opcua,
        line,
        reconnect=sources.reconnect,
        command_timeout_s=sources.command_timeout_s,
        clock=SystemClock(),
    )


def mqtt_setup(
    sources: SourcesConfig, broker: MqttService, session: SimSession
) -> tuple[SourcesConfig, MqttAdapter, MqttPublisher]:
    assert sources.mqtt is not None
    prefix = f"test/{uuid.uuid4().hex[:12]}/line1"  # isolates concurrent/previous runs
    mqtt_cfg = sources.mqtt.model_copy(
        update={
            "host": broker.host,
            "port": broker.port,
            "topic_prefix": prefix,
            "client_id": f"linetwin-test-{uuid.uuid4().hex[:8]}",
            "session_expiry_s": 10,
        }
    )
    assert mqtt_cfg.qos == 1  # QoS 0 may legitimately drop messages (ADR 0007)
    srcs = fast(sources).model_copy(update={"mqtt": mqtt_cfg})
    adapter = MqttAdapter(
        mqtt_cfg,
        session.line,
        reconnect=srcs.reconnect,
        command_timeout_s=srcs.command_timeout_s,
        clock=SystemClock(),
    )
    publisher = MqttPublisher(
        session,
        MqttPublisherOptions(
            host=broker.host,
            port=broker.port,
            topic_prefix=prefix,
            qos=mqtt_cfg.qos,
            keepalive_s=mqtt_cfg.keepalive_s,
            client_id=f"linetwin-sim-{uuid.uuid4().hex[:8]}",
            reconnect=srcs.reconnect,
            command_timeout_s=srcs.command_timeout_s,
            buffer_size=100_000,
        ),
    )
    return srcs, adapter, publisher


# -- identical sequences --------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
async def test_file_replay_matches_simulated_run(
    fmt: str, sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    path = sim_export.jsonl if fmt == "jsonl" else sim_export.csv
    adapter, file_cfg = file_adapter(sources_cfg, line_cfg, path, fmt)
    srcs = sources_cfg.model_copy(update={"file": file_cfg})  # mapping matches the format
    collector = StreamCollector(adapter, line_cfg, srcs)
    expected = expected_run(sim_export, line_cfg)
    await collector.start()
    try:
        await collector.until(lambda: adapter.health().status is AdapterStatus.EOF, TIMEOUT_S)
    finally:
        await collector.stop()
    assert canonical_order(collector.out, [m.id for m in line_cfg.machines]) == expected


async def test_mock_server_matches_simulated_run(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export, SPEED)
    server = ServerHandle(create_app(session))
    await server.start()
    srcs = fast(sources_cfg)
    assert srcs.mock is not None
    adapter = MockServerAdapter(
        srcs.mock.model_copy(update={"base_url": server.base_url}),
        line_cfg,
        reconnect=srcs.reconnect,
        command_timeout_s=srcs.command_timeout_s,
        clock=SystemClock(),
    )
    expected = expected_run(sim_export, line_cfg)
    try:
        out = await collect_live(StreamCollector(adapter, line_cfg, srcs), session, len(expected))
    finally:
        await server.stop()
    assert canonical_order(out, [m.id for m in line_cfg.machines]) == expected


async def test_opcua_matches_simulated_run(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export, SPEED)
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/linetwin/"
    srcs = opcua_sources(sources_cfg, endpoint)
    assert srcs.opcua is not None
    server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
    await server.start()
    expected = expected_run(sim_export, line_cfg)
    try:
        collector = StreamCollector(opcua_adapter(srcs, line_cfg), line_cfg, srcs)
        out = await collect_live(collector, session, len(expected))
    finally:
        await server.stop()
    assert server.events_fired == len(expected)
    assert canonical_order(out, [m.id for m in line_cfg.machines]) == expected


@pytest.mark.services
async def test_mqtt_matches_simulated_run(
    mqtt_broker: MqttService,
    sim_export: SimExport,
    line_cfg: LineConfig,
    sources_cfg: SourcesConfig,
) -> None:
    import asyncio

    session = matching_session(sim_export, SPEED)
    srcs, adapter, publisher = mqtt_setup(sources_cfg, mqtt_broker, session)
    pub_task = asyncio.create_task(publisher.run())
    expected = expected_run(sim_export, line_cfg)
    try:
        await asyncio.wait_for(publisher.connected.wait(), TIMEOUT_S)
        out = await collect_live(StreamCollector(adapter, line_cfg, srcs), session, len(expected))
    finally:
        publisher.stop()
        await asyncio.wait_for(pub_task, TIMEOUT_S)
    assert canonical_order(out, [m.id for m in line_cfg.machines]) == expected
    assert publisher.dropped == 0


# -- commands -------------------------------------------------------------------------------
async def check_commands(send: Any, session: SimSession) -> None:
    """SET_SPEED is executed and reaches the simulator; an invalid RESET is rejected."""
    ok = sample_command(
        machine_id="CNC-01",
        type=CommandType.SET_SPEED,
        params={"speed": 8800.0},
        command_id=f"c-{uuid.uuid4().hex[:6]}",
        idempotency_key=uuid.uuid4().hex,
    )
    ack: CommandAck = await send(ok)
    assert (ack.command_id, ack.status) == (ok.command_id, AckStatus.EXECUTED)
    assert session.sim.machine("CNC-01").speed_set == 8800.0
    bad = sample_command(
        machine_id="CNC-01",
        type=CommandType.RESET,
        params={},
        command_id=f"c-{uuid.uuid4().hex[:6]}",
        idempotency_key=uuid.uuid4().hex,
    )
    ack = await send(bad)
    assert ack.status is AckStatus.REJECTED
    assert ack.reason == "machine is not faulted"


async def test_commands_over_opcua(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export, 5.0)
    session.max_ticks = None  # keep running while commands are sent
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/linetwin/"
    srcs = opcua_sources(sources_cfg, endpoint)
    assert srcs.opcua is not None
    server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
    await server.start()
    session.start()
    collector = StreamCollector(opcua_adapter(srcs, line_cfg), line_cfg, srcs)
    await collector.start()
    try:
        await collector.wait_connected()
        await check_commands(collector.adapter.send_command, session)
        assert server.commands == 2
    finally:
        await collector.stop()
        await session.stop()
        await server.stop()


async def test_commands_over_mock(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export, 5.0)
    session.max_ticks = None
    server = ServerHandle(create_app(session))
    await server.start()
    session.start()
    srcs = fast(sources_cfg)
    assert srcs.mock is not None
    adapter = MockServerAdapter(
        srcs.mock.model_copy(update={"base_url": server.base_url}),
        line_cfg,
        reconnect=srcs.reconnect,
        command_timeout_s=srcs.command_timeout_s,
        clock=SystemClock(),
    )
    collector = StreamCollector(adapter, line_cfg, srcs)
    await collector.start()
    try:
        await collector.wait_connected()
        await check_commands(adapter.send_command, session)
    finally:
        await collector.stop()
        await session.stop()
        await server.stop()


@pytest.mark.services
async def test_commands_over_mqtt(
    mqtt_broker: MqttService,
    sim_export: SimExport,
    line_cfg: LineConfig,
    sources_cfg: SourcesConfig,
) -> None:
    import asyncio

    session = matching_session(sim_export, 5.0)
    session.max_ticks = None
    srcs, adapter, publisher = mqtt_setup(sources_cfg, mqtt_broker, session)
    pub_task = asyncio.create_task(publisher.run())
    session.start()
    collector = StreamCollector(adapter, line_cfg, srcs)
    await collector.start()
    try:
        await asyncio.wait_for(publisher.connected.wait(), TIMEOUT_S)
        await collector.wait_connected()
        await check_commands(adapter.send_command, session)
        assert publisher.commands == 2
    finally:
        await collector.stop()
        publisher.stop()
        await asyncio.wait_for(pub_task, TIMEOUT_S)
        await session.stop()


# -- OPC UA robustness ----------------------------------------------------------------------
async def test_opcua_reconnects_after_server_restart(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export, 20.0)
    session.max_ticks = None
    port = free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/linetwin/"
    srcs = opcua_sources(sources_cfg, endpoint)
    assert srcs.opcua is not None
    srcs = srcs.model_copy(update={"opcua": srcs.opcua.model_copy(update={"watchdog_s": 0.5})})
    assert srcs.opcua is not None
    server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
    await server.start()
    session.start()
    collector = StreamCollector(opcua_adapter(srcs, line_cfg), line_cfg, srcs)
    await collector.start()
    try:
        await collector.until(lambda: len(collector.seqs("CNC-01")) >= 20)
        await server.stop()
        await collector.until(
            lambda: collector.adapter.health().status is AdapterStatus.RECONNECTING
        )
        server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
        await server.start()
        resumed_from = session.sim.machine("CNC-01").seq
        await collector.until(lambda: any(s > resumed_from + 10 for s in collector.seqs("CNC-01")))
    finally:
        await collector.stop()
        await session.stop()
        await server.stop()
    assert collector.adapter.health().reconnects >= 1
    seqs = collector.seqs("CNC-01")
    assert seqs == sorted(seqs)  # still in order; the outage shows up as a gap


async def test_opcua_single_station_line(
    sim_export_single: SimExport, single_line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = matching_session(sim_export_single, SPEED)
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/linetwin/"
    srcs = opcua_sources(sources_cfg, endpoint)
    assert srcs.opcua is not None
    server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
    await server.start()
    expected = expected_run(sim_export_single, single_line_cfg)
    try:
        collector = StreamCollector(opcua_adapter(srcs, single_line_cfg), single_line_cfg, srcs)
        out = await collect_live(collector, session, len(expected))
    finally:
        await server.stop()
    assert canonical_order(out, ["CNV-01"]) == expected


async def test_partially_missing_reading_over_opcua(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    """A missing sensor (null -> OPC UA Double NaN) arrives as null with quality SUSPECT."""
    session = matching_session(sim_export, SPEED)  # not started: readings injected by hand
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/linetwin/"
    srcs = opcua_sources(sources_cfg, endpoint)
    assert srcs.opcua is not None
    server = OpcUaServer(session, endpoint, srcs.opcua.namespace, srcs.command_timeout_s)
    await server.start()
    readings = load_jsonl_readings(sim_export.jsonl)
    cnc = [r for r in readings if r.machine_id == "CNC-01"][:3]
    cnc[1] = cnc[1].model_copy(update={"temp": None, "vib": None})
    collector = StreamCollector(opcua_adapter(srcs, line_cfg), line_cfg, srcs)
    await collector.start()
    try:
        await collector.wait_connected()
        for r in cnc:
            await server._on_reading(r)
        await collector.until(lambda: len(collector.out) >= 3)
    finally:
        await collector.stop()
        await server.stop()
    got = {r.seq: r for r in collector.out}
    partial = got[cnc[1].seq]
    assert (partial.temp, partial.vib, partial.quality) == (None, None, Quality.SUSPECT)
    assert partial.speed == cnc[1].speed
    assert got[cnc[0].seq] == cnc[0]
    m = collector.pipeline.metrics
    assert m.value("missing_sensor", source="opcua", sensor="temp") == 1
