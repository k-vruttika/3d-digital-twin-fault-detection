"""MockServerAdapter against the real mock server (uvicorn, in-process, free port).

Covers WebSocket streaming, backfill from /history on connect, REST fallback while the
WebSocket is unavailable, reconnect with backoff after a server restart, and commands.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from twin_backend.adapters.mock import MockServerAdapter
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import SystemClock
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import MemoryQuarantine
from twin_backend.testing import drain_readings
from twin_common.config.line import LineConfig
from twin_common.config.sources import MockSourceConfig, SourcesConfig
from twin_common.contracts import Reading
from twin_common.enums import AckStatus, AdapterKind, AdapterStatus, Capability, CommandType
from twin_common.testing import sample_command
from twin_sim.publishers.http import ServerHandle
from twin_sim.publishers.mock_server import create_app
from twin_sim.publishers.session import SessionOptions, SimSession

SPEED = 50.0  # 20 ms per simulated second
TIMEOUT_S = 15.0
FAST_RECONNECT = {"initial_s": 0.05, "max_s": 0.2, "jitter": 0.0}


def session_for(
    config_dir: Path, line: str = "line.yaml", scenario: str = "quick.yaml"
) -> SimSession:
    return SimSession.from_options(
        SessionOptions(
            line_path=config_dir / line,
            scenario_path=config_dir / "scenarios" / scenario,
            sim_config_path=config_dir / "simulator.yaml",
            seed=5,
            speed=SPEED,
            duration_s=100_000,
            history_size=5000,
        )
    )


class Rig:
    def __init__(
        self, sources: SourcesConfig, line: LineConfig, base_url: str, **mock_overrides: object
    ) -> None:
        assert sources.mock is not None
        self.cfg: MockSourceConfig = sources.mock.model_copy(
            update={"base_url": base_url, "rest_poll_s": 0.05, **mock_overrides}
        )
        self.clock = SystemClock()
        self.adapter = MockServerAdapter(
            self.cfg,
            line,
            reconnect=sources.reconnect.model_copy(update=FAST_RECONNECT),
            command_timeout_s=sources.command_timeout_s,
            clock=self.clock,
        )
        self.bus = InMemoryEventBus(queue_size=1_000_000)
        self.sub = self.bus.subscribe(Topic.READINGS)
        self.quarantine = MemoryQuarantine(1000)
        ingest = sources.ingest.model_copy(update={"lateness_window_s": 0.5})
        self.pipeline = IngestPipeline(
            line,
            ingest,
            self.bus,
            self.quarantine,
            self.clock,
            {AdapterKind.MOCK: self.cfg.mapping},
        )
        self.via: dict[str, int] = defaultdict(int)
        self._pump: asyncio.Task[None] | None = None
        self.out: list[Reading] = []

    async def start(self) -> None:
        await self.adapter.start()
        self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for raw in self.adapter.stream():
            self.via[raw.meta.get("via", "?")] += 1
            await self.pipeline.process(raw)

    async def until(self, cond: Callable[[], bool], timeout: float = TIMEOUT_S) -> None:
        async def poll() -> None:
            while True:
                await self.pipeline.tick()
                self.out.extend(drain_readings(self.sub))
                if cond():
                    return
                await asyncio.sleep(0.02)

        await asyncio.wait_for(poll(), timeout)

    async def stop(self) -> None:
        await self.adapter.stop()
        if self._pump is not None:
            await asyncio.wait_for(self._pump, 5)

    def seqs(self, machine: str) -> list[int]:
        return [r.seq for r in self.out if r.machine_id == machine]


@pytest.fixture
async def server(config_dir: Path) -> AsyncIterator[tuple[ServerHandle, SimSession]]:
    session = session_for(config_dir)
    handle = ServerHandle(create_app(session))
    await handle.start()
    yield handle, session
    await handle.stop()
    await session.stop()


def contiguous_from_one(seqs: list[int]) -> bool:
    return seqs == list(range(1, len(seqs) + 1))


async def test_stream_and_backfill_from_start(
    server: tuple[ServerHandle, SimSession], line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    handle, session = server
    session.start()
    await asyncio.sleep(0.5)  # ~25 readings per machine exist before the adapter connects
    rig = Rig(sources_cfg, line_cfg, handle.base_url)
    await rig.start()
    try:
        await rig.until(lambda: all(len(rig.seqs(m.id)) >= 60 for m in line_cfg.machines))
    finally:
        await rig.stop()
    for m in line_cfg.machines:
        assert contiguous_from_one(rig.seqs(m.id)), m.id  # backfill closed the startup gap
    assert rig.via["backfill"] > 0
    assert rig.via["ws"] > 0
    assert rig.adapter.health().connected is False  # stopped
    assert rig.adapter.capabilities == {
        Capability.COMMANDS,
        Capability.HISTORY,
        Capability.SCENARIO,
        Capability.WHATIF,
    }
    lat = rig.pipeline.metrics.registry.get_sample_value(
        "linetwin_ingest_e2e_latency_seconds_count", {"source": "mock", "basis": "source"}
    )
    assert lat is not None  # WS envelopes carry sentAt
    assert lat > 0


async def test_reconnect_after_server_restart_fills_the_gap(
    server: tuple[ServerHandle, SimSession], line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    handle, session = server
    session.start()
    rig = Rig(sources_cfg, line_cfg, handle.base_url)
    await rig.start()
    try:
        await rig.until(lambda: len(rig.seqs("CNC-01")) >= 20)
        await handle.stop()  # server down, simulation keeps running
        await rig.until(lambda: rig.adapter.health().status is AdapterStatus.RECONNECTING)
        await asyncio.sleep(0.5)
        await handle.start()  # same port
        before = session.sim.machine("CNC-01").seq
        await rig.until(lambda: len(rig.seqs("CNC-01")) >= before + 10)
    finally:
        await rig.stop()
    assert rig.adapter.health().reconnects >= 1
    for m in line_cfg.machines:
        assert contiguous_from_one(rig.seqs(m.id)), m.id  # nothing lost across the outage
    assert rig.pipeline.metrics.value("gap_seqs", source="mock") == 0


async def test_rest_fallback_when_websocket_unavailable(
    server: tuple[ServerHandle, SimSession], line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    handle, session = server
    session.start()
    rig = Rig(sources_cfg, line_cfg, handle.base_url, ws_path="/no-such-stream")
    await rig.start()
    try:
        await rig.until(lambda: all(len(rig.seqs(m.id)) >= 30 for m in line_cfg.machines))
    finally:
        await rig.stop()
    assert rig.via["rest"] > 0
    assert rig.via["ws"] == 0
    assert rig.adapter.health().reconnects >= 1
    for m in line_cfg.machines:
        assert contiguous_from_one(rig.seqs(m.id)), m.id


async def test_commands_reach_the_simulator(
    server: tuple[ServerHandle, SimSession], line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    handle, session = server
    session.start()
    rig = Rig(sources_cfg, line_cfg, handle.base_url)
    await rig.start()
    try:
        cmd = sample_command(
            machine_id="CNC-01", type=CommandType.SET_SPEED, params={"speed": 8500.0}
        )
        ack = await rig.adapter.send_command(cmd)
        assert (ack.command_id, ack.status) == (cmd.command_id, AckStatus.EXECUTED)
        assert session.sim.machine("CNC-01").speed_set == 8500.0
        bad = sample_command(
            machine_id="CNC-01",
            type=CommandType.RESET,
            params={},
            command_id="c2",
            idempotency_key="k2",
        )
        ack = await rig.adapter.send_command(bad)
        assert ack.status is AckStatus.REJECTED
        assert ack.reason == "machine is not faulted"
    finally:
        await rig.stop()


async def test_command_when_server_down_is_rejected(
    line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = Rig(sources_cfg, line_cfg, "http://127.0.0.1:9")  # nothing listens on port 9
    await rig.start()
    try:
        ack = await rig.adapter.send_command(sample_command())
        assert ack.status is AckStatus.REJECTED
        assert "command transport failed" in (ack.reason or "")
    finally:
        await rig.stop()


async def test_single_station_line(
    config_dir: Path, single_line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    session = session_for(config_dir, "line_single.yaml", "quick_single.yaml")
    handle = ServerHandle(create_app(session))
    await handle.start()
    session.start()
    rig = Rig(sources_cfg, single_line_cfg, handle.base_url)
    await rig.start()
    try:
        await rig.until(lambda: len(rig.seqs("CNV-01")) >= 30)
    finally:
        await rig.stop()
        await handle.stop()
        await session.stop()
    assert {r.machine_id for r in rig.out} == {"CNV-01"}
    assert contiguous_from_one(rig.seqs("CNV-01"))
