"""SourceManager: active adapter from sources.yaml, pump + ticker, commands, hot switch."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from twin_backend.adapters.base import AdapterError
from twin_backend.adapters.manager import SourceManager
from twin_backend.adapters.registry import build_adapter
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import SystemClock
from twin_backend.testing import (
    SimExport,
    canonical_order,
    drain_readings,
    load_jsonl_readings,
    matching_session,
)
from twin_common.config.line import LineConfig
from twin_common.config.sources import SourcesConfig
from twin_common.contracts import Command, CommandAck, Reading
from twin_common.enums import AckStatus, AdapterKind, AdapterStatus, CommandType
from twin_common.testing import sample_command
from twin_sim.publishers.http import ServerHandle
from twin_sim.publishers.mock_server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 15.0


def file_sources(sources: SourcesConfig, export: SimExport, **file_kw: object) -> SourcesConfig:
    assert sources.file is not None
    file_cfg = sources.file.model_copy(
        update={"path": str(export.csv), "loop": False, "replay_speed": 1000.0, **file_kw}
    )
    ingest = sources.ingest.model_copy(update={"tick_s": 0.02, "lateness_window_s": 0.2})
    return sources.model_copy(update={"file": file_cfg, "ingest": ingest})


async def until(cond: object, timeout: float = TIMEOUT_S) -> None:
    async def poll() -> None:
        while not cond():  # type: ignore[operator]
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def test_file_source_end_to_end(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    bus = InMemoryEventBus(queue_size=1_000_000)
    sub = bus.subscribe(Topic.READINGS)
    mgr = SourceManager(file_sources(sources_cfg, sim_export), line_cfg, bus, root=REPO_ROOT)
    await mgr.start()
    try:
        assert mgr.active_kind is AdapterKind.FILE
        await asyncio.wait_for(mgr.wait_idle(), TIMEOUT_S)
        health = mgr.health()
        assert health is not None
        assert health.status is AdapterStatus.EOF
    finally:
        await mgr.stop()
    expected = load_jsonl_readings(sim_export.jsonl)
    machines = [m.id for m in line_cfg.machines]
    assert canonical_order(drain_readings(sub), machines) == canonical_order(expected, machines)


async def test_commands_rejected_by_file_source_and_logged_on_bus(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    bus = InMemoryEventBus(queue_size=1000)
    cmds = bus.subscribe(Topic.COMMANDS)
    mgr = SourceManager(file_sources(sources_cfg, sim_export), line_cfg, bus, root=REPO_ROOT)
    await mgr.start()
    try:
        ack = await mgr.send_command(sample_command())
    finally:
        await mgr.stop()
    assert ack.status is AckStatus.REJECTED
    assert "file adapter" in (ack.reason or "")
    logged = drain_readings(cmds)
    assert isinstance(logged[0], Command)
    assert isinstance(logged[1], CommandAck)


async def test_hot_switch_file_to_mock_is_seamless(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    """File replay of a run, then a hot switch to a mock server streaming the same run:
    the overlap is deduplicated and every machine's sequence stays complete."""
    session = matching_session(sim_export, speed=40.0)
    server = ServerHandle(create_app(session))
    await server.start()
    session.start()
    assert sources_cfg.mock is not None
    sources = file_sources(sources_cfg, sim_export, replay_speed=20.0)
    sources = sources.model_copy(
        update={
            "mock": sources_cfg.mock.model_copy(update={"base_url": server.base_url}),
            "reconnect": sources_cfg.reconnect.model_copy(update={"initial_s": 0.05}),
        }
    )
    bus = InMemoryEventBus(queue_size=1_000_000)
    sub = bus.subscribe(Topic.READINGS)
    mgr = SourceManager(sources, line_cfg, bus, root=REPO_ROOT)
    out: list[Reading] = []
    n_machines = len(line_cfg.machines)
    try:
        await mgr.start()

        def some_file_data() -> bool:
            out.extend(drain_readings(sub))
            return len(out) >= 10 * n_machines

        await until(some_file_data)
        await mgr.switch(AdapterKind.MOCK)
        assert mgr.active_kind is AdapterKind.MOCK
        assert mgr.switches == 1
        total = len(load_jsonl_readings(sim_export.jsonl))

        def done() -> bool:
            out.extend(drain_readings(sub))
            return len(out) >= total

        await until(done, timeout=30)
        ack = await mgr.send_command(
            sample_command(machine_id="CNV-01", type=CommandType.SET_LOAD, params={"load": 50.0})
        )
    finally:
        await mgr.stop()
        await server.stop()
        await session.stop()
    assert ack.status in {AckStatus.EXECUTED, AckStatus.TIMEOUT}  # run may already be over
    assert len(out) == total
    machines = [m.id for m in line_cfg.machines]
    expected = load_jsonl_readings(sim_export.jsonl)
    assert canonical_order(out, machines) == canonical_order(expected, machines)
    assert mgr.pipeline.metrics.value("duplicates", source="mock") > 0  # overlap removed
    assert n_machines == 4


async def test_failed_switch_keeps_current_source(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    sources = file_sources(sources_cfg, sim_export, loop=True, replay_speed=1.0)
    mgr = SourceManager(sources, line_cfg, InMemoryEventBus(queue_size=100), root=REPO_ROOT)
    await mgr.start()
    try:
        broken = sources.model_copy(
            update={"file": sources.file.model_copy(update={"path": "missing.csv"})}  # type: ignore[union-attr]
        )
        with pytest.raises(AdapterError, match="replay file not found"):
            await mgr.switch(AdapterKind.FILE, broken)
        assert mgr.active_kind is AdapterKind.FILE
        assert mgr.switches == 0
        no_opcua = sources.model_copy(update={"opcua": None})
        with pytest.raises(AdapterError, match=r"no .opcua:. section"):
            await mgr.switch(AdapterKind.OPCUA, no_opcua)
        assert mgr.active_kind is AdapterKind.FILE
    finally:
        await mgr.stop()


def test_registry_wraps_chaos(sources_cfg: SourcesConfig, line_cfg: LineConfig) -> None:
    from twin_backend.adapters.chaos import ChaosAdapter

    sources = sources_cfg.model_copy(
        update={"chaos": sources_cfg.chaos.model_copy(update={"enabled": True})}
    )
    adapter = build_adapter(
        AdapterKind.FILE, sources, line_cfg, root=REPO_ROOT, clock=SystemClock()
    )
    assert isinstance(adapter, ChaosAdapter)
    assert adapter.kind is AdapterKind.FILE


def test_file_path_imports_no_network_or_db_library() -> None:
    """Minimal mode: building and running the file adapter loads no broker/DB/ML/HTTP lib."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from twin_backend.adapters.manager import SourceManager\n"
        "from twin_backend.adapters.registry import build_adapter\n"
        "from twin_backend.clock import SystemClock\n"
        "from twin_common.config import load_line_config, load_sources_config\n"
        "from twin_common.enums import AdapterKind\n"
        "s = load_sources_config('config/sources.yaml')\n"
        "build_adapter(AdapterKind.FILE, s, load_line_config('config/line.yaml'),"
        " root=Path('.'), clock=SystemClock())\n"
        "bad = [m for m in ('aiomqtt', 'paho', 'asyncua', 'websockets', 'httpx', 'asyncpg',"
        " 'sqlalchemy', 'torch', 'sklearn', 'xgboost', 'mlflow', 'twin_ml', 'fastapi',"
        " 'uvicorn') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
