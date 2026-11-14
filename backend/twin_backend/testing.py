"""Pytest fixtures for tests that need the Docker Compose services (MQTT broker, TimescaleDB).

Registered as a plugin by the root ``conftest.py``. Not used by production code.

Each fixture probes its service once per session. If it is unreachable the test is skipped
with a hint to run ``uv run poe up``; with ``LINETWIN_REQUIRE_SERVICES=1`` (set in CI on
Ubuntu) it fails instead, so service tests can never be skipped silently there.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest

from twin_common.config import load_sources_config
from twin_common.config.settings import TwinSettings
from twin_common.config.sources import SourcesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/twin_backend/testing.py -> repo
REQUIRE_SERVICES_ENV = "LINETWIN_REQUIRE_SERVICES"
START_HINT = "start services with: uv run poe up"
PROBE_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class MqttService:
    host: str
    port: int


@dataclass(frozen=True)
class PgService:
    dsn: str  # contains the password: never print it
    display: str  # host:port/db, safe to show


def _unavailable(what: str, where: str, detail: str) -> NoReturn:
    message = f"{what} not reachable at {where} ({detail}) - {START_HINT}"
    if os.environ.get(REQUIRE_SERVICES_ENV) == "1":
        pytest.fail(f"{message} [{REQUIRE_SERVICES_ENV}=1]", pytrace=False)
    pytest.skip(message)


def probe_mqtt(host: str, port: int, timeout_s: float = PROBE_TIMEOUT_S) -> str | None:
    """MQTT CONNECT/CONNACK round trip. Returns None if OK, else a short reason.

    Uses paho's thread-based client rather than aiomqtt, which needs a selector event loop
    (not the Windows default).
    """
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"linetwin-probe-{os.getpid()}")
    client.connect_timeout = timeout_s
    connected = threading.Event()
    result: dict[str, Any] = {}

    def on_connect(_c: Any, _u: Any, _f: Any, reason_code: Any, _p: Any) -> None:
        result["rc"] = reason_code
        connected.set()

    client.on_connect = on_connect
    try:
        client.connect(host, port, keepalive=10)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    client.loop_start()
    try:
        if not connected.wait(timeout_s):
            return f"no CONNACK within {timeout_s:.0f} s"
        if result["rc"].is_failure:
            return f"broker refused the connection: {result['rc']}"
        return None
    finally:
        client.disconnect()
        client.loop_stop()


async def _probe_pg_async(dsn: str, timeout_s: float) -> str | None:
    import asyncpg

    try:
        conn = await asyncpg.connect(dsn, timeout=timeout_s)
    except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        return f"{type(exc).__name__}: {exc}"
    await conn.close()
    return None


def probe_pg(dsn: str, timeout_s: float = PROBE_TIMEOUT_S) -> str | None:
    """Open and close one connection. Returns None if OK, else a short reason."""
    return asyncio.run(_probe_pg_async(dsn, timeout_s))


@pytest.fixture(scope="session")
def mqtt_broker() -> MqttService:
    """The MQTT broker from config/sources.yaml (Docker Compose: `uv run poe up`)."""
    mqtt_cfg = load_sources_config(REPO_ROOT / "config" / "sources.yaml").mqtt
    if mqtt_cfg is None:
        pytest.fail("config/sources.yaml has no mqtt section", pytrace=False)
    reason = probe_mqtt(mqtt_cfg.host, mqtt_cfg.port)
    if reason is not None:
        _unavailable("MQTT broker", f"{mqtt_cfg.host}:{mqtt_cfg.port}", reason)
    return MqttService(mqtt_cfg.host, mqtt_cfg.port)


@pytest.fixture(scope="session")
def timescale_db() -> PgService:
    """TimescaleDB from TwinSettings / .env (Docker Compose: `uv run poe up`)."""
    settings = TwinSettings(_env_file=REPO_ROOT / ".env")
    display = f"{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    reason = probe_pg(settings.pg_dsn)
    if reason is not None:
        _unavailable("TimescaleDB", display, reason)
    return PgService(settings.pg_dsn, display)


# ---------------------------------------------------------------- Phase 2: ingest fixtures
@dataclass(frozen=True)
class SimExport:
    """One simulated run exported as JSONL and CSV (same plan, same readings)."""

    dir: Path
    jsonl: Path
    csv: Path
    line_path: Path
    scenario_path: Path
    seed: int
    duration_s: float


EXPORT_DURATION_S = 120.0
EXPORT_SEED = 11


def _export(tmp: Path, line_name: str, scenario_name: str) -> SimExport:
    from twin_common.config import load_line_config
    from twin_sim.batch import generate
    from twin_sim.config import load_scenario, load_sim_config

    config = REPO_ROOT / "config"
    line_path = config / line_name
    scenario_path = config / "scenarios" / scenario_name
    scenario = load_scenario(scenario_path).model_copy(update={"duration_s": EXPORT_DURATION_S})
    generate(
        line=load_line_config(line_path),
        sim=load_sim_config(config / "simulator.yaml"),
        scenario=scenario,
        runs=1,
        base_seed=EXPORT_SEED,
        out_dir=tmp,
        exports=("csv", "jsonl"),
        workers=1,
    )
    return SimExport(
        tmp,
        tmp / "run_0000.jsonl",
        tmp / "run_0000.csv",
        line_path,
        scenario_path,
        EXPORT_SEED,
        EXPORT_DURATION_S,
    )


@pytest.fixture(scope="session")
def sim_export(tmp_path_factory: pytest.TempPathFactory) -> SimExport:
    """120 s of the full line (quick scenario), exported as JSONL + CSV."""
    return _export(tmp_path_factory.mktemp("sim_export"), "line.yaml", "quick.yaml")


@pytest.fixture(scope="session")
def sim_export_single(tmp_path_factory: pytest.TempPathFactory) -> SimExport:
    """120 s of the single-station line (quick_single scenario), exported as JSONL + CSV."""
    return _export(
        tmp_path_factory.mktemp("sim_export_single"), "line_single.yaml", "quick_single.yaml"
    )


@pytest.fixture(scope="session")
def sources_cfg() -> SourcesConfig:
    """The shipped config/sources.yaml (tests derive variants with model_copy)."""
    return load_sources_config(REPO_ROOT / "config" / "sources.yaml")


def drain_readings(sub: Any) -> list[Any]:
    """Everything currently queued on a bus subscription."""
    out: list[Any] = []
    while (msg := sub.get_nowait()) is not None:
        out.append(msg)
    return out


async def replay_into(adapter: Any, pipeline: Any, *, tick: bool = True) -> None:
    """Start ``adapter``, push its whole stream through ``pipeline``, flush at the end.

    With ``tick`` the pipeline's housekeeping runs after every message (on the pipeline clock).
    """
    await adapter.start()
    try:
        async for raw in adapter.stream():
            await pipeline.process(raw)
            if tick:
                await pipeline.tick()
        await pipeline.flush()
    finally:
        await adapter.stop()


def load_jsonl_readings(path: Path) -> list[Any]:
    from twin_common.contracts import Reading

    return [
        Reading.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_order(readings: list[Any], machine_order: list[str]) -> list[Any]:
    """Sort readings by (ts, line machine order), the order every adapter must reproduce."""
    rank = {m: i for i, m in enumerate(machine_order)}
    return sorted(readings, key=lambda r: (r.ts, rank.get(r.machine_id, len(rank)), r.seq))


def matching_session(export: SimExport, speed: float, history_size: int = 10_000) -> Any:
    """A live SimSession that replays exactly the run in ``export`` (same plan and seed)."""
    from twin_common.config import load_line_config
    from twin_sim.config import load_scenario, load_sim_config
    from twin_sim.publishers.session import SimSession, build_simulator

    scenario = load_scenario(export.scenario_path).model_copy(
        update={"duration_s": export.duration_s}
    )
    sim = build_simulator(
        load_line_config(export.line_path),
        load_sim_config(REPO_ROOT / "config" / "simulator.yaml"),
        scenario,
        export.seed,
        0,
    )
    session = SimSession(sim, speed, history_size)
    session.max_ticks = round(export.duration_s)
    return session


def free_port() -> int:
    """A TCP port that was free a moment ago (for servers that need an explicit port)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class StreamCollector:
    """Runs an adapter into a real-clock ingest pipeline and collects the bus output."""

    def __init__(
        self,
        adapter: Any,
        line: Any,
        sources: SourcesConfig,
        *,
        lateness_window_s: float = 0.5,
    ) -> None:

        from twin_backend.adapters.registry import mappings_from
        from twin_backend.bus import InMemoryEventBus, Topic
        from twin_backend.clock import SystemClock
        from twin_backend.ingest.pipeline import IngestPipeline
        from twin_backend.ingest.quarantine import MemoryQuarantine

        self.adapter = adapter
        self.clock = SystemClock()
        self.bus = InMemoryEventBus(queue_size=1_000_000)
        self.sub = self.bus.subscribe(Topic.READINGS)
        self.quarantine = MemoryQuarantine(100_000)
        ingest = sources.ingest.model_copy(update={"lateness_window_s": lateness_window_s})
        self.pipeline = IngestPipeline(
            line, ingest, self.bus, self.quarantine, self.clock, mappings_from(sources)
        )
        self.out: list[Any] = []
        self._pump: asyncio.Task[None] | None = None

    async def start(self) -> None:
        import asyncio

        await self.adapter.start()
        self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for raw in self.adapter.stream():
            await self.pipeline.process(raw)

    async def until(self, cond: Any, timeout: float = 20.0) -> None:
        """Poll (ticking the pipeline) until ``cond()`` is true."""
        import asyncio

        async def poll() -> None:
            while True:
                await self.pipeline.tick()
                self.out.extend(drain_readings(self.sub))
                if cond():
                    return
                await asyncio.sleep(0.02)

        await asyncio.wait_for(poll(), timeout)

    async def wait_connected(self, timeout: float = 20.0) -> None:
        await self.until(lambda: self.adapter.health().connected, timeout)

    async def stop(self) -> None:
        import asyncio

        await self.adapter.stop()
        if self._pump is not None:
            await asyncio.wait_for(self._pump, 10)
        await self.pipeline.flush()
        self.out.extend(drain_readings(self.sub))

    def seqs(self, machine_id: str) -> list[int]:
        return [r.seq for r in self.out if r.machine_id == machine_id]


# ---------------------------------------------------------------- Phase 3: storage fixtures
# Backends every store-level test runs on. TimescaleDB tests carry the ``services`` marker and
# skip (or fail with LINETWIN_REQUIRE_SERVICES=1) when the database is not reachable.
STORE_BACKENDS: list[Any] = ["sqlite", pytest.param("timescale", marks=pytest.mark.services)]
TEST_DB_PREFIX = "linetwin_test_"


def storage_config(**overrides: Any) -> Any:
    """config/storage.yaml with top-level overrides (e.g. writer=..., snapshots=...)."""
    from twin_common.config import load_storage_config

    cfg = load_storage_config(REPO_ROOT / "config" / "storage.yaml")
    return cfg.model_copy(update=overrides)


def test_timescale_config() -> Any:
    """The shipped TimescaleDB settings, with refresh jobs scheduled hourly so no background
    refresh runs in the middle of a test (tests run the jobs explicitly)."""
    ts = storage_config().timescale
    aggs = ts.aggregates
    return ts.model_copy(
        update={
            "aggregates": aggs.model_copy(
                update={
                    "m1": aggs.m1.model_copy(update={"schedule_s": 3600.0}),
                    "m10": aggs.m10.model_copy(update={"schedule_s": 3600.0}),
                }
            )
        }
    )


test_timescale_config.__test__ = False  # type: ignore[attr-defined]  # not a test


def run_in_thread(coro: Any) -> Any:
    """Run a coroutine to completion on a fresh event loop in a worker thread (usable from
    sync fixtures, whatever loop pytest-asyncio has running)."""
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _create_test_db(admin_dsn: str, name: str, test_dsn: str) -> None:
    from twin_backend.storage.timescale import TimescaleEventStore, create_database

    await create_database(admin_dsn, name)
    store = TimescaleEventStore(test_dsn, test_timescale_config())
    await store.connect()
    try:
        await store.migrate()
    finally:
        await store.close()


async def _drop_test_db(admin_dsn: str, name: str) -> None:
    from twin_backend.storage.timescale import drop_database

    await drop_database(admin_dsn, name)


@pytest.fixture(scope="session")
def timescale_test_db(timescale_db: PgService) -> Any:
    """A throw-away TimescaleDB database for this test session.

    ``linetwin_test_<random hex>`` is created next to the configured database (never the
    configured database itself), migrated, and dropped with ``WITH (FORCE)`` at session end, so
    tests never touch real data.
    """
    import secrets

    settings = TwinSettings(_env_file=REPO_ROOT / ".env")
    name = f"{TEST_DB_PREFIX}{secrets.token_hex(6)}"
    if name == settings.pg_db:  # impossible by construction; never risk the real database
        pytest.fail("refusing to use the configured database for tests", pytrace=False)
    test_settings = settings.model_copy(update={"pg_db": name})
    run_in_thread(_create_test_db(settings.pg_dsn, name, test_settings.pg_dsn))
    try:
        yield PgService(test_settings.pg_dsn, f"{settings.pg_host}:{settings.pg_port}/{name}")
    finally:
        run_in_thread(_drop_test_db(settings.pg_dsn, name))


async def open_store(backend: str, tmp: Path, *, dsn: str | None = None) -> Any:
    """A connected, migrated, empty event store of ``backend``."""
    if backend == "sqlite":
        from twin_backend.storage.sqlite import SqliteEventStore

        store: Any = SqliteEventStore(tmp / "events.sqlite")
        await store.connect()
        await store.migrate()
        return store
    if backend == "timescale":
        if dsn is None:
            raise ValueError("the timescale store needs the test database DSN")
        from twin_backend.storage.timescale import TimescaleEventStore

        store = TimescaleEventStore(dsn, test_timescale_config())
        await store.connect()
        await store.migrate()
        await store.delete_all()
        return store
    raise ValueError(f"unknown store backend {backend!r}")


@pytest.fixture(params=STORE_BACKENDS)
def store_backend(request: pytest.FixtureRequest) -> str:
    """Backend name, once per backend in ``STORE_BACKENDS``."""
    backend: str = request.param
    return backend


@pytest.fixture
def store_factory(store_backend: str, request: pytest.FixtureRequest) -> Any:
    """``await store_factory(tmp_dir)`` opens an empty store of the current backend.

    TimescaleDB stores share the session's test database and are emptied on open, so open the
    next one only after you are done reading the previous one.
    """
    dsn = request.getfixturevalue("timescale_test_db").dsn if store_backend == "timescale" else None

    async def factory(tmp: Path) -> Any:
        return await open_store(store_backend, tmp, dsn=dsn)

    return factory


@pytest.fixture
async def event_store(store_factory: Any, tmp_path: Path) -> Any:
    """An empty event store, once per backend in ``STORE_BACKENDS``."""
    store = await store_factory(tmp_path)
    try:
        yield store
    finally:
        await store.close()


# ---------------------------------------------------------------- Phase 4: runtime / API fixtures
def settings_for(dsn: str | None = None, **overrides: Any) -> Any:
    """TwinSettings that ignore the local .env; ``dsn`` points the TimescaleDB store at it."""
    from urllib.parse import urlparse

    from twin_common.config.settings import TwinSettings

    fields: dict[str, Any] = {"config_dir": REPO_ROOT / "config"}
    if dsn is not None:
        u = urlparse(dsn)
        fields |= {
            "pg_host": u.hostname or "",
            "pg_port": u.port or 5432,
            "pg_db": u.path.lstrip("/"),
            "pg_user": u.username or "",
            "pg_password": u.password or "",
        }
    return TwinSettings(_env_file=None, **(fields | overrides))


def file_runtime(
    export: SimExport,
    tmp: Path,
    backend: str = "sqlite",
    dsn: str | None = None,
    *,
    replay_speed: float = 1e6,
    loop: bool = False,
    settings: Any = None,
    **build_kw: Any,
) -> Any:
    """A Runtime replaying ``export`` (JSONL) through the file adapter into ``backend``."""
    from twin_backend.runtime import build_runtime
    from twin_common.config.sources import FieldMapping

    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    assert sources.file is not None
    file_cfg = sources.file.model_copy(
        update={
            "path": str(export.jsonl),
            "format": "jsonl",
            "loop": loop,
            "replay_speed": replay_speed,
            "mapping": FieldMapping(),
        }
    )
    ingest = sources.ingest.model_copy(update={"tick_s": 0.01, "lateness_window_s": 0.05})
    sources = sources.model_copy(update={"file": file_cfg, "ingest": ingest})
    base = storage_config()
    st = base.model_copy(
        update={
            "backend": backend,
            "sqlite": base.sqlite.model_copy(update={"path": str(tmp / "store.sqlite")}),
            "snapshots": base.snapshots.model_copy(update={"settle_s": 0.6}),
        }
    )
    return build_runtime(
        settings if settings is not None else settings_for(dsn),
        root=REPO_ROOT,
        sources=sources,
        storage_cfg=st,
        **build_kw,
    )


API_OP_KEY = "op-key-0123456789abcdef"
API_VIEW_KEY = "viewer-key-0123456789abcdef"
API_KEYS = {
    "api_keys_operator": f"alice:{API_OP_KEY}",
    "api_keys_viewer": f"victor:{API_VIEW_KEY}",
}


@asynccontextmanager
async def running_api(
    export: SimExport,
    tmp: Path,
    *,
    backend: str = "sqlite",
    dsn: str | None = None,
    keys: bool = True,
    wait_idle: bool = True,
    **runtime_kw: Any,
) -> AsyncIterator[tuple[Any, Any]]:
    """A uvicorn server on a file-source runtime; yields (runtime, ServerHandle)."""
    from pydantic import SecretStr

    from twin_backend.api.app import create_app
    from twin_backend.api.server import ServerHandle

    secret_keys = {k: SecretStr(v) for k, v in API_KEYS.items()} if keys else {}
    rt = file_runtime(
        export, tmp, backend, dsn, settings=settings_for(dsn, **secret_keys), **runtime_kw
    )
    server = ServerHandle(create_app(rt))
    await server.start()
    try:
        if wait_idle:
            await rt.manager.wait_idle()
            await rt.engine.drain()
            await rt.storage.flush()
        yield rt, server
    finally:
        await server.stop()


def api_client(server: Any, key: str | None = API_VIEW_KEY) -> Any:
    """An httpx client for a ``running_api`` server, sending ``key`` as X-API-Key."""
    import httpx

    headers = {"X-API-Key": key} if key else {}
    return httpx.AsyncClient(base_url=server.base_url, headers=headers, timeout=10.0)


def require_chromium() -> None:
    """Skip when Playwright's Chromium is missing; fail instead with LINETWIN_REQUIRE_SERVICES=1."""
    from twin_backend.e2e_stack import chromium_missing

    reason = chromium_missing()
    if reason is None:
        return
    if os.environ.get(REQUIRE_SERVICES_ENV) == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest.fixture
def runtime_dsn(store_backend: str, request: pytest.FixtureRequest) -> str | None:
    """The test database DSN for the timescale backend (None for sqlite). The database is
    emptied first. Resolved in a sync fixture, outside the test's event loop."""
    if store_backend != "timescale":
        return None
    dsn: str = request.getfixturevalue("timescale_test_db").dsn

    async def empty() -> None:
        store = await open_store("timescale", REPO_ROOT, dsn=dsn)
        await store.close()

    run_in_thread(empty())
    return dsn
