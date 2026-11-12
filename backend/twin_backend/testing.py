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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest

from twin_common.config import load_sources_config
from twin_common.config.settings import TwinSettings

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
