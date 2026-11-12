"""The Docker Compose services are reachable and correctly configured.

Skipped when the services are not running (start them with `uv run poe up`); required in CI
on Ubuntu via LINETWIN_REQUIRE_SERVICES=1.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

import pytest

from twin_backend.testing import MqttService, PgService

pytestmark = pytest.mark.services

ROUND_TRIP_TIMEOUT_S = 5.0
EXPECTED_PG_MAJOR = 16


def test_mqtt_publish_subscribe_round_trip(mqtt_broker: MqttService) -> None:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    topic = f"linetwin/test/{uuid.uuid4().hex}"
    payload = b"round-trip"
    subscribed, received = threading.Event(), threading.Event()
    got: list[bytes] = []

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"linetwin-test-{os.getpid()}")

    def on_connect(c: Any, _u: Any, _f: Any, _rc: Any, _p: Any) -> None:
        c.subscribe(topic, qos=1)

    def on_subscribe(_c: Any, _u: Any, _mid: Any, _rc: Any, _p: Any) -> None:
        subscribed.set()

    def on_message(_c: Any, _u: Any, msg: Any) -> None:
        got.append(msg.payload)
        received.set()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(mqtt_broker.host, mqtt_broker.port, keepalive=10)
    client.loop_start()
    try:
        assert subscribed.wait(ROUND_TRIP_TIMEOUT_S), "SUBACK not received"
        client.publish(topic, payload, qos=1).wait_for_publish(ROUND_TRIP_TIMEOUT_S)
        assert received.wait(ROUND_TRIP_TIMEOUT_S), "published message not received back"
    finally:
        client.disconnect()
        client.loop_stop()
    assert got == [payload]


async def test_timescaledb_extension_available(timescale_db: PgService) -> None:
    import asyncpg

    conn = await asyncpg.connect(timescale_db.dsn, timeout=5)
    try:
        major = await conn.fetchval("SELECT current_setting('server_version_num')::int / 10000")
        installed = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        )
        available = await conn.fetchval(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb'"
        )
    finally:
        await conn.close()
    assert major == EXPECTED_PG_MAJOR
    assert available is not None, "timescaledb is not available on this server"
    assert installed is not None, f"timescaledb not created in {timescale_db.display}"
