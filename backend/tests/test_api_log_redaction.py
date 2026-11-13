"""No log line ever contains an API key.

Captures the structlog output and uvicorn's standard-library logs (including access logs)
while clients use valid and invalid keys over REST headers, a (wrongly used) query parameter
and the WebSocket auth message, and while code logs a key on purpose. None of the key strings
may appear anywhere in the captured text.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import structlog
import websockets
from pydantic import SecretStr

from twin_backend.api.app import create_app
from twin_backend.api.server import ServerHandle
from twin_backend.testing import SimExport, file_runtime, settings_for
from twin_common.logging import (
    REDACTED,
    SecretFilter,
    configure_logging,
    get_logger,
    redact_text,
)

OP_KEY = "op-secret-key-0123456789"
VIEW_KEY = "view-secret-key-0123456789"
WRONG_KEY = "wrong-secret-key-0123456789"
ALL_KEYS = (OP_KEY, VIEW_KEY, WRONG_KEY)


@pytest.fixture
def captured() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    configure_logging("DEBUG", stream=buf)
    handler = logging.StreamHandler(buf)
    handler.addFilter(SecretFilter())
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield buf
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        structlog.reset_defaults()


def test_query_parameters_and_registered_values_are_masked() -> None:
    assert redact_text("GET /health?apiKey=abc123&x=1") == f"GET /health?apiKey={REDACTED}&x=1"
    assert redact_text("api_key=zzz password=hunter2") == f"api_key={REDACTED} password={REDACTED}"


async def test_no_key_reaches_any_log_line(
    sim_export: SimExport, tmp_path: Path, captured: io.StringIO
) -> None:
    settings = settings_for(
        api_keys_operator=SecretStr(f"alice:{OP_KEY}"),
        api_keys_viewer=SecretStr(f"victor:{VIEW_KEY}"),
    )
    rt = file_runtime(sim_export, tmp_path, settings=settings)
    server = ServerHandle(create_app(rt), access_log=True)
    await server.start()
    try:
        await rt.manager.wait_idle()
        async with httpx.AsyncClient(base_url=server.base_url) as c:
            for key in ALL_KEYS:
                await c.get("/machines", headers={"X-API-Key": key})
                await c.get(f"/health?apiKey={key}")  # wrong place for a key, still masked
            await c.post("/alerts/x/ack", headers={"X-API-Key": OP_KEY})
        for key in (VIEW_KEY, WRONG_KEY):
            async with websockets.connect(server.ws_url, close_timeout=0.5) as ws:
                await ws.send(json.dumps({"type": "auth", "apiKey": key}))
                await ws.recv()
        log = get_logger("test")
        log.info("deliberate", api_key=OP_KEY, apiKey=VIEW_KEY, note=f"contains {OP_KEY}")
        log.info(f"message with {VIEW_KEY} inside")
    finally:
        await server.stop()
    text = captured.getvalue()
    assert '"GET /machines HTTP/1.1"' in text  # access logs were captured
    assert "deliberate" in text
    assert REDACTED in text
    leaked = [k for k in ALL_KEYS if k in text]
    assert leaked == [], f"keys leaked into logs: {leaked}"
