"""Minimal mode serves the API (CLAUDE.md section 1).

Starts ``python -m twin_backend serve --mode minimal`` as a separate process (a plain
``subprocess.Popen``: LineTwin never uses asyncio subprocesses, ADR 0008) on the sim-quick
replay file with SQLite and no API keys, as on a fresh clone. Checks over HTTP and WebSocket:

- the API runs and serves live state for every machine;
- no broker, database driver or ML module was imported by the server process;
- read access works without a key (loopback, no keys) and writes are refused;
- on Windows the running server uses a selector event loop.

CI runs it on the ``generate --quick`` output via ``uv run poe minimal-data-check``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from twin_backend.testing import free_port

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_S = 90.0
MACHINES = ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    preset = os.environ.get("LINETWIN_MINIMAL_DATA_DIR")
    if preset:
        return Path(preset)
    out = tmp_path_factory.mktemp("sim_quick_p4")
    subprocess.run(
        [
            *(sys.executable, "-m", "twin_sim", "generate", "--quick", "--runs", "1"),
            *("--duration", "300", "--workers", "1", "--export", "csv", "--out", str(out)),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def server(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = free_port()
    tmp = tmp_path_factory.mktemp("minimal_api")
    log_path = tmp / "server.log"
    env = dict(os.environ)
    # As on a fresh clone: no API keys (real environment variables override any local .env).
    env["LINETWIN_API_KEYS_OPERATOR"] = ""
    env["LINETWIN_API_KEYS_VIEWER"] = ""
    cmd = [
        sys.executable,
        "-m",
        "twin_backend",
        "serve",
        "--mode",
        "minimal",
        "--file",
        str((data_dir / "run_0000.csv").resolve()),
        "--replay-speed",
        "20",
        "--sqlite-path",
        str(tmp / "minimal.sqlite"),
        "--port",
        str(port),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited early:\n{log_path.read_text(encoding='utf-8')}")
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"server did not start:\n{log_path.read_text(encoding='utf-8')}")
            time.sleep(0.2)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)


def _wait_for(fn: Any, timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        result = fn()
        if result:
            return result
        if time.monotonic() > deadline:
            pytest.fail("condition not met in time")
        time.sleep(0.2)


def test_health_wiring_and_no_heavy_modules(server: str) -> None:
    health = httpx.get(f"{server}/health", params={"modules": "true"}).json()
    assert health["store"] == "sqlite"
    assert health["source"]["kind"] == "file"
    assert health["auth"] == {"keysConfigured": False, "anonymousRead": True, "operatorKeys": False}
    assert health["loadedOptionalModules"] == []  # no broker, DB driver or ML library
    if sys.platform == "win32":
        assert "Selector" in health["eventLoop"]


def test_serves_live_state_for_every_machine(server: str) -> None:
    def all_known() -> dict[str, Any] | None:
        body: dict[str, Any] = httpx.get(f"{server}/machines").json()
        return body if all(m["state"] for m in body["machines"]) else None

    body = _wait_for(all_known)
    assert [m["machineId"] for m in body["machines"]] == MACHINES
    assert body["line"]["status"] in ("RUNNING", "DEGRADED", "STOPPED")
    one = httpx.get(f"{server}/machines/CNC-01/state").json()
    assert one["kind"] == "CNC"
    assert one["lastReading"]["machineId"] == "CNC-01"


def test_history_comes_from_sqlite(server: str) -> None:
    def rows() -> list[Any] | None:
        r = httpx.get(
            f"{server}/history",
            params={
                "from": "2000-01-01T00:00:00Z",
                "to": "2100-01-01T00:00:00Z",
                "resolution": "raw",
            },
        ).json()
        readings: list[Any] = r.get("readings") or []
        return readings if len(readings) >= 8 else None

    readings = _wait_for(rows)
    assert {r["machineId"] for r in readings} <= set(MACHINES)


def test_writes_refused_without_keys(server: str) -> None:
    r = httpx.post(f"{server}/sources/switch", json={"kind": "mock"})
    assert r.status_code == 403
    assert "LINETWIN_API_KEYS_OPERATOR" in r.json()["detail"]


def test_websocket_streams_readings_anonymously(server: str) -> None:
    url = server.replace("http://", "ws://") + "/stream"

    async def run() -> list[dict[str, Any]]:
        got: list[dict[str, Any]] = []
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "auth"}))
            auth = json.loads(await ws.recv())
            assert auth["type"] == "auth"
            assert auth["data"]["role"] == "viewer"
            await ws.send(json.dumps({"type": "subscribe", "channels": ["readings"]}))
            while len([m for m in got if m["type"] == "reading"]) < 4:
                got.append(json.loads(await asyncio.wait_for(ws.recv(), 20)))
        return got

    from twin_common import aio

    messages = aio.run(run())
    types = [m["type"] for m in messages]
    assert types[:2] == ["subscribed", "snapshot"]
    assert all(m["data"]["machineId"] in MACHINES for m in messages if m["type"] == "reading")
