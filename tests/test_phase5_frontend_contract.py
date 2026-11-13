"""The frontend contract in minimal mode (Phase 5).

Starts ``python -m twin_backend serve --mode minimal --no-loop`` in a fresh process on a short
replay file (SQLite, no keys, no Docker), as the 3D dev page sees it:

- ``GET /line`` equals the committed frontend fixture (the frontend tests use that file);
- every WebSocket frame validates against the published ``WsServerMessage`` schema, from which
  the frontend types are generated;
- when the replay ends, every machine goes STALE and then OFFLINE after ``stale_after_s`` /
  ``offline_after_s`` of silence (``sources.yaml``), which the twin shows as semi-transparent
  and then greyed out.
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

from twin_backend.api.ws_models import SERVER_ADAPTER
from twin_backend.testing import free_port
from twin_common.config import load_sources_config

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_S = 90.0
RUN_S = 40
REPLAY_SPEED = 4.0
STALENESS = load_sources_config(REPO_ROOT / "config" / "sources.yaml").ingest
# Wall-clock slack for process scheduling and the staleness tick.
SLACK_S = 2.5


@pytest.fixture(scope="module")
def replay_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sim_p5")
    subprocess.run(
        [
            *(sys.executable, "-m", "twin_sim", "generate", "--quick", "--runs", "1"),
            *("--duration", str(RUN_S), "--workers", "1", "--export", "csv", "--out", str(out)),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return out / "run_0000.csv"


@pytest.fixture(scope="module")
def server(replay_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = free_port()
    tmp = tmp_path_factory.mktemp("p5_api")
    log_path = tmp / "server.log"
    env = dict(os.environ)
    env["LINETWIN_API_KEYS_OPERATOR"] = ""
    env["LINETWIN_API_KEYS_VIEWER"] = ""
    cmd = [
        *(sys.executable, "-m", "twin_backend", "serve", "--mode", "minimal", "--no-loop"),
        *("--file", str(replay_file.resolve()), "--replay-speed", str(REPLAY_SPEED)),
        *("--sqlite-path", str(tmp / "p5.sqlite"), "--port", str(port)),
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
            time.sleep(0.1)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)


def test_line_matches_frontend_fixture(server: str) -> None:
    fixture = json.loads(
        (REPO_ROOT / "frontend/src/engine/__fixtures__/line.json").read_text(encoding="utf-8")
    )
    assert httpx.get(f"{server}/line").json() == fixture


async def _watch_until_offline(url: str, machines: list[str]) -> dict[str, Any]:
    last_reading: dict[str, float] = {}
    stale_at: dict[str, float] = {}
    offline_at: dict[str, float] = {}
    frames = 0
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "auth"}))
        await ws.send(json.dumps({"type": "subscribe"}))  # all machines, all channels
        deadline = time.monotonic() + RUN_S / REPLAY_SPEED + STALENESS.offline_after_s + 30
        while len(offline_at) < len(machines):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(f"not every machine went OFFLINE: {sorted(offline_at)}")
            msg = json.loads(await asyncio.wait_for(ws.recv(), remaining))
            SERVER_ADAPTER.validate_python(msg)  # the published schema
            frames += 1
            now = time.monotonic()
            if msg["type"] == "reading":
                last_reading[msg["data"]["machineId"]] = now
            elif msg["type"] == "state":
                mid, state = msg["data"]["machineId"], msg["data"]["state"]
                if state == "STALE":
                    stale_at.setdefault(mid, now)
                elif state == "OFFLINE":
                    offline_at.setdefault(mid, now)
    return {"last": last_reading, "stale": stale_at, "offline": offline_at, "frames": frames}


def test_ws_frames_follow_schema_and_machines_go_stale_then_offline(server: str) -> None:
    machines = [m["id"] for m in httpx.get(f"{server}/line").json()["machines"]]
    ws_url = server.replace("http://", "ws://") + "/stream"
    seen = asyncio.run(_watch_until_offline(ws_url, machines))
    assert seen["frames"] > len(machines)
    for mid in machines:
        assert mid in seen["last"], f"{mid}: no reading seen"
        assert mid in seen["stale"], f"{mid}: no STALE before OFFLINE"
        stale = seen["stale"][mid] - seen["last"][mid]
        offline = seen["offline"][mid] - seen["last"][mid]
        assert seen["stale"][mid] <= seen["offline"][mid]
        assert STALENESS.stale_after_s - 0.5 <= stale <= STALENESS.stale_after_s + SLACK_S, stale
        assert STALENESS.offline_after_s - 0.5 <= offline <= STALENESS.offline_after_s + SLACK_S, (
            offline
        )
