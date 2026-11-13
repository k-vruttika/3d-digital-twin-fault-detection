"""A slow WebSocket client does not slow down other clients or ingestion.

Units: the per-client Outbox coalesces and drops the oldest; a client whose send times out is
closed. Live: a file replay at 40x speed (160 readings/s) streams to one client that never
reads, one that reads slowly, and one normal client. The normal client receives every reading,
ingestion keeps its pace with nothing dropped anywhere, and the slow clients' buffers stay
bounded.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import websockets

from twin_backend.api.app import create_app
from twin_backend.api.auth import Principal
from twin_backend.api.server import ServerHandle
from twin_backend.api.ws import CHANNELS, CLOSE_TOO_SLOW, Client, Outbox, WsHub
from twin_backend.testing import SimExport, file_runtime, settings_for
from twin_common.enums import Role

REPLAY_SPEED = 40.0
RUN_S = 4.0


def test_outbox_coalesces_latest_and_drops_oldest_ordered() -> None:
    box = Outbox(maxsize=3)
    for i in range(10):
        box.put_latest(("readings", "CNC-01"), f"r{i}")
    box.put_latest(("readings", "CNV-01"), "c0")
    for i in range(5):
        box.put(f"s{i}")
    assert len(box) == 5  # 3 ordered + 2 latest
    out = box.take()
    assert json.loads(out[0]) == {"type": "dropped", "data": {"count": 2}}
    assert out[1:] == ["s2", "s3", "s4", "r9", "c0"]
    assert box.coalesced == 9
    assert box.dropped == 2
    box.put("s5")
    assert box.take() == ["s5"]  # the drop was reported once
    assert box.take() == []


class _StuckSocket:
    """A WebSocket whose sends never complete (a client that stopped reading)."""

    def __init__(self) -> None:
        self.closed_with: int | None = None
        self.application_state = None

    async def send_text(self, _text: str) -> None:
        await asyncio.Event().wait()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed_with = code


async def test_client_whose_send_times_out_is_closed(sim_export: SimExport, tmp_path: Path) -> None:
    rt = file_runtime(sim_export, tmp_path, settings=settings_for())
    app = create_app(rt, manage_runtime=False)
    hub: WsHub = app.state.ctx.hub
    hub.cfg = hub.cfg.model_copy(update={"send_timeout_s": 0.05})
    ws = _StuckSocket()
    client = Client(ws, Principal("t", Role.VIEWER), Outbox(4), channels=frozenset(CHANNELS))  # type: ignore[arg-type]
    hub.clients.add(client)
    client.outbox.put("x")
    await asyncio.wait_for(hub._send_loop(client), 5)
    assert ws.closed_with == CLOSE_TOO_SLOW
    assert client not in hub.clients
    assert rt.metrics.registry.get_sample_value("linetwin_ws_clients_closed_slow_total") == 1.0


async def _auth_subscribe(url: str, channels: list[str]) -> Any:
    ws = await websockets.connect(url, max_queue=1, close_timeout=0.5)
    await ws.send(json.dumps({"type": "auth"}))
    await ws.recv()
    await ws.send(json.dumps({"type": "subscribe", "channels": channels}))
    return ws


async def test_slow_client_does_not_slow_others_or_ingestion(
    sim_export: SimExport, tmp_path: Path
) -> None:
    rt = file_runtime(
        sim_export, tmp_path, settings=settings_for(), replay_speed=REPLAY_SPEED, loop=True
    )
    server = ServerHandle(create_app(rt))
    await server.start()
    hub: WsHub = server.app.state.ctx.hub
    fast_readings: list[dict[str, Any]] = []
    try:
        never = await _auth_subscribe(server.ws_url, list(CHANNELS))  # never reads again
        slow = await _auth_subscribe(server.ws_url, list(CHANNELS))
        fast = await websockets.connect(server.ws_url, close_timeout=0.5)
        await fast.send(json.dumps({"type": "auth"}))
        await fast.recv()
        await fast.send(json.dumps({"type": "subscribe", "channels": ["readings"]}))

        published_at_start: float | None = None
        max_outbox = 0

        async def read_fast() -> None:
            nonlocal published_at_start
            async for raw in fast:
                msg = json.loads(raw)
                if msg["type"] == "snapshot":
                    published_at_start = rt.metrics.value("published", source="file")
                elif msg["type"] == "reading":
                    fast_readings.append(msg["data"])

        async def read_slow() -> None:
            while True:
                await slow.recv()
                await asyncio.sleep(0.5)

        tasks = [asyncio.create_task(read_fast()), asyncio.create_task(read_slow())]
        t0 = time.monotonic()
        while time.monotonic() - t0 < RUN_S:
            await asyncio.sleep(0.05)
            for c in hub.clients:
                max_outbox = max(max_outbox, len(c.outbox))
        elapsed = time.monotonic() - t0
        published_end = rt.metrics.value("published", source="file")
        await asyncio.sleep(0.5)  # let the fast client take what was sent
        for task in tasks:
            task.cancel()
        await never.close()
        await slow.close()
        await fast.close()
    finally:
        await server.stop()

    # ingestion kept its pace: ~160 readings/s (4 machines x replay speed)
    assert published_at_start is not None
    expected = len(rt.engine.machine_ids) * REPLAY_SPEED * elapsed
    assert published_end >= 0.8 * expected, (published_end, expected)
    # the fast client got every reading published while it was subscribed (by machine/seq)
    seen = {(r["machineId"], r["seq"], r["ts"]) for r in fast_readings}
    assert len(fast_readings) >= published_end - published_at_start - len(rt.engine.machine_ids)
    assert len(seen) == len(fast_readings)
    # nobody else lost anything: tracker, engine and storage writer kept up
    assert rt.engine.tracker.dropped == 0
    assert rt.storage.metrics.dropped_total() == 0
    # the slow clients' buffers stayed bounded: ordered queue + one value per coalesced key
    bound = hub.cfg.client_queue_size + 2 * len(rt.engine.machine_ids) + 2
    assert max_outbox <= bound, (max_outbox, bound)
