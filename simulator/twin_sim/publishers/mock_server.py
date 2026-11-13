"""Mock machine server (FastAPI): the simulated line over REST and WebSocket.

- ``GET /info``: line id and machine ids
- ``GET /state``: latest Reading per machine
- ``GET /history?machineId=&sinceSeq=&limit=``: Readings with seq > sinceSeq, oldest first
- ``WS /stream``: every Reading as ``{"reading": {...}, "sentAt": "<wall-clock ISO>"}``
- ``POST /command``: a ``Command``; responds with the simulator's ``CommandAck``
- ``POST /scenario``: a ``ScenarioRequest`` (simulation only, ADR 0015); responds with a
  ``ScenarioAck``

A WebSocket client that cannot keep up loses messages from its queue; it can fill the gap
from ``/history`` (the backend MockServerAdapter does so after every reconnect).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from twin_common.contracts import Command, CommandAck, Reading, ScenarioAck, ScenarioRequest
from twin_common.enums import AckStatus
from twin_common.logging import get_logger
from twin_sim.publishers.scenario import timeout_ack
from twin_sim.publishers.session import SimSession

log = get_logger(__name__)

DEFAULT_COMMAND_TIMEOUT_S = 5.0
DEFAULT_HISTORY_LIMIT = 1000
WS_QUEUE_SIZE = 10_000  # per client; full -> messages dropped, client backfills from /history


def envelope(reading: Reading, sent_at: datetime) -> str:
    return f'{{"reading":{reading.model_dump_json()},"sentAt":"{sent_at.isoformat()}"}}'


def create_app(
    session: SimSession,
    *,
    own_session: bool = False,
    command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
) -> FastAPI:
    """Build the app. With ``own_session`` the app starts and stops the simulation itself."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if own_session:
            session.start()
        try:
            yield
        finally:
            if own_session:
                await session.stop()

    app = FastAPI(title="LineTwin mock machine server", lifespan=lifespan)
    app.state.ws_clients = 0

    @app.get("/info")
    async def info() -> dict[str, Any]:
        return {"lineId": session.line.line_id, "machines": session.machine_ids}

    @app.get("/state")
    async def state() -> list[Reading]:
        return session.history.latest()

    @app.get("/history")
    async def history(
        machine_id: Annotated[str | None, Query(alias="machineId")] = None,
        since_seq: Annotated[int, Query(alias="sinceSeq", ge=0)] = 0,
        limit: Annotated[int, Query(ge=1)] = DEFAULT_HISTORY_LIMIT,
    ) -> list[Reading]:
        return session.history.since(machine_id, since_seq, limit)

    @app.post("/command")
    async def command(cmd: Command) -> CommandAck:
        ack = await session.submit(cmd, command_timeout_s)
        if ack is None:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.TIMEOUT,
                reason="the simulator is not running or did not apply the command in time",
                ts=datetime.now(UTC),
            )
        return ack

    @app.post("/scenario")
    async def scenario(req: ScenarioRequest) -> ScenarioAck:
        ack = await session.submit_scenario(req, command_timeout_s)
        return ack if ack is not None else timeout_ack(req)

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=WS_QUEUE_SIZE)

        def listener(reading: Reading) -> None:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(envelope(reading, datetime.now(UTC)))

        async def pump() -> None:
            while True:
                await ws.send_text(await queue.get())

        session.add_listener(listener)
        app.state.ws_clients += 1
        pump_task = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            session.remove_listener(listener)
            app.state.ws_clients -= 1
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task

    return app
