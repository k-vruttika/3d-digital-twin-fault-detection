"""WebSocket stream ``/stream``: fan-out of bus events to subscribed clients (ADR 0013).

Protocol (JSON text frames; see docs/api.md, "WebSocket stream"):

1. The client's first message is ``{"type": "auth", "apiKey": "..."}`` (``apiKey`` may be
   omitted when anonymous read is allowed). Without it the server waits ``auth_timeout_s``,
   then grants anonymous read access if allowed or closes with code 4401.
2. ``{"type": "subscribe", "machines": [...] | null, "channels": [...]}`` selects what to
   receive; the server answers with ``subscribed`` and a ``snapshot``.
3. The server streams ``reading``, ``state``, ``line``, ``dataQuality``, ``alert``,
   ``command``, ``ack`` and ``prediction`` messages.

Backpressure: the hub never waits for a client. Every client has an :class:`Outbox`; readings,
line views and data-quality reports are *coalesced* (only the newest per machine is kept until
the client takes it), and ordered messages (states, alerts, commands) go into a bounded queue
that drops the oldest and reports the loss with a ``dropped`` message. A client whose send
takes longer than ``send_timeout_s`` is closed. A slow client therefore only delays itself,
never other clients or ingestion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from twin_backend.api.auth import Authenticator, Principal
from twin_backend.bus.base import EventBus, Subscription, Topic
from twin_backend.engine.data_quality import DataQualityReport
from twin_backend.engine.twin import TwinEngine
from twin_common.config.api import WsConfig
from twin_common.contracts import Alert, Command, CommandAck, Prediction, Reading, StateChange
from twin_common.logging import get_logger

log = get_logger(__name__)

CLOSE_UNAUTHORIZED = 4401
CLOSE_TOO_SLOW = 4408

CHANNELS = ("readings", "states", "line", "dataQuality", "alerts", "commands", "predictions")
COALESCED = frozenset({"readings", "line", "dataQuality"})
HUB_TOPICS = (Topic.READINGS, Topic.STATES, Topic.PREDICTIONS, Topic.COMMANDS, Topic.ALERTS)


def _channel_and_type(message: BaseModel) -> tuple[str, str] | None:
    if isinstance(message, Reading):
        return "readings", "reading"
    if isinstance(message, StateChange):
        return "states", "state"
    if isinstance(message, Prediction):
        return "predictions", "prediction"
    if isinstance(message, Alert):
        return "alerts", "alert"
    if isinstance(message, Command):
        return "commands", "command"
    if isinstance(message, CommandAck):
        return "commands", "ack"
    return None


def envelope(msg_type: str, data: Any) -> str:
    return json.dumps({"type": msg_type, "data": data}, separators=(",", ":"))


class Outbox:
    """Per-client buffer: coalesced latest-per-key values plus a bounded ordered queue."""

    def __init__(self, maxsize: int) -> None:
        self._ordered: deque[str] = deque()
        self._maxsize = maxsize
        self._latest: dict[tuple[str, str], str] = {}
        self._event = asyncio.Event()
        self.dropped = 0
        self.coalesced = 0
        self._reported_dropped = 0

    def __len__(self) -> int:
        return len(self._ordered) + len(self._latest)

    def put(self, text: str) -> None:
        if len(self._ordered) >= self._maxsize:
            self._ordered.popleft()
            self.dropped += 1
        self._ordered.append(text)
        self._event.set()

    def put_latest(self, key: tuple[str, str], text: str) -> None:
        if key in self._latest:
            self.coalesced += 1
            del self._latest[key]  # re-insert: keeps arrival order of the newest values
        self._latest[key] = text
        self._event.set()

    def take(self) -> list[str]:
        """Everything queued now: a ``dropped`` notice, ordered messages, then latest values."""
        out: list[str] = []
        if self.dropped > self._reported_dropped:
            out.append(envelope("dropped", {"count": self.dropped - self._reported_dropped}))
            self._reported_dropped = self.dropped
        out.extend(self._ordered)
        out.extend(self._latest.values())
        self._ordered.clear()
        self._latest.clear()
        self._event.clear()
        return out

    async def wait(self) -> None:
        await self._event.wait()


@dataclass(eq=False)
class Client:
    ws: WebSocket
    principal: Principal
    outbox: Outbox
    machines: frozenset[str] | None = None  # None = all
    channels: frozenset[str] = frozenset()
    sender: asyncio.Task[None] | None = field(default=None, repr=False)

    def wants(self, channel: str, machine_id: str | None) -> bool:
        if channel not in self.channels:
            return False
        return self.machines is None or machine_id is None or machine_id in self.machines


class SubscribeMessage(BaseModel):
    type: str
    machines: list[str] | None = None
    channels: list[str] = list(CHANNELS)


class HubMetrics:
    def __init__(self, registry: CollectorRegistry | None) -> None:
        self.clients = Gauge(
            "linetwin_ws_clients", "Connected WebSocket clients", registry=registry
        )
        self.sent = Counter("linetwin_ws_messages_sent", "Messages sent", registry=registry)
        self.dropped = Counter(
            "linetwin_ws_messages_dropped",
            "Ordered messages dropped for slow clients",
            registry=registry,
        )
        self.coalesced = Counter(
            "linetwin_ws_messages_coalesced",
            "Messages replaced by a newer value before a slow client took them",
            registry=registry,
        )
        self.closed_slow = Counter(
            "linetwin_ws_clients_closed_slow",
            "Clients closed because a send timed out",
            registry=registry,
        )


class WsHub:
    def __init__(
        self,
        bus: EventBus,
        engine: TwinEngine,
        auth: Authenticator,
        cfg: WsConfig,
        *,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.bus = bus
        self.engine = engine
        self.auth = auth
        self.cfg = cfg
        self.metrics = HubMetrics(registry)
        self.clients: set[Client] = set()
        self._subs: list[Subscription] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._listener: Callable[[str, BaseModel], None] = self._on_engine

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> None:
        if self._tasks:
            return
        self._subs = [self.bus.subscribe(t, maxsize=self.cfg.hub_queue_size) for t in HUB_TOPICS]
        self._tasks = [asyncio.create_task(self._pump(s), name=f"ws-{s.topic}") for s in self._subs]
        self.engine.add_listener(self._listener)

    async def stop(self) -> None:
        self.engine.remove_listener(self._listener)
        for sub in self._subs:
            sub.close()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for client in list(self.clients):
            await self._close(client, 1001)

    # -- fan-out (never waits) ---------------------------------------------------------------
    async def _pump(self, sub: Subscription) -> None:
        async for message in sub:
            self.publish(message)

    def publish(self, message: BaseModel) -> None:
        kind = _channel_and_type(message)
        if kind is None:
            return
        channel, msg_type = kind
        # Acks carry no machine id: they go to every client subscribed to "commands".
        machine_id = getattr(message, "machine_id", None)
        self._fanout(channel, msg_type, machine_id, message.model_dump(mode="json"))

    def _on_engine(self, channel: str, payload: BaseModel) -> None:
        if channel == "line":
            self._fanout("line", "line", None, payload.model_dump(mode="json"))
        elif channel == "dataQuality" and isinstance(payload, DataQualityReport):
            self._fanout(
                "dataQuality", "dataQuality", payload.machine_id, payload.model_dump(mode="json")
            )

    def _fanout(self, channel: str, msg_type: str, machine_id: str | None, data: Any) -> None:
        text: str | None = None
        for client in self.clients:
            if not client.wants(channel, machine_id):
                continue
            if text is None:
                text = envelope(msg_type, data)  # serialise once for every client
            if channel in COALESCED:
                before = client.outbox.coalesced
                client.outbox.put_latest((channel, machine_id or ""), text)
                if client.outbox.coalesced > before:
                    self.metrics.coalesced.inc()
            else:
                before = client.outbox.dropped
                client.outbox.put(text)
                if client.outbox.dropped > before:
                    self.metrics.dropped.inc()

    # -- one connection ----------------------------------------------------------------------
    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        principal = await self._authenticate(ws)
        if principal is None:
            return
        client = Client(ws, principal, Outbox(self.cfg.client_queue_size))
        self.clients.add(client)
        self.metrics.clients.set(len(self.clients))
        client.sender = asyncio.create_task(self._send_loop(client), name="ws-sender")
        try:
            while True:
                raw = await ws.receive_text()
                await self._handle(client, raw)
        except WebSocketDisconnect:
            pass
        finally:
            await self._close(client, None)

    async def _authenticate(self, ws: WebSocket) -> Principal | None:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=self.cfg.auth_timeout_s)
        except TimeoutError:
            if self.auth.anonymous_read:
                principal = self.auth.resolve(None)
                await ws.send_text(envelope("auth", self._auth_info(principal)))
                return principal
            await self._refuse(ws, 'authentication required: send {"type": "auth", ...} first')
            return None
        except WebSocketDisconnect:
            return None
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            msg = None
        if not isinstance(msg, dict) or msg.get("type") != "auth":
            await self._refuse(ws, 'the first message must be {"type": "auth", "apiKey": ...}')
            return None
        key = msg.get("apiKey")
        principal = self.auth.resolve(key if isinstance(key, str) else None)
        if principal is None:
            await self._refuse(ws, "invalid or missing API key")
            return None
        await ws.send_text(envelope("auth", self._auth_info(principal)))
        return principal

    @staticmethod
    def _auth_info(principal: Principal | None) -> dict[str, Any]:
        assert principal is not None
        return {"principal": principal.name, "role": principal.role.value}

    @staticmethod
    async def _refuse(ws: WebSocket, reason: str) -> None:
        with contextlib.suppress(Exception):
            await ws.send_text(envelope("error", {"reason": reason}))
            await ws.close(code=CLOSE_UNAUTHORIZED, reason="unauthorized")

    async def _handle(self, client: Client, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            client.outbox.put(envelope("error", {"reason": "invalid JSON"}))
            return
        msg_type = data.get("type") if isinstance(data, dict) else None
        if msg_type == "ping":
            client.outbox.put(envelope("pong", {}))
            return
        if msg_type == "auth":
            client.outbox.put(envelope("error", {"reason": "already authenticated"}))
            return
        if msg_type != "subscribe":
            client.outbox.put(envelope("error", {"reason": f"unknown message type {msg_type!r}"}))
            return
        try:
            sub = SubscribeMessage.model_validate(data)
        except ValidationError as exc:
            client.outbox.put(envelope("error", {"reason": str(exc)}))
            return
        unknown_ch = sorted(set(sub.channels) - set(CHANNELS))
        unknown_m = sorted(set(sub.machines or ()) - set(self.engine.machine_ids))
        if unknown_ch or unknown_m:
            client.outbox.put(
                envelope(
                    "error",
                    {
                        "reason": "unknown channels/machines",
                        "channels": unknown_ch,
                        "machines": unknown_m,
                    },
                )
            )
            return
        client.machines = frozenset(sub.machines) if sub.machines is not None else None
        client.channels = frozenset(sub.channels)
        client.outbox.put(
            envelope(
                "subscribed",
                {"machines": sub.machines, "channels": sorted(client.channels)},
            )
        )
        client.outbox.put(envelope("snapshot", self._snapshot(client)))

    def _snapshot(self, client: Client) -> dict[str, Any]:
        state = self.engine.snapshot()
        machines = {
            m: v.model_dump(mode="json")
            for m, v in state.machines.items()
            if client.machines is None or m in client.machines
        }
        dq = {
            m: r.model_dump(mode="json")
            for m, r in self.engine.data_quality_all().items()
            if client.machines is None or m in client.machines
        }
        alerts = [
            a.model_dump(mode="json")
            for a in self.engine.alerts.current()
            if client.machines is None or a.machine_id in client.machines
        ]
        return {
            "asOf": state.as_of.isoformat() if state.as_of else None,
            "machines": machines,
            "line": self.engine.line_model().model_dump(mode="json"),
            "dataQuality": dq,
            "alerts": alerts,
        }

    async def _send_loop(self, client: Client) -> None:
        try:
            while True:
                await client.outbox.wait()
                for text in client.outbox.take():
                    await asyncio.wait_for(client.ws.send_text(text), self.cfg.send_timeout_s)
                    self.metrics.sent.inc()
        except TimeoutError:
            self.metrics.closed_slow.inc()
            log.warning("ws.client_too_slow", principal=client.principal.name)
            self.clients.discard(client)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    client.ws.close(code=CLOSE_TOO_SLOW, reason="too slow"), self.cfg.send_timeout_s
                )
        except (WebSocketDisconnect, RuntimeError):
            self.clients.discard(client)

    async def _close(self, client: Client, code: int | None) -> None:
        self.clients.discard(client)
        self.metrics.clients.set(len(self.clients))
        sender, client.sender = client.sender, None
        if sender is not None:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
        if code is not None and client.ws.application_state is WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.ws.close(code=code), self.cfg.send_timeout_s)
