"""OPC UA adapter (asyncua client): ReadingEvents in, method calls out.

- **Discovery:** machines are found as ``Objects/{line_id}/{machineId}`` in namespace
  ``namespace``, with machine ids taken from the loaded line config.
- **Telemetry:** the adapter subscribes to ``ReadingEventType`` events on each machine node.
  Every event carries a complete reading. Events are queued on the server (``event_queue_size``)
  and on the client (unbounded), so none are dropped while connected.
- **sentAt:** the event ``Time`` (server wall clock when fired) becomes ``RawMessage.sent_at``.
- **Liveness:** with no event for ``watchdog_s`` the server state is read. A failure, a
  subscription status change or a lost transport triggers a reconnect with exponential backoff.
  Events fired while the adapter was disconnected are not recovered; the reorder buffer counts
  them as a gap.
- **Commands:** OPC UA method calls such as ``Stop(commandId, idempotencyKey, issuedBy)``.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

from asyncua import ua
from asyncua.client.client import Client
from asyncua.client.ua_client import UaClientState
from asyncua.common.node import Node
from asyncua.common.subscription import OpcEvent, StatusChangeEvent
from asyncua.ua.object_ids import ObjectIds
from asyncua.ua.uaerrors import UaError

from twin_backend.adapters.base import BaseAdapter
from twin_backend.clock import Clock
from twin_common.config.line import LineConfig
from twin_common.config.sources import OpcUaSourceConfig, ReconnectConfig
from twin_common.contracts import Command, CommandAck
from twin_common.enums import AckStatus, AdapterKind, Capability
from twin_common.logging import get_logger
from twin_common.opcua_names import COMMAND_METHODS, FIELD_BROWSE_NAMES, READING_EVENT_TYPE

log = get_logger(__name__)

_PAYLOAD_FIELDS = frozenset(FIELD_BROWSE_NAMES.values())
_TRANSPORT_ERRORS = (OSError, TimeoutError, ConnectionError, UaError)


def _aware(ts: Any) -> datetime | None:
    if not isinstance(ts, datetime):
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


class OpcUaAdapter(BaseAdapter):
    kind = AdapterKind.OPCUA
    capabilities = frozenset({Capability.COMMANDS})

    def __init__(
        self,
        cfg: OpcUaSourceConfig,
        line: LineConfig,
        *,
        reconnect: ReconnectConfig,
        command_timeout_s: float,
        clock: Clock,
    ) -> None:
        super().__init__(clock, reconnect)
        self.cfg = cfg
        self.line_id = line.line_id
        self.machine_ids = [m.id for m in line.machines]
        self.command_timeout_s = command_timeout_s
        self._client: Client | None = None
        self._machines: dict[str, Node] = {}
        self._ns = 0

    async def _discover(self, client: Client) -> Node:
        self._ns = await client.get_namespace_index(self.cfg.namespace)
        ns = self._ns
        line_node = await client.nodes.objects.get_child(f"{ns}:{self.line_id}")
        self._machines = {mid: await line_node.get_child(f"{ns}:{mid}") for mid in self.machine_ids}
        base_event = client.get_node(ObjectIds.BaseEventType)
        return await base_event.get_child(f"{ns}:{READING_EVENT_TYPE}")

    async def _run(self) -> None:
        while not self._stopping:
            client = Client(url=self.cfg.endpoint, timeout=self.command_timeout_s)
            try:
                await client.connect()
                try:
                    etype = await self._discover(client)
                    sub = await client.create_subscription(
                        self.cfg.publishing_interval_ms, queue_maxsize=0
                    )
                    for node in self._machines.values():
                        await sub.subscribe_events(node, etype, queuesize=self.cfg.event_queue_size)
                    self._client = client
                    self._mark_connected()
                    log.info("opcua.connected", endpoint=self.cfg.endpoint)
                    await self._consume(client, sub)
                finally:
                    self._client = None
                    with contextlib.suppress(*_TRANSPORT_ERRORS):
                        await asyncio.wait_for(client.disconnect(), self.command_timeout_s)
            except _TRANSPORT_ERRORS as exc:
                await self._wait_reconnect(exc)

    async def _consume(self, client: Client, sub: Any) -> None:
        while not self._stopping:
            ev = await sub.next_event(timeout=self.cfg.watchdog_s)
            if ev is None:
                await self._probe(client)
                continue
            if isinstance(ev, StatusChangeEvent):
                raise ConnectionError(f"subscription status changed: {ev.notification.Status}")
            if isinstance(ev, OpcEvent):
                fields = ev.event.get_event_props_as_fields_dict()
                payload = {
                    name: variant.Value  # Double NaN = missing sensor value
                    for name, variant in fields.items()
                    if name in _PAYLOAD_FIELDS
                }
                await self._emit(
                    payload,
                    sent_at=_aware(getattr(ev.event, "Time", None)),
                    meta={"node": str(getattr(ev.event, "SourceName", ""))},
                )

    async def _probe(self, client: Client) -> None:
        if client.state is not UaClientState.CONNECTED:
            raise ConnectionError(f"client state {client.state.name}")
        state_node = client.get_node(ObjectIds.Server_ServerStatus_State)
        await asyncio.wait_for(state_node.read_value(), self.cfg.watchdog_s)

    # -- commands ----------------------------------------------------------------------------
    def _ack(self, cmd: Command, status: AckStatus, reason: str | None) -> CommandAck:
        return CommandAck(
            command_id=cmd.command_id, status=status, reason=reason, ts=self.clock.now()
        )

    async def send_command(self, cmd: Command) -> CommandAck:
        node = self._machines.get(cmd.machine_id) if self._client is not None else None
        if node is None:
            reason = (
                "OPC UA adapter is not connected"
                if self._client is None
                else f"machine '{cmd.machine_id}' has no OPC UA node"
            )
            return self._ack(cmd, AckStatus.REJECTED, reason)
        method, param = COMMAND_METHODS[cmd.type]
        args: list[Any] = [cmd.command_id, cmd.idempotency_key, cmd.issued_by]
        if param is not None:
            args.append(ua.Variant(float(cmd.params[param]), ua.VariantType.Double))
        try:
            result = await asyncio.wait_for(
                node.call_method(f"{self._ns}:{method}", *args), self.command_timeout_s
            )
        except TimeoutError:
            return self._ack(cmd, AckStatus.TIMEOUT, f"no ack within {self.command_timeout_s:g} s")
        except (OSError, ConnectionError, UaError) as exc:
            return self._ack(cmd, AckStatus.REJECTED, f"method call failed: {exc}")
        status, reason = result if isinstance(result, list) else (result, "")
        try:
            ack_status = AckStatus(status)
        except ValueError:
            return self._ack(cmd, AckStatus.REJECTED, f"unexpected method result {result!r}")
        return self._ack(cmd, ack_status, reason or None)
