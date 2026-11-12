"""MQTT adapter (aiomqtt, MQTT v5): telemetry in, commands out.

- **Telemetry:** subscribes to ``{prefix}/+/telemetry`` at the configured QoS (1 by default:
  at least once, duplicates removed by dedup; ADR 0007). The connection uses a persistent
  session (``clean_start=False``, session expiry ``session_expiry_s``), so the broker queues
  telemetry while the adapter reconnects.
- **sentAt:** the MQTT v5 user property ``sentAt``, when present, becomes ``RawMessage.sent_at``.
- **Commands:** published on ``{prefix}/{machineId}/cmd``. The adapter waits for the matching
  ``commandId`` on ``{prefix}/+/cmd/ack``, or answers TIMEOUT after ``command_timeout_s``.

aiomqtt needs a selector event loop on Windows; start the program with
``twin_common.aio.run`` (ADR 0008).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import aiomqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from twin_backend.adapters.base import BaseAdapter
from twin_backend.clock import Clock
from twin_common.aio import require_selector_loop
from twin_common.config.line import LineConfig
from twin_common.config.sources import MqttSourceConfig, ReconnectConfig
from twin_common.contracts import Command, CommandAck
from twin_common.enums import AckStatus, AdapterKind, Capability
from twin_common.logging import get_logger
from twin_common.topics import cmd_topic

log = get_logger(__name__)

SENT_AT_PROPERTY = "sentAt"
CMD_QOS = 1


def _sent_at(properties: Any) -> datetime | None:
    for key, value in getattr(properties, "UserProperty", None) or []:
        if key == SENT_AT_PROPERTY:
            try:
                return datetime.fromisoformat(value)
            except (TypeError, ValueError):
                return None
    return None


class MqttAdapter(BaseAdapter):
    kind = AdapterKind.MQTT
    capabilities = frozenset({Capability.COMMANDS})

    def __init__(
        self,
        cfg: MqttSourceConfig,
        line: LineConfig,
        *,
        reconnect: ReconnectConfig,
        command_timeout_s: float,
        clock: Clock,
    ) -> None:
        super().__init__(clock, reconnect)
        self.cfg = cfg
        self.machine_ids = {m.id for m in line.machines}
        self.command_timeout_s = command_timeout_s
        prefix = cfg.topic_prefix.rstrip("/")
        self.telemetry_filter = f"{prefix}/+/telemetry"
        self.ack_filter = f"{prefix}/+/cmd/ack"
        self._client: aiomqtt.Client | None = None
        self._pending: dict[str, asyncio.Future[CommandAck]] = {}

    async def _prepare(self) -> None:
        require_selector_loop("MQTT adapter (aiomqtt)")

    def _make_client(self) -> aiomqtt.Client:
        props = Properties(PacketTypes.CONNECT)  # type: ignore[no-untyped-call]
        props.SessionExpiryInterval = self.cfg.session_expiry_s
        return aiomqtt.Client(
            self.cfg.host,
            self.cfg.port,
            identifier=self.cfg.client_id,
            protocol=aiomqtt.ProtocolVersion.V5,
            keepalive=self.cfg.keepalive_s,
            clean_start=False,
            properties=props,
        )

    async def _run(self) -> None:
        while not self._stopping:
            try:
                async with self._make_client() as client:
                    await client.subscribe(self.telemetry_filter, self.cfg.qos)
                    await client.subscribe(self.ack_filter, CMD_QOS)
                    self._client = client
                    self._mark_connected()
                    log.info("mqtt.connected", host=self.cfg.host, port=self.cfg.port)
                    async for message in client.messages:
                        await self._dispatch(message)
            except aiomqtt.MqttError as exc:
                self._client = None
                await self._wait_reconnect(exc)
            finally:
                self._client = None

    async def _dispatch(self, message: aiomqtt.Message) -> None:
        payload = message.payload
        if not isinstance(payload, bytes | str):
            payload = b"" if payload is None else str(payload)
        topic = message.topic.value
        if message.topic.matches(self.ack_filter):
            self._resolve_ack(payload)
            return
        await self._emit(
            payload,
            sent_at=_sent_at(message.properties),
            meta={"topic": topic, "qos": str(message.qos)},
        )

    def _resolve_ack(self, payload: bytes | str) -> None:
        try:
            ack = CommandAck.model_validate_json(payload)
        except ValueError:
            log.warning("mqtt.bad_ack", payload=str(payload)[:200])
            return
        fut = self._pending.pop(ack.command_id, None)
        if fut is not None and not fut.done():
            fut.set_result(ack)

    async def send_command(self, cmd: Command) -> CommandAck:
        client = self._client
        if client is None:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.REJECTED,
                reason="MQTT adapter is not connected",
                ts=self.clock.now(),
            )
        fut: asyncio.Future[CommandAck] = asyncio.get_running_loop().create_future()
        self._pending[cmd.command_id] = fut
        try:
            await client.publish(
                cmd_topic(cmd.machine_id, self.cfg.topic_prefix),
                cmd.model_dump_json(),
                qos=CMD_QOS,
            )
            return await asyncio.wait_for(fut, timeout=self.command_timeout_s)
        except TimeoutError:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.TIMEOUT,
                reason=f"no ack within {self.command_timeout_s:g} s",
                ts=self.clock.now(),
            )
        except aiomqtt.MqttError as exc:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.REJECTED,
                reason=f"command transport failed: {exc}",
                ts=self.clock.now(),
            )
        finally:
            self._pending.pop(cmd.command_id, None)
