"""MQTT publisher: the simulated line on the CLAUDE.md topics (aiomqtt, MQTT v5).

- ``{prefix}/{machineId}/telemetry``: every Reading as JSON (QoS from the options), with an
  MQTT v5 user property ``sentAt`` (wall clock, ISO-8601) for end-to-end latency.
- ``{prefix}/{machineId}/state``: ``{"machineId", "state", "seq", "ts"}`` on every state
  change, retained, so a new subscriber sees the current state at once.
- ``{prefix}/+/cmd``: Commands are applied through the RealtimeRunner and acknowledged on
  ``{prefix}/{machineId}/cmd/ack``.

Readings pass through a bounded queue drained by one task: simulation ticks never wait for the
broker, publish order is kept, and readings survive a short broker outage (the publisher
reconnects with exponential backoff). aiomqtt needs a selector event loop on Windows, so start
the publisher with ``twin_common.aio.run`` (ADR 0008).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import aiomqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
from pydantic import ValidationError

from twin_common.aio import require_selector_loop
from twin_common.config.sources import ReconnectConfig
from twin_common.contracts import Command, CommandAck, Reading
from twin_common.enums import AckStatus, MachineState
from twin_common.logging import get_logger
from twin_common.topics import cmd_ack_topic, state_topic, telemetry_topic
from twin_sim.publishers.session import SimSession

log = get_logger(__name__)

SENT_AT_PROPERTY = "sentAt"
CMD_QOS = 1


@dataclass(frozen=True)
class MqttPublisherOptions:
    host: str
    port: int
    topic_prefix: str
    qos: int
    keepalive_s: int
    client_id: str
    reconnect: ReconnectConfig
    command_timeout_s: float
    buffer_size: int  # readings kept while the broker is unreachable (oldest dropped)


def publish_properties(sent_at: datetime) -> Properties:
    props = Properties(PacketTypes.PUBLISH)  # type: ignore[no-untyped-call]
    props.UserProperty = [(SENT_AT_PROPERTY, sent_at.isoformat())]
    return props


class MqttPublisher:
    def __init__(self, session: SimSession, opts: MqttPublisherOptions) -> None:
        self.session = session
        self.opts = opts
        self._buffer: deque[Reading] = deque(maxlen=opts.buffer_size)
        self._wakeup = asyncio.Event()
        self._last_state: dict[str, MachineState] = {}
        self._stop = asyncio.Event()
        self.connected = asyncio.Event()
        self.published = 0
        self.dropped = 0
        self.commands = 0
        session.add_listener(self._on_reading)

    def _on_reading(self, reading: Reading) -> None:
        if len(self._buffer) == self._buffer.maxlen:
            self.dropped += 1
        self._buffer.append(reading)
        self._wakeup.set()

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    async def run(self) -> None:
        """Publish until ``stop()``; reconnects with exponential backoff."""
        require_selector_loop("MQTT publisher (aiomqtt)")
        attempt = 0
        cfg = self.opts.reconnect
        while not self._stop.is_set():
            try:
                async with aiomqtt.Client(
                    self.opts.host,
                    self.opts.port,
                    identifier=self.opts.client_id,
                    protocol=aiomqtt.ProtocolVersion.V5,
                    keepalive=self.opts.keepalive_s,
                ) as client:
                    attempt = 0
                    await client.subscribe(f"{self.opts.topic_prefix.rstrip('/')}/+/cmd", CMD_QOS)
                    self.connected.set()
                    log.info("mqtt_pub.connected", host=self.opts.host, port=self.opts.port)
                    await self._serve(client)
                    return
            except aiomqtt.MqttError as exc:
                self.connected.clear()
                delay = min(cfg.max_s, cfg.initial_s * cfg.multiplier**attempt)
                attempt += 1
                log.warning("mqtt_pub.reconnecting", error=str(exc), delay_s=delay)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)

    async def _serve(self, client: aiomqtt.Client) -> None:
        pub = asyncio.create_task(self._publish_loop(client))
        cmds = asyncio.create_task(self._command_loop(client))
        stop = asyncio.create_task(self._stop.wait())
        try:
            done, _ = await asyncio.wait({pub, cmds, stop}, return_when=asyncio.FIRST_COMPLETED)
            if stop in done:
                await self._flush(client)
            for task in done:
                if task is not stop:
                    task.result()  # re-raise MqttError -> reconnect
        finally:
            for task in (pub, cmds, stop):
                task.cancel()
            for task in (pub, cmds, stop):
                with contextlib.suppress(asyncio.CancelledError, aiomqtt.MqttError):
                    await task

    async def _publish_one(self, client: aiomqtt.Client, reading: Reading) -> None:
        prefix = self.opts.topic_prefix
        await client.publish(
            telemetry_topic(reading.machine_id, prefix),
            reading.model_dump_json(),
            qos=self.opts.qos,
            properties=publish_properties(datetime.now(UTC)),
        )
        if self._last_state.get(reading.machine_id) is not reading.state:
            self._last_state[reading.machine_id] = reading.state
            state = {
                "machineId": reading.machine_id,
                "state": reading.state.value,
                "seq": reading.seq,
                "ts": reading.ts.isoformat(),
            }
            await client.publish(
                state_topic(reading.machine_id, prefix),
                json.dumps(state),
                qos=self.opts.qos,
                retain=True,
            )
        self.published += 1

    async def _flush(self, client: aiomqtt.Client) -> None:
        while self._buffer:
            await self._publish_one(client, self._buffer[0])
            self._buffer.popleft()

    async def _publish_loop(self, client: aiomqtt.Client) -> None:
        while True:
            # Only drop a reading from the buffer once the broker has accepted it.
            await self._flush(client)
            self._wakeup.clear()
            if not self._buffer:
                await self._wakeup.wait()

    async def _command_loop(self, client: aiomqtt.Client) -> None:
        async for message in client.messages:
            payload = message.payload
            if not isinstance(payload, bytes | str):
                continue
            await self._handle_command(client, payload)

    async def _handle_command(self, client: aiomqtt.Client, payload: bytes | str) -> None:
        try:
            cmd = Command.model_validate_json(payload)
        except ValidationError as exc:
            log.warning("mqtt_pub.bad_command", error=str(exc).splitlines()[0])
            return
        self.commands += 1
        ack = await self.session.submit(cmd, self.opts.command_timeout_s)
        if ack is None:
            ack = CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.TIMEOUT,
                reason="the simulator is not running or did not apply the command in time",
                ts=datetime.now(UTC),
            )
        await client.publish(
            cmd_ack_topic(cmd.machine_id, self.opts.topic_prefix),
            ack.model_dump_json(),
            qos=CMD_QOS,
        )
