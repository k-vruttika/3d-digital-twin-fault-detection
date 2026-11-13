"""MQTT publisher: the simulated line on the CLAUDE.md topics (aiomqtt, MQTT v5).

- ``{prefix}/{machineId}/telemetry``: every Reading as JSON (QoS from the options), with an
  MQTT v5 user property ``sentAt`` (wall clock, ISO-8601) for end-to-end latency.
- ``{prefix}/{machineId}/state``: ``{"machineId", "state", "seq", "ts"}`` on every state
  change, retained, so a new subscriber sees the current state at once.
- ``{prefix}/+/cmd``: Commands are applied through the RealtimeRunner and acknowledged on
  ``{prefix}/{machineId}/cmd/ack``.
- ``{prefix}/scenario``: ScenarioRequests (simulation only, ADR 0015), acknowledged with a
  ScenarioAck on ``{prefix}/scenario/ack``.

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
from twin_common.contracts import Command, CommandAck, Reading, ScenarioRequest
from twin_common.enums import AckStatus, MachineState
from twin_common.logging import get_logger
from twin_common.topics import (
    cmd_ack_topic,
    scenario_ack_topic,
    scenario_topic,
    state_topic,
    telemetry_topic,
)
from twin_sim.publishers.scenario import timeout_ack
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
        # Readings not yet accepted by the broker. Only the publish loop removes items (one
        # drainer, guarded by _drain_lock); the reading being sent is taken out first, so the
        # bound below can never evict it, and a failed send puts it back at the front.
        self._buffer: deque[Reading] = deque()
        self._drain_lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._last_state: dict[str, MachineState] = {}
        self._stop = asyncio.Event()
        self.connected = asyncio.Event()
        self.published = 0
        self.dropped = 0
        self.commands = 0
        session.add_listener(self._on_reading)

    def _on_reading(self, reading: Reading) -> None:
        while len(self._buffer) >= self.opts.buffer_size:
            self._buffer.popleft()  # broker unreachable for too long: drop the oldest
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
                    await client.subscribe(scenario_topic(self.opts.topic_prefix), CMD_QOS)
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
        """Run the publish loop (sole owner of the buffer) and the command loop.

        Returns once the publish loop has drained the buffer after ``stop()``; raises
        MqttError (-> reconnect) if either loop loses the connection.
        """
        pub = asyncio.create_task(self._publish_loop(client))
        cmds = asyncio.create_task(self._command_loop(client))
        try:
            done, _ = await asyncio.wait({pub, cmds}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()  # re-raise MqttError -> reconnect
        finally:
            for task in (pub, cmds):
                task.cancel()
            for task in (pub, cmds):
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
            # Only after the broker accepted it, so a failed send is retried with the reading.
            self._last_state[reading.machine_id] = reading.state
        self.published += 1

    async def _drain(self, client: aiomqtt.Client) -> None:
        """Send buffered readings oldest first, one at a time.

        The lock guarantees a single drainer. A reading counts as sent only once the broker
        accepted it; if the publish fails or is cancelled, it goes back to the front of the
        buffer, so the next connection resends it first (at least once, order kept).
        """
        async with self._drain_lock:
            while self._buffer:
                reading = self._buffer.popleft()
                try:
                    await self._publish_one(client, reading)
                except BaseException:
                    self._buffer.appendleft(reading)
                    raise

    async def _publish_loop(self, client: aiomqtt.Client) -> None:
        """Sole owner of the buffer: drains it, sleeps until new readings arrive, and on
        ``stop()`` drains what is left and returns."""
        while True:
            self._wakeup.clear()
            await self._drain(client)
            if self._stop.is_set():
                return
            if not self._buffer:
                await self._wakeup.wait()

    async def _command_loop(self, client: aiomqtt.Client) -> None:
        scenario = scenario_topic(self.opts.topic_prefix)
        async for message in client.messages:
            payload = message.payload
            if not isinstance(payload, bytes | str):
                continue
            if message.topic.matches(scenario):
                await self._handle_scenario(client, payload)
            else:
                await self._handle_command(client, payload)

    async def _handle_scenario(self, client: aiomqtt.Client, payload: bytes | str) -> None:
        try:
            req = ScenarioRequest.model_validate_json(payload)
        except ValidationError as exc:
            log.warning("mqtt_pub.bad_scenario", error=str(exc).splitlines()[0])
            return
        ack = await self.session.submit_scenario(req, self.opts.command_timeout_s)
        await client.publish(
            scenario_ack_topic(self.opts.topic_prefix),
            (ack if ack is not None else timeout_ack(req)).model_dump_json(),
            qos=CMD_QOS,
        )

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
