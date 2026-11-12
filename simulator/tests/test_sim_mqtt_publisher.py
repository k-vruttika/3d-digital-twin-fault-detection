"""MqttPublisher buffer handling, with a fake aiomqtt client (no broker needed).

Regression tests for a CI failure: on stop, the publisher flushed the buffer while its publish
task was still sending, so two coroutines drained the same deque. The result was duplicate
sends, lost readings or ``IndexError: pop from an empty deque``. A slow fake publish makes the
interleaving deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import aiomqtt
import pytest

from twin_common.config.sources import ReconnectConfig
from twin_common.contracts import Reading
from twin_common.testing import sample_reading
from twin_sim.publishers.mqtt import MqttPublisher, MqttPublisherOptions

PREFIX = "test/line1"


class FakeSession:
    """Just enough of SimSession for the publisher."""

    def __init__(self) -> None:
        self.listeners: list[Any] = []

    def add_listener(self, listener: Any) -> None:
        self.listeners.append(listener)


SLOW_PUBACK_ITERATIONS = 5  # loop iterations until the slow publish is acknowledged


class FakeClient:
    """Records publishes like aiomqtt: every publish yields to the event loop, and the first
    ``slow_calls`` telemetry publishes are acknowledged only after SLOW_PUBACK_ITERATIONS
    loop iterations (a slow PUBACK). Purely deterministic: no wall-clock sleeps."""

    def __init__(self, slow_calls: int = 1, fail_seq: int | None = None) -> None:
        self.in_publish = asyncio.Event()
        self.slow_calls = slow_calls
        self.fail_seq = fail_seq
        self.telemetry: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def publish(
        self, topic: str, payload: str, qos: int = 0, retain: bool = False, properties: Any = None
    ) -> None:
        await asyncio.sleep(0)
        if not topic.endswith("/telemetry"):
            return
        seq = Reading.model_validate_json(payload).seq
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.slow_calls > 0:
                self.slow_calls -= 1
                self.in_publish.set()
                for _ in range(SLOW_PUBACK_ITERATIONS):
                    await asyncio.sleep(0)
            if seq == self.fail_seq:
                self.fail_seq = None
                raise aiomqtt.MqttError("broker went away")
            self.telemetry.append(seq)
        finally:
            self.concurrent -= 1

    @property
    def messages(self) -> AsyncIterator[Any]:
        async def never() -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield  # pragma: no cover

        return never()


def make_publisher(buffer_size: int = 100) -> tuple[MqttPublisher, FakeSession]:
    session = FakeSession()
    opts = MqttPublisherOptions(
        host="127.0.0.1",
        port=1883,
        topic_prefix=PREFIX,
        qos=1,
        keepalive_s=30,
        client_id="fake",
        reconnect=ReconnectConfig(initial_s=0.1, max_s=1.0, multiplier=2.0, jitter=0.0, seed=1),
        command_timeout_s=1.0,
        buffer_size=buffer_size,
    )
    return MqttPublisher(session, opts), session  # type: ignore[arg-type]


def feed(session: FakeSession, seqs: range) -> None:
    for seq in seqs:
        for listener in session.listeners:
            listener(sample_reading(machine_id="CNV-01", seq=seq))


async def test_stop_during_slow_publish_sends_each_reading_once_in_order() -> None:
    """The CI race: stop() while the publish task waits for a PUBACK."""
    publisher, session = make_publisher()
    client = FakeClient(slow_calls=1)
    feed(session, range(1, 6))
    serve = asyncio.create_task(publisher._serve(client))  # type: ignore[arg-type]
    await asyncio.wait_for(client.in_publish.wait(), 1)  # seq 1 waits for its PUBACK
    feed(session, range(6, 9))  # readings keep arriving while stopping
    publisher.stop()
    await asyncio.wait_for(serve, 2)
    assert client.telemetry == list(range(1, 9))  # nothing lost, nothing twice, in order
    assert client.max_concurrent == 1  # only one drainer ever publishes
    assert publisher.published == 8


async def test_buffer_overflow_never_evicts_the_reading_in_flight() -> None:
    publisher, session = make_publisher(buffer_size=3)
    client = FakeClient(slow_calls=1)
    feed(session, range(1, 2))
    serve = asyncio.create_task(publisher._serve(client))  # type: ignore[arg-type]
    await asyncio.wait_for(client.in_publish.wait(), 1)  # seq 1 in flight, buffer empty
    feed(session, range(2, 8))  # 6 readings into a buffer of 3: the oldest 3 are dropped
    publisher.stop()
    await asyncio.wait_for(serve, 2)
    assert client.telemetry == [1, 5, 6, 7]
    assert publisher.dropped == 3
    assert publisher.published == 4


async def test_failed_publish_keeps_the_reading_for_the_next_connection() -> None:
    publisher, session = make_publisher()
    feed(session, range(1, 5))
    first = FakeClient(slow_calls=0, fail_seq=3)  # connection drops while sending seq 3
    with pytest.raises(aiomqtt.MqttError):
        await asyncio.wait_for(publisher._serve(first), 2)  # type: ignore[arg-type]
    assert first.telemetry == [1, 2]
    second = FakeClient(slow_calls=0)  # reconnected
    publisher.stop()
    await asyncio.wait_for(publisher._serve(second), 2)  # type: ignore[arg-type]
    assert second.telemetry == [3, 4]  # seq 3 resent first, order kept
    assert publisher.published == 4
