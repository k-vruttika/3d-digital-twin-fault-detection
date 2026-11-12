"""In-memory event bus: fan-out, topic isolation, drop-oldest, close."""

from __future__ import annotations

import asyncio

import pytest

from twin_backend.bus import EventBus, InMemoryEventBus, Topic
from twin_common.testing import sample_reading


async def test_fan_out_to_every_subscriber() -> None:
    bus = InMemoryEventBus(queue_size=10)
    a, b = bus.subscribe(Topic.READINGS), bus.subscribe(Topic.READINGS)
    r = sample_reading()
    await bus.publish(Topic.READINGS, r)
    assert await a.get() == r
    assert await b.get() == r


async def test_topics_are_isolated() -> None:
    bus = InMemoryEventBus(queue_size=10)
    states = bus.subscribe(Topic.STATES)
    await bus.publish(Topic.READINGS, sample_reading())
    assert states.get_nowait() is None
    assert bus.published[Topic.READINGS] == 1


async def test_slow_subscriber_drops_oldest_and_never_blocks() -> None:
    bus = InMemoryEventBus(queue_size=3)
    sub = bus.subscribe(Topic.READINGS)
    for seq in range(5):
        await bus.publish(Topic.READINGS, sample_reading(seq=seq))
    assert sub.dropped == 2
    got = [sub.get_nowait() for _ in range(3)]
    assert [r.seq for r in got if r is not None] == [2, 3, 4]  # type: ignore[attr-defined]


async def test_iteration_ends_on_close() -> None:
    bus: EventBus = InMemoryEventBus(queue_size=10)
    sub = bus.subscribe(Topic.READINGS)
    received: list[int] = []

    async def consume() -> None:
        async for msg in sub:
            received.append(msg.seq)  # type: ignore[attr-defined]

    task = asyncio.create_task(consume())
    await bus.publish(Topic.READINGS, sample_reading(seq=1))
    await asyncio.sleep(0)
    await bus.close()
    await asyncio.wait_for(task, timeout=1)
    assert received == [1]


async def test_closed_subscription_is_removed() -> None:
    bus = InMemoryEventBus(queue_size=10)
    sub = bus.subscribe(Topic.ALERTS)
    assert bus.subscriber_count(Topic.ALERTS) == 1
    sub.close()
    assert bus.subscriber_count(Topic.ALERTS) == 0
    with pytest.raises(StopAsyncIteration):
        await sub.get()


def test_queue_size_validated() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        InMemoryEventBus(queue_size=0)


async def test_per_subscription_maxsize_overrides_default() -> None:
    bus = InMemoryEventBus(queue_size=100)
    small = bus.subscribe(Topic.READINGS, maxsize=2)
    big = bus.subscribe(Topic.READINGS)
    for seq in range(5):
        await bus.publish(Topic.READINGS, sample_reading(seq=seq))
    assert (small.maxsize, small.dropped, small.qsize()) == (2, 3, 2)
    assert (big.dropped, big.qsize()) == (0, 5)
    with pytest.raises(ValueError, match="maxsize"):
        bus.subscribe(Topic.READINGS, maxsize=0)


async def test_messages_are_stamped_at_publish() -> None:
    from twin_backend.clock import FakeClock

    clock = FakeClock()
    bus = InMemoryEventBus(queue_size=10, clock=clock)
    sub = bus.subscribe(Topic.READINGS)
    published_at = clock.now()
    await bus.publish(Topic.READINGS, sample_reading())
    clock.advance(5)  # a slow consumer does not change the stamp
    stamp, message = await sub.get_stamped()
    assert stamp == published_at
    assert message == sample_reading()
    await bus.publish(Topic.READINGS, sample_reading(seq=1))
    got = sub.get_stamped_nowait()
    assert got is not None
    assert got[0] == clock.now()
    assert sub.get_stamped_nowait() is None
