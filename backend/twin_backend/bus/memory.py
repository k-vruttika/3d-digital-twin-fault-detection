"""In-process event bus: one bounded asyncio queue per subscriber, drop-oldest on overflow."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator

from pydantic import BaseModel

from twin_backend.bus.base import Topic


class _MemorySubscription:
    def __init__(self, bus: InMemoryEventBus, topic: Topic, maxsize: int) -> None:
        self.topic = topic
        self._bus = bus
        self._items: deque[BaseModel] = deque()
        self._maxsize = maxsize
        self._event = asyncio.Event()
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def closed(self) -> bool:
        return self._closed

    def _offer(self, message: BaseModel) -> None:
        if self._closed:
            return
        if len(self._items) >= self._maxsize:
            self._items.popleft()
            self._dropped += 1
        self._items.append(message)
        self._event.set()

    async def get(self) -> BaseModel:
        while not self._items:
            if self._closed:
                raise StopAsyncIteration
            self._event.clear()
            await self._event.wait()
        return self._items.popleft()

    def get_nowait(self) -> BaseModel | None:
        return self._items.popleft() if self._items else None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._event.set()
            self._bus._remove(self)

    def __aiter__(self) -> AsyncIterator[BaseModel]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[BaseModel]:
        while True:
            try:
                yield await self.get()
            except StopAsyncIteration:
                return


class InMemoryEventBus:
    """Fan-out to every subscriber of a topic; never blocks the publisher."""

    def __init__(self, queue_size: int) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.queue_size = queue_size
        self._subs: dict[Topic, list[_MemorySubscription]] = defaultdict(list)
        self.published: dict[Topic, int] = defaultdict(int)

    async def publish(self, topic: Topic, message: BaseModel) -> None:
        self.published[topic] += 1
        for sub in list(self._subs[topic]):
            sub._offer(message)

    def subscribe(self, topic: Topic) -> _MemorySubscription:
        sub = _MemorySubscription(self, topic, self.queue_size)
        self._subs[topic].append(sub)
        return sub

    def subscriber_count(self, topic: Topic) -> int:
        return len(self._subs[topic])

    def _remove(self, sub: _MemorySubscription) -> None:
        subs = self._subs[sub.topic]
        if sub in subs:
            subs.remove(sub)

    async def close(self) -> None:
        for subs in list(self._subs.values()):
            for sub in list(subs):
                sub.close()
