"""Event bus interface. The in-memory implementation can later be swapped for Redis (ADR 0006).

Messages are Pydantic contract models (Reading, StateChange, Prediction, ...). Publishing never
blocks on slow subscribers: each subscriber has a bounded queue and loses its oldest messages
when it falls behind (``Subscription.dropped`` counts them).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class Topic(StrEnum):
    READINGS = "readings"
    STATES = "states"
    PREDICTIONS = "predictions"
    COMMANDS = "commands"
    ALERTS = "alerts"


class Subscription(Protocol):
    topic: Topic

    @property
    def dropped(self) -> int:
        """Messages lost because this subscriber's queue was full."""
        ...

    async def get(self) -> BaseModel:
        """Next message; raises ``StopAsyncIteration`` once the subscription is closed."""
        ...

    def get_nowait(self) -> BaseModel | None:
        """Next message if one is queued, else None."""
        ...

    def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[BaseModel]: ...


class EventBus(Protocol):
    async def publish(self, topic: Topic, message: BaseModel) -> None: ...

    def subscribe(self, topic: Topic) -> Subscription: ...

    async def close(self) -> None:
        """Close every subscription (their iterators end)."""
        ...
