"""Event bus interface. The in-memory implementation can later be swapped for Redis (ADR 0006).

Messages are Pydantic contract models (Reading, StateChange, Prediction, ...). Publishing never
blocks on slow subscribers: each subscriber has a bounded queue and loses its oldest messages
when it falls behind (``Subscription.dropped`` counts them).

Every message is stamped with the wall-clock time it was published (``get_stamped``); the
event store keeps it as ``ingested_at`` (metadata for latency measurement and debugging).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class Topic(StrEnum):
    READINGS = "readings"
    STATES = "states"
    PREDICTIONS = "predictions"
    COMMANDS = "commands"
    ALERTS = "alerts"
    # Quarantined messages (QuarantineRecord), for the engine's data-quality status. The store
    # gets them through its quarantine sink, not from this topic.
    QUARANTINE = "quarantine"


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

    async def get_stamped(self) -> tuple[datetime, BaseModel]:
        """Like ``get`` but also returns the wall-clock time the message was published."""
        ...

    def get_stamped_nowait(self) -> tuple[datetime, BaseModel] | None: ...

    def qsize(self) -> int:
        """Messages currently queued."""
        ...

    def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[BaseModel]: ...


class EventBus(Protocol):
    async def publish(self, topic: Topic, message: BaseModel) -> None: ...

    def subscribe(self, topic: Topic, *, maxsize: int | None = None) -> Subscription:
        """New subscription; ``maxsize`` overrides the bus default queue bound."""
        ...

    async def close(self) -> None:
        """Close every subscription (their iterators end)."""
        ...
