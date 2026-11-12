"""Event store interface (ADR 0009). Implementations: SQLite (minimal mode, tests) and
TimescaleDB (services). Both have the same semantics:

- **Append-only event streams** (:class:`Stream`). Every row keeps the event's own ``ts``
  (event time) and an ``ingested_at`` wall-clock stamp (metadata only: never used by replay
  or history queries).
- **Idempotent inserts:** re-inserting the same event (same natural key) is a no-op, so a
  retried batch or a replayed export never duplicates rows.
- **Time ranges:** ``fetch`` is half-open ``[t0, t1)``; ``fetch_replay`` is ``(after, until]``,
  which is what "state at t from the snapshot taken at s" needs.
- **History buckets** start in ``[t0, t1)`` and contain all readings of that bucket. Averages,
  minima and maxima ignore missing (null) sensor values; per-sensor counts say how many values
  each statistic is based on. LATE and SUSPECT readings are included and counted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from twin_backend.ingest.quarantine import QuarantineRecord
from twin_common.contracts import Alert, Command, CommandAck, Prediction, Reading, StateChange
from twin_common.enums import SensorName


class Stream(StrEnum):
    """One event table each."""

    READINGS = "readings"
    STATE_CHANGES = "state_changes"
    PREDICTIONS = "predictions"
    COMMANDS = "commands"
    COMMAND_ACKS = "command_acks"
    ALERTS = "alerts"
    QUARANTINE = "quarantine"


STREAM_TYPES: dict[Stream, type[BaseModel]] = {
    Stream.READINGS: Reading,
    Stream.STATE_CHANGES: StateChange,
    Stream.PREDICTIONS: Prediction,
    Stream.COMMANDS: Command,
    Stream.COMMAND_ACKS: CommandAck,
    Stream.ALERTS: Alert,
    Stream.QUARANTINE: QuarantineRecord,
}

# Streams folded into the line state by reconstruct (the others are history, not state).
STATE_STREAMS: tuple[Stream, ...] = (Stream.READINGS, Stream.STATE_CHANGES, Stream.PREDICTIONS)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Stored(Generic[T]):
    """An event plus the wall-clock time it was published on the bus."""

    item: T
    ingested_at: datetime


Resolution = Literal["1s", "1m", "10m"]
RESOLUTION_SECONDS: dict[str, int] = {"1s": 1, "1m": 60, "10m": 600}


@dataclass(frozen=True, slots=True)
class SensorStats:
    count: int
    mean: float | None
    min: float | None
    max: float | None


@dataclass(frozen=True, slots=True)
class HistoryBucket:
    bucket: datetime
    machine_id: str
    n: int
    n_late: int
    n_suspect: int
    sensors: dict[SensorName, SensorStats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Snapshot:
    ts: datetime
    state_json: str
    ingested_at: datetime


class EventStore(Protocol):
    backend: str

    async def connect(self) -> None: ...

    async def migrate(self) -> list[str]:
        """Apply pending migrations; returns the versions applied now."""
        ...

    async def close(self) -> None: ...

    async def insert(self, stream: Stream, rows: Sequence[Stored[Any]]) -> int:
        """Insert a batch in one transaction; returns rows actually added (duplicates skipped)."""
        ...

    async def fetch(
        self,
        stream: Stream,
        t0: datetime | None = None,
        t1: datetime | None = None,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[Stored[Any]]:
        """Rows with ``t0 <= ts < t1`` in (ts, machine, key) order."""
        ...

    async def fetch_replay(
        self, streams: Sequence[Stream], after: datetime | None, until: datetime
    ) -> list[Stored[Any]]:
        """Rows with ``after < ts <= until`` from several streams, ordered by ts."""
        ...

    async def count(self, stream: Stream) -> int: ...

    async def min_max_ts(self, streams: Sequence[Stream]) -> tuple[datetime, datetime] | None:
        """Oldest and newest event ts over ``streams`` (None when empty)."""
        ...

    async def history(
        self,
        t0: datetime,
        t1: datetime,
        resolution: Resolution,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[HistoryBucket]: ...

    async def save_snapshot(self, snapshot: Snapshot) -> None: ...

    async def latest_snapshot(self, at_or_before: datetime | None = None) -> Snapshot | None: ...

    async def snapshot_times(self) -> list[datetime]: ...

    async def invalidate_snapshots_from(self, ts: datetime) -> int:
        """Delete snapshots with ``ts >= ts`` (an event at or before them arrived later)."""
        ...

    async def delete_all(self) -> None:
        """Empty every table (tests and benchmarks only)."""
        ...


# ---------------------------------------------------------------- shared helpers
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def to_us(ts: datetime) -> int:
    """Exact microseconds since the epoch (UTC)."""
    delta = ts - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def from_us(us: int) -> datetime:
    return EPOCH + timedelta(microseconds=us)


def bucket_bounds(t0: datetime, t1: datetime, width_s: int) -> tuple[datetime, datetime]:
    """Event-time range covering exactly the buckets whose start lies in ``[t0, t1)``."""
    w = width_s * 1_000_000

    def ceil(ts: datetime) -> int:
        return math.ceil(to_us(ts) / w) * w

    return from_us(ceil(t0)), from_us(ceil(t1))


def sort_key(stored: Stored[Any]) -> tuple[datetime, int, str]:
    """Deterministic replay order across streams: ts, then stream rank, then a stable key."""
    item = stored.item
    rank = _REPLAY_RANK.get(type(item), len(_REPLAY_RANK))
    return (item.ts, rank, getattr(item, "machine_id", None) or "")


_REPLAY_RANK: dict[type[BaseModel], int] = {Reading: 0, StateChange: 1, Prediction: 2}
