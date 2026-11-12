"""StorageWriter: event bus -> batched, retried inserts into the event store.

- **Never blocks ingestion.** The writer reads from its own bus subscriptions, bounded at
  ``writer.queue_size``. The bus never waits for a subscriber: when the writer falls behind,
  the oldest queued rows are dropped and counted in
  ``linetwin_storage_dropped_writes_total{stream, reason="queue_full"}``. Quarantine records
  (not on the bus) go into a bounded queue with the same policy.
- **Batches:** every ``flush_interval_s`` everything queued is written in chunks of at most
  ``batch_size`` rows, one transaction per chunk.
- **Retries:** a failed chunk is retried ``retry.attempts`` times with doubling backoff, then
  dropped and counted (``reason="write_error"``).
- ``ingested_at`` is the time the message was published on the bus (not when it was written).
- ``stop()`` writes everything still queued.

The ``commands`` topic carries both Command and CommandAck; they go to their own tables, and
queue drops on that topic are counted under ``stream="commands"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from twin_backend.bus.base import EventBus, Subscription, Topic
from twin_backend.clock import Clock
from twin_backend.ingest.quarantine import QuarantineRecord
from twin_backend.storage.base import EventStore, Stored, Stream
from twin_backend.storage.metrics import StorageMetrics
from twin_common.config.storage import WriterConfig
from twin_common.contracts import Alert, Command, CommandAck, Prediction, Reading, StateChange
from twin_common.logging import get_logger

log = get_logger(__name__)

# Bus topic -> stream label used for queue-drop metrics.
TOPIC_LABEL: dict[Topic, Stream] = {
    Topic.READINGS: Stream.READINGS,
    Topic.STATES: Stream.STATE_CHANGES,
    Topic.PREDICTIONS: Stream.PREDICTIONS,
    Topic.COMMANDS: Stream.COMMANDS,
    Topic.ALERTS: Stream.ALERTS,
}

_TYPE_STREAM: dict[type[BaseModel], Stream] = {
    Reading: Stream.READINGS,
    StateChange: Stream.STATE_CHANGES,
    Prediction: Stream.PREDICTIONS,
    Command: Stream.COMMANDS,
    CommandAck: Stream.COMMAND_ACKS,
    Alert: Stream.ALERTS,
}

OnCommitted = Callable[[Stream, Sequence[Stored[Any]]], Awaitable[None]]


class StorageWriter:
    def __init__(
        self,
        store: EventStore,
        bus: EventBus,
        cfg: WriterConfig,
        *,
        clock: Clock,
        metrics: StorageMetrics,
        on_committed: OnCommitted | None = None,
    ) -> None:
        self.store = store
        self.bus = bus
        self.cfg = cfg
        self.clock = clock
        self.metrics = metrics
        self.on_committed = on_committed
        self._subs: dict[Topic, Subscription] = {}
        self._seen_drops: dict[Topic, int] = {}
        self._quarantine: deque[Stored[QuarantineRecord]] = deque()
        self._task: asyncio.Task[None] | None = None
        self._flush_lock = asyncio.Lock()

    # -- input -------------------------------------------------------------------------------
    def subscribe(self) -> None:
        """Subscribe to the bus. Called by ``start()``; separate so tests can drive ``flush()``."""
        if not self._subs:
            for topic in TOPIC_LABEL:
                self._subs[topic] = self.bus.subscribe(topic, maxsize=self.cfg.queue_size)
                self._seen_drops[topic] = 0

    def offer_quarantine(self, record: QuarantineRecord) -> None:
        """Queue a quarantine record; never waits (drops the oldest when full)."""
        if len(self._quarantine) >= self.cfg.queue_size:
            self._quarantine.popleft()
            self.metrics.dropped.labels(stream=Stream.QUARANTINE.value, reason="queue_full").inc()
        self._quarantine.append(Stored(record, record.ts))

    def queued(self) -> int:
        return sum(s.qsize() for s in self._subs.values()) + len(self._quarantine)

    def _sync_drop_metrics(self) -> None:
        self.metrics.queue_depth.labels(stream=Stream.QUARANTINE.value).set(len(self._quarantine))
        for topic, sub in self._subs.items():
            self.metrics.queue_depth.labels(stream=TOPIC_LABEL[topic].value).set(sub.qsize())
            delta = sub.dropped - self._seen_drops[topic]
            if delta:
                self._seen_drops[topic] = sub.dropped
                self.metrics.dropped.labels(
                    stream=TOPIC_LABEL[topic].value, reason="queue_full"
                ).inc(delta)

    def _drain(self) -> dict[Stream, list[Stored[Any]]]:
        out: dict[Stream, list[Stored[Any]]] = {}
        for topic, sub in self._subs.items():
            while (got := sub.get_stamped_nowait()) is not None:
                stamp, message = got
                stream = _TYPE_STREAM.get(type(message))
                if stream is None:
                    log.warning("storage.unknown_message", topic=topic.value, type=type(message))
                    continue
                out.setdefault(stream, []).append(Stored(message, stamp))
        if self._quarantine:
            out[Stream.QUARANTINE] = list(self._quarantine)
            self._quarantine.clear()
        return out

    # -- output ------------------------------------------------------------------------------
    async def _write(self, stream: Stream, chunk: Sequence[Stored[Any]]) -> None:
        delay = self.cfg.retry.backoff_s
        for attempt in range(1, self.cfg.retry.attempts + 1):
            started = time.perf_counter()
            try:
                added = await self.store.insert(stream, chunk)
            except Exception as exc:
                self.metrics.write_errors.labels(stream=stream.value).inc()
                error = f"{type(exc).__name__}: {exc}"
                if attempt == self.cfg.retry.attempts:
                    self.metrics.dropped.labels(stream=stream.value, reason="write_error").inc(
                        len(chunk)
                    )
                    log.error(
                        "storage.batch_dropped", stream=stream.value, rows=len(chunk), error=error
                    )
                    return
                log.warning(
                    "storage.batch_retry", stream=stream.value, attempt=attempt, error=error
                )
                await self.clock.sleep(delay)
                delay *= 2
                continue
            self.metrics.batch_seconds.labels(stream=stream.value).observe(
                time.perf_counter() - started
            )
            self.metrics.written.labels(stream=stream.value).inc(added)
            if added < len(chunk):
                self.metrics.duplicates.labels(stream=stream.value).inc(len(chunk) - added)
            if self.on_committed is not None:
                await self.on_committed(stream, chunk)
            return

    async def flush(self) -> int:
        """Write everything queued right now. Returns the number of rows handed to the store."""
        async with self._flush_lock:
            self._sync_drop_metrics()
            batches = self._drain()
            size = self.cfg.batch_size
            total = 0
            for stream, rows in batches.items():
                for i in range(0, len(rows), size):
                    chunk = rows[i : i + size]
                    await self._write(stream, chunk)
                    total += len(chunk)
            return total

    # -- lifecycle ---------------------------------------------------------------------------
    async def _loop(self) -> None:
        while True:
            await self.clock.sleep(self.cfg.flush_interval_s)
            try:
                await self.flush()
            except Exception as exc:  # never let the writer die silently
                log.error("storage.flush_failed", error=f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        self.subscribe()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="storage-writer")

    async def stop(self) -> None:
        """Stop the loop (never mid-batch), then write everything still queued."""
        async with self._flush_lock:
            task, self._task = self._task, None
            if task is not None:
                task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for sub in self._subs.values():
            sub.close()
        while await self.flush():
            pass
