"""Periodic full line snapshots for fast reconstruction (ADR 0010).

Snapshots sit at event-time multiples of ``interval_s`` (aligned to the epoch, so the same data
always gives the same snapshot times). Each one is computed with ``reconstruct()`` from the
previous snapshot plus the events since, so a snapshot always equals what replay would give.

Only event time older than the store watermark minus ``settle_s`` is snapshotted, so data still
held by the reorder buffer or the writer is normally in. Data that arrives even later (e.g. an
MQTT backlog after an outage) invalidates the affected snapshots: the storage service deletes
every snapshot at or after the oldest such event, and they are rebuilt on the next run. The
shared ``lock`` keeps a snapshot computed before such an invalidation from being saved after it.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Iterable
from datetime import datetime, timedelta

from twin_backend.clock import Clock
from twin_backend.storage.base import (
    STATE_STREAMS,
    EventStore,
    Snapshot,
    from_us,
    to_us,
)
from twin_backend.storage.metrics import StorageMetrics
from twin_backend.storage.replay import reconstruct
from twin_common.config.storage import SnapshotConfig
from twin_common.logging import get_logger

log = get_logger(__name__)


def next_boundary(ts: datetime, interval_s: float, *, strictly_after: bool) -> datetime:
    w = round(interval_s * 1_000_000)
    us = to_us(ts)
    k = us // w + 1 if strictly_after else math.ceil(us / w)
    return from_us(k * w)


class Snapshotter:
    def __init__(
        self,
        store: EventStore,
        machine_ids: Iterable[str],
        cfg: SnapshotConfig,
        clock: Clock,
        *,
        lock: asyncio.Lock,
        metrics: StorageMetrics,
    ) -> None:
        self.store = store
        self.machine_ids = list(machine_ids)
        self.cfg = cfg
        self.clock = clock
        self.lock = lock
        self.metrics = metrics
        self._task: asyncio.Task[None] | None = None

    async def run_once(self, *, settle_s: float | None = None) -> int:
        """Take every snapshot that is due now. Returns how many were saved."""
        span = await self.store.min_max_ts(STATE_STREAMS)
        if span is None:
            return 0
        oldest, newest = span
        settle = self.cfg.settle_s if settle_s is None else settle_s
        limit = newest - timedelta(seconds=settle)
        saved = 0
        while True:
            async with self.lock:
                last = await self.store.latest_snapshot()
                if last is not None:
                    due = next_boundary(last.ts, self.cfg.interval_s, strictly_after=True)
                else:
                    due = next_boundary(oldest, self.cfg.interval_s, strictly_after=False)
                if due > limit:
                    return saved
                state = await reconstruct(self.store, self.machine_ids, due)
                await self.store.save_snapshot(
                    Snapshot(due, state.model_dump_json(), self.clock.now())
                )
            saved += 1
            self.metrics.snapshots.inc()

    async def _loop(self) -> None:
        while True:
            await self.clock.sleep(self.cfg.check_every_s)
            try:
                await self.run_once()
            except Exception as exc:
                log.error("storage.snapshot_failed", error=f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        if self.cfg.enabled and self._task is None:
            self._task = asyncio.create_task(self._loop(), name="storage-snapshots")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
