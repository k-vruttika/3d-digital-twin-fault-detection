"""StorageService: event store + writer + snapshotter, as one unit the SourceManager runs.

``build_storage`` picks the backend from ``config/storage.yaml``. The TimescaleDB module (and
with it asyncpg) is imported only when that backend is selected, so minimal mode (SQLite)
never loads a database driver beyond the standard library.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry

from twin_backend.bus.base import EventBus
from twin_backend.clock import Clock, SystemClock
from twin_backend.engine.line_state import LineState
from twin_backend.ingest.quarantine import QuarantineRecord, QuarantineSink
from twin_backend.storage.base import STATE_STREAMS, EventStore, Stored, Stream
from twin_backend.storage.metrics import StorageMetrics
from twin_backend.storage.replay import reconstruct
from twin_backend.storage.snapshots import Snapshotter
from twin_backend.storage.writer import StorageWriter
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings
from twin_common.config.storage import StorageConfig
from twin_common.contracts import Reading
from twin_common.enums import Quality
from twin_common.logging import get_logger

log = get_logger(__name__)


class StoreQuarantine:
    """Quarantine sink that keeps the existing sink and also persists every record."""

    def __init__(self, inner: QuarantineSink, writer: StorageWriter) -> None:
        self.inner = inner
        self.writer = writer
        self.total = 0

    async def put(self, record: QuarantineRecord) -> None:
        await self.inner.put(record)
        self.writer.offer_quarantine(record)
        self.total += 1


class StorageService:
    def __init__(
        self,
        store: EventStore,
        cfg: StorageConfig,
        bus: EventBus,
        machine_ids: Iterable[str],
        *,
        clock: Clock | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.machine_ids = list(machine_ids)
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.metrics = StorageMetrics(registry)
        self.lock = asyncio.Lock()
        self.writer = StorageWriter(
            store,
            bus,
            cfg.writer,
            clock=self.clock,
            metrics=self.metrics,
            on_committed=self._on_committed,
        )
        self.snapshotter = Snapshotter(
            store,
            self.machine_ids,
            cfg.snapshots,
            self.clock,
            lock=self.lock,
            metrics=self.metrics,
        )
        self._started = False

    def quarantine_sink(self, inner: QuarantineSink) -> StoreQuarantine:
        return StoreQuarantine(inner, self.writer)

    async def _on_committed(self, stream: Stream, rows: Sequence[Stored[Any]]) -> None:
        """Delete snapshots that an event committed just now should have been part of."""
        if stream not in STATE_STREAMS:
            return
        stamps = [
            r.item.ts
            for r in rows
            if not (isinstance(r.item, Reading) and r.item.quality is Quality.LATE)
        ]
        if not stamps:
            return
        oldest = min(stamps)
        async with self.lock:
            latest = await self.store.latest_snapshot()
            if latest is not None and oldest <= latest.ts:
                removed = await self.store.invalidate_snapshots_from(oldest)
                self.metrics.snapshot_invalidations.inc(removed)
                log.info("storage.snapshots_invalidated", since=oldest.isoformat(), n=removed)

    async def start(self) -> None:
        if self._started:
            return
        await self.store.connect()
        await self.store.migrate()
        self.writer.start()
        self.snapshotter.start()
        self._started = True
        log.info("storage.started", backend=self.store.backend)

    async def flush(self) -> None:
        await self.writer.flush()

    async def stop(self) -> None:
        if not self._started:
            return
        await self.snapshotter.stop()
        await self.writer.stop()
        await self.store.close()
        self._started = False
        log.info("storage.stopped", backend=self.store.backend)

    async def reconstruct(self, t: datetime) -> LineState:
        return await reconstruct(self.store, self.machine_ids, t)


def build_store(cfg: StorageConfig, *, root: Path, settings: TwinSettings | None = None) -> Any:
    """The configured EventStore (not yet connected)."""
    if cfg.backend == "sqlite":
        from twin_backend.storage.sqlite import SqliteEventStore

        path = Path(cfg.sqlite.path)
        return SqliteEventStore(
            path if path.is_absolute() else root / path,
            busy_timeout_s=cfg.sqlite.busy_timeout_s,
        )
    if cfg.timescale is None:  # enforced by StorageConfig; kept for the type checker
        raise ConfigError("storage backend 'timescale' needs a 'timescale:' section")
    from twin_backend.storage.timescale import TimescaleEventStore

    settings = settings if settings is not None else TwinSettings()
    return TimescaleEventStore(settings.pg_dsn, cfg.timescale)


def build_storage(
    cfg: StorageConfig,
    bus: EventBus,
    machine_ids: Iterable[str],
    *,
    root: Path,
    settings: TwinSettings | None = None,
    clock: Clock | None = None,
    registry: CollectorRegistry | None = None,
) -> StorageService:
    store: EventStore = build_store(cfg, root=root, settings=settings)
    return StorageService(store, cfg, bus, machine_ids, clock=clock, registry=registry)
