"""StorageWriter: batching, routing, retries, and backpressure that never blocks ingestion."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from twin_backend.adapters.base import RawMessage
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock, SystemClock
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import MemoryQuarantine, QuarantineRecord
from twin_backend.storage.base import Stored, Stream
from twin_backend.storage.metrics import StorageMetrics
from twin_backend.storage.writer import StorageWriter
from twin_backend.testing import REPO_ROOT, storage_config
from twin_common.config import load_line_config
from twin_common.config.sources import FieldMapping, SourcesConfig
from twin_common.config.storage import WriterConfig, WriterRetryConfig
from twin_common.enums import AdapterKind, QuarantineReason
from twin_common.testing import SAMPLE_TS, sample_ack, sample_command, sample_reading


class FakeStore:
    """Records batches; can block until released and fail a number of attempts."""

    backend = "fake"

    def __init__(self, *, fail: int = 0, blocked: bool = False) -> None:
        self.batches: list[tuple[Stream, list[Stored[Any]]]] = []
        self.fail = fail
        self.attempts = 0
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.waiting = asyncio.Event()

    async def insert(self, stream: Stream, rows: Sequence[Stored[Any]]) -> int:
        self.attempts += 1
        self.waiting.set()
        await self.release.wait()
        if self.fail > 0:
            self.fail -= 1
            raise OSError("disk on fire")
        self.batches.append((stream, list(rows)))
        return len(rows)

    def rows(self, stream: Stream) -> list[Stored[Any]]:
        return [r for s, batch in self.batches if s is stream for r in batch]


def writer_cfg(**kw: Any) -> WriterConfig:
    base: WriterConfig = storage_config().writer
    retry = kw.pop("retry", WriterRetryConfig(attempts=3, backoff_s=0.0))
    cfg: WriterConfig = base.model_copy(update={"retry": retry, **kw})
    return cfg


def make_writer(
    store: Any, bus: InMemoryEventBus, **kw: Any
) -> tuple[StorageWriter, StorageMetrics]:
    metrics = StorageMetrics()
    clock = kw.pop("clock", FakeClock(auto_advance=True))
    writer = StorageWriter(store, bus, writer_cfg(**kw), clock=clock, metrics=metrics)
    return writer, metrics


async def test_batches_and_ingested_at_from_publish_time() -> None:
    clock = FakeClock()
    bus = InMemoryEventBus(1000, clock=clock)
    store = FakeStore()
    writer, metrics = make_writer(store, bus, batch_size=3)
    writer.subscribe()
    stamps = []
    for seq in range(10):
        stamps.append(clock.now())
        await bus.publish(Topic.READINGS, sample_reading(seq=seq))
        clock.advance(1)
    assert await writer.flush() == 10
    assert [len(b) for _, b in store.batches] == [3, 3, 3, 1]
    assert [r.ingested_at for r in store.rows(Stream.READINGS)] == stamps
    assert metrics.value("written", stream="readings") == 10
    assert await writer.flush() == 0


async def test_commands_and_acks_go_to_their_own_streams() -> None:
    bus = InMemoryEventBus(1000)
    store = FakeStore()
    writer, _ = make_writer(store, bus)
    writer.subscribe()
    await bus.publish(Topic.COMMANDS, sample_command())
    await bus.publish(Topic.COMMANDS, sample_ack())
    await writer.flush()
    assert [r.item for r in store.rows(Stream.COMMANDS)] == [sample_command()]
    assert [r.item for r in store.rows(Stream.COMMAND_ACKS)] == [sample_ack()]


async def test_failed_batch_is_retried_then_written() -> None:
    bus = InMemoryEventBus(1000)
    store = FakeStore(fail=2)
    writer, metrics = make_writer(store, bus)
    writer.subscribe()
    await bus.publish(Topic.READINGS, sample_reading())
    await writer.flush()
    assert store.attempts == 3
    assert len(store.rows(Stream.READINGS)) == 1
    assert metrics.value("write_errors", stream="readings") == 2
    assert metrics.dropped_total() == 0


async def test_batch_dropped_after_all_retries_and_counted() -> None:
    bus = InMemoryEventBus(1000)
    store = FakeStore(fail=99)
    writer, metrics = make_writer(store, bus, batch_size=4)
    writer.subscribe()
    for seq in range(6):
        await bus.publish(Topic.READINGS, sample_reading(seq=seq))
    await writer.flush()
    assert store.rows(Stream.READINGS) == []
    assert metrics.value("dropped_writes", stream="readings", reason="write_error") == 6
    assert metrics.value("write_errors", stream="readings") == 6  # 2 batches x 3 attempts


async def test_quarantine_records_are_queued_bounded() -> None:
    bus = InMemoryEventBus(1000)
    store = FakeStore()
    writer, metrics = make_writer(store, bus, queue_size=2)
    writer.subscribe()
    record = QuarantineRecord(
        ts=SAMPLE_TS,
        source=AdapterKind.FILE,
        reason=QuarantineReason.SCHEMA_ERROR,
        stage="validate",
        detail="bad",
        machine_id=None,
        payload="x",
        meta={},
    )
    for i in range(3):
        writer.offer_quarantine(record.model_copy(update={"detail": f"bad {i}"}))
    await writer.flush()
    assert [r.item.detail for r in store.rows(Stream.QUARANTINE)] == ["bad 1", "bad 2"]
    assert metrics.value("dropped_writes", stream="quarantine", reason="queue_full") == 1


def _pipeline(bus: InMemoryEventBus, clock: FakeClock, sources: SourcesConfig) -> IngestPipeline:
    line = load_line_config(REPO_ROOT / "config" / "line.yaml")
    return IngestPipeline(
        line,
        sources.ingest,
        bus,
        MemoryQuarantine(100),
        clock,
        {AdapterKind.FILE: FieldMapping()},
    )


def _raw(seq: int, clock: FakeClock) -> RawMessage:
    reading = sample_reading(seq=seq, ts=SAMPLE_TS + timedelta(seconds=seq))
    return RawMessage(
        source=AdapterKind.FILE,
        payload=reading.model_dump(mode="json"),
        received_mono=clock.monotonic(),
        received_at=clock.now(),
    )


async def test_slow_store_never_blocks_ingestion_and_drops_are_counted(
    sources_cfg: SourcesConfig,
) -> None:
    """The store is blocked for the whole ingest run; ingestion still completes at once."""
    n, queue = 2000, 50
    clock = FakeClock()
    bus = InMemoryEventBus(1_000_000, clock=clock)
    store = FakeStore(blocked=True)
    writer, metrics = make_writer(
        store, bus, queue_size=queue, batch_size=10, flush_interval_s=0.001, clock=SystemClock()
    )
    writer.start()
    pipeline = _pipeline(bus, clock, sources_cfg)
    readings_out = bus.subscribe(Topic.READINGS, maxsize=n)

    await pipeline.process(_raw(0, clock))  # first message: held for the lateness window
    clock.advance(sources_cfg.ingest.lateness_window_s)
    await pipeline.tick()
    await asyncio.wait_for(store.waiting.wait(), 5)  # the writer is now stuck in insert()

    async def ingest_rest() -> None:
        for seq in range(1, n):
            await pipeline.process(_raw(seq, clock))

    # Must finish without the store ever making progress.
    await asyncio.wait_for(ingest_rest(), 10)
    assert pipeline.metrics.value("published", source="file") == n
    assert readings_out.qsize() == n  # other subscribers got everything
    assert len(store.batches) == 0

    store.release.set()
    await writer.stop()
    written = len(store.rows(Stream.READINGS))
    dropped = metrics.value("dropped_writes", stream="readings", reason="queue_full")
    assert dropped > 0
    assert written + dropped == n
    assert written <= queue + 10  # one batch in flight + one full queue
    # What survived is the newest data (drop-oldest), in order.
    seqs = [r.item.seq for r in store.rows(Stream.READINGS)]
    assert seqs == sorted(seqs)
    assert seqs[-1] == n - 1


async def test_stop_drains_everything_queued() -> None:
    bus = InMemoryEventBus(1000)
    store = FakeStore()
    writer, _ = make_writer(store, bus, flush_interval_s=3600, clock=SystemClock())
    writer.start()
    for seq in range(25):
        await bus.publish(Topic.READINGS, sample_reading(seq=seq))
    await writer.stop()
    assert len(store.rows(Stream.READINGS)) == 25
    assert bus.subscriber_count(Topic.READINGS) == 0
