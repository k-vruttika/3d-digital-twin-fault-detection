"""Ingest pipeline: RawMessage -> normalize -> validate -> dedup -> reorder -> event bus.

Rejected messages go to the quarantine sink with a reason. ``tick()`` (called every
``ingest.tick_s`` by the source manager, or by tests after advancing a fake clock) releases
messages whose lateness window expired and emits STALE/OFFLINE state changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from twin_backend.adapters.base import RawMessage
from twin_backend.bus.base import EventBus, Topic
from twin_backend.clock import Clock
from twin_backend.ingest.dedup import Deduplicator, DedupResult
from twin_backend.ingest.errors import RejectedMessageError
from twin_backend.ingest.metrics import IngestMetrics
from twin_backend.ingest.normalizer import Normalizer
from twin_backend.ingest.quarantine import QuarantineRecord, QuarantineSink, payload_for_record
from twin_backend.ingest.reorder import Released, ReorderBuffer
from twin_backend.ingest.staleness import StalenessMonitor
from twin_backend.ingest.validator import Validator
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping, IngestConfig
from twin_common.contracts import Reading
from twin_common.enums import AdapterKind, Quality
from twin_common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Pending:
    reading: Reading
    source: AdapterKind
    received_mono: float
    sent_at: datetime | None


class IngestPipeline:
    def __init__(
        self,
        line: LineConfig,
        cfg: IngestConfig,
        bus: EventBus,
        quarantine: QuarantineSink,
        clock: Clock,
        mappings: Mapping[AdapterKind, FieldMapping],
        metrics: IngestMetrics | None = None,
    ) -> None:
        self.line = line
        self.cfg = cfg
        self.bus = bus
        self.quarantine = quarantine
        self.clock = clock
        self.metrics = metrics if metrics is not None else IngestMetrics()
        self._normalizers = {kind: Normalizer(m, line) for kind, m in mappings.items()}
        self._default_normalizer = Normalizer(FieldMapping(), line)
        self.validator = Validator(line, cfg.out_of_range)
        self.dedup = Deduplicator(cfg.dedup_window, cfg.seq_reset_threshold)
        self.reorder: ReorderBuffer[_Pending] = ReorderBuffer(cfg.lateness_window_s)
        self.staleness = StalenessMonitor(
            [m.id for m in line.machines], cfg.stale_after_s, cfg.offline_after_s, clock
        )

    def set_mapping(self, kind: AdapterKind, mapping: FieldMapping) -> None:
        self._normalizers[kind] = Normalizer(mapping, self.line)

    async def process(self, raw: RawMessage) -> None:
        src = raw.source.value
        self.metrics.messages_in.labels(source=src).inc()
        normalizer = self._normalizers.get(raw.source, self._default_normalizer)
        try:
            data = normalizer.normalize(raw)
            reading = self.validator.validate(data)
        except RejectedMessageError as rej:
            await self._quarantine(raw, rej)
            return
        if reading.quality is Quality.SUSPECT:
            self.metrics.suspect.labels(source=src).inc()

        result = self.dedup.check(reading.machine_id, reading.seq)
        if result is DedupResult.DUPLICATE:
            self.metrics.duplicates.labels(source=src).inc()
            return
        if result is DedupResult.RESET:
            self.metrics.source_resets.labels(source=src).inc()
            log.warning("ingest.source_reset", machine=reading.machine_id, seq=reading.seq)
            for released in self.reorder.reset(reading.machine_id):
                await self._publish(reading.machine_id, released)

        if not self.reorder.is_late(reading.machine_id, reading.seq):
            change = self.staleness.seen(reading.machine_id, reading.state)
            if change is not None:
                await self.bus.publish(Topic.STATES, change)

        pending = _Pending(reading, raw.source, raw.received_mono, raw.sent_at)
        now = self.clock.monotonic()
        for released in self.reorder.push(reading.machine_id, reading.seq, pending, now):
            await self._publish(reading.machine_id, released)

    async def tick(self) -> None:
        for machine_id, released in self.reorder.tick(self.clock.monotonic()):
            await self._publish(machine_id, released)
        for change in self.staleness.tick():
            await self.bus.publish(Topic.STATES, change)

    async def flush(self) -> None:
        """Release everything held (end of a finite stream)."""
        for machine_id, released in self.reorder.flush():
            await self._publish(machine_id, released)

    async def _publish(self, machine_id: str, released: Released[_Pending]) -> None:
        pending = released.item
        src = pending.source.value
        reading = pending.reading
        if released.late:
            reading = reading.model_copy(update={"quality": Quality.LATE})
            self.metrics.late.labels(source=src).inc()
        if released.reordered:
            self.metrics.reordered.labels(source=src).inc()
        if released.gap_before:
            self.metrics.gaps.labels(source=src).inc(released.gap_before)
            log.info(
                "ingest.gap", machine=machine_id, missing=released.gap_before, seq=released.seq
            )
        await self.bus.publish(Topic.READINGS, reading)
        self.metrics.published.labels(source=src).inc()
        if pending.sent_at is not None:
            latency = (self.clock.now() - pending.sent_at).total_seconds()
            basis = "source"
        else:
            latency = self.clock.monotonic() - pending.received_mono
            basis = "receive"
        self.metrics.latency.labels(source=src, basis=basis).observe(max(0.0, latency))

    async def _quarantine(self, raw: RawMessage, rej: RejectedMessageError) -> None:
        self.metrics.quarantined.labels(source=raw.source.value, reason=rej.reason.value).inc()
        record = QuarantineRecord(
            ts=self.clock.now(),
            source=raw.source,
            reason=rej.reason,
            stage=rej.stage,
            detail=rej.detail,
            machine_id=rej.machine_id,
            payload=payload_for_record(raw.payload),
            meta=dict(raw.meta),
        )
        await self.quarantine.put(record)
