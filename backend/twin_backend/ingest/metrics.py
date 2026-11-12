"""Ingestion metrics (Prometheus). Each pipeline owns a registry, so tests stay isolated; the
API (Phase 4) serves it at ``/metrics``.

End-to-end latency is measured up to the bus publish, from the source's send time when the
transport carries one (``basis="source"``), otherwise from when the adapter received the
message (``basis="receive"``).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

from twin_common.enums import AdapterKind, QuarantineReason

PREFIX = "linetwin_ingest"


class IngestMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.messages_in = Counter(
            f"{PREFIX}_messages_in", "Raw messages received", ["source"], registry=r
        )
        self.published = Counter(
            f"{PREFIX}_published", "Readings published on the bus", ["source"], registry=r
        )
        self.quarantined = Counter(
            f"{PREFIX}_quarantined", "Messages quarantined", ["source", "reason"], registry=r
        )
        self.duplicates = Counter(
            f"{PREFIX}_duplicates", "Duplicate (machineId, seq) dropped", ["source"], registry=r
        )
        self.late = Counter(
            f"{PREFIX}_late", "Readings beyond the lateness window", ["source"], registry=r
        )
        self.reordered = Counter(
            f"{PREFIX}_reordered", "Readings put back in seq order", ["source"], registry=r
        )
        self.gaps = Counter(
            f"{PREFIX}_gap_seqs", "Seqs given up after the lateness window", ["source"], registry=r
        )
        self.suspect = Counter(
            f"{PREFIX}_suspect",
            "Readings flagged SUSPECT (out of range or partially missing)",
            ["source"],
            registry=r,
        )
        self.partial = Counter(
            f"{PREFIX}_partial",
            "Readings kept with some sensor values missing (quality SUSPECT)",
            ["source"],
            registry=r,
        )
        self.missing_sensor = Counter(
            f"{PREFIX}_missing_sensor",
            "Missing sensor values in kept readings, per sensor",
            ["source", "sensor"],
            registry=r,
        )
        self.source_resets = Counter(
            f"{PREFIX}_source_resets", "Seq restarts detected", ["source"], registry=r
        )
        self.latency = Histogram(
            f"{PREFIX}_e2e_latency_seconds",
            "Send/receive -> bus publish",
            ["source", "basis"],
            registry=r,
        )

    def value(self, name: str, **labels: str) -> float:
        """Current value of a counter sample, e.g. ``value("duplicates", source="file")``."""
        sample = self.registry.get_sample_value(f"{PREFIX}_{name}_total", labels)
        return 0.0 if sample is None else sample

    def quarantined_by_reason(self, source: AdapterKind) -> dict[QuarantineReason, float]:
        return {
            reason: self.value("quarantined", source=source.value, reason=reason.value)
            for reason in QuarantineReason
        }
