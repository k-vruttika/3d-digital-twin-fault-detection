"""Storage metrics (Prometheus), registered on the same registry as the ingest metrics so the
API (Phase 4) serves them together at ``/metrics``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

PREFIX = "linetwin_storage"


class StorageMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.written = Counter(
            f"{PREFIX}_written", "Rows written to the event store", ["stream"], registry=r
        )
        self.duplicates = Counter(
            f"{PREFIX}_duplicates",
            "Rows skipped by the store because the event already existed",
            ["stream"],
            registry=r,
        )
        self.dropped = Counter(
            f"{PREFIX}_dropped_writes",
            "Rows that never reached the store (queue_full: writer fell behind; "
            "write_error: batch failed after all retries)",
            ["stream", "reason"],
            registry=r,
        )
        self.write_errors = Counter(
            f"{PREFIX}_write_errors", "Failed batch insert attempts", ["stream"], registry=r
        )
        self.batch_seconds = Histogram(
            f"{PREFIX}_batch_seconds", "Duration of one batch insert", ["stream"], registry=r
        )
        self.queue_depth = Gauge(
            f"{PREFIX}_queue_depth", "Rows waiting for the writer", ["stream"], registry=r
        )
        self.snapshots = Counter(f"{PREFIX}_snapshots", "Line snapshots saved", registry=r)
        self.snapshot_invalidations = Counter(
            f"{PREFIX}_snapshot_invalidations",
            "Snapshots deleted because older data arrived after them",
            registry=r,
        )

    def value(self, name: str, **labels: str) -> float:
        """Current value of a counter sample, e.g. ``value("dropped_writes", stream=...)``."""
        sample = self.registry.get_sample_value(f"{PREFIX}_{name}_total", labels)
        return 0.0 if sample is None else sample

    def dropped_total(self) -> float:
        total = 0.0
        for metric in self.registry.collect():
            if metric.name == f"{PREFIX}_dropped_writes":
                total += sum(s.value for s in metric.samples if s.name.endswith("_total"))
        return total
