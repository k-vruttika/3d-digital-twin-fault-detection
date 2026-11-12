"""Schema for ``config/storage.yaml``: event store backend, writer, snapshots (Phase 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel


class SqliteStoreConfig(ConfigModel):
    path: str = Field(min_length=1, description="relative to the repository root")
    busy_timeout_s: float = Field(gt=0, description="wait this long for a locked database")


class AggregatePolicyConfig(ConfigModel):
    """Refresh policy of one continuous aggregate (TimescaleDB)."""

    start_offset_s: float = Field(gt=0, description="refresh window reaches back this far")
    end_offset_s: float = Field(ge=0, description="newest data left to real-time aggregation")
    schedule_s: float = Field(gt=0, description="how often the refresh job runs")

    @model_validator(mode="after")
    def _ordered(self) -> AggregatePolicyConfig:
        if self.end_offset_s >= self.start_offset_s:
            raise ValueError(
                f"end_offset_s ({self.end_offset_s}) must be < start_offset_s "
                f"({self.start_offset_s})"
            )
        return self


class AggregatesConfig(ConfigModel):
    m1: AggregatePolicyConfig
    m10: AggregatePolicyConfig


class CompressionConfig(ConfigModel):
    enabled: bool = False
    after_s: float = Field(gt=0, description="compress chunks older than this")


class RetentionConfig(ConfigModel):
    enabled: bool = False
    drop_after_s: float = Field(gt=0, description="drop raw chunks older than this")


class ChunkIntervalConfig(ConfigModel):
    readings: float = Field(gt=0)
    default: float = Field(gt=0, description="all other event tables")


class TimescaleStoreConfig(ConfigModel):
    pool_min: int = Field(ge=1)
    pool_max: int = Field(ge=1)
    command_timeout_s: float = Field(gt=0)
    chunk_interval_s: ChunkIntervalConfig
    max_late_arrival_s: float = Field(
        gt=0, description="LATE readings this old must still reach the aggregates"
    )
    aggregates: AggregatesConfig
    compression: CompressionConfig
    retention: RetentionConfig

    @model_validator(mode="after")
    def _consistent(self) -> TimescaleStoreConfig:
        errors: list[str] = []
        if self.pool_min > self.pool_max:
            errors.append(f"pool_min ({self.pool_min}) must be <= pool_max ({self.pool_max})")
        for name, agg in (("m1", self.aggregates.m1), ("m10", self.aggregates.m10)):
            need = self.max_late_arrival_s + agg.end_offset_s
            if agg.start_offset_s < need:
                errors.append(
                    f"aggregates.{name}.start_offset_s ({agg.start_offset_s:g}) must be >= "
                    f"max_late_arrival_s + end_offset_s ({need:g}) so late readings are "
                    "re-aggregated"
                )
        if self.retention.enabled and self.retention.drop_after_s <= self.max_late_arrival_s:
            errors.append("retention.drop_after_s must be > max_late_arrival_s")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class WriterRetryConfig(ConfigModel):
    attempts: int = Field(ge=1, description="tries per batch before it is dropped")
    backoff_s: float = Field(ge=0, description="first retry delay; doubles per attempt")


class WriterConfig(ConfigModel):
    batch_size: int = Field(ge=1)
    flush_interval_s: float = Field(gt=0, description="max time a row waits for its batch")
    queue_size: int = Field(ge=1, description="per stream; oldest dropped when full")
    retry: WriterRetryConfig


class SnapshotConfig(ConfigModel):
    enabled: bool = True
    interval_s: float = Field(gt=0, description="event-time spacing of line snapshots")
    settle_s: float = Field(
        ge=0, description="only snapshot event time older than the watermark minus this"
    )
    check_every_s: float = Field(gt=0, description="wall-clock period of the snapshot check")


class StorageConfig(ConfigModel):
    backend: Literal["sqlite", "timescale"]
    sqlite: SqliteStoreConfig
    timescale: TimescaleStoreConfig | None = None
    writer: WriterConfig
    snapshots: SnapshotConfig

    @model_validator(mode="after")
    def _backend_configured(self) -> StorageConfig:
        if self.backend == "timescale" and self.timescale is None:
            raise ValueError("backend is 'timescale' but no 'timescale:' section is configured")
        return self
