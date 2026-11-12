"""Schema for ``config/sources.yaml``: active adapter, per-adapter settings and ingest tuning."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.contracts import Reading
from twin_common.enums import AdapterKind, SensorName
from twin_common.topics import DEFAULT_TOPIC_PREFIX
from twin_common.units import KNOWN_UNITS

_READING_WIRE_KEYS: frozenset[str] = frozenset(
    (f.alias or name) for name, f in Reading.model_fields.items()
)


class FieldMapping(ConfigModel):
    """How a source's payload maps onto the canonical Reading (camelCase wire names).

    ``fields`` maps canonical key -> source key; keys not listed are taken as-is.
    ``units`` maps sensor -> the unit the source sends; values are converted to the line
    catalog unit. With ``drop_unmapped`` source keys that are neither mapped nor canonical
    are dropped; otherwise they are kept and fail validation as unknown fields.
    """

    fields: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    drop_unmapped: bool = False

    @model_validator(mode="after")
    def _known_keys(self) -> FieldMapping:
        errors: list[str] = []
        for key in self.fields:
            if key not in _READING_WIRE_KEYS:
                errors.append(
                    f"fields: '{key}' is not a Reading field "
                    f"(allowed: {', '.join(sorted(_READING_WIRE_KEYS))})"
                )
        sensors = {s.value for s in SensorName}
        for sensor, unit in self.units.items():
            if sensor not in sensors:
                errors.append(f"units: '{sensor}' is not a sensor ({', '.join(sorted(sensors))})")
            if unit not in KNOWN_UNITS:
                errors.append(
                    f"units: unknown unit '{unit}' for '{sensor}' "
                    f"(known: {', '.join(sorted(KNOWN_UNITS))})"
                )
        if errors:
            raise ValueError("; ".join(errors))
        return self


class FileSourceConfig(ConfigModel):
    path: str = Field(min_length=1, description="Relative to the repository root")
    format: Literal["csv", "jsonl", "json"]
    replay_speed: float = Field(gt=0, description="1.0 = real time")
    loop: bool = False
    start_offset_s: float = Field(default=0.0, ge=0, description="skip this much event time")
    mapping: FieldMapping = Field(default_factory=FieldMapping)


class MockSourceConfig(ConfigModel):
    base_url: str = Field(min_length=1)
    ws_path: str = Field(min_length=1)
    history_path: str = Field(min_length=1)
    command_path: str = Field(min_length=1)
    rest_poll_s: float = Field(gt=0, description="REST fallback poll period while WS is down")
    history_limit: int = Field(ge=1, description="max readings per /history request")
    mapping: FieldMapping = Field(default_factory=FieldMapping)


class MqttSourceConfig(ConfigModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    topic_prefix: str = DEFAULT_TOPIC_PREFIX
    client_id: str = Field(min_length=1)
    qos: Literal[0, 1, 2]
    keepalive_s: int = Field(gt=0)
    mapping: FieldMapping = Field(default_factory=FieldMapping)


class OpcUaSourceConfig(ConfigModel):
    endpoint: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    mapping: FieldMapping = Field(default_factory=FieldMapping)


class ReconnectConfig(ConfigModel):
    """Exponential backoff: delay_n = min(max_s, initial_s * multiplier^n) * (1 +/- jitter)."""

    initial_s: float = Field(gt=0)
    max_s: float = Field(gt=0)
    multiplier: float = Field(ge=1)
    jitter: float = Field(ge=0, le=1)
    seed: int

    @model_validator(mode="after")
    def _ordered(self) -> ReconnectConfig:
        if self.initial_s > self.max_s:
            raise ValueError(f"initial_s ({self.initial_s}) must be <= max_s ({self.max_s})")
        return self


class QuarantineConfig(ConfigModel):
    kind: Literal["memory", "jsonl"]
    path: str | None = Field(default=None, description="jsonl only; relative to the repo root")
    max_records: int = Field(ge=1, description="memory sink capacity (oldest dropped)")

    @model_validator(mode="after")
    def _path_for_jsonl(self) -> QuarantineConfig:
        if self.kind == "jsonl" and not self.path:
            raise ValueError("quarantine kind 'jsonl' needs a 'path'")
        return self


class IngestConfig(ConfigModel):
    lateness_window_s: float = Field(gt=0, description="max hold time for out-of-order data")
    dedup_window: int = Field(ge=1, description="seqs remembered per machine for dedup")
    seq_reset_threshold: int = Field(
        ge=1, description="seq drop larger than this = source restarted"
    )
    stale_after_s: float = Field(gt=0)
    offline_after_s: float = Field(gt=0)
    tick_s: float = Field(gt=0, description="period of reorder/staleness housekeeping")
    out_of_range: Literal["suspect", "quarantine"]
    quarantine: QuarantineConfig

    @model_validator(mode="after")
    def _ordered(self) -> IngestConfig:
        errors: list[str] = []
        if self.stale_after_s >= self.offline_after_s:
            errors.append(
                f"stale_after_s ({self.stale_after_s}) must be < offline_after_s "
                f"({self.offline_after_s})"
            )
        if self.seq_reset_threshold > self.dedup_window:
            # Any seq older than the dedup window must be classified as a source restart,
            # never silently re-accepted as new.
            errors.append(
                f"seq_reset_threshold ({self.seq_reset_threshold}) must be <= dedup_window "
                f"({self.dedup_window})"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self


class BusConfig(ConfigModel):
    queue_size: int = Field(ge=1, description="per-subscriber queue; oldest dropped when full")


class ChaosConfig(ConfigModel):
    """Seeded transport faults applied by the ChaosAdapter wrapper (tests and demos)."""

    enabled: bool = False
    seed: int
    drop: float = Field(ge=0, le=1)
    duplicate: float = Field(ge=0, le=1)
    delay: float = Field(ge=0, le=1)
    delay_s: float = Field(ge=0)
    reorder: float = Field(ge=0, le=1)
    reorder_max_displacement: int = Field(ge=1, description="max positions a message moves")
    corrupt: float = Field(ge=0, le=1)


class SourcesConfig(ConfigModel):
    active: AdapterKind
    command_timeout_s: float = Field(gt=0)
    file: FileSourceConfig | None = None
    mock: MockSourceConfig | None = None
    mqtt: MqttSourceConfig | None = None
    opcua: OpcUaSourceConfig | None = None
    reconnect: ReconnectConfig
    ingest: IngestConfig
    bus: BusConfig
    chaos: ChaosConfig

    @model_validator(mode="after")
    def _active_configured(self) -> SourcesConfig:
        self.require(self.active)
        return self

    def require(self, kind: AdapterKind) -> None:
        if getattr(self, kind.value) is None:
            raise ValueError(
                f"active adapter is '{kind.value}' but no '{kind.value}:' section is configured"
            )
