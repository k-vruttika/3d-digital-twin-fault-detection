"""Canonical data contracts (CLAUDE.md section 6).

Python attributes are snake_case; the JSON wire format is camelCase. Models are immutable and
reject unknown fields. All timestamps must be timezone-aware and are normalized to UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from twin_common.enums import (
    AckStatus,
    CommandType,
    HealthClass,
    MachineState,
    Quality,
    Reliability,
    SensorName,
)

READING_SCHEMA_VERSION: Literal[2] = 2

# Numerical tolerance for "probabilities sum to 1". This is floating-point slack, not a
# domain threshold, so it lives here rather than in YAML config.
PROB_SUM_TOLERANCE = 1e-6


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
ParamValue = bool | int | float | str


class ContractModel(BaseModel):
    """Base for wire contracts: camelCase JSON, immutable, strict about unknown fields."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )


class Reading(ContractModel):
    """One telemetry sample from one machine."""

    schema_version: Literal[2] = READING_SCHEMA_VERSION
    machine_id: NonEmptyStr
    seq: int = Field(ge=0, description="Monotonic per-machine sequence number")
    ts: UtcDatetime
    state: MachineState
    speed: float = Field(ge=0, description="RPM")
    temp: float = Field(description="degrees C")
    vib: float = Field(ge=0, description="mm/s RMS")
    load: float = Field(ge=0, description="percent")
    current: float = Field(ge=0, description="A")
    run_time: float = Field(ge=0, description="seconds since start")
    quality: Quality = Quality.OK


class Attribution(ContractModel):
    """Contribution of one sensor (and the physical part it maps to) to a prediction."""

    sensor: SensorName
    part: NonEmptyStr
    weight: float


class Prediction(ContractModel):
    """Model output for one machine at one horizon."""

    machine_id: NonEmptyStr
    ts: UtcDatetime
    horizon_s: float = Field(gt=0)
    probs: dict[HealthClass, Annotated[float, Field(ge=0, le=1)]]
    calibrated: bool
    conformal_set: list[HealthClass]
    ttf_s: float | None = Field(default=None, ge=0, description="None: no fault expected")
    ttf_interval: tuple[float, float] | None = None
    anomaly_score: float
    attributions: list[Attribution] = Field(default_factory=list)
    reliability: Reliability
    model_version: NonEmptyStr

    @field_validator("probs")
    @classmethod
    def _probs_complete_and_normalized(
        cls, v: dict[HealthClass, float]
    ) -> dict[HealthClass, float]:
        missing = set(HealthClass) - set(v)
        if missing:
            raise ValueError(f"probs missing classes: {sorted(m.value for m in missing)}")
        total = sum(v.values())
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"probs must sum to 1 (got {total:.6f})")
        # Canonical key order for deterministic JSON.
        return {c: v[c] for c in HealthClass}

    @field_validator("conformal_set")
    @classmethod
    def _conformal_unique_canonical(cls, v: list[HealthClass]) -> list[HealthClass]:
        if len(set(v)) != len(v):
            raise ValueError("conformalSet contains duplicates")
        return [c for c in HealthClass if c in v]

    @field_validator("ttf_interval")
    @classmethod
    def _ttf_interval_ordered(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None:
            lo, hi = v
            if lo < 0 or lo > hi:
                raise ValueError(f"ttfInterval must satisfy 0 <= lo <= hi (got [{lo}, {hi}])")
        return v


# Parameters each command type must carry. Value limits are enforced by the safety rules.
REQUIRED_COMMAND_PARAMS: dict[CommandType, tuple[str, ...]] = {
    CommandType.SET_SPEED: ("speed",),
    CommandType.SET_LOAD: ("load",),
}


class Command(ContractModel):
    """Operator command sent towards a machine."""

    command_id: NonEmptyStr
    idempotency_key: NonEmptyStr
    machine_id: NonEmptyStr
    type: CommandType
    params: dict[str, ParamValue] = Field(default_factory=dict)
    issued_by: NonEmptyStr
    ts: UtcDatetime

    @model_validator(mode="after")
    def _required_params(self) -> Command:
        for key in REQUIRED_COMMAND_PARAMS.get(self.type, ()):
            value = self.params.get(key)
            if value is None:
                raise ValueError(f"{self.type.value} requires param '{key}'")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{self.type.value} param '{key}' must be a number")
        return self


class CommandAck(ContractModel):
    """Acknowledgement of a command by the command bus or the machine."""

    command_id: NonEmptyStr
    status: AckStatus
    reason: str | None = None
    ts: UtcDatetime
