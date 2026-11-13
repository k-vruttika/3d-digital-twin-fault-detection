"""Schema for ``config/commands.yaml``: safety rules, parameter limits, idempotency (Phase 4).

STOP is always allowed (fail-safe principle, ADR 0014): the evaluator exempts it in code and
this schema rejects any rule that names STOP, so configuration cannot disable it.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.enums import CommandType, MachineKind, MachineState

FAIL_SAFE_COMMANDS: frozenset[CommandType] = frozenset({CommandType.STOP})


class SafetyRule(ConfigModel):
    """Reject ``types`` while the machine is in ``forbidden_states``, or unless it is in
    ``required_states``. Exactly one of the two must be given."""

    types: list[CommandType] = Field(min_length=1)
    forbidden_states: list[MachineState] | None = None
    required_states: list[MachineState] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _one_condition(self) -> SafetyRule:
        if (self.forbidden_states is None) == (self.required_states is None):
            raise ValueError("give exactly one of forbidden_states / required_states")
        safe = sorted(t.value for t in self.types if t in FAIL_SAFE_COMMANDS)
        if safe:
            raise ValueError(
                f"{safe} must always be allowed (fail-safe principle); remove it from types"
            )
        return self


class Range(ConfigModel):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class KindLimits(ConfigModel):
    speed_rpm: Range
    load_pct: Range


class CommandsConfig(ConfigModel):
    idempotency_ttl_s: float = Field(gt=0, description="remember idempotency keys this long")
    idempotency_max_entries: int = Field(ge=1)
    limits: dict[MachineKind, KindLimits]
    rules: list[SafetyRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_kinds(self) -> CommandsConfig:
        missing = sorted(k.value for k in MachineKind if k not in self.limits)
        if missing:
            raise ValueError(f"limits: missing entries for machine kinds {missing}")
        return self
