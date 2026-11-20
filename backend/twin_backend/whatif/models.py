"""Request and response of ``POST /whatif`` (camelCase JSON, ADR 0031)."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from twin_backend.engine.line_state import StateModel
from twin_common.contracts import ParamValue
from twin_common.enums import CommandType, HealthClass, MachineState


class WhatIfPayload(StateModel):
    """Body of ``POST /whatif``: one change to one machine, applied at the fork point (now)."""

    machine_id: str = Field(min_length=1)
    type: CommandType
    params: dict[str, ParamValue] = Field(default_factory=dict)
    horizon_s: int | None = Field(
        default=None, gt=0, description="projection length (default and maximum: whatif.yaml)"
    )


class ProjectionPoint(StateModel):
    """One machine at one second: the simulated reading and, with a model, its prediction."""

    t_s: int = Field(description="seconds after the fork point (history: zero or negative)")
    ts: datetime
    state: MachineState
    speed: float | None = None
    temp: float | None = None
    vib: float | None = None
    load: float | None = None
    current: float | None = None
    gate: str | None = Field(
        default=None, description="predict | warming_up | waiting | stopped; None without a model"
    )
    p_warning: float | None = Field(default=None, description="calibrated, primary horizon")
    p_fault: float | None = Field(default=None, description="calibrated, primary horizon")
    top_class: HealthClass | None = None
    alarm: bool | None = Field(default=None, description="the evaluated alarm rule")
    unusual: bool | None = None
    ttf_s: float | None = None


class MachineProjection(StateModel):
    points: list[ProjectionPoint] = Field(description="t_s > 0, oldest first")
    ghost: dict[str, MachineState] = Field(
        description='projected simulator state at each ghost horizon ("10", "30") and "end"'
    )
    first_warning_s: int | None = Field(default=None, description="simulator WARNING (truth)")
    first_fault_s: int | None = Field(default=None, description="simulator FAULT (truth)")
    first_alarm_s: int | None = Field(
        default=None, description="model alarm (None: none, or no model)"
    )
    max_p_fault: float | None = None


class Variant(StateModel):
    machines: dict[str, MachineProjection]


class WhatIfResponse(StateModel):
    as_of: datetime
    machine_id: str
    type: CommandType
    params: dict[str, ParamValue]
    horizon_s: int
    ghost_horizons_s: list[int]
    primary_horizon_s: int | None = Field(default=None, description="horizon of p_* and alarm")
    model: str | None = Field(default=None, description="model version scoring the projection")
    model_note: str | None = Field(default=None, description="why there are no predictions")
    history: dict[str, list[ProjectionPoint]] = Field(description="t_s <= 0, per machine")
    baseline: Variant = Field(description="nothing changed")
    scenario: Variant = Field(description="the change applied now")
    elapsed_ms: int
