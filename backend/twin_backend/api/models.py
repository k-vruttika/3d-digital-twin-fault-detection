"""Response models of the REST API (camelCase JSON)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from twin_backend.engine.data_quality import DataQualityReport
from twin_backend.engine.line_state import LineState, StateModel
from twin_backend.engine.twin import LineViewModel, RootCauseView
from twin_common.contracts import Alert, Prediction, Reading, StateChange
from twin_common.enums import (
    AdapterKind,
    AdapterStatus,
    Capability,
    CommandType,
    MachineKind,
    MachineState,
    Role,
    SensorName,
)


class MachineStateResponse(StateModel):
    machine_id: str
    kind: MachineKind
    name: str
    state: MachineState | None = None
    state_ts: datetime | None = None
    reported_state: MachineState | None = None
    root_cause: RootCauseView
    data_quality: DataQualityReport
    last_reading: Reading | None = None
    last_state_change: StateChange | None = None
    predictions: dict[str, Prediction] = Field(default_factory=dict)
    alerts: list[Alert] = Field(default_factory=list)


class MachinesResponse(StateModel):
    line_id: str
    as_of: datetime | None = None
    line: LineViewModel
    machines: list[MachineStateResponse]


class SensorStatsModel(StateModel):
    count: int
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class HistoryBucketModel(StateModel):
    bucket: datetime
    machine_id: str
    n: int
    n_late: int
    n_suspect: int
    sensors: dict[SensorName, SensorStatsModel]


class HistoryResponse(StateModel):
    resolution: Literal["raw", "1s", "1m", "10m"]
    start: datetime
    end: datetime
    machines: list[str] | None = None
    readings: list[Reading] | None = None
    buckets: list[HistoryBucketModel] | None = None


class AdapterHealthModel(StateModel):
    kind: AdapterKind
    status: AdapterStatus
    connected: bool
    messages: int
    reconnects: int
    last_message_at: datetime | None = None
    last_error: str | None = None


class SourcesResponse(StateModel):
    active: AdapterKind | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    health: AdapterHealthModel | None = None
    configured: list[AdapterKind]
    switches: int
    ingest: IngestStatsModel | None = Field(
        default=None, description="ingestion counters of the active source"
    )


class SwitchRequest(StateModel):
    kind: AdapterKind


class AlertsResponse(StateModel):
    alerts: list[Alert]


class AuthInfo(StateModel):
    keys_configured: bool
    anonymous_read: bool
    operator_keys: bool


class HealthResponse(StateModel):
    status: Literal["ok", "degraded"]
    line_id: str
    machines: int
    source: AdapterHealthModel | None = None
    store: str | None = None
    event_loop: str
    auth: AuthInfo
    loaded_optional_modules: list[str] | None = None


# ---------------------------------------------------------------- GET /line (Phase 5)
class SensorSpecModel(StateModel):
    unit: str
    min: float
    max: float


class MachineLayoutModel(StateModel):
    position: tuple[float, float, float] | None = Field(
        default=None, description="[x, y, z] in metres, y up"
    )
    rotation_deg: float | None = Field(default=None, description="about the vertical axis")


class LineMachineModel(StateModel):
    id: str
    kind: MachineKind
    name: str
    sensors: list[str]
    parts: list[str]
    sensor_parts: dict[str, str] = Field(description="sensor -> part it is highlighted on")
    layout: MachineLayoutModel | None = Field(
        default=None, description="placement override; null = automatic layout"
    )


class LineBufferModel(StateModel):
    id: str
    from_: str = Field(alias="from")
    to: str
    capacity: int


class TopologyModel(StateModel):
    order: list[str] = Field(description="machine ids in material-flow (config) order")
    upstream: dict[str, str | None] = Field(description="machine -> feeder; null = SOURCE")
    downstream: dict[str, str | None] = Field(description="machine -> receiver; null = SINK")


class LineResponse(StateModel):
    line_id: str
    sensors: dict[str, SensorSpecModel]
    machines: list[LineMachineModel]
    buffers: list[LineBufferModel]
    topology: TopologyModel


# ---------------------------------------------------------------- Phase 6 (dashboard)
class AuthMeResponse(StateModel):
    principal: str
    role: Role
    anonymous: bool


class RangeModel(StateModel):
    min: float
    max: float


class KindLimitsModel(StateModel):
    speed_rpm: RangeModel
    load_pct: RangeModel


class SafetyRuleModel(StateModel):
    types: list[CommandType]
    forbidden_states: list[MachineState] | None = None
    required_states: list[MachineState] | None = None
    reason: str


class CommandLimitsResponse(StateModel):
    limits: dict[MachineKind, KindLimitsModel] = Field(
        description="parameter limits per machine kind (commands.yaml)"
    )
    rules: list[SafetyRuleModel] = Field(description="state rules; STOP is always allowed")
    always_allowed: list[CommandType] = Field(description="commands no rule can block")


class ReplayRangeResponse(StateModel):
    start: datetime | None = Field(default=None, description="oldest stored reading/state ts")
    end: datetime | None = Field(default=None, description="newest stored reading/state ts")


class ReplayResponse(LineState):
    """``reconstruct(t)`` plus what the twin derives from it (root causes, alerts as of t)."""

    line: LineViewModel
    alerts: list[Alert] = Field(
        default_factory=list, description="unresolved alerts as of t (highest rev <= t)"
    )


class LatencyModel(StateModel):
    count: int
    mean_s: float | None = None
    p95_s: float | None = Field(default=None, description="upper bound of the p95 histogram bucket")


class IngestStatsModel(StateModel):
    """Counters of the active source since the backend started (IngestMetrics)."""

    messages_in: int
    published: int
    late: int
    late_pct: float | None = None
    quarantined: int
    quarantined_by_reason: dict[str, int]
    partial: int
    partial_pct: float | None = None
    duplicates: int
    latency: LatencyModel


class FaultOptionModel(StateModel):
    fault_type: str
    kinds: list[MachineKind]
    gradual: bool = Field(description="progresses over progressionS; false = immediate (jam)")
    held_out: bool = Field(description="excluded from training data (unseen-fault experiments)")


class ScenarioOptionsResponse(StateModel):
    faults: list[FaultOptionModel]
    progression_s: RangeModel | None = Field(
        default=None, description="configured progression range of gradual faults"
    )


# ---------------------------------------------------------------- Phase 11: predictions
GateStatus = Literal["predict", "warming_up", "waiting", "stopped", "unavailable"]


class ModelCardModel(StateModel):
    training_data: str
    real_data: str
    limitations: list[str]


class BundleInfoModel(StateModel):
    """A loaded deploy bundle (one registered MLflow model version, ADR 0029)."""

    registered: str
    version: str
    label: str
    kind: Literal["supervised", "anomaly"]
    model_type: str
    seed: int
    dataset_key: str
    built_at: str
    checks: dict[str, str] = Field(description="how the bundle reproduced the evaluated run")
    card: ModelCardModel
    horizons: list[int] | None = None
    ttf_cap_s: float | None = None
    sustain_s: float | None = None
    alpha: float | None = Field(default=None, description="conformal miscoverage level")
    percentile: float | None = Field(default=None, description="autoencoder threshold")


class InferenceInfoModel(StateModel):
    enabled: bool
    state: Literal["off", "starting", "ready", "restarting", "failed", "stopped"]
    reason: str | None = None
    since: datetime | None = None
    model: str | None = Field(default=None, description="config/inference.yaml models key")
    model_version: str | None = None
    window_s: int | None = None
    primary_horizon_s: int | None = None
    prediction_stale_s: float | None = None
    supervised: BundleInfoModel | None = None
    anomaly: BundleInfoModel | None = None


class MachineGateModel(StateModel):
    status: GateStatus
    run_len: int = Field(description="consecutive operating readings (capped at the window)")
    window: int
    last_prediction_ts: datetime | None = None
    alarm: bool = Field(description="supervised alarm (the evaluated rule)")
    predicted_fault: bool
    unusual: bool = Field(description="autoencoder alarm without a supervised alarm")


class MachinePredictionModel(StateModel):
    gate: MachineGateModel
    predictions: dict[str, Prediction] = Field(
        default_factory=dict, description="newest prediction per horizon"
    )


class PredictResponse(StateModel):
    inference: InferenceInfoModel
    machines: dict[str, MachinePredictionModel] = Field(default_factory=dict)


class PredictionsResponse(StateModel):
    predictions: list[Prediction]


SourcesResponse.model_rebuild()
