"""Schema for ``config/engine.yaml``: state transitions, data quality, alerts (Phase 4)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.enums import AlertSeverity, MachineState


class AlertKind(StrEnum):
    """Alert kinds raised by the twin engine."""

    MACHINE_WARNING = "machine_warning"
    MACHINE_FAULT = "machine_fault"
    MACHINE_OFFLINE = "machine_offline"
    DATA_QUALITY = "data_quality"
    LINE_STOPPAGE = "line_stoppage"
    # Phase 11 (ADR 0030): raised from the inference service's conditions (source "inference")
    PREDICTED_WARNING = "predicted_warning"
    PREDICTED_FAULT = "predicted_fault"
    UNUSUAL_BEHAVIOUR = "unusual_behaviour"
    MODEL_DRIFT = "model_drift"


INFERENCE_ALERT_KINDS = frozenset(
    {
        AlertKind.PREDICTED_WARNING,
        AlertKind.PREDICTED_FAULT,
        AlertKind.UNUSUAL_BEHAVIOUR,
        AlertKind.MODEL_DRIFT,
    }
)


class DataQualityConfig(ConfigModel):
    window_s: float = Field(gt=0, description="rolling wall-clock window for the rates")
    min_samples: int = Field(ge=1, description="judge rates only with at least this many messages")
    max_late_pct: float = Field(ge=0, le=100)
    max_quarantine_pct: float = Field(ge=0, le=100)
    max_partial_pct: float = Field(ge=0, le=100)
    stale_is_degraded: bool = True
    recover_after_s: float = Field(ge=0, description="must stay OK this long to return to OK")
    evaluate_every_s: float = Field(gt=0)


class AlertKindConfig(ConfigModel):
    enabled: bool = True
    severity: AlertSeverity
    raise_after_s: float = Field(
        ge=0, default=0.0, description="condition must persist this long (event time) to raise"
    )
    source: str | None = Field(
        default=None, min_length=1, description="Alert.source override (default: alerts.source)"
    )


class AlertsConfig(ConfigModel):
    source: str = Field(min_length=1, description="Alert.source of engine alerts")
    kinds: dict[AlertKind, AlertKindConfig]

    @model_validator(mode="after")
    def _all_kinds(self) -> AlertsConfig:
        missing = sorted(k.value for k in AlertKind if k not in self.kinds)
        if missing:
            raise ValueError(f"kinds: missing entries for {missing}")
        return self

    def source_of(self, kind: AlertKind) -> str:
        return self.kinds[kind].source or self.source

    def sources(self) -> set[str]:
        return {self.source} | {k.source for k in self.kinds.values() if k.source}


class EngineConfig(ConfigModel):
    # from-state -> allowed to-states (self-transitions are not transitions).
    transitions: dict[MachineState, list[MachineState]]
    data_quality: DataQualityConfig
    alerts: AlertsConfig

    @model_validator(mode="after")
    def _complete(self) -> EngineConfig:
        missing = sorted(s.value for s in MachineState if s not in self.transitions)
        if missing:
            raise ValueError(f"transitions: every MachineState needs an entry; missing {missing}")
        for src, targets in self.transitions.items():
            if src in targets:
                raise ValueError(f"transitions.{src.value}: lists itself")
            if len(set(targets)) != len(targets):
                raise ValueError(f"transitions.{src.value}: duplicate targets")
        return self

    def allowed(self, src: MachineState, dst: MachineState) -> bool:
        return src == dst or dst in self.transitions[src]
