"""Valid sample contracts for tests in every package. Not used by production code."""

from __future__ import annotations

from datetime import UTC, datetime

from twin_common.contracts import (
    Alert,
    Attribution,
    Command,
    CommandAck,
    Prediction,
    Reading,
    StateChange,
)
from twin_common.enums import (
    AckStatus,
    AlertSeverity,
    CommandType,
    HealthClass,
    MachineState,
    Quality,
    Reliability,
    SensorName,
)

SAMPLE_TS = datetime(2026, 1, 15, 8, 30, 0, 123000, tzinfo=UTC)


def sample_reading(**overrides: object) -> Reading:
    data: dict[str, object] = {
        "machine_id": "CNC-01",
        "seq": 42,
        "ts": SAMPLE_TS,
        "state": MachineState.RUNNING,
        "speed": 8000.0,
        "temp": 55.5,
        "vib": 2.3,
        "load": 62.0,
        "current": 14.2,
        "run_time": 3600.0,
        "quality": Quality.OK,
    }
    data.update(overrides)
    return Reading.model_validate(data)


def sample_prediction(**overrides: object) -> Prediction:
    data: dict[str, object] = {
        "machine_id": "CNC-01",
        "ts": SAMPLE_TS,
        "horizon_s": 30.0,
        "probs": {HealthClass.NORMAL: 0.7, HealthClass.WARNING: 0.2, HealthClass.FAULT: 0.1},
        "calibrated": True,
        "conformal_set": [HealthClass.NORMAL, HealthClass.WARNING],
        "ttf_s": 120.0,
        "ttf_interval": (90.0, 160.0),
        "anomaly_score": 0.35,
        "attributions": [
            Attribution(sensor=SensorName.VIB, part="spindle_bearing", weight=0.6),
            Attribution(sensor=SensorName.TEMP, part="spindle_motor", weight=0.25),
        ],
        "reliability": Reliability.OK,
        "model_version": "xgb-2026.01.15-abc123",
    }
    data.update(overrides)
    return Prediction.model_validate(data)


def sample_command(**overrides: object) -> Command:
    data: dict[str, object] = {
        "command_id": "cmd-0001",
        "idempotency_key": "idem-0001",
        "machine_id": "CNV-01",
        "type": CommandType.SET_SPEED,
        "params": {"speed": 1200.0},
        "issued_by": "operator.alice",
        "ts": SAMPLE_TS,
    }
    data.update(overrides)
    return Command.model_validate(data)


def sample_ack(**overrides: object) -> CommandAck:
    data: dict[str, object] = {
        "command_id": "cmd-0001",
        "status": AckStatus.EXECUTED,
        "reason": None,
        "ts": SAMPLE_TS,
    }
    data.update(overrides)
    return CommandAck.model_validate(data)


def sample_state_change(**overrides: object) -> StateChange:
    data: dict[str, object] = {
        "machine_id": "CNC-01",
        "state": MachineState.STALE,
        "previous_state": MachineState.RUNNING,
        "ts": SAMPLE_TS,
        "reason": "no data for 3.0 s (limit 3 s)",
    }
    data.update(overrides)
    return StateChange.model_validate(data)


def sample_alert(**overrides: object) -> Alert:
    data: dict[str, object] = {
        "alert_id": "alert-0001",
        "machine_id": "CNC-01",
        "ts": SAMPLE_TS,
        "severity": AlertSeverity.WARNING,
        "kind": "fault_predicted",
        "message": "FAULT likely within 30 s",
        "source": "inference",
    }
    data.update(overrides)
    return Alert.model_validate(data)
