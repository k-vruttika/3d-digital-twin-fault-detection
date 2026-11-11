"""Contracts reject malformed data with clear errors."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from twin_common import (
    CommandType,
    HealthClass,
    MachineState,
    Reading,
    to_health_class,
)
from twin_common.enums import UnlabelableStateError, is_labelable
from twin_common.testing import sample_command as make_command
from twin_common.testing import sample_prediction as make_prediction
from twin_common.testing import sample_reading as make_reading
from twin_common.topics import cmd_ack_topic, cmd_topic, state_topic, telemetry_topic


def test_reading_rejects_unknown_field() -> None:
    payload = json.loads(make_reading().model_dump_json())
    payload["pressure"] = 1.0
    with pytest.raises(ValidationError, match="pressure"):
        Reading.model_validate(payload)


def test_reading_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        make_reading(ts=datetime(2026, 1, 15, 8, 30))


def test_reading_rejects_wrong_schema_version() -> None:
    payload = json.loads(make_reading().model_dump_json())
    payload["schemaVersion"] = 1
    with pytest.raises(ValidationError, match="schemaVersion"):
        Reading.model_validate(payload)


@pytest.mark.parametrize("field", ["speed", "vib", "current", "load", "run_time", "seq"])
def test_reading_rejects_negative(field: str) -> None:
    with pytest.raises(ValidationError):
        make_reading(**{field: -1})


def test_reading_rejects_unknown_state() -> None:
    with pytest.raises(ValidationError):
        make_reading(state="EXPLODED")


def test_reading_is_immutable() -> None:
    r = make_reading()
    with pytest.raises(ValidationError):
        r.speed = 1.0  # type: ignore[misc]


def test_prediction_probs_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        make_prediction(
            probs={HealthClass.NORMAL: 0.5, HealthClass.WARNING: 0.2, HealthClass.FAULT: 0.1}
        )


def test_prediction_probs_must_cover_all_classes() -> None:
    with pytest.raises(ValidationError, match="missing classes"):
        make_prediction(probs={HealthClass.NORMAL: 0.8, HealthClass.WARNING: 0.2})


def test_prediction_ttf_interval_ordered() -> None:
    with pytest.raises(ValidationError, match="ttfInterval"):
        make_prediction(ttf_interval=(200.0, 100.0))


def test_prediction_conformal_set_canonical_and_unique() -> None:
    p = make_prediction(conformal_set=[HealthClass.FAULT, HealthClass.NORMAL])
    assert p.conformal_set == [HealthClass.NORMAL, HealthClass.FAULT]
    with pytest.raises(ValidationError, match="duplicates"):
        make_prediction(conformal_set=[HealthClass.FAULT, HealthClass.FAULT])


def test_set_speed_requires_speed_param() -> None:
    with pytest.raises(ValidationError, match="requires param 'speed'"):
        make_command(type=CommandType.SET_SPEED, params={})


def test_set_load_param_must_be_number() -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        make_command(type=CommandType.SET_LOAD, params={"load": "high"})
    with pytest.raises(ValidationError, match="must be a number"):
        make_command(type=CommandType.SET_LOAD, params={"load": True})


def test_command_requires_non_empty_ids() -> None:
    with pytest.raises(ValidationError):
        make_command(idempotency_key="")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (MachineState.RUNNING, HealthClass.NORMAL),
        (MachineState.IDLE, HealthClass.NORMAL),
        (MachineState.STARVED, HealthClass.NORMAL),
        (MachineState.BLOCKED, HealthClass.NORMAL),
        (MachineState.MAINTENANCE, HealthClass.NORMAL),
        (MachineState.WARNING, HealthClass.WARNING),
        (MachineState.FAULT, HealthClass.FAULT),
    ],
)
def test_health_mapping(state: MachineState, expected: HealthClass) -> None:
    assert to_health_class(state) is expected
    assert is_labelable(state)


@pytest.mark.parametrize("state", [MachineState.OFFLINE, MachineState.STALE])
def test_unlabelable_states(state: MachineState) -> None:
    assert not is_labelable(state)
    with pytest.raises(UnlabelableStateError, match=state.value):
        to_health_class(state)


def test_mqtt_topics() -> None:
    assert telemetry_topic("CNC-01") == "plant/line1/CNC-01/telemetry"
    assert state_topic("CNC-01") == "plant/line1/CNC-01/state"
    assert cmd_topic("CNC-01") == "plant/line1/CNC-01/cmd"
    assert cmd_ack_topic("CNC-01", prefix="plant/line2/") == "plant/line2/CNC-01/cmd/ack"
    for bad in ["", "a/b", "+", "#"]:
        with pytest.raises(ValueError, match="invalid machine id"):
            telemetry_topic(bad)
