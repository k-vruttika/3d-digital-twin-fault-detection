"""model -> JSON -> model round trips for every contract in CLAUDE.md section 6."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from twin_common import (
    AckStatus,
    Attribution,
    CommandType,
    MachineState,
    Quality,
    Reading,
    Reliability,
    SensorName,
)
from twin_common.testing import sample_ack, sample_command, sample_prediction, sample_reading

CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*$")

make_reading = sample_reading
make_prediction = sample_prediction
make_command = sample_command
make_ack = sample_ack


CASES: list[BaseModel] = [
    make_reading(),
    *[make_reading(state=s) for s in MachineState],
    *[make_reading(quality=q) for q in Quality],
    make_prediction(),
    make_prediction(ttf_s=None, ttf_interval=None, attributions=[], calibrated=False),
    make_prediction(conformal_set=[], reliability=Reliability.DEGRADED),
    Attribution(sensor=SensorName.CURRENT, part="motor", weight=-0.1),
    make_command(),
    *[
        make_command(type=t, params={})
        for t in CommandType
        if t not in (CommandType.SET_SPEED, CommandType.SET_LOAD)
    ],
    make_command(type=CommandType.SET_LOAD, params={"load": 70}),
    make_command(
        type=CommandType.SCHEDULE_MAINTENANCE, params={"at": "2026-01-16T06:00:00Z", "slotMin": 30}
    ),
    *[make_ack(status=s) for s in AckStatus],
    make_ack(status=AckStatus.REJECTED, reason="speed above safety limit"),
]


def _ids(case: BaseModel) -> str:
    return type(case).__name__


@pytest.mark.parametrize("model", CASES, ids=_ids)
def test_roundtrip_json(model: BaseModel) -> None:
    raw = model.model_dump_json()
    restored = type(model).model_validate_json(raw)
    assert restored == model
    assert restored.model_dump_json() == raw


@pytest.mark.parametrize("model", CASES, ids=_ids)
def test_roundtrip_python_dict(model: BaseModel) -> None:
    restored = type(model).model_validate(model.model_dump(mode="json"))
    assert restored == model


def _all_keys(obj: object) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(k)
            keys.extend(_all_keys(v))
    elif isinstance(obj, list):
        for v in obj:
            keys.extend(_all_keys(v))
    return keys


@pytest.mark.parametrize("model", [make_reading(), make_prediction(), make_command(), make_ack()])
def test_wire_keys_are_camel_case(model: BaseModel) -> None:
    payload = json.loads(model.model_dump_json())
    # probs keys are class names (NORMAL, ...) and params keys are free-form.
    payload.pop("probs", None)
    payload.pop("params", None)
    for key in _all_keys(payload):
        assert CAMEL.match(key), key


def test_reading_wire_format() -> None:
    payload = json.loads(make_reading().model_dump_json())
    assert payload["schemaVersion"] == 2
    assert payload["machineId"] == "CNC-01"
    assert payload["runTime"] == 3600.0
    assert payload["ts"] == "2026-01-15T08:30:00.123000Z"
    assert set(payload) == {
        "schemaVersion",
        "machineId",
        "seq",
        "ts",
        "state",
        "speed",
        "temp",
        "vib",
        "load",
        "current",
        "runTime",
        "quality",
    }


def test_prediction_wire_format() -> None:
    payload = json.loads(make_prediction().model_dump_json())
    assert set(payload["probs"]) == {"NORMAL", "WARNING", "FAULT"}
    assert payload["horizonS"] == 30.0
    assert payload["ttfS"] == 120.0
    assert payload["ttfInterval"] == [90.0, 160.0]
    assert payload["conformalSet"] == ["NORMAL", "WARNING"]
    assert payload["modelVersion"] == "xgb-2026.01.15-abc123"
    assert payload["attributions"][0] == {
        "sensor": "vib",
        "part": "spindle_bearing",
        "weight": 0.6,
    }


def test_camel_case_json_input_accepted() -> None:
    raw = make_reading().model_dump_json()
    assert Reading.model_validate_json(raw).machine_id == "CNC-01"


def test_timestamps_normalized_to_utc() -> None:
    cet = timezone(timedelta(hours=1))
    r = make_reading(ts=datetime(2026, 1, 15, 9, 30, tzinfo=cet))
    assert r.ts == datetime(2026, 1, 15, 8, 30, tzinfo=UTC)
    assert r.ts.utcoffset() == timedelta(0)
