"""Normalizer (decode, field mapping, units) and Validator (schema, machine, ranges)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from twin_backend.adapters.base import RawMessage
from twin_backend.clock import FakeClock
from twin_backend.ingest.errors import RejectedMessageError
from twin_backend.ingest.normalizer import Normalizer
from twin_backend.ingest.validator import Validator
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping
from twin_common.enums import AdapterKind, Quality, QuarantineReason
from twin_common.testing import sample_reading

CLOCK = FakeClock()


def raw(payload: Any) -> RawMessage:
    return RawMessage(AdapterKind.FILE, payload, CLOCK.monotonic(), CLOCK.now())


def wire(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = sample_reading().model_dump(mode="json")
    data.update(overrides)
    return data


# -- normalizer ------------------------------------------------------------------------------
def test_identity_mapping_accepts_dict_str_bytes(line_cfg: LineConfig) -> None:
    n = Normalizer(FieldMapping(), line_cfg)
    d = wire()
    assert n.normalize(raw(d)) == d
    assert n.normalize(raw(json.dumps(d))) == d
    assert n.normalize(raw(json.dumps(d).encode())) == d


@pytest.mark.parametrize("payload", [b"\xff\xfe", "{not json", "[1, 2]", '"text"'])
def test_decode_errors(line_cfg: LineConfig, payload: str | bytes) -> None:
    with pytest.raises(RejectedMessageError) as info:
        Normalizer(FieldMapping(), line_cfg).normalize(raw(payload))
    assert info.value.reason is QuarantineReason.DECODE_ERROR


def test_field_mapping_and_drop_unmapped(line_cfg: LineConfig) -> None:
    mapping = FieldMapping(
        fields={"machineId": "machine_id", "runTime": "run_time", "temp": "Temperature"},
        drop_unmapped=True,
    )
    src = wire()
    src["machine_id"] = src.pop("machineId")
    src["run_time"] = src.pop("runTime")
    src["Temperature"] = src.pop("temp")
    src["fault_type"] = "none"  # label column: dropped
    out = Normalizer(mapping, line_cfg).normalize(raw(src))
    assert out == wire()


def test_unmapped_keys_kept_without_drop(line_cfg: LineConfig) -> None:
    out = Normalizer(FieldMapping(), line_cfg).normalize(raw(wire(extra=1)))
    assert out["extra"] == 1  # the validator rejects it as an unknown field


def test_unit_conversion(line_cfg: LineConfig) -> None:
    mapping = FieldMapping(units={"temp": "degF", "vib": "in/s", "current": "mA"})
    out = Normalizer(mapping, line_cfg).normalize(raw(wire(temp=212.0, vib=0.1, current="1500")))
    assert out["temp"] == pytest.approx(100.0)
    assert out["vib"] == pytest.approx(2.54)
    assert out["current"] == pytest.approx(1.5)


def test_unit_conversion_error_is_mapping_error(line_cfg: LineConfig) -> None:
    n = Normalizer(FieldMapping(units={"temp": "degF"}), line_cfg)
    with pytest.raises(RejectedMessageError) as info:
        n.normalize(raw(wire(temp="hot")))
    assert info.value.reason is QuarantineReason.MAPPING_ERROR
    assert info.value.machine_id == "CNC-01"


def test_impossible_unit_conversion_rejected_at_construction(line_cfg: LineConfig) -> None:
    with pytest.raises(ValueError, match=r"mapping.units.temp: no conversion from 'in/s'"):
        Normalizer(FieldMapping(units={"temp": "in/s"}), line_cfg)


# -- validator -------------------------------------------------------------------------------
def test_valid_reading(line_cfg: LineConfig) -> None:
    r = Validator(line_cfg, "suspect").validate(wire())
    assert r == sample_reading()


def test_csv_strings_are_coerced(line_cfg: LineConfig) -> None:
    data = {k: str(v) for k, v in wire().items() if k != "schemaVersion"}  # CSV has none
    data["ts"] = "2026-01-01 00:00:01.000Z"  # pyarrow CSV timestamp format
    r = Validator(line_cfg, "suspect").validate(data)
    assert r.seq == 42
    assert r.ts.isoformat() == "2026-01-01T00:00:01+00:00"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"temp": float("nan")}, QuarantineReason.NON_FINITE),
        ({"speed": "inf"}, QuarantineReason.NON_FINITE),
        ({"runTime": float("-inf")}, QuarantineReason.NON_FINITE),
        ({"seq": "#corrupt#"}, QuarantineReason.SCHEMA_ERROR),
        ({"state": "EXPLODED"}, QuarantineReason.SCHEMA_ERROR),
        ({"ts": "2026-01-01T00:00:00"}, QuarantineReason.SCHEMA_ERROR),  # naive
        ({"vib": -1.0}, QuarantineReason.SCHEMA_ERROR),
        ({"extra": 1}, QuarantineReason.SCHEMA_ERROR),
        ({"machineId": "PRESS-9"}, QuarantineReason.UNKNOWN_MACHINE),
    ],
)
def test_rejections(
    line_cfg: LineConfig, overrides: dict[str, Any], reason: QuarantineReason
) -> None:
    with pytest.raises(RejectedMessageError) as info:
        Validator(line_cfg, "suspect").validate(wire(**overrides))
    assert info.value.reason is reason
    assert info.value.detail


def test_missing_field_is_schema_error(line_cfg: LineConfig) -> None:
    data = wire()
    del data["seq"]
    with pytest.raises(RejectedMessageError, match="SCHEMA_ERROR: seq"):
        Validator(line_cfg, "suspect").validate(data)


def test_out_of_range_suspect_mode(line_cfg: LineConfig) -> None:
    r = Validator(line_cfg, "suspect").validate(wire(temp=180.0))
    assert r.quality is Quality.SUSPECT
    assert r.temp == 180.0


def test_out_of_range_quarantine_mode(line_cfg: LineConfig) -> None:
    with pytest.raises(RejectedMessageError) as info:
        Validator(line_cfg, "quarantine").validate(wire(temp=180.0, current=150.0))
    assert info.value.reason is QuarantineReason.OUT_OF_RANGE
    assert "temp=180.0 outside [-20.0, 150.0] degC" in info.value.detail
    assert "current=150.0" in info.value.detail


def test_single_line_rejects_full_line_machines(single_line_cfg: LineConfig) -> None:
    v = Validator(single_line_cfg, "suspect")
    assert v.validate(wire(machineId="CNV-01")).machine_id == "CNV-01"
    with pytest.raises(RejectedMessageError) as info:
        v.validate(wire(machineId="CNC-01"))
    assert info.value.reason is QuarantineReason.UNKNOWN_MACHINE
    assert "line1_single" in info.value.detail
