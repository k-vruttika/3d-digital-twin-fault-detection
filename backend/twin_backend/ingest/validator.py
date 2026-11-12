"""Validator: canonical dict -> Reading, plus line-config checks (machine known, sensor ranges).

Missing sensor values (absent key, ``null``, NaN, +/-inf or an empty string) become ``None``.
A reading with some of its machine's sensors missing is kept with ``quality=SUSPECT``; one with
none of them present is rejected (``NO_SENSOR_DATA``). Decode/schema errors, a non-finite
``runTime`` and unknown machines always reject. Values outside the line catalog's physical
range are kept as SUSPECT or rejected, depending on ``ingest.out_of_range`` (ADR 0007).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from twin_backend.ingest.errors import RejectedMessageError
from twin_backend.ingest.normalizer import is_missing, machine_id_of
from twin_common.config.line import LineConfig
from twin_common.contracts import Reading
from twin_common.enums import Quality, QuarantineReason, SensorName

STAGE = "validate"
_SENSOR_KEYS: tuple[str, ...] = tuple(s.value for s in SensorName)
_MAX_ERRORS_SHOWN = 5


def _summarize(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False)[:_MAX_ERRORS_SHOWN]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    more = exc.error_count() - len(parts)
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


@dataclass(frozen=True, slots=True)
class Validated:
    reading: Reading
    missing: tuple[str, ...] = ()  # machine sensors without a value
    out_of_range: tuple[str, ...] = ()


class Validator:
    def __init__(self, line: LineConfig, out_of_range: Literal["suspect", "quarantine"]) -> None:
        self.line = line
        self.out_of_range = out_of_range
        self._machines = {m.id: m for m in line.machines}

    def validate(self, data: Mapping[str, Any]) -> Reading:
        return self.check(data).reading

    def check(self, data: Mapping[str, Any]) -> Validated:
        mid = machine_id_of(data)
        clean = dict(data)
        for key in _SENSOR_KEYS:
            if key in clean and is_missing(clean[key]):
                clean[key] = None
        run_time = clean.get("runTime")
        if run_time is not None and not isinstance(run_time, bool):
            try:
                finite = math.isfinite(float(run_time))
            except (TypeError, ValueError):
                finite = True  # left to the schema check
            if not finite:
                raise RejectedMessageError(
                    QuarantineReason.NON_FINITE, f"runTime={run_time!r} is not finite", STAGE, mid
                )
        try:
            reading = Reading.model_validate(clean)
        except ValidationError as exc:
            raise RejectedMessageError(
                QuarantineReason.SCHEMA_ERROR, _summarize(exc), STAGE, mid
            ) from exc

        machine = self._machines.get(reading.machine_id)
        if machine is None:
            raise RejectedMessageError(
                QuarantineReason.UNKNOWN_MACHINE,
                f"machine '{reading.machine_id}' is not on line '{self.line.line_id}' "
                f"(known: {', '.join(self._machines)})",
                STAGE,
                reading.machine_id,
            )

        missing = tuple(s for s in machine.sensors if getattr(reading, s) is None)
        if len(missing) == len(machine.sensors):
            raise RejectedMessageError(
                QuarantineReason.NO_SENSOR_DATA,
                f"no valid sensor value (missing: {', '.join(missing)})",
                STAGE,
                reading.machine_id,
            )

        violations: list[str] = []
        flagged: list[str] = []
        for sensor in machine.sensors:
            value = getattr(reading, sensor)
            if value is None:
                continue
            spec = self.line.sensors[sensor]
            if not spec.min <= value <= spec.max:
                flagged.append(sensor)
                violations.append(f"{sensor}={value} outside [{spec.min}, {spec.max}] {spec.unit}")
        if violations and self.out_of_range == "quarantine":
            raise RejectedMessageError(
                QuarantineReason.OUT_OF_RANGE, "; ".join(violations), STAGE, reading.machine_id
            )
        if missing or flagged:
            reading = reading.model_copy(update={"quality": Quality.SUSPECT})
        return Validated(reading, missing, tuple(flagged))
