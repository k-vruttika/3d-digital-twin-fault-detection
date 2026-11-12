"""Validator: canonical dict -> Reading, plus line-config checks (machine known, sensor ranges).

Schema errors, non-finite numbers and unknown machines always reject. Values outside the
line catalog's physical range are either kept with ``quality=SUSPECT`` or rejected, depending
on ``ingest.out_of_range``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from twin_backend.ingest.errors import RejectedMessageError
from twin_backend.ingest.normalizer import machine_id_of
from twin_common.config.line import LineConfig
from twin_common.contracts import Reading
from twin_common.enums import Quality, QuarantineReason, SensorName

STAGE = "validate"
_NUMERIC_KEYS: tuple[str, ...] = (*(s.value for s in SensorName), "runTime")
_MAX_ERRORS_SHOWN = 5


def _summarize(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False)[:_MAX_ERRORS_SHOWN]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    more = exc.error_count() - len(parts)
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


class Validator:
    def __init__(self, line: LineConfig, out_of_range: Literal["suspect", "quarantine"]) -> None:
        self.line = line
        self.out_of_range = out_of_range
        self._machines = {m.id: m for m in line.machines}

    def validate(self, data: Mapping[str, Any]) -> Reading:
        mid = machine_id_of(data)
        for key in _NUMERIC_KEYS:
            value = data.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue  # left to the schema check
            if not math.isfinite(number):
                raise RejectedMessageError(
                    QuarantineReason.NON_FINITE, f"{key}={value!r} is not finite", STAGE, mid
                )
        try:
            reading = Reading.model_validate(dict(data))
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

        violations = []
        for sensor in machine.sensors:
            spec = self.line.sensors[sensor]
            value = getattr(reading, sensor)
            if not spec.min <= value <= spec.max:
                violations.append(f"{sensor}={value} outside [{spec.min}, {spec.max}] {spec.unit}")
        if violations:
            if self.out_of_range == "quarantine":
                raise RejectedMessageError(
                    QuarantineReason.OUT_OF_RANGE, "; ".join(violations), STAGE, reading.machine_id
                )
            return reading.model_copy(update={"quality": Quality.SUSPECT})
        return reading
