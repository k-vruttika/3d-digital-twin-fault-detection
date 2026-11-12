"""Normalizer: decode a raw payload, apply the per-adapter field mapping and unit conversion.

The output is a dict keyed by canonical Reading wire names (camelCase), ready for the
validator. Values are not type-checked here (CSV values are still strings).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from twin_backend.adapters.base import Payload, RawMessage
from twin_backend.ingest.errors import RejectedMessageError
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping
from twin_common.contracts import Reading
from twin_common.enums import QuarantineReason
from twin_common.units import Converter, UnitConversionError, converter

STAGE = "normalize"

READING_WIRE_KEYS: tuple[str, ...] = tuple(
    (f.alias or name) for name, f in Reading.model_fields.items()
)


def machine_id_of(data: Mapping[str, Any]) -> str | None:
    value = data.get("machineId")
    return value if isinstance(value, str) and value else None


class Normalizer:
    """Maps one adapter's payloads onto canonical Reading keys and units."""

    def __init__(self, mapping: FieldMapping, line: LineConfig) -> None:
        self.mapping = mapping
        # canonical key -> source key
        self._source_key = {k: mapping.fields.get(k, k) for k in READING_WIRE_KEYS}
        self._converters: dict[str, Converter] = {}
        for sensor, unit in mapping.units.items():
            spec = line.sensors.get(sensor)
            if spec is None:
                raise ValueError(f"mapping.units: sensor '{sensor}' is not in the line catalog")
            try:
                self._converters[sensor] = converter(unit, spec.unit)
            except UnitConversionError as exc:
                raise ValueError(f"mapping.units.{sensor}: {exc}") from exc

    @staticmethod
    def decode(payload: Payload) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            return payload
        try:
            text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RejectedMessageError(
                QuarantineReason.DECODE_ERROR, f"not valid UTF-8 JSON: {exc}", STAGE
            ) from exc
        if not isinstance(data, dict):
            raise RejectedMessageError(
                QuarantineReason.DECODE_ERROR,
                f"expected a JSON object, got {type(data).__name__}",
                STAGE,
            )
        return data

    def normalize(self, raw: RawMessage) -> dict[str, Any]:
        src = self.decode(raw.payload)
        out: dict[str, Any] = {}
        consumed: set[str] = set()
        for canonical, source_key in self._source_key.items():
            if source_key in src:
                out[canonical] = src[source_key]
                consumed.add(source_key)
        if not self.mapping.drop_unmapped:
            for key, value in src.items():
                if key not in consumed:
                    out.setdefault(key, value)
        for sensor, convert in self._converters.items():
            if sensor not in out:
                continue
            try:
                out[sensor] = convert(float(out[sensor]))
            except (TypeError, ValueError) as exc:
                raise RejectedMessageError(
                    QuarantineReason.MAPPING_ERROR,
                    f"cannot convert {sensor}={out[sensor]!r}: {exc}",
                    STAGE,
                    machine_id=machine_id_of(out),
                ) from exc
        return out
