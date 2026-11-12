"""Unit conversions used by the ingest normalizer (source unit -> canonical unit).

Canonical units are those of the line sensor catalog (``config/line.yaml``): RPM, degC, mm/s,
%, A. The factors below are physical definitions, not tunable thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Converter = Callable[[float], float]

_IN_TO_MM = 25.4
_RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
_KELVIN_OFFSET = 273.15


def _identity(v: float) -> float:
    return v


def _scale(factor: float) -> Converter:
    def convert(v: float) -> float:
        return v * factor

    return convert


# (source unit, canonical unit) -> converter
CONVERSIONS: dict[tuple[str, str], Converter] = {
    ("degF", "degC"): lambda v: (v - 32.0) * 5.0 / 9.0,
    ("K", "degC"): lambda v: v - _KELVIN_OFFSET,
    ("rad/s", "RPM"): _scale(_RAD_S_TO_RPM),
    ("Hz", "RPM"): _scale(60.0),
    ("in/s", "mm/s"): _scale(_IN_TO_MM),
    ("m/s", "mm/s"): _scale(1000.0),
    ("mA", "A"): _scale(1e-3),
    ("fraction", "%"): _scale(100.0),
}

KNOWN_UNITS: frozenset[str] = frozenset(u for pair in CONVERSIONS for u in pair)


class UnitConversionError(ValueError):
    """No conversion is defined between two units."""


def converter(source_unit: str, canonical_unit: str) -> Converter:
    """Return a function converting ``source_unit`` values into ``canonical_unit``."""
    if source_unit == canonical_unit:
        return _identity
    try:
        return CONVERSIONS[(source_unit, canonical_unit)]
    except KeyError:
        supported = sorted(f"{a}->{b}" for a, b in CONVERSIONS if b == canonical_unit)
        raise UnitConversionError(
            f"no conversion from '{source_unit}' to '{canonical_unit}' "
            f"(supported into {canonical_unit}: {', '.join(supported) or 'none'})"
        ) from None
