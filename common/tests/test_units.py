"""Unit conversion table used by the ingest normalizer."""

from __future__ import annotations

import math

import pytest

from twin_common.units import KNOWN_UNITS, UnitConversionError, converter


@pytest.mark.parametrize(
    ("src", "dst", "value", "expected"),
    [
        ("degF", "degC", 212.0, 100.0),
        ("degF", "degC", 32.0, 0.0),
        ("K", "degC", 273.15, 0.0),
        ("rad/s", "RPM", 2 * math.pi, 60.0),
        ("Hz", "RPM", 25.0, 1500.0),
        ("in/s", "mm/s", 1.0, 25.4),
        ("m/s", "mm/s", 0.002, 2.0),
        ("mA", "A", 1500.0, 1.5),
        ("fraction", "%", 0.62, 62.0),
        ("RPM", "RPM", 1234.5, 1234.5),
    ],
)
def test_conversions(src: str, dst: str, value: float, expected: float) -> None:
    assert converter(src, dst)(value) == pytest.approx(expected)


def test_unknown_conversion_lists_supported() -> None:
    with pytest.raises(UnitConversionError, match=r"no conversion from 'mm/s' to 'degC'.*degF"):
        converter("mm/s", "degC")


def test_known_units_include_canonical() -> None:
    assert {"RPM", "degC", "mm/s", "%", "A"} <= KNOWN_UNITS
