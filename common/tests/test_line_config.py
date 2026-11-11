"""line.yaml loading and validation errors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from twin_common.config import ConfigError, LineConfig, load_line_config
from twin_common.enums import MachineKind

WriteYaml = Callable[[dict[str, Any], str], Path]


def _load(write_yaml: WriteYaml, data: dict[str, Any]) -> LineConfig:
    return load_line_config(write_yaml(data, "line.yaml"))


def test_real_line_config_loads(config_dir: Path) -> None:
    cfg = load_line_config(config_dir / "line.yaml")
    assert [m.id for m in cfg.machines] == ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]
    assert [m.kind for m in cfg.machines] == [
        MachineKind.CONVEYOR,
        MachineKind.CNC,
        MachineKind.INSPECTION,
        MachineKind.CONVEYOR,
    ]
    assert [(b.from_machine, b.to_machine) for b in cfg.buffers] == [
        ("CNV-01", "CNC-01"),
        ("CNC-01", "INSP-01"),
        ("INSP-01", "CNV-02"),
    ]
    assert all(b.capacity > 0 for b in cfg.buffers)
    assert (cfg.load_ranges.low.min, cfg.load_ranges.low.max) == (30, 50)
    assert (cfg.load_ranges.medium.min, cfg.load_ranges.medium.max) == (50, 75)
    assert (cfg.load_ranges.high.min, cfg.load_ranges.high.max) == (75, 95)
    cnc = cfg.machine("CNC-01")
    assert cnc.sensor_parts["vib"] == "spindle_bearing"
    assert cnc.sensor_parts["speed"] == "spindle"
    assert cfg.machine("CNV-01").sensor_parts["speed"] == "belt"
    assert cfg.machine("CNV-01").sensor_parts["temp"] == "motor"
    with pytest.raises(KeyError):
        cfg.machine("NOPE")


def test_unknown_sensor(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["machines"][1]["sensors"].append("pressure")
    line_dict["machines"][1]["sensor_parts"]["pressure"] = "spindle"
    with pytest.raises(ConfigError) as exc:
        _load(write_yaml, line_dict)
    msg = str(exc.value)
    assert "line.yaml" in msg
    assert "machines[1] (CNC-01).sensors: unknown sensor 'pressure'" in msg
    assert "known: current, load, speed, temp, vib" in msg


def test_catalog_sensor_must_be_reading_field(
    write_yaml: WriteYaml, line_dict: dict[str, Any]
) -> None:
    line_dict["sensors"]["pressure"] = {"unit": "bar", "min": 0, "max": 10}
    with pytest.raises(ConfigError, match="'pressure' is not a Reading sensor field"):
        _load(write_yaml, line_dict)


@pytest.mark.parametrize(("end", "index"), [("from", 0), ("to", 2)])
def test_buffer_links_unknown_machine(
    write_yaml: WriteYaml, line_dict: dict[str, Any], end: str, index: int
) -> None:
    line_dict["buffers"][index][end] = "LATHE-99"
    with pytest.raises(ConfigError) as exc:
        _load(write_yaml, line_dict)
    assert f"buffers[{index}]" in str(exc.value)
    assert f".{end}: unknown machine 'LATHE-99'" in str(exc.value)


def test_buffer_self_link(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["buffers"][0]["to"] = "CNV-01"
    with pytest.raises(ConfigError, match="links machine 'CNV-01' to itself"):
        _load(write_yaml, line_dict)


@pytest.mark.parametrize("capacity", [-5, 0])
def test_buffer_capacity_must_be_positive(
    write_yaml: WriteYaml, line_dict: dict[str, Any], capacity: int
) -> None:
    line_dict["buffers"][1]["capacity"] = capacity
    with pytest.raises(ConfigError) as exc:
        _load(write_yaml, line_dict)
    msg = str(exc.value)
    assert "buffers[1].capacity" in msg
    assert "greater than or equal to 1" in msg


def test_duplicate_machine_id(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["machines"][3]["id"] = "CNV-01"
    with pytest.raises(ConfigError, match=r"duplicate machine id\(s\): \['CNV-01'\]"):
        _load(write_yaml, line_dict)


def test_sensor_mapped_to_unknown_part(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["machines"][0]["sensor_parts"]["vib"] = "gearbox"
    with pytest.raises(ConfigError, match=r"part 'gearbox' .* is not one of this machine's parts"):
        _load(write_yaml, line_dict)


def test_sensor_without_part_mapping(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    del line_dict["machines"][2]["sensor_parts"]["temp"]
    with pytest.raises(ConfigError, match=r"sensors without a part mapping \['temp'\]"):
        _load(write_yaml, line_dict)


def test_load_ranges_overlap(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["load_ranges"]["low"]["max"] = 60
    with pytest.raises(ConfigError, match="load ranges overlap: low ends at 60"):
        _load(write_yaml, line_dict)


def test_load_ranges_gap(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["load_ranges"]["high"]["min"] = 80
    with pytest.raises(ConfigError, match="gap between load ranges"):
        _load(write_yaml, line_dict)


def test_load_range_out_of_bounds(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["load_ranges"]["high"]["max"] = 130
    with pytest.raises(ConfigError, match=r"load_ranges\.high"):
        _load(write_yaml, line_dict)


def test_unknown_top_level_key(write_yaml: WriteYaml, line_dict: dict[str, Any]) -> None:
    line_dict["conveyor_colour"] = "red"
    with pytest.raises(ConfigError, match="conveyor_colour"):
        _load(write_yaml, line_dict)


def test_multiple_errors_reported_together(
    write_yaml: WriteYaml, line_dict: dict[str, Any]
) -> None:
    line_dict["machines"][0]["sensors"].append("humidity")
    line_dict["machines"][0]["sensor_parts"]["humidity"] = "motor"
    line_dict["buffers"][2]["to"] = "GHOST"
    with pytest.raises(ConfigError) as exc:
        _load(write_yaml, line_dict)
    assert "humidity" in str(exc.value)
    assert "GHOST" in str(exc.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_line_config(tmp_path / "nope.yaml")


def test_yaml_syntax_error(tmp_path: Path) -> None:
    path = tmp_path / "line.yaml"
    path.write_text("machines: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML syntax error"):
        load_line_config(path)


def test_top_level_not_mapping(tmp_path: Path) -> None:
    path = tmp_path / "line.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_line_config(path)
