"""Phase 0 integration: shipped config files agree with the shared contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from twin_common import MachineState, Reading, SensorName
from twin_common.config import TwinSettings, load_line_config, load_sources_config
from twin_common.topics import telemetry_topic

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = TwinSettings(config_dir=REPO_ROOT / "config")


def test_configs_load_from_settings() -> None:
    line = load_line_config(SETTINGS.line_config_path)
    sources = load_sources_config(SETTINGS.sources_config_path)
    assert len(line.machines) == 4
    assert sources.active.value == "file"


def test_every_machine_sensor_is_a_reading_field_with_a_part() -> None:
    line = load_line_config(SETTINGS.line_config_path)
    reading_fields = set(Reading.model_fields)
    for machine in line.machines:
        for sensor in machine.sensors:
            assert sensor in reading_fields, (machine.id, sensor)
            assert sensor in {s.value for s in SensorName}
            assert machine.sensor_parts[sensor] in machine.parts


def test_reading_can_be_built_for_every_configured_machine() -> None:
    line = load_line_config(SETTINGS.line_config_path)
    sources = load_sources_config(SETTINGS.sources_config_path)
    assert sources.mqtt is not None
    for i, machine in enumerate(line.machines):
        values = {s: line.sensors[s].min for s in machine.sensors}
        reading = Reading.model_validate(
            {
                "machineId": machine.id,
                "seq": i,
                "ts": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "state": MachineState.IDLE.value,
                "runTime": 0.0,
                **values,
            }
        )
        assert Reading.model_validate_json(reading.model_dump_json()) == reading
        assert telemetry_topic(machine.id, sources.mqtt.topic_prefix).endswith(
            f"/{machine.id}/telemetry"
        )
