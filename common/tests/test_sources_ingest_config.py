"""sources.yaml: ingest, reconnect, chaos, bus and field-mapping sections (Phase 2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from twin_common.config import ConfigError, load_sources_config

WriteYaml = Callable[[dict[str, Any], str], Path]


def test_shipped_ingest_defaults(config_dir: Path) -> None:
    cfg = load_sources_config(config_dir / "sources.yaml")
    assert cfg.ingest.offline_after_s == 5.0
    assert cfg.ingest.stale_after_s < cfg.ingest.offline_after_s
    assert cfg.ingest.out_of_range == "suspect"
    assert cfg.chaos.enabled is False
    assert cfg.mqtt is not None
    assert cfg.mqtt.qos == 1  # ADR 0007: telemetry at least once
    assert cfg.file is not None
    assert cfg.file.format == "csv"
    assert cfg.file.mapping.fields["machineId"] == "machine_id"
    assert cfg.mock is not None
    assert cfg.mock.ws_path == "/stream"


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("ingest", "stale_after_s", 6.0, "stale_after_s .* must be < offline_after_s"),
        ("ingest", "seq_reset_threshold", 99999, "seq_reset_threshold .* must be <= dedup_window"),
        ("ingest", "out_of_range", "ignore", "out_of_range"),
        ("ingest", "lateness_window_s", 0, "lateness_window_s"),
        ("reconnect", "initial_s", 60.0, r"initial_s \(60.0\) must be <= max_s"),
        ("reconnect", "jitter", 1.5, "jitter"),
        ("chaos", "duplicate", 1.2, "duplicate"),
        ("chaos", "reorder_max_displacement", 0, "reorder_max_displacement"),
        ("bus", "queue_size", 0, "queue_size"),
    ],
)
def test_invalid_values(
    write_yaml: WriteYaml,
    sources_dict: dict[str, Any],
    section: str,
    key: str,
    value: object,
    match: str,
) -> None:
    sources_dict[section][key] = value
    with pytest.raises(ConfigError, match=match):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))


def test_jsonl_quarantine_needs_path(write_yaml: WriteYaml, sources_dict: dict[str, Any]) -> None:
    sources_dict["ingest"]["quarantine"] = {"kind": "jsonl", "path": None, "max_records": 10}
    with pytest.raises(ConfigError, match="quarantine kind 'jsonl' needs a 'path'"):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))


def test_mapping_rejects_unknown_field_and_unit(
    write_yaml: WriteYaml, sources_dict: dict[str, Any]
) -> None:
    sources_dict["file"]["mapping"] = {
        "fields": {"temperature": "Temp"},
        "units": {"temp": "furlongs", "pressure": "bar"},
    }
    with pytest.raises(ConfigError) as info:
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))
    msg = str(info.value)
    assert "'temperature' is not a Reading field" in msg
    assert "unknown unit 'furlongs'" in msg
    assert "'pressure' is not a sensor" in msg


def test_missing_ingest_section(write_yaml: WriteYaml, sources_dict: dict[str, Any]) -> None:
    del sources_dict["ingest"]
    with pytest.raises(ConfigError, match="ingest"):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))
