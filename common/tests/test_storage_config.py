"""config/storage.yaml schema: shipped file loads, cross-field rules give readable errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from twin_common.config import ConfigError, TwinSettings, load_storage_config

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "storage.yaml"


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "storage.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def shipped() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    return data


def test_shipped_config_loads_with_sqlite_default() -> None:
    cfg = load_storage_config(SHIPPED)
    assert cfg.backend == "sqlite"  # minimal mode default
    assert cfg.timescale is not None
    assert cfg.timescale.aggregates.m1.start_offset_s >= cfg.timescale.max_late_arrival_s
    assert TwinSettings().storage_config_path == Path("config") / "storage.yaml"


def test_timescale_backend_needs_its_section(tmp_path: Path) -> None:
    data = shipped()
    data["backend"] = "timescale"
    del data["timescale"]
    with pytest.raises(ConfigError, match="no 'timescale:' section"):
        load_storage_config(write(tmp_path, data))


def test_refresh_window_must_cover_late_arrivals(tmp_path: Path) -> None:
    data = shipped()
    data["timescale"]["aggregates"]["m1"]["start_offset_s"] = 600
    with pytest.raises(ConfigError, match=r"aggregates\.m1\.start_offset_s"):
        load_storage_config(write(tmp_path, data))


def test_end_offset_below_start_offset(tmp_path: Path) -> None:
    data = shipped()
    data["timescale"]["aggregates"]["m10"]["end_offset_s"] = 20000
    with pytest.raises(ConfigError, match="end_offset_s"):
        load_storage_config(write(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("writer", "batch_size"), 0, "batch_size"),
        (("writer", "queue_size"), 0, "queue_size"),
        (("snapshots", "interval_s"), 0, "interval_s"),
        (("timescale", "pool_min"), 9, "pool_min"),
        (("backend",), "mongodb", "backend"),
    ],
)
def test_field_errors(tmp_path: Path, path: tuple[str, ...], value: Any, match: str) -> None:
    data = shipped()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=match):
        load_storage_config(write(tmp_path, data))
