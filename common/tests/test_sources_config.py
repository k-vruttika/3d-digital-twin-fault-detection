"""sources.yaml loading and validation errors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from twin_common.config import ConfigError, TwinSettings, load_sources_config
from twin_common.enums import AdapterKind

WriteYaml = Callable[[dict[str, Any], str], Path]


def test_real_sources_config_loads_and_defaults_to_file(config_dir: Path) -> None:
    cfg = load_sources_config(config_dir / "sources.yaml")
    # Minimal mode: the shipped default must not need a broker or database.
    assert cfg.active is AdapterKind.FILE
    assert cfg.file is not None
    assert cfg.mqtt is not None
    assert cfg.mqtt.topic_prefix == "plant/line1"


def test_active_adapter_requires_section(
    write_yaml: WriteYaml, sources_dict: dict[str, Any]
) -> None:
    sources_dict["active"] = "opcua"
    del sources_dict["opcua"]
    with pytest.raises(ConfigError, match="active adapter is 'opcua' but no 'opcua:' section"):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))


def test_unknown_adapter(write_yaml: WriteYaml, sources_dict: dict[str, Any]) -> None:
    sources_dict["active"] = "kafka"
    with pytest.raises(ConfigError, match="active"):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))


def test_invalid_mqtt_port(write_yaml: WriteYaml, sources_dict: dict[str, Any]) -> None:
    sources_dict["mqtt"]["port"] = 70000
    with pytest.raises(ConfigError, match=r"mqtt\.port"):
        load_sources_config(write_yaml(sources_dict, "sources.yaml"))


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LINETWIN_CONFIG_DIR", str(tmp_path))
    s = TwinSettings()
    assert s.line_config_path == tmp_path / "line.yaml"
    assert s.sources_config_path == tmp_path / "sources.yaml"
