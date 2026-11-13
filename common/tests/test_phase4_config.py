"""engine.yaml, api.yaml, commands.yaml and the Phase 4 settings (line file, API keys)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from twin_common.config import (
    ConfigError,
    TwinSettings,
    load_api_config,
    load_commands_config,
    load_engine_config,
    load_model,
)
from twin_common.config.commands import CommandsConfig
from twin_common.config.engine import AlertKind, EngineConfig
from twin_common.enums import CommandType, MachineKind, MachineState, Role

CONFIG = Path(__file__).resolve().parents[2] / "config"


def _write(tmp: Path, name: str, data: Any) -> Path:
    p = tmp / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _raw(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))
    return data


def test_shipped_configs_load() -> None:
    engine = load_engine_config(CONFIG / "engine.yaml")
    assert set(engine.transitions) == set(MachineState)
    assert set(engine.alerts.kinds) == set(AlertKind)
    assert engine.allowed(MachineState.RUNNING, MachineState.RUNNING)
    assert not engine.allowed(MachineState.IDLE, MachineState.FAULT)
    assert not engine.allowed(MachineState.OFFLINE, MachineState.STALE)
    api = load_api_config(CONFIG / "api.yaml")
    assert api.host == "127.0.0.1"  # loopback by default
    assert "http://localhost:5173" in api.cors_origins
    cmds = load_commands_config(CONFIG / "commands.yaml")
    assert set(cmds.limits) == set(MachineKind)
    assert all(CommandType.STOP not in r.types for r in cmds.rules)


def test_engine_requires_every_state_and_kind(tmp_path: Path) -> None:
    data = _raw("engine.yaml")
    del data["transitions"]["STALE"]
    with pytest.raises(ConfigError, match="missing"):
        load_model(_write(tmp_path, "e.yaml", data), EngineConfig)
    data = _raw("engine.yaml")
    data["transitions"]["IDLE"].append("IDLE")
    with pytest.raises(ConfigError, match="lists itself"):
        load_model(_write(tmp_path, "e.yaml", data), EngineConfig)
    data = _raw("engine.yaml")
    del data["alerts"]["kinds"]["line_stoppage"]
    with pytest.raises(ConfigError, match="line_stoppage"):
        load_model(_write(tmp_path, "e.yaml", data), EngineConfig)


def test_commands_config_rejects_rules_against_stop(tmp_path: Path) -> None:
    data = _raw("commands.yaml")
    data["rules"].append(
        {"types": ["STOP"], "forbidden_states": ["OFFLINE"], "reason": "should not be allowed"}
    )
    with pytest.raises(ConfigError, match="fail-safe"):
        load_model(_write(tmp_path, "c.yaml", data), CommandsConfig)


def test_commands_rule_needs_exactly_one_condition(tmp_path: Path) -> None:
    for extra in ({}, {"forbidden_states": ["FAULT"], "required_states": ["IDLE"]}):
        data = _raw("commands.yaml")
        data["rules"] = [{"types": ["START"], "reason": "r", **extra}]
        with pytest.raises(ConfigError, match="exactly one"):
            load_model(_write(tmp_path, "c.yaml", data), CommandsConfig)
    data = _raw("commands.yaml")
    del data["limits"]["CNC"]
    with pytest.raises(ConfigError, match="CNC"):
        load_model(_write(tmp_path, "c.yaml", data), CommandsConfig)


def test_line_file_and_api_keys_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LINETWIN_LINE_FILE", "LINETWIN_API_KEYS_OPERATOR", "LINETWIN_API_KEYS_VIEWER"):
        monkeypatch.delenv(var, raising=False)
    s = TwinSettings(_env_file=tmp_path / "missing.env")
    assert s.line_config_path == Path("config") / "line.yaml"
    assert s.api_keys() == []
    env = tmp_path / ".env"
    env.write_text(
        "LINETWIN_LINE_FILE=line_single.yaml\n"
        "LINETWIN_API_KEYS_OPERATOR=alice:op-key-1234\n"
        "LINETWIN_API_KEYS_VIEWER=view-key-1234,bob:view-key-5678\n",
        encoding="utf-8",
    )
    s = TwinSettings(_env_file=env)
    assert s.line_config_path.name == "line_single.yaml"
    keys = s.api_keys()
    assert [(k.principal, k.role) for k in keys] == [
        ("alice", Role.OPERATOR),
        ("viewer-1", Role.VIEWER),
        ("bob", Role.VIEWER),
    ]
    assert "op-key-1234" not in repr(s)  # SecretStr hides the keys


def test_env_example_has_empty_key_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LINETWIN_API_KEYS_OPERATOR", "LINETWIN_API_KEYS_VIEWER"):
        monkeypatch.delenv(var, raising=False)
    example = CONFIG.parent / ".env.example"
    text = example.read_text(encoding="utf-8")
    assert "LINETWIN_API_KEYS_OPERATOR=" in text
    assert "LINETWIN_API_KEYS_VIEWER=" in text
    assert TwinSettings(_env_file=example).api_keys() == []
