"""YAML config loaders that turn every failure into one readable ``ConfigError``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from twin_common.config.api import ApiConfig
from twin_common.config.commands import CommandsConfig
from twin_common.config.engine import EngineConfig
from twin_common.config.line import LineConfig
from twin_common.config.sources import SourcesConfig
from twin_common.config.storage import StorageConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigError(Exception):
    """A config file is missing, unparsable or invalid."""


def _format_loc(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out


def format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source}: invalid configuration ({exc.error_count()} error(s))"]
    for err in exc.errors(include_url=False):
        loc = _format_loc(tuple(err["loc"]))
        msg = str(err["msg"]).removeprefix("Value error, ")
        prefix = f"{loc}: " if loc else ""
        for i, msg_line in enumerate(msg.splitlines()):
            lines.append(f"  - {prefix if i == 0 else ''}{msg_line}")
    return "\n".join(lines)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML syntax error: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_model(path: str | Path, model: type[ModelT]) -> ModelT:
    p = Path(path)
    data = _read_yaml(p)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(str(p), exc)) from exc


def load_line_config(path: str | Path) -> LineConfig:
    return load_model(path, LineConfig)


def load_sources_config(path: str | Path) -> SourcesConfig:
    return load_model(path, SourcesConfig)


def load_storage_config(path: str | Path) -> StorageConfig:
    return load_model(path, StorageConfig)


def load_engine_config(path: str | Path) -> EngineConfig:
    return load_model(path, EngineConfig)


def load_api_config(path: str | Path) -> ApiConfig:
    return load_model(path, ApiConfig)


def load_commands_config(path: str | Path) -> CommandsConfig:
    return load_model(path, CommandsConfig)
