from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def line_dict() -> dict[str, Any]:
    """A fresh, mutable copy of the real config/line.yaml."""
    data: dict[str, Any] = yaml.safe_load((CONFIG_DIR / "line.yaml").read_text(encoding="utf-8"))
    return data


@pytest.fixture
def sources_dict() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CONFIG_DIR / "sources.yaml").read_text(encoding="utf-8"))
    return data


@pytest.fixture
def write_yaml(tmp_path: Path) -> Callable[[dict[str, Any], str], Path]:
    def _write(data: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _write
