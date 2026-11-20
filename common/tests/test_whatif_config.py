"""config/whatif.yaml (Phase 11 Part C, ADR 0031)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from twin_common.config.loader import read_yaml
from twin_common.config.whatif import WhatIfConfig, load_whatif_config

CONFIG = Path(__file__).resolve().parents[2] / "config"


def test_the_shipped_config_loads() -> None:
    cfg = load_whatif_config(CONFIG / "whatif.yaml")
    assert cfg.horizon_default_s <= cfg.horizon_max_s
    assert cfg.ghost_horizons_s == [10, 30]


@pytest.mark.parametrize(
    "change",
    [
        {"horizon_default_s": 999},  # above the maximum
        {"ghost_horizons_s": [10, 9999]},  # beyond the maximum
        {"ghost_horizons_s": []},
        {"history_s": 0},
        {"unknown_key": 1},
    ],
)
def test_inconsistent_settings_are_rejected(change: dict[str, object]) -> None:
    data = read_yaml(CONFIG / "whatif.yaml") | change
    with pytest.raises(ValidationError):
        WhatIfConfig.model_validate(data)
