"""Schema for ``config/whatif.yaml``: what-if projections (Phase 11 Part C, ADR 0031)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.config.loader import load_model


class WhatIfConfig(ConfigModel):
    horizon_default_s: int = Field(gt=0, description="projection length when none is requested")
    horizon_max_s: int = Field(gt=0, description="longest projection a request may ask for")
    history_s: int = Field(ge=1, description="recent readings sent along (fills the model window)")
    request_timeout_s: float = Field(gt=0, description="the simulator fork")
    rows_per_step: int = Field(ge=1, description="readings per machine per batch to the workers")
    ghost_horizons_s: list[int] = Field(
        min_length=1, description="projected states shown next to 'now'"
    )

    @model_validator(mode="after")
    def _consistent(self) -> WhatIfConfig:
        if self.horizon_default_s > self.horizon_max_s:
            raise ValueError("horizon_default_s must not exceed horizon_max_s")
        if any(h <= 0 or h > self.horizon_max_s for h in self.ghost_horizons_s):
            raise ValueError("ghost_horizons_s must be within (0, horizon_max_s]")
        return self


def load_whatif_config(path: str | Path) -> WhatIfConfig:
    return load_model(path, WhatIfConfig)
