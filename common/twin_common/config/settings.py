"""Process settings from environment variables (prefix ``LINETWIN_``)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TwinSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LINETWIN_", extra="ignore")

    config_dir: Path = Path("config")
    log_level: str = "INFO"

    @property
    def line_config_path(self) -> Path:
        return self.config_dir / "line.yaml"

    @property
    def sources_config_path(self) -> Path:
        return self.config_dir / "sources.yaml"
