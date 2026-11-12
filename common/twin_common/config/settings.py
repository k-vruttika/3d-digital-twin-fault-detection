"""Process settings from environment variables (prefix ``LINETWIN_``) and ``.env``.

The same ``.env`` file (see ``.env.example``) is read by Docker Compose, so the database
credentials used by the services and by the code always agree. Real environment variables
take precedence over ``.env``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TwinSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LINETWIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_dir: Path = Path("config")
    log_level: str = "INFO"

    # TimescaleDB / PostgreSQL (docker-compose.yml defaults; override in .env).
    pg_host: str = "127.0.0.1"
    pg_port: int = Field(default=5432, ge=1, le=65535)
    pg_db: str = "linetwin"
    pg_user: str = "linetwin"
    pg_password: SecretStr = SecretStr("linetwin-dev")

    @property
    def line_config_path(self) -> Path:
        return self.config_dir / "line.yaml"

    @property
    def sources_config_path(self) -> Path:
        return self.config_dir / "sources.yaml"

    @property
    def storage_config_path(self) -> Path:
        return self.config_dir / "storage.yaml"

    def _pg_location(self) -> str:
        user = quote(self.pg_user, safe="")
        password = quote(self.pg_password.get_secret_value(), safe="")
        return f"{user}:{password}@{self.pg_host}:{self.pg_port}/{quote(self.pg_db, safe='')}"

    @property
    def pg_dsn(self) -> str:
        """libpq/asyncpg connection string (contains the password; never log it)."""
        return f"postgresql://{self._pg_location()}"

    @property
    def database_url(self) -> str:
        """SQLAlchemy async URL (contains the password; never log it)."""
        return f"postgresql+asyncpg://{self._pg_location()}"
