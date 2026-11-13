"""Process settings from environment variables (prefix ``LINETWIN_``) and ``.env``.

The same ``.env`` file (see ``.env.example``) is read by Docker Compose, so the database
credentials used by the services and by the code always agree. Real environment variables
take precedence over ``.env``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from twin_common.enums import Role


@dataclass(frozen=True, slots=True)
class ApiKey:
    """One configured API key. ``principal`` is recorded as ``issuedBy`` / ``ackedBy``."""

    principal: str
    role: Role
    secret: SecretStr


def parse_api_keys(raw: str, role: Role) -> list[ApiKey]:
    """``"alice:key1,bob:key2"`` or ``"key1,key2"`` -> keys; without a name the principal is
    ``<role>-<n>``. Empty entries are ignored."""
    keys: list[ApiKey] = []
    for n, entry in enumerate((e.strip() for e in raw.split(",")), start=1):
        if not entry:
            continue
        name, sep, secret = entry.partition(":")
        if not sep:
            name, secret = f"{role.value}-{n}", entry
        name, secret = name.strip(), secret.strip()
        if not name or not secret:
            raise ValueError(f"malformed API key entry #{n} for role {role.value}")
        keys.append(ApiKey(principal=name, role=role, secret=SecretStr(secret)))
    return keys


class TwinSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LINETWIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_dir: Path = Path("config")
    # Line layout file inside config_dir (line.yaml = full line, line_single.yaml = one station).
    line_file: str = "line.yaml"
    log_level: str = "INFO"

    # API keys (never commit them): "name:key,name2:key2" per role. Empty = no keys of that role.
    api_keys_operator: SecretStr = SecretStr("")
    api_keys_viewer: SecretStr = SecretStr("")

    # TimescaleDB / PostgreSQL (docker-compose.yml defaults; override in .env).
    pg_host: str = "127.0.0.1"
    pg_port: int = Field(default=5432, ge=1, le=65535)
    pg_db: str = "linetwin"
    pg_user: str = "linetwin"
    pg_password: SecretStr = SecretStr("linetwin-dev")

    @property
    def line_config_path(self) -> Path:
        return self.config_dir / self.line_file

    def api_keys(self) -> list[ApiKey]:
        """All configured API keys (operator keys first)."""
        return parse_api_keys(
            self.api_keys_operator.get_secret_value(), Role.OPERATOR
        ) + parse_api_keys(self.api_keys_viewer.get_secret_value(), Role.VIEWER)

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
