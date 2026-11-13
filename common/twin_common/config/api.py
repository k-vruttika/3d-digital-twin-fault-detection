"""Schema for ``config/api.yaml``: HTTP server, CORS, WebSocket streaming (Phase 4)."""

from __future__ import annotations

from pydantic import Field

from twin_common.config.line import ConfigModel


class WsConfig(ConfigModel):
    client_queue_size: int = Field(
        ge=1, description="ordered messages (states, alerts, commands) kept per slow client"
    )
    send_timeout_s: float = Field(gt=0, description="a send taking longer closes the client")
    auth_timeout_s: float = Field(gt=0, description="wait this long for the auth message")
    hub_queue_size: int = Field(ge=1, description="bus subscription bound of the WS hub")


class ApiConfig(ConfigModel):
    host: str = Field(min_length=1, description="bind address; keep loopback unless keys are set")
    port: int = Field(ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    max_history_rows: int = Field(ge=1, description="raw history / export row limit")
    ws: WsConfig
