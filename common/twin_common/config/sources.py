"""Schema for ``config/sources.yaml``: which adapter is active and per-adapter settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.enums import AdapterKind
from twin_common.topics import DEFAULT_TOPIC_PREFIX


class FileSourceConfig(ConfigModel):
    path: str = Field(min_length=1, description="Relative to the repository root")
    format: Literal["csv", "json"]
    replay_speed: float = Field(gt=0, description="1.0 = real time")
    loop: bool = False


class MockSourceConfig(ConfigModel):
    base_url: str = Field(min_length=1)
    ws_path: str = Field(min_length=1)


class MqttSourceConfig(ConfigModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    topic_prefix: str = DEFAULT_TOPIC_PREFIX
    client_id: str = Field(min_length=1)
    qos: Literal[0, 1, 2]
    keepalive_s: int = Field(gt=0)


class OpcUaSourceConfig(ConfigModel):
    endpoint: str = Field(min_length=1)
    namespace: str = Field(min_length=1)


class SourcesConfig(ConfigModel):
    active: AdapterKind
    file: FileSourceConfig | None = None
    mock: MockSourceConfig | None = None
    mqtt: MqttSourceConfig | None = None
    opcua: OpcUaSourceConfig | None = None

    @model_validator(mode="after")
    def _active_configured(self) -> SourcesConfig:
        if getattr(self, self.active.value) is None:
            raise ValueError(
                f"active adapter is '{self.active.value}' but no '{self.active.value}:' "
                "section is configured"
            )
        return self
