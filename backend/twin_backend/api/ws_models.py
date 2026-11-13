"""Typed WebSocket frames of ``/stream`` (ADR 0013, ADR 0017).

Every frame is ``{"type": ..., "data": ...}`` (client frames carry their fields at the top
level). The models here are the single definition of the protocol: the hub builds its control
frames (auth, subscribed, snapshot, error, dropped, pong) through them, and they are merged into
the OpenAPI components (``WsClientMessage`` / ``WsServerMessage``), from which the frontend's
TypeScript types are generated. Streamed frames (reading, state, ...) are serialised once per
message by the hub; their shape is the same contract model wrapped in the envelope below.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel

from twin_backend.engine.data_quality import DataQualityReport
from twin_backend.engine.line_state import MachineView
from twin_backend.engine.twin import LineViewModel
from twin_common.contracts import Alert, Command, CommandAck, Prediction, Reading, StateChange
from twin_common.enums import Role

Channel = Literal["readings", "states", "line", "dataQuality", "alerts", "commands", "predictions"]
CHANNELS: tuple[Channel, ...] = (
    "readings",
    "states",
    "line",
    "dataQuality",
    "alerts",
    "commands",
    "predictions",
)


class _Model(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class _ClientModel(_Model):
    model_config = ConfigDict(extra="ignore")


class _ServerModel(_Model):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- client -> server
class WsAuthRequest(_ClientModel):
    """First message of every connection. ``apiKey`` may be omitted for anonymous read."""

    type: Literal["auth"]
    api_key: str | None = None


class WsSubscribeRequest(_ClientModel):
    """Select machines (``null`` = all) and channels (default: all); repeat to change."""

    type: Literal["subscribe"]
    machines: list[str] | None = None
    channels: list[Channel] = Field(default_factory=lambda: list(CHANNELS))


class WsPingRequest(_ClientModel):
    type: Literal["ping"]


WsClientMessage = Annotated[
    WsAuthRequest | WsSubscribeRequest | WsPingRequest, Field(discriminator="type")
]


# ---------------------------------------------------------------- server -> client: payloads
class WsAuthInfo(_ServerModel):
    principal: str
    role: Role


class WsSubscribedInfo(_ServerModel):
    machines: list[str] | None = None
    channels: list[Channel]


class WsSnapshotData(_ServerModel):
    as_of: datetime | None = None
    machines: dict[str, MachineView]
    line: LineViewModel
    data_quality: dict[str, DataQualityReport]
    alerts: list[Alert]


class WsDroppedInfo(_ServerModel):
    count: int


class WsErrorInfo(_ServerModel):
    reason: str
    channels: list[str] | None = None
    machines: list[str] | None = None


class WsEmpty(_ServerModel):
    pass


# ---------------------------------------------------------------- server -> client: frames
class WsAuthMessage(_ServerModel):
    type: Literal["auth"]
    data: WsAuthInfo


class WsSubscribedMessage(_ServerModel):
    type: Literal["subscribed"]
    data: WsSubscribedInfo


class WsSnapshotMessage(_ServerModel):
    type: Literal["snapshot"]
    data: WsSnapshotData


class WsReadingMessage(_ServerModel):
    type: Literal["reading"]
    data: Reading


class WsStateMessage(_ServerModel):
    type: Literal["state"]
    data: StateChange


class WsLineMessage(_ServerModel):
    type: Literal["line"]
    data: LineViewModel


class WsDataQualityMessage(_ServerModel):
    type: Literal["dataQuality"]
    data: DataQualityReport


class WsAlertMessage(_ServerModel):
    type: Literal["alert"]
    data: Alert


class WsCommandMessage(_ServerModel):
    type: Literal["command"]
    data: Command


class WsAckMessage(_ServerModel):
    type: Literal["ack"]
    data: CommandAck


class WsPredictionMessage(_ServerModel):
    type: Literal["prediction"]
    data: Prediction


class WsDroppedMessage(_ServerModel):
    type: Literal["dropped"]
    data: WsDroppedInfo


class WsErrorMessage(_ServerModel):
    type: Literal["error"]
    data: WsErrorInfo


class WsPongMessage(_ServerModel):
    type: Literal["pong"]
    data: WsEmpty


WsServerMessage = Annotated[
    WsAuthMessage
    | WsSubscribedMessage
    | WsSnapshotMessage
    | WsReadingMessage
    | WsStateMessage
    | WsLineMessage
    | WsDataQualityMessage
    | WsAlertMessage
    | WsCommandMessage
    | WsAckMessage
    | WsPredictionMessage
    | WsDroppedMessage
    | WsErrorMessage
    | WsPongMessage,
    Field(discriminator="type"),
]

CLIENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(WsClientMessage)
SERVER_ADAPTER: TypeAdapter[Any] = TypeAdapter(WsServerMessage)


def frame(message: _ServerModel) -> str:
    """Serialise a server frame (compact JSON, camelCase, nulls kept)."""
    return message.model_dump_json(by_alias=True)


REF_TEMPLATE = "#/components/schemas/{model}"


def openapi_components() -> dict[str, dict[str, Any]]:
    """JSON schemas of both unions and every model they use, as OpenAPI components."""
    out: dict[str, dict[str, Any]] = {}
    for name, adapter, mode in (
        ("WsClientMessage", CLIENT_ADAPTER, "validation"),
        ("WsServerMessage", SERVER_ADAPTER, "serialization"),
    ):
        schema = adapter.json_schema(ref_template=REF_TEMPLATE, mode=mode)  # type: ignore[arg-type]
        defs = schema.pop("$defs", {})
        out.update(defs)
        out[name] = schema
    return out
