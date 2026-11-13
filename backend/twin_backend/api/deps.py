"""Shared request context and dependencies (auth, store access)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from twin_backend.api.auth import NO_OPERATOR_KEYS, Authenticator, Principal
from twin_backend.api.ws import WsHub
from twin_backend.runtime import Runtime
from twin_backend.storage.base import EventStore
from twin_backend.storage.service import StorageService
from twin_common.enums import Role

API_KEY_HEADER = "X-API-Key"
_api_key = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="API key (viewer or operator). Optional for reads when the server runs "
    "without keys on a loopback address.",
)


@dataclass
class ApiContext:
    runtime: Runtime
    auth: Authenticator
    hub: WsHub


def get_ctx(request: Request) -> ApiContext:
    ctx: ApiContext = request.app.state.ctx
    return ctx


Ctx = Annotated[ApiContext, Depends(get_ctx)]


def get_principal(ctx: Ctx, key: Annotated[str | None, Security(_api_key)]) -> Principal:
    principal = ctx.auth.resolve(key)
    if principal is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    return principal


Viewer = Annotated[Principal, Depends(get_principal)]


def require_operator(ctx: Ctx, principal: Viewer) -> Principal:
    if not principal.has(Role.OPERATOR):
        detail = (
            NO_OPERATOR_KEYS
            if not ctx.auth.operators_configured
            else f"the {principal.role.value} role cannot do this; an operator key is required"
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail)
    return principal


Operator = Annotated[Principal, Depends(require_operator)]


def get_storage(ctx: Ctx) -> StorageService:
    storage = ctx.runtime.storage
    if storage is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "the event store is disabled in this runtime"
        )
    return storage


Storage = Annotated[StorageService, Depends(get_storage)]


def get_store(storage: Storage) -> EventStore:
    store: EventStore = storage.store
    return store


Store = Annotated[EventStore, Depends(get_store)]


def check_machines(ctx: ApiContext, machines: list[str] | None) -> list[str] | None:
    """Split comma lists and reject unknown machine ids (404)."""
    if not machines:
        return None
    ids = [m for item in machines for m in item.split(",") if m]
    unknown = sorted(set(ids) - set(ctx.runtime.engine.machine_ids))
    if unknown:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown machine(s): {unknown}")
    return ids
