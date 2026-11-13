"""FastAPI application factory (Phase 4, ADR 0013).

``create_app(runtime)`` wires the REST routers, the ``/stream`` WebSocket hub, CORS and auth
around a built :class:`~twin_backend.runtime.Runtime`. The app's lifespan starts and stops the
runtime (storage, source, engine) unless ``manage_runtime=False`` (tests that drive it).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from twin_backend.api.auth import Authenticator
from twin_backend.api.deps import API_KEY_HEADER, ApiContext
from twin_backend.api.routes import (
    alerts,
    commands,
    history,
    line,
    machines,
    scenario,
    sources,
    system,
)
from twin_backend.api.ws import WsHub
from twin_backend.api.ws_models import openapi_components
from twin_backend.runtime import Runtime

API_VERSION = "0.4.0"
DESCRIPTION = """\
LineTwin backend: live state of the production-line twin, event-store queries, alerts,
operator commands, source management and (simulation only) scenario injection.

**Auth:** send an API key in the `X-API-Key` header. Viewers can read; operators can also send
commands, switch sources and acknowledge alerts. Without configured keys the server serves
read-only access on a loopback address only. WebSocket clients authenticate with their first
message (see the WebSocket reference in docs/api.md).
"""


def create_app(
    runtime: Runtime, *, host: str | None = None, manage_runtime: bool = True
) -> FastAPI:
    bind_host = host if host is not None else runtime.api_cfg.host
    auth = Authenticator(runtime.settings.api_keys(), bind_host=bind_host)
    hub = WsHub(
        runtime.bus,
        runtime.engine,
        auth,
        runtime.api_cfg.ws,
        registry=runtime.metrics.registry,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        auth.log_startup()
        if manage_runtime:
            await runtime.start()
        hub.start()
        try:
            yield
        finally:
            await hub.stop()
            if manage_runtime:
                await runtime.stop()

    app = FastAPI(
        title="LineTwin API",
        version=API_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.ctx = ApiContext(runtime=runtime, auth=auth, hub=hub)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.api_cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=[API_KEY_HEADER, "Content-Type"],
        expose_headers=["Idempotent-Replay"],
    )
    for module in (line, machines, history, commands, alerts, sources, scenario, system):
        app.include_router(module.router)

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await hub.serve(ws)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = with_ws_schemas(FastAPI.openapi(app))
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


def with_ws_schemas(spec: dict[str, Any]) -> dict[str, Any]:
    """Add the WebSocket frame models (``WsClientMessage``, ``WsServerMessage``) to the schema.

    OpenAPI has no WebSocket operations, but the components are what the frontend types are
    generated from (ADR 0017). A model that is also used by a REST route must produce the same
    schema in both places; a conflict is a bug and fails loudly.
    """
    schemas: dict[str, Any] = spec.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in openapi_components().items():
        if name in schemas:
            if _normalised(schemas[name]) != _normalised(schema):
                raise RuntimeError(f"OpenAPI component {name!r} differs between REST and WebSocket")
            continue  # same model: keep FastAPI's rendering
        schemas[name] = _normalised(schema)
    return spec


def _normalised(node: Any) -> Any:
    """Drop ``"default": null`` (FastAPI omits it; pydantic writes it) recursively."""
    if isinstance(node, dict):
        return {k: _normalised(v) for k, v in node.items() if not (k == "default" and v is None)}
    if isinstance(node, list):
        return [_normalised(v) for v in node]
    return node
