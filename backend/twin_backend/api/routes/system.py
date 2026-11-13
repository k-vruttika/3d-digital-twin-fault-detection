"""GET /health, GET /metrics, and the Phase 11 stubs GET /predict, POST /whatif."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from twin_backend.api.deps import Ctx, Viewer
from twin_backend.api.models import AuthInfo, HealthResponse
from twin_backend.api.routes.sources import health_model
from twin_common.enums import AdapterStatus

router = APIRouter(tags=["system"])

# Modules minimal mode must not load (no broker, database server or ML).
OPTIONAL_MODULES = (
    "asyncpg",
    "aiomqtt",
    "paho",
    "asyncua",
    "torch",
    "sklearn",
    "xgboost",
    "onnxruntime",
    "shap",
    "captum",
)
NOT_IMPLEMENTED = "available from Phase 11 (inference service); not implemented yet"


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    summary="Liveness and wiring (no key needed)",
)
async def health(
    ctx: Ctx,
    modules: Annotated[
        bool, Query(description="also list which optional heavy modules are loaded")
    ] = False,
) -> HealthResponse:
    h = ctx.runtime.manager.health()
    ok = h is not None and h.status not in (AdapterStatus.FAILED, AdapterStatus.STOPPED)
    storage = ctx.runtime.storage
    return HealthResponse(
        status="ok" if ok else "degraded",
        line_id=ctx.runtime.line.line_id,
        machines=len(ctx.runtime.engine.machine_ids),
        source=health_model(h),
        store=storage.store.backend if storage is not None else None,
        event_loop=type(asyncio.get_running_loop()).__name__,
        auth=AuthInfo(
            keys_configured=bool(ctx.auth.keys),
            anonymous_read=ctx.auth.anonymous_read,
            operator_keys=ctx.auth.operators_configured,
        ),
        loaded_optional_modules=[m for m in OPTIONAL_MODULES if m in sys.modules]
        if modules
        else None,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics (text exposition format)",
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
async def metrics(ctx: Ctx, _: Viewer) -> Response:
    return Response(generate_latest(ctx.runtime.metrics.registry), media_type=CONTENT_TYPE_LATEST)


@router.get(
    "/predict",
    summary="Fault predictions (Phase 11; returns 501 until then)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses={501: {"description": NOT_IMPLEMENTED}},
)
async def predict(_: Viewer) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, NOT_IMPLEMENTED)


@router.post(
    "/whatif",
    summary="What-if simulation (Phase 11; returns 501 until then)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses={501: {"description": NOT_IMPLEMENTED}},
)
async def whatif(_: Viewer) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, NOT_IMPLEMENTED)
