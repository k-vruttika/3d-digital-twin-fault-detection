"""GET /health, GET /metrics and POST /whatif (what-if projections, ADR 0031)."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from twin_backend.adapters.base import WhatIfError, WhatIfUnsupportedError
from twin_backend.api.deps import Ctx, Viewer, check_machines
from twin_backend.api.models import AuthInfo, HealthResponse
from twin_backend.api.routes.sources import health_model
from twin_backend.whatif.models import WhatIfPayload, WhatIfResponse
from twin_backend.whatif.service import WhatIfRejectedError
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
    "mlflow",
    "twin_ml",
    "onnxruntime",
    "shap",
    "captum",
)
NOT_CONFIGURED = "what-if is not configured (no config/whatif.yaml)"


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


@router.post(
    "/whatif",
    response_model=WhatIfResponse,
    response_model_exclude_none=True,
    summary="What-if projection of one change (simulation only)",
    description=(
        "Forks a copy of the simulation behind the active source, applies one command to the "
        "copy now, fast-forwards it and an unchanged copy, and (with live inference on) scores "
        "both projections with the deployed model. Nothing is sent to the machine: apply the "
        "change for real with `POST /commands`. The change must pass the same safety rules as "
        "a command."
    ),
    responses={
        404: {"description": "Unknown machine"},
        409: {"description": "The active source does not support what-if (simulation only)"},
        422: {"description": "Not allowed by the safety rules, or rejected by the simulator"},
        502: {"description": "The simulator behind the source could not run the projection"},
    },
)
async def whatif(body: WhatIfPayload, ctx: Ctx, _: Viewer) -> WhatIfResponse:
    check_machines(ctx, [body.machine_id])
    svc = ctx.runtime.whatif
    if svc is None:
        raise HTTPException(status.HTTP_409_CONFLICT, NOT_CONFIGURED)
    try:
        return await svc.run(body)
    except WhatIfUnsupportedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except WhatIfRejectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except WhatIfError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
