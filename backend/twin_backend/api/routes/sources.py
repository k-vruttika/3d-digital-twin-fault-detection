"""Data sources: GET /sources, POST /sources/switch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from twin_backend.adapters.base import AdapterHealth
from twin_backend.api.deps import ApiContext, Ctx, Operator, Viewer
from twin_backend.api.models import AdapterHealthModel, SourcesResponse, SwitchRequest
from twin_common.enums import AdapterKind
from twin_common.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["sources"])


def health_model(h: AdapterHealth | None) -> AdapterHealthModel | None:
    if h is None:
        return None
    return AdapterHealthModel(
        kind=h.kind,
        status=h.status,
        connected=h.connected,
        messages=h.messages,
        reconnects=h.reconnects,
        last_message_at=h.last_message_at,
        last_error=h.last_error,
    )


def sources_response(ctx: ApiContext) -> SourcesResponse:
    mgr = ctx.runtime.manager
    adapter = mgr.adapter
    return SourcesResponse(
        active=mgr.active_kind,
        capabilities=sorted(adapter.capabilities) if adapter is not None else [],
        health=health_model(mgr.health()),
        configured=[k for k in AdapterKind if getattr(mgr.sources, k.value, None) is not None],
        switches=mgr.switches,
    )


@router.get("/sources", response_model=SourcesResponse, summary="Active source and its health")
async def get_sources(ctx: Ctx, _: Viewer) -> SourcesResponse:
    return sources_response(ctx)


@router.post(
    "/sources/switch",
    response_model=SourcesResponse,
    summary="Hot-switch to another source (operator); a failing source keeps the current one",
    responses={409: {"description": "The new source could not start"}},
)
async def switch_source(body: SwitchRequest, ctx: Ctx, principal: Operator) -> SourcesResponse:
    mgr = ctx.runtime.manager
    if getattr(mgr.sources, body.kind.value, None) is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"source {body.kind.value!r} is not configured in sources.yaml",
        )
    try:
        await mgr.switch(body.kind)
    except Exception as exc:  # the old source keeps running (SourceManager.switch)
        log.warning("api.switch_failed", kind=body.kind.value, error=repr(exc))
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"could not start the {body.kind.value} source ({type(exc).__name__}: {exc}); "
            f"still using {mgr.active_kind.value if mgr.active_kind else 'no source'}",
        ) from exc
    log.info("api.source_switched", kind=body.kind.value, by=principal.name)
    return sources_response(ctx)
