"""Data sources: GET /sources, POST /sources/switch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from twin_backend.adapters.base import AdapterHealth
from twin_backend.api.deps import ApiContext, Ctx, Operator, Viewer
from twin_backend.api.models import (
    AdapterHealthModel,
    IngestStatsModel,
    LatencyModel,
    SourcesResponse,
    SwitchRequest,
)
from twin_backend.ingest.metrics import PREFIX, IngestMetrics
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


def _pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole > 0 else None


def latency_stats(metrics: IngestMetrics, source: str) -> LatencyModel:
    """Count, mean and p95 (bucket upper bound) of the e2e latency histogram, all bases."""
    name = f"{PREFIX}_e2e_latency_seconds"
    count = 0.0
    total = 0.0
    cumulative: dict[float, float] = {}
    for family in metrics.registry.collect():
        if family.name != name:
            continue
        for sample in family.samples:
            if sample.labels.get("source") != source:
                continue
            if sample.name == f"{name}_count":
                count += sample.value
            elif sample.name == f"{name}_sum":
                total += sample.value
            elif sample.name == f"{name}_bucket":
                le = float(sample.labels["le"])
                cumulative[le] = cumulative.get(le, 0.0) + sample.value
    if count <= 0:
        return LatencyModel(count=0)
    p95 = next(
        (le for le in sorted(cumulative) if cumulative[le] >= 0.95 * count and le != float("inf")),
        None,
    )
    return LatencyModel(count=int(count), mean_s=total / count, p95_s=p95)


def ingest_stats(metrics: IngestMetrics, kind: AdapterKind) -> IngestStatsModel:
    src = kind.value
    published = int(metrics.value("published", source=src))
    late = int(metrics.value("late", source=src))
    partial = int(metrics.value("partial", source=src))
    by_reason = {r.value: int(v) for r, v in metrics.quarantined_by_reason(kind).items()}
    return IngestStatsModel(
        messages_in=int(metrics.value("messages_in", source=src)),
        published=published,
        late=late,
        late_pct=_pct(late, published),
        quarantined=sum(by_reason.values()),
        quarantined_by_reason=by_reason,
        partial=partial,
        partial_pct=_pct(partial, published),
        duplicates=int(metrics.value("duplicates", source=src)),
        latency=latency_stats(metrics, src),
    )


def sources_response(ctx: ApiContext) -> SourcesResponse:
    mgr = ctx.runtime.manager
    adapter = mgr.adapter
    active = mgr.active_kind
    return SourcesResponse(
        active=active,
        capabilities=sorted(adapter.capabilities) if adapter is not None else [],
        health=health_model(mgr.health()),
        configured=[k for k in AdapterKind if getattr(mgr.sources, k.value, None) is not None],
        switches=mgr.switches,
        ingest=ingest_stats(ctx.runtime.metrics, active) if active is not None else None,
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
