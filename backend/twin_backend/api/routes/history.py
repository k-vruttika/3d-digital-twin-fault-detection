"""Event-store queries: GET /history, GET /replay, GET /export."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import AwareDatetime

from twin_backend.api.deps import Ctx, Storage, Store, Viewer, check_machines
from twin_backend.api.models import (
    HistoryBucketModel,
    HistoryResponse,
    ReplayRangeResponse,
    ReplayResponse,
    SensorStatsModel,
)
from twin_backend.engine.propagation import line_view
from twin_backend.engine.twin import LineViewModel
from twin_backend.storage.base import EventStore, Stream
from twin_backend.storage.export import order_readings, readings_to_csv, readings_to_jsonl
from twin_common.contracts import Alert

router = APIRouter(tags=["history"])
# fetch() is half-open [t0, t1); one microsecond (the store's resolution) makes ts == t inclusive.
_EPSILON = timedelta(microseconds=1)

From = Annotated[AwareDatetime, Query(alias="from", description="inclusive, ISO-8601 with zone")]
To = Annotated[AwareDatetime, Query(alias="to", description="exclusive, ISO-8601 with zone")]
Machines = Annotated[
    list[str] | None,
    Query(description="machine ids (repeat the parameter or separate with commas); all if omitted"),
]


def _utc(ts: datetime) -> datetime:
    return ts.astimezone(UTC)


def _range(t0: datetime, t1: datetime) -> tuple[datetime, datetime]:
    t0, t1 = _utc(t0), _utc(t1)
    if t1 <= t0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "'to' must be after 'from'")
    return t0, t1


def _too_many(n: int, limit: int) -> HTTPException:
    return HTTPException(
        status.HTTP_413_CONTENT_TOO_LARGE,
        f"{n} readings in this range exceed the limit of {limit} "
        "(api.yaml max_history_rows); narrow the range or use a coarser resolution",
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    response_model_exclude_none=True,
    summary="Readings in a time range: raw, or 1 s / 1 min / 10 min buckets",
)
async def history(
    ctx: Ctx,
    store: Store,
    _: Viewer,
    t0: From,
    t1: To,
    resolution: Annotated[
        Literal["raw", "1s", "1m", "10m"], Query(description="raw rows or bucket width")
    ] = "1m",
    machines: Machines = None,
) -> HistoryResponse:
    t0, t1 = _range(t0, t1)
    ids = check_machines(ctx, machines)
    limit = ctx.runtime.api_cfg.max_history_rows
    if resolution == "raw":
        rows = await store.fetch(Stream.READINGS, t0, t1, machine_ids=ids)
        if len(rows) > limit:
            raise _too_many(len(rows), limit)
        return HistoryResponse(
            resolution="raw", start=t0, end=t1, machines=ids, readings=[r.item for r in rows]
        )
    buckets = await store.history(t0, t1, resolution, machine_ids=ids)
    if len(buckets) > limit:
        raise _too_many(len(buckets), limit)
    return HistoryResponse(
        resolution=resolution,
        start=t0,
        end=t1,
        machines=ids,
        buckets=[
            HistoryBucketModel(
                bucket=b.bucket,
                machine_id=b.machine_id,
                n=b.n,
                n_late=b.n_late,
                n_suspect=b.n_suspect,
                sensors={
                    s: SensorStatsModel(count=v.count, mean=v.mean, min=v.min, max=v.max)
                    for s, v in b.sensors.items()
                },
            )
            for b in buckets
        ],
    )


@router.get(
    "/replay",
    response_model=ReplayResponse,
    summary="Line state at event time t (reconstruct(t)) with root causes and alerts as of t",
    description=(
        "`machines` and `predictions` equal `reconstruct(t)` (snapshot + replay). `line` holds "
        "the root causes the twin engine derives from those states (same propagation as live), "
        "`alerts` the alerts that were unresolved at t (highest rev with ts <= t). Data quality "
        "uses a wall-clock window and is not replayed."
    ),
)
async def replay(
    ctx: Ctx,
    storage: Storage,
    store: Store,
    _: Viewer,
    t: Annotated[AwareDatetime, Query(description="event time, ISO-8601 with zone")],
) -> ReplayResponse:
    t = _utc(t)
    state = await storage.reconstruct(t)
    topo = ctx.runtime.engine.topology
    view = line_view({m: v.state for m, v in state.machines.items()}, topo)
    return ReplayResponse(
        as_of=state.as_of,
        machines=state.machines,
        predictions=state.predictions,
        line=LineViewModel.of(view),
        alerts=await alerts_as_of(store, t),
    )


async def alerts_as_of(store: EventStore, t: datetime) -> list[Alert]:
    """Unresolved alerts at event time t: the highest rev per alert among events with ts <= t."""
    latest: dict[str, Alert] = {}
    for row in await store.fetch(Stream.ALERTS, None, t + _EPSILON):
        alert: Alert = row.item
        cur = latest.get(alert.alert_id)
        if cur is None or alert.rev > cur.rev:
            latest[alert.alert_id] = alert
    return sorted((a for a in latest.values() if not a.resolved), key=lambda a: (a.ts, a.alert_id))


@router.get(
    "/replay/range",
    response_model=ReplayRangeResponse,
    summary="Event-time span of the stored readings and state changes (replay scrubber bounds)",
)
async def replay_range(store: Store, _: Viewer) -> ReplayRangeResponse:
    span = await store.min_max_ts([Stream.READINGS, Stream.STATE_CHANGES])
    if span is None:
        return ReplayRangeResponse()
    return ReplayRangeResponse(start=span[0], end=span[1])


@router.get(
    "/export",
    summary="Readings in a time range as CSV or JSONL (replayable by the file adapter)",
    response_class=Response,
    responses={
        200: {
            "content": {"text/csv": {}, "application/x-ndjson": {}},
            "description": "Canonical Reading rows ordered by (ts, line order, seq)",
        }
    },
)
async def export(
    ctx: Ctx,
    store: Store,
    _: Viewer,
    t0: From,
    t1: To,
    format: Annotated[Literal["csv", "jsonl"], Query(description="file format")] = "csv",
    machines: Machines = None,
) -> Response:
    t0, t1 = _range(t0, t1)
    ids = check_machines(ctx, machines)
    rows = await store.fetch(Stream.READINGS, t0, t1, machine_ids=ids)
    limit = ctx.runtime.api_cfg.max_history_rows
    if len(rows) > limit:
        raise _too_many(len(rows), limit)
    readings = order_readings(rows, ctx.runtime.engine.machine_ids)
    if format == "csv":
        body, media = readings_to_csv(readings), "text/csv"
    else:
        body, media = readings_to_jsonl(readings), "application/x-ndjson"
    name = f"linetwin_{t0:%Y%m%dT%H%M%SZ}_{t1:%Y%m%dT%H%M%SZ}.{format}"
    return Response(
        body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'}
    )
