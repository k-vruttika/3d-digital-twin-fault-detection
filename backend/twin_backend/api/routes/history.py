"""Event-store queries: GET /history, GET /replay, GET /export."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import AwareDatetime

from twin_backend.api.deps import Ctx, Storage, Store, Viewer, check_machines
from twin_backend.api.models import HistoryBucketModel, HistoryResponse, SensorStatsModel
from twin_backend.engine.line_state import LineState
from twin_backend.storage.base import Stream
from twin_backend.storage.export import order_readings, readings_to_csv, readings_to_jsonl

router = APIRouter(tags=["history"])

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
    response_model=LineState,
    summary="Line state at event time t (snapshot + replay, equals reconstruct(t))",
)
async def replay(
    storage: Storage,
    _: Viewer,
    t: Annotated[AwareDatetime, Query(description="event time, ISO-8601 with zone")],
) -> LineState:
    return await storage.reconstruct(_utc(t))


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
