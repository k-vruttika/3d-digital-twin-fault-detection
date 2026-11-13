"""Alerts: GET /alerts, GET /alerts/history, POST /alerts/{id}/ack (ADR 0011)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import AwareDatetime

from twin_backend.api.deps import Ctx, Operator, Store, Viewer, check_machines
from twin_backend.api.models import AlertsResponse
from twin_backend.storage.base import Stream
from twin_common.contracts import Alert

router = APIRouter(tags=["alerts"])


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="Unresolved alerts (active, or cleared but not yet acknowledged)",
)
async def list_alerts(
    ctx: Ctx,
    _: Viewer,
    active: Annotated[bool | None, Query(description="only active / only cleared")] = None,
    acknowledged: Annotated[bool | None, Query(description="filter by acknowledgement")] = None,
    machine: Annotated[str | None, Query(description="machine id")] = None,
) -> AlertsResponse:
    if machine is not None:
        check_machines(ctx, [machine])
    return AlertsResponse(
        alerts=ctx.runtime.engine.alerts.current(
            active=active, acknowledged=acknowledged, machine_id=machine
        )
    )


@router.get(
    "/alerts/history",
    response_model=AlertsResponse,
    summary="Stored alert lifecycle events in a time range (every raise, clear, ack)",
)
async def alert_history(
    ctx: Ctx,
    store: Store,
    _: Viewer,
    t0: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    t1: Annotated[AwareDatetime | None, Query(alias="to")] = None,
    machine: Annotated[str | None, Query(description="machine id")] = None,
) -> AlertsResponse:
    ids = check_machines(ctx, [machine] if machine else None)
    rows = await store.fetch(Stream.ALERTS, t0, t1, machine_ids=ids)
    return AlertsResponse(alerts=[r.item for r in rows])


@router.post(
    "/alerts/{alert_id}/ack",
    response_model=Alert,
    summary="Acknowledge an alert (operator); repeating it changes nothing",
    responses={404: {"description": "Unknown or already resolved alert"}},
)
async def acknowledge(alert_id: str, ctx: Ctx, principal: Operator) -> Alert:
    result = ctx.runtime.engine.acknowledge(alert_id, principal.name)
    if result.alert is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no unresolved alert {alert_id!r} (unknown or resolved)"
        )
    await ctx.runtime.engine.flush()  # publish now, so the ack is on the bus before we answer
    return result.alert
