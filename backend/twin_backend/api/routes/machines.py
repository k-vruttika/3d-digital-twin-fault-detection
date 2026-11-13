"""Live machine state: GET /machines, GET /machines/{id}/state."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from twin_backend.api.deps import ApiContext, Ctx, Viewer
from twin_backend.api.models import MachinesResponse, MachineStateResponse
from twin_backend.engine.line_state import LineState
from twin_backend.engine.twin import RootCauseView

router = APIRouter(tags=["machines"])


def machine_response(ctx: ApiContext, state: LineState, machine_id: str) -> MachineStateResponse:
    engine = ctx.runtime.engine
    cfg = next(m for m in ctx.runtime.line.machines if m.id == machine_id)
    view = state.machines[machine_id]
    rc = engine.line_view().root_causes.get(machine_id)
    return MachineStateResponse(
        machine_id=machine_id,
        kind=cfg.kind,
        name=cfg.name,
        state=view.state,
        state_ts=view.state_ts,
        reported_state=view.reported_state,
        root_cause=RootCauseView(cause=rc.cause, unknown=rc.unknown)
        if rc is not None
        else RootCauseView(unknown=True),
        data_quality=engine.data_quality(machine_id),
        last_reading=view.last_reading,
        last_state_change=view.last_state_change,
        predictions=dict(state.predictions.get(machine_id, {})),
        alerts=engine.alerts.current(machine_id=machine_id),
    )


@router.get("/machines", response_model=MachinesResponse, summary="Live state of every machine")
async def list_machines(ctx: Ctx, _: Viewer) -> MachinesResponse:
    state = ctx.runtime.engine.snapshot()
    return MachinesResponse(
        line_id=ctx.runtime.line.line_id,
        as_of=state.as_of,
        line=ctx.runtime.engine.line_model(),
        machines=[machine_response(ctx, state, m) for m in ctx.runtime.engine.machine_ids],
    )


@router.get(
    "/machines/{machine_id}/state",
    response_model=MachineStateResponse,
    summary="Live state of one machine",
    responses={404: {"description": "Unknown machine"}},
)
async def machine_state(machine_id: str, ctx: Ctx, _: Viewer) -> MachineStateResponse:
    if machine_id not in ctx.runtime.engine.machine_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown machine {machine_id!r}")
    return machine_response(ctx, ctx.runtime.engine.snapshot(), machine_id)
