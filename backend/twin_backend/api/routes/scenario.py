"""Scenario injection (simulation only, ADR 0015): POST /scenario, GET /scenario[/{id}].

Real equipment would not offer this; it works only when the active source is a simulator
publisher (mock, mqtt, opcua). Requests are stored for audit, rejected ones included.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import AwareDatetime, ValidationError

from twin_backend.api.deps import Ctx, Operator, Viewer, check_machines
from twin_backend.commands.scenarios import ScenarioPayload, ScenarioRecord
from twin_backend.engine.line_state import StateModel

router = APIRouter(tags=["scenario (simulation only)"])


class ScenariosResponse(StateModel):
    requests: list[ScenarioRecord]


@router.post(
    "/scenario",
    response_model=ScenarioRecord,
    summary="Inject a fault or load a scenario (operator; simulation only)",
    description=(
        "Simulation-only capability: asks the simulator behind the active source to start a "
        "fault now (`INJECT_FAULT`) or to schedule a scenario file's events from now on "
        "(`LOAD_SCENARIO`). The request and its ack are stored, so injected faults are "
        "distinguishable from real ones. The simulator's verdict is in the ack (`EXECUTED`, "
        "`REJECTED` with a reason, `TIMEOUT`)."
    ),
    responses={
        403: {"description": "Not an operator"},
        409: {"description": "The active source does not support scenario injection"},
        422: {"description": "Inconsistent request"},
    },
)
async def inject(body: ScenarioPayload, ctx: Ctx, principal: Operator) -> ScenarioRecord:
    if body.machine_id is not None:
        check_machines(ctx, [body.machine_id])
    svc = ctx.runtime.scenarios
    try:
        req = svc.build(body, principal.name)
    except ValidationError as exc:
        msgs = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, msgs) from exc
    result = await svc.submit(req)
    if not result.supported:
        raise HTTPException(status.HTTP_409_CONFLICT, result.ack.reason)
    return ScenarioRecord(request=result.request, acks=[result.ack])


@router.get(
    "/scenario",
    response_model=ScenariosResponse,
    summary="Audit trail of scenario requests in an event-time range",
)
async def list_requests(
    ctx: Ctx,
    _: Viewer,
    t0: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    t1: Annotated[AwareDatetime | None, Query(alias="to")] = None,
    machine: Annotated[str | None, Query(description="machine id")] = None,
) -> ScenariosResponse:
    if machine is not None:
        check_machines(ctx, [machine])
    return ScenariosResponse(requests=await ctx.runtime.scenarios.history(t0, t1, machine))


@router.get(
    "/scenario/{request_id}",
    response_model=ScenarioRecord,
    summary="One scenario request and its ack",
    responses={404: {"description": "Unknown request"}},
)
async def get_request(request_id: str, ctx: Ctx, _: Viewer) -> ScenarioRecord:
    record = await ctx.runtime.scenarios.get(request_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown scenario request {request_id!r}")
    return record
