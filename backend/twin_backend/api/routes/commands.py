"""Operator commands: POST /commands, GET /commands/{id}, GET /commands (ADR 0014)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import AwareDatetime

from twin_backend.api.deps import Ctx, Operator, Viewer, check_machines
from twin_backend.api.models import (
    CommandLimitsResponse,
    KindLimitsModel,
    RangeModel,
    SafetyRuleModel,
)
from twin_backend.commands.service import (
    CommandRecord,
    CommandRequest,
    IdempotencyConflictError,
    UnknownMachineError,
)
from twin_backend.engine.line_state import StateModel
from twin_common.config.commands import FAIL_SAFE_COMMANDS, CommandsConfig
from twin_common.contracts import CommandAck

router = APIRouter(tags=["commands"])
REPLAY_HEADER = "Idempotent-Replay"


class CommandsResponse(StateModel):
    commands: list[CommandRecord]


@router.post(
    "/commands",
    response_model=CommandAck,
    summary="Send a command (operator); answers with the final ack",
    description=(
        "Every outcome of the command itself is a 200 with the final `CommandAck`: `EXECUTED`, "
        "`REJECTED` (safety rule, unsupported by the active source, or refused by the machine; "
        "see `reason`) or `TIMEOUT`. STOP is always allowed by the safety rules. Repeating an "
        "`idempotencyKey` with the same request returns the original ack without sending "
        f"again (response header `{REPLAY_HEADER}: true`)."
    ),
    responses={
        403: {"description": "Not an operator"},
        404: {"description": "Unknown machine"},
        409: {"description": "idempotencyKey already used for a different request"},
    },
)
async def send_command(
    body: CommandRequest, ctx: Ctx, principal: Operator, response: Response
) -> CommandAck:
    try:
        result = await ctx.runtime.commands.submit(body, principal.name)
    except UnknownMachineError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown machine {exc.args[0]!r}") from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"idempotencyKey {exc.args[0]!r} was already used for a different request",
        ) from exc
    except ValueError as exc:  # Command contract (e.g. a non-numeric speed)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    response.headers[REPLAY_HEADER] = "true" if result.replay else "false"
    return result.ack


def limits_response(cfg: CommandsConfig) -> CommandLimitsResponse:
    return CommandLimitsResponse(
        limits={
            kind: KindLimitsModel(
                speed_rpm=RangeModel(min=lim.speed_rpm.min, max=lim.speed_rpm.max),
                load_pct=RangeModel(min=lim.load_pct.min, max=lim.load_pct.max),
            )
            for kind, lim in cfg.limits.items()
        },
        rules=[
            SafetyRuleModel(
                types=list(r.types),
                forbidden_states=r.forbidden_states,
                required_states=r.required_states,
                reason=r.reason,
            )
            for r in cfg.rules
        ],
        always_allowed=sorted(FAIL_SAFE_COMMANDS),
    )


@router.get(
    "/commands/limits",
    response_model=CommandLimitsResponse,
    summary="Parameter limits per machine kind and the safety rules (commands.yaml)",
)
async def command_limits(ctx: Ctx, _: Viewer) -> CommandLimitsResponse:
    return limits_response(ctx.runtime.commands_cfg)


@router.get(
    "/commands/{command_id}",
    response_model=CommandRecord,
    summary="A command and every ack it got (audit trail)",
    responses={404: {"description": "Unknown command"}},
)
async def get_command(command_id: str, ctx: Ctx, _: Viewer) -> CommandRecord:
    record = await ctx.runtime.commands.get(command_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown command {command_id!r}")
    return record


@router.get(
    "/commands",
    response_model=CommandsResponse,
    summary="Audit trail: commands (and their acks) in an event-time range",
)
async def list_commands(
    ctx: Ctx,
    _: Viewer,
    t0: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    t1: Annotated[AwareDatetime | None, Query(alias="to")] = None,
    machines: Annotated[list[str] | None, Query(description="machine ids")] = None,
) -> CommandsResponse:
    ids = check_machines(ctx, machines)
    return CommandsResponse(commands=await ctx.runtime.commands.history(t0, t1, ids))
