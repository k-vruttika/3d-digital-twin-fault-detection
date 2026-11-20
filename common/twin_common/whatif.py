"""Wire contracts between the backend and a simulator for what-if forks (ADR 0031).

The backend asks the simulator behind the active source to fork a copy, apply one command at
the fork point and fast-forward; the simulator answers with the projected readings of the
unchanged copy (``baseline``) and of the changed copy (``scenario``), plus the recent real
readings so a model window can be filled. Simulation only: real equipment cannot be forked.
"""

from __future__ import annotations

from pydantic import Field

from twin_common.contracts import ContractModel, NonEmptyStr, ParamValue, Reading, UtcDatetime
from twin_common.enums import AckStatus, CommandType


class WhatIfSimRequest(ContractModel):
    """Backend -> simulator: fork now, apply ``type`` to ``machine_id``, run ``horizon_s``."""

    machine_id: NonEmptyStr
    type: CommandType
    params: dict[str, ParamValue] = Field(default_factory=dict)
    horizon_s: int = Field(gt=0)
    history_s: int = Field(ge=1)


class WhatIfSimResult(ContractModel):
    """Simulator -> backend. Readings are oldest first, all machines, 1 Hz."""

    as_of: UtcDatetime = Field(description="the fork point: simulator time of the snapshot")
    action_status: AckStatus = Field(description="the simulator's verdict on the change")
    action_reason: str | None = None
    history: list[Reading] = Field(description="the last readings before the fork point")
    baseline: list[Reading] = Field(description="the copy without the change")
    scenario: list[Reading] = Field(description="the copy with the change applied at the fork")
