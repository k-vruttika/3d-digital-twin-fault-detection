"""What-if forks of a running simulator (ADR 0031; simulation only).

``take_fork`` runs at a tick boundary (through ``RealtimeRunner.submit_call``) and only copies
the simulator state. ``run_fork`` then needs no access to the live simulator, so it can run in
a worker thread while the live simulation keeps its pace: it builds two copies from the
snapshot, applies the change to one of them, and fast-forwards both ``horizon_s`` seconds.
The copies restore the random-number streams as well, so before the change the two copies
are identical, and the unchanged copy is exactly what the live simulation would do next.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from twin_common.contracts import Command, Reading
from twin_common.enums import AckStatus
from twin_common.whatif import WhatIfSimRequest, WhatIfSimResult
from twin_sim.line import LineSimulator, SimSnapshot

WHATIF_ISSUER = "what-if"


@dataclass(frozen=True)
class Fork:
    snapshot: SimSnapshot
    as_of: datetime


def take_fork(sim: LineSimulator) -> Fork:
    """Copy the simulator state (call at a tick boundary)."""
    return Fork(sim.snapshot(), sim.timestamp(sim.t))


def _command(req: WhatIfSimRequest, ts: datetime) -> Command:
    return Command(
        command_id="whatif",
        idempotency_key="whatif",
        machine_id=req.machine_id,
        type=req.type,
        params=req.params,
        issued_by=WHATIF_ISSUER,
        ts=ts,
    )


def _run(sim: LineSimulator, horizon_s: int) -> list[Reading]:
    out: list[Reading] = []
    for frame in sim.fast_forward(horizon_s, record=True):
        out.extend(sim.readings(frame))
    return out


def run_fork(
    template: LineSimulator,
    fork: Fork,
    req: WhatIfSimRequest,
    history: list[Reading],
) -> WhatIfSimResult:
    """Fast-forward the unchanged and the changed copy; ``template`` only supplies configs."""
    baseline = LineSimulator.from_snapshot(template.line, template.cfg, fork.snapshot)
    changed = LineSimulator.from_snapshot(template.line, template.cfg, fork.snapshot)
    ack = changed.apply_command(_command(req, fork.as_of))
    unchanged = _run(baseline, req.horizon_s)
    # a rejected change leaves the copy as it was: the projection equals the baseline
    scenario = unchanged if ack.status is AckStatus.REJECTED else _run(changed, req.horizon_s)
    return WhatIfSimResult(
        as_of=fork.as_of,
        action_status=ack.status,
        action_reason=ack.reason,
        history=history,
        baseline=unchanged,
        scenario=scenario,
    )
