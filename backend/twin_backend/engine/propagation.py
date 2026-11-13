"""Line-level propagation: which machine is the root cause of a STARVED or BLOCKED machine.

Mirrors the simulator's rule (``twin_sim.line.LineSimulator._root_cause``) on the topology from
the line config:

- RUNNING has no root cause;
- STARVED follows the upstream link, BLOCKED the downstream link, until it reaches a RUNNING
  machine (that machine is the bottleneck) or a machine that is itself stopped for another
  reason (IDLE, WARNING, FAULT, MAINTENANCE: that machine is the cause). The start of the line
  is ``SOURCE`` and the end ``SINK``;
- any other state is its own root cause.

The twin adds one case the simulator does not have: a neighbour whose data link is down (STALE,
OFFLINE) or not yet known is reported as the cause with ``unknown=True``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from twin_backend.engine.topology import SINK_ID, SOURCE_ID, Topology
from twin_common.enums import MachineState

_LINK_DOWN = (MachineState.STALE, MachineState.OFFLINE)
_WAITING = (MachineState.STARVED, MachineState.BLOCKED)


@dataclass(frozen=True, slots=True)
class RootCause:
    cause: str | None  # machine id, SOURCE, SINK, or None (RUNNING / no state yet)
    unknown: bool = False  # the cause's state is not known (link down / no data)


class LineStatus(StrEnum):
    RUNNING = "RUNNING"  # every machine RUNNING
    DEGRADED = "DEGRADED"  # some machine not RUNNING, at least one RUNNING
    STOPPED = "STOPPED"  # no machine RUNNING
    UNKNOWN = "UNKNOWN"  # some machine has no state yet


@dataclass(frozen=True, slots=True)
class LineView:
    status: LineStatus
    root_causes: dict[str, RootCause]
    # root-cause machine -> the other machines it makes STARVED/BLOCKED (flow order)
    impacted: dict[str, list[str]] = field(default_factory=dict)


def root_cause(
    machine_id: str, states: Mapping[str, MachineState | None], topo: Topology
) -> RootCause:
    st = states.get(machine_id)
    if st is None:
        return RootCause(None, unknown=True)
    if st is MachineState.RUNNING:
        return RootCause(None)
    if st not in _WAITING:
        return RootCause(machine_id, unknown=st in _LINK_DOWN)
    current, depth = machine_id, 0
    while True:
        cur_state = states.get(current)
        starved = cur_state is MachineState.STARVED
        nxt = topo.neighbour_for(current, starved)
        if nxt is None:
            return RootCause(SOURCE_ID if starved else SINK_ID)
        nxt_state = states.get(nxt)
        if depth >= len(topo.machine_ids) or nxt_state is MachineState.RUNNING:
            return RootCause(nxt)
        if nxt_state is None or nxt_state in _LINK_DOWN:
            return RootCause(nxt, unknown=True)
        if nxt_state not in _WAITING:
            return RootCause(nxt)
        current, depth = nxt, depth + 1


def line_view(states: Mapping[str, MachineState | None], topo: Topology) -> LineView:
    causes = {m: root_cause(m, states, topo) for m in topo.machine_ids}
    impacted: dict[str, list[str]] = {}
    for m in topo.machine_ids:
        rc = causes[m]
        if states.get(m) in _WAITING and rc.cause is not None and rc.cause in topo.upstream:
            impacted.setdefault(rc.cause, []).append(m)
    values = [states.get(m) for m in topo.machine_ids]
    if any(v is None for v in values):
        status = LineStatus.UNKNOWN
    elif all(v is MachineState.RUNNING for v in values):
        status = LineStatus.RUNNING
    elif any(v is MachineState.RUNNING for v in values):
        status = LineStatus.DEGRADED
    else:
        status = LineStatus.STOPPED
    return LineView(status, causes, impacted)
