"""Per-machine state machine on top of the LineState fold.

The fold decides the effective state; this layer only checks each observed transition against
the allowed transitions in ``config/engine.yaml``. An invalid transition is logged and counted,
never rejected: the data is the truth, and a skipped intermediate state (1 Hz sampling) or a
source bug must not stop the twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter

from twin_common.config.engine import EngineConfig
from twin_common.enums import MachineState
from twin_common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Transition:
    machine_id: str
    previous: MachineState | None
    state: MachineState
    valid: bool


@dataclass
class MachineStateMachine:
    machine_id: str
    cfg: EngineConfig
    state: MachineState | None = None
    transitions: int = 0
    invalid: int = 0

    def observe(self, new: MachineState | None) -> Transition | None:
        """Record the machine's effective state; returns the transition if it changed."""
        if new is None or new == self.state:
            return None
        prev = self.state
        self.state = new
        self.transitions += 1
        valid = prev is None or self.cfg.allowed(prev, new)
        if not valid:
            self.invalid += 1
        return Transition(self.machine_id, prev, new, valid)


class StateMachines:
    """One state machine per machine plus the invalid-transition metric."""

    def __init__(
        self, machine_ids: list[str], cfg: EngineConfig, registry: CollectorRegistry | None
    ) -> None:
        self.machines = {m: MachineStateMachine(m, cfg) for m in machine_ids}
        self.invalid_total = Counter(
            "linetwin_engine_invalid_transitions",
            "State transitions not allowed by config/engine.yaml (applied anyway)",
            ["machine", "from_state", "to_state"],
            registry=registry,
        )
        self.transitions_total = Counter(
            "linetwin_engine_transitions",
            "Observed machine state transitions",
            ["machine"],
            registry=registry,
        )

    def observe(self, machine_id: str, new: MachineState | None) -> Transition | None:
        sm = self.machines.get(machine_id)
        if sm is None:
            return None
        tr = sm.observe(new)
        if tr is None:
            return None
        self.transitions_total.labels(machine=machine_id).inc()
        if not tr.valid:
            assert tr.previous is not None
            self.invalid_total.labels(
                machine=machine_id, from_state=tr.previous.value, to_state=tr.state.value
            ).inc()
            log.warning(
                "engine.invalid_transition",
                machine=machine_id,
                from_state=tr.previous.value,
                to_state=tr.state.value,
            )
        return tr

    def invalid_count(self) -> int:
        return sum(sm.invalid for sm in self.machines.values())
