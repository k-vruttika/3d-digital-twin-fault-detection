"""Staleness monitor: a machine silent for ``stale_after_s`` is STALE, for ``offline_after_s``
OFFLINE. When data returns, the machine goes back to the state its reading reports.

Time comes from the injectable clock (monotonic seconds), so tests advance a fake clock instead
of sleeping. Machines that never reported count from the monitor's start.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from twin_backend.clock import Clock
from twin_common.contracts import StateChange
from twin_common.enums import MachineState


class LinkStatus(Enum):
    LIVE = "live"
    STALE = "stale"
    OFFLINE = "offline"


@dataclass
class _Link:
    last_seen: float
    status: LinkStatus
    last_state: MachineState | None  # last state reported by a reading


class StalenessMonitor:
    def __init__(
        self,
        machine_ids: list[str],
        stale_after_s: float,
        offline_after_s: float,
        clock: Clock,
    ) -> None:
        if not 0 < stale_after_s < offline_after_s:
            raise ValueError("need 0 < stale_after_s < offline_after_s")
        self.stale_after_s = stale_after_s
        self.offline_after_s = offline_after_s
        self.clock = clock
        start = clock.monotonic()
        self._links = {m: _Link(start, LinkStatus.LIVE, None) for m in machine_ids}

    def status(self, machine_id: str) -> LinkStatus:
        return self._links[machine_id].status

    def effective_state(self, machine_id: str) -> MachineState | None:
        link = self._links[machine_id]
        if link.status is LinkStatus.STALE:
            return MachineState.STALE
        if link.status is LinkStatus.OFFLINE:
            return MachineState.OFFLINE
        return link.last_state

    def seen(self, machine_id: str, state: MachineState) -> StateChange | None:
        """A live (not late) reading arrived. Returns a recovery StateChange if it was down."""
        link = self._links.get(machine_id)
        if link is None:
            return None
        link.last_seen = self.clock.monotonic()
        link.last_state = state
        if link.status is LinkStatus.LIVE:
            return None
        previous = self.effective_state(machine_id)
        link.status = LinkStatus.LIVE
        return StateChange(
            machine_id=machine_id,
            state=state,
            previous_state=previous,
            ts=self.clock.now(),
            reason="data resumed",
        )

    def tick(self) -> list[StateChange]:
        now = self.clock.monotonic()
        changes: list[StateChange] = []
        for machine_id, link in self._links.items():
            silent = now - link.last_seen
            if silent >= self.offline_after_s:
                target = LinkStatus.OFFLINE
            elif silent >= self.stale_after_s:
                target = LinkStatus.STALE
            else:
                continue
            if target is link.status:
                continue
            previous = self.effective_state(machine_id)
            link.status = target
            state = MachineState.OFFLINE if target is LinkStatus.OFFLINE else MachineState.STALE
            limit = self.offline_after_s if target is LinkStatus.OFFLINE else self.stale_after_s
            changes.append(
                StateChange(
                    machine_id=machine_id,
                    state=state,
                    previous_state=previous,
                    ts=self.clock.now(),
                    reason=f"no data for {silent:.1f} s (limit {limit:g} s)",
                )
            )
        return changes
