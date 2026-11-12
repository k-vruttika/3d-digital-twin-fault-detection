"""Staleness monitor: a machine silent for ``stale_after_s`` is STALE, for ``offline_after_s``
OFFLINE. When data returns, the machine goes back to the state its reading reports.

Silence is measured on the injectable clock (monotonic seconds), so tests advance a fake clock
instead of sleeping. Machines that never reported count from the monitor's start.

**Timestamps are on the event-time axis** (ADR 0007 amendment, ADR 0010), the same axis as
``Reading.ts``, so stored state changes and readings can be replayed together:

- STALE / OFFLINE: the machine's last live reading ts + wall silence x ``time_scale``
  (event seconds per wall second: the file adapter's replay speed, 1.0 for live sources);
- recovery: the ts of the reading that resumed the data;
- a machine that never reported: the line-wide watermark (the newest reading ts of any
  machine, extrapolated the same way); before any reading at all, ``clock.now()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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
    last_ts: datetime | None = None  # event time of that reading


class StalenessMonitor:
    def __init__(
        self,
        machine_ids: list[str],
        stale_after_s: float,
        offline_after_s: float,
        clock: Clock,
        *,
        time_scale: float = 1.0,
    ) -> None:
        if not 0 < stale_after_s < offline_after_s:
            raise ValueError("need 0 < stale_after_s < offline_after_s")
        self.stale_after_s = stale_after_s
        self.offline_after_s = offline_after_s
        self.clock = clock
        self.time_scale = time_scale
        start = clock.monotonic()
        self._links = {m: _Link(start, LinkStatus.LIVE, None) for m in machine_ids}
        # Line-wide watermark: newest reading ts seen and the monotonic time it arrived.
        self._watermark: tuple[datetime, float] | None = None

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        if value <= 0:
            raise ValueError("time_scale must be > 0")
        self._time_scale = value

    def status(self, machine_id: str) -> LinkStatus:
        return self._links[machine_id].status

    def effective_state(self, machine_id: str) -> MachineState | None:
        link = self._links[machine_id]
        if link.status is LinkStatus.STALE:
            return MachineState.STALE
        if link.status is LinkStatus.OFFLINE:
            return MachineState.OFFLINE
        return link.last_state

    def seen(
        self, machine_id: str, state: MachineState, ts: datetime | None = None
    ) -> StateChange | None:
        """A live (not late) reading arrived. Returns a recovery StateChange if it was down.

        ``ts`` is the reading's event time; without it the wall clock stands in.
        """
        link = self._links.get(machine_id)
        if link is None:
            return None
        now_mono = self.clock.monotonic()
        event_ts = ts if ts is not None else self.clock.now()
        link.last_seen = now_mono
        link.last_state = state
        link.last_ts = event_ts
        if self._watermark is None or event_ts >= self._watermark[0]:
            self._watermark = (event_ts, now_mono)
        if link.status is LinkStatus.LIVE:
            return None
        previous = self.effective_state(machine_id)
        link.status = LinkStatus.LIVE
        return StateChange(
            machine_id=machine_id,
            state=state,
            previous_state=previous,
            ts=event_ts,
            reason="data resumed",
        )

    def _event_time(self, link: _Link, silent: float, now_mono: float) -> datetime:
        if link.last_ts is not None:
            return link.last_ts + timedelta(seconds=silent * self._time_scale)
        if self._watermark is not None:
            wm_ts, wm_mono = self._watermark
            return wm_ts + timedelta(seconds=(now_mono - wm_mono) * self._time_scale)
        return self.clock.now()

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
                    ts=self._event_time(link, silent, now),
                    reason=f"no data for {silent:.1f} s (limit {limit:g} s)",
                )
            )
        return changes
