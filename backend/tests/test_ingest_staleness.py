"""Staleness monitor: STALE / OFFLINE transitions at exact fake-clock times, and recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from twin_backend.clock import FakeClock
from twin_backend.ingest.staleness import LinkStatus, StalenessMonitor
from twin_common.enums import MachineState

STALE_S = 3.0
OFFLINE_S = 5.0


def monitor(clock: FakeClock, machines: list[str] | None = None) -> StalenessMonitor:
    return StalenessMonitor(machines or ["A", "B"], STALE_S, OFFLINE_S, clock)


def test_transitions_fire_exactly_at_thresholds() -> None:
    clock = FakeClock()
    mon = monitor(clock, ["A"])
    assert mon.seen("A", MachineState.RUNNING) is None
    clock.advance(STALE_S - 0.01)
    assert mon.tick() == []
    clock.advance(0.01)  # exactly 3 s
    [stale] = mon.tick()
    assert (stale.state, stale.previous_state) == (MachineState.STALE, MachineState.RUNNING)
    assert stale.ts == clock.now()
    assert mon.tick() == []  # no repeat
    clock.advance(OFFLINE_S - STALE_S - 0.01)
    assert mon.tick() == []
    clock.advance(0.01)  # exactly 5 s
    [offline] = mon.tick()
    assert (offline.state, offline.previous_state) == (MachineState.OFFLINE, MachineState.STALE)
    assert mon.status("A") is LinkStatus.OFFLINE
    assert mon.effective_state("A") is MachineState.OFFLINE


def test_jump_straight_to_offline() -> None:
    clock = FakeClock()
    mon = monitor(clock, ["A"])
    mon.seen("A", MachineState.IDLE)
    clock.advance(10)
    [change] = mon.tick()
    assert change.state is MachineState.OFFLINE
    assert change.previous_state is MachineState.IDLE


def test_recovery_returns_to_reported_state() -> None:
    clock = FakeClock()
    mon = monitor(clock, ["A"])
    mon.seen("A", MachineState.RUNNING)
    clock.advance(OFFLINE_S)
    mon.tick()
    clock.advance(1)
    change = mon.seen("A", MachineState.WARNING)
    assert change is not None
    assert (change.state, change.previous_state) == (MachineState.WARNING, MachineState.OFFLINE)
    assert mon.status("A") is LinkStatus.LIVE
    assert mon.seen("A", MachineState.WARNING) is None


def test_data_keeps_machine_live_and_machines_independent() -> None:
    clock = FakeClock()
    mon = monitor(clock)
    for _ in range(10):
        clock.advance(1)
        mon.seen("A", MachineState.RUNNING)
        changes = mon.tick()
        assert all(c.machine_id == "B" for c in changes)
    assert mon.status("A") is LinkStatus.LIVE
    assert mon.status("B") is LinkStatus.OFFLINE  # never reported: counts from the start


def test_unknown_machine_ignored() -> None:
    mon = monitor(FakeClock())
    assert mon.seen("ZZZ", MachineState.RUNNING) is None


def test_thresholds_validated() -> None:
    with pytest.raises(ValueError, match="stale_after_s"):
        StalenessMonitor(["A"], 5.0, 5.0, FakeClock())


# ---------------------------------------------------------------- event-time stamps (Phase 3)
EVENT_T0 = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)  # sim time, far from the wall clock


@pytest.mark.parametrize("scale", [1.0, 10.0, 0.5])
def test_stale_and_offline_are_stamped_in_event_time(scale: float) -> None:
    clock = FakeClock(datetime(2026, 11, 12, tzinfo=UTC))  # wall clock: months later
    mon = StalenessMonitor(["A"], STALE_S, OFFLINE_S, clock, time_scale=scale)
    mon.seen("A", MachineState.RUNNING, EVENT_T0)
    clock.advance(STALE_S)
    [stale] = mon.tick()
    assert stale.ts == EVENT_T0 + timedelta(seconds=STALE_S * scale)
    clock.advance(OFFLINE_S - STALE_S)
    [offline] = mon.tick()
    assert offline.ts == EVENT_T0 + timedelta(seconds=OFFLINE_S * scale)


def test_recovery_carries_the_resuming_reading_ts() -> None:
    clock = FakeClock()
    mon = StalenessMonitor(["A"], STALE_S, OFFLINE_S, clock)
    mon.seen("A", MachineState.RUNNING, EVENT_T0)
    clock.advance(STALE_S)
    mon.tick()
    resumed = EVENT_T0 + timedelta(seconds=7)
    change = mon.seen("A", MachineState.IDLE, resumed)
    assert change is not None
    assert change.ts == resumed
    assert change.previous_state is MachineState.STALE


def test_silent_machine_uses_the_line_watermark() -> None:
    clock = FakeClock()
    mon = StalenessMonitor(["A", "B"], STALE_S, OFFLINE_S, clock, time_scale=2.0)
    clock.advance(1.0)
    mon.seen("A", MachineState.RUNNING, EVENT_T0)  # B never reports
    clock.advance(STALE_S - 1.0)  # B silent for 3 s since start, A for 2 s
    [b_stale] = mon.tick()
    assert b_stale.machine_id == "B"
    # watermark (A's reading) + 2 s of wall time since it arrived x scale 2
    assert b_stale.ts == EVENT_T0 + timedelta(seconds=4.0)


def test_before_any_reading_the_wall_clock_stands_in() -> None:
    clock = FakeClock()
    mon = StalenessMonitor(["A"], STALE_S, OFFLINE_S, clock)
    clock.advance(STALE_S)
    [stale] = mon.tick()
    assert stale.ts == clock.now()


def test_time_scale_must_be_positive() -> None:
    mon = monitor(FakeClock())
    with pytest.raises(ValueError, match="time_scale"):
        mon.time_scale = 0.0
