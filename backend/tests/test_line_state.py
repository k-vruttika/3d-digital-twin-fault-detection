"""LineState reducer: rules, LATE handling, precedence, and order independence."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.engine.line_state import (
    LineState,
    LiveStateTracker,
    StateEvent,
    apply_event,
    fold,
    new_line_state,
)
from twin_common.enums import MachineState, Quality
from twin_common.testing import (
    SAMPLE_TS,
    sample_prediction,
    sample_reading,
    sample_state_change,
)

MACHINES = ["CNV-01", "CNC-01"]


def at(seconds: float) -> datetime:
    return SAMPLE_TS + timedelta(seconds=seconds)


def reading(machine: str, seq: int, t: float, **kw: object) -> StateEvent:
    return sample_reading(machine_id=machine, seq=seq, ts=at(t), **kw)


def test_latest_reading_wins_and_sets_state() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, reading("CNC-01", 2, 2, state=MachineState.WARNING))
    apply_event(state, reading("CNC-01", 1, 1, state=MachineState.RUNNING))  # older: ignored
    view = state.machines["CNC-01"]
    assert view.last_reading is not None
    assert view.last_reading.seq == 2
    assert view.state is MachineState.WARNING
    assert view.reported_state is MachineState.WARNING
    assert view.state_ts == at(2)
    assert state.machines["CNV-01"].state is None


def test_late_readings_are_never_applied() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, reading("CNC-01", 1, 1))
    changed = apply_event(
        state, reading("CNC-01", 0, 5, quality=Quality.LATE, state=MachineState.FAULT)
    )
    assert not changed
    view = state.machines["CNC-01"]
    assert view.last_reading is not None
    assert view.last_reading.seq == 1
    assert view.state is MachineState.RUNNING


def test_suspect_partial_readings_are_applied() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, reading("CNC-01", 1, 1, temp=None, quality=Quality.SUSPECT))
    view = state.machines["CNC-01"]
    assert view.last_reading is not None
    assert view.last_reading.temp is None


def test_stale_change_overrides_older_reading_until_newer_data() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, reading("CNC-01", 1, 1))
    apply_event(
        state,
        sample_state_change(machine_id="CNC-01", state=MachineState.STALE, ts=at(4)),
    )
    view = state.machines["CNC-01"]
    assert view.state is MachineState.STALE
    assert view.reported_state is MachineState.RUNNING
    apply_event(state, reading("CNC-01", 2, 5, state=MachineState.IDLE))
    assert state.machines["CNC-01"].state is MachineState.IDLE


def test_tie_goes_to_state_change() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, reading("CNC-01", 1, 3))
    apply_event(
        state, sample_state_change(machine_id="CNC-01", state=MachineState.OFFLINE, ts=at(3))
    )
    assert state.machines["CNC-01"].state is MachineState.OFFLINE


def test_newest_prediction_per_horizon() -> None:
    state = new_line_state(MACHINES)
    apply_event(state, sample_prediction(horizon_s=10.0, ts=at(2), model_version="b"))
    apply_event(state, sample_prediction(horizon_s=10.0, ts=at(1), model_version="z"))
    apply_event(state, sample_prediction(horizon_s=30.0, ts=at(1)))
    preds = state.predictions["CNC-01"]
    assert set(preds) == {"10", "30"}
    assert preds["10"].model_version == "b"


def _events(rng: random.Random) -> list[StateEvent]:
    events: list[StateEvent] = []
    for machine in MACHINES:
        for seq in range(30):
            quality = Quality.LATE if rng.random() < 0.1 else Quality.OK
            events.append(
                reading(
                    machine,
                    seq,
                    seq,
                    quality=quality,
                    state=rng.choice([MachineState.RUNNING, MachineState.WARNING]),
                    temp=None if rng.random() < 0.2 else 50.0 + seq,
                )
            )
        for t in (7.5, 12.0, 21.5):
            events.append(
                sample_state_change(
                    machine_id=machine,
                    state=rng.choice([MachineState.STALE, MachineState.OFFLINE]),
                    ts=at(t),
                )
            )
    for t in range(0, 30, 3):
        events.append(sample_prediction(ts=at(t), horizon_s=rng.choice([10.0, 30.0])))
    return events


def test_fold_is_order_independent() -> None:
    rng = random.Random(3)
    events = _events(rng)
    reference = fold(new_line_state(MACHINES), events).body()
    for seed in range(20):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert fold(new_line_state(MACHINES), shuffled).body() == reference


def test_state_json_round_trip() -> None:
    state = fold(new_line_state(MACHINES), _events(random.Random(5)))
    state.as_of = at(100)
    restored = LineState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.watermark() == state.watermark()


async def test_live_tracker_follows_the_bus() -> None:
    bus = InMemoryEventBus(100)
    seen: list[StateEvent] = []
    tracker = LiveStateTracker(bus, MACHINES, on_apply=lambda e, _s: seen.append(e))
    await bus.publish(Topic.READINGS, reading("CNC-01", 1, 1))
    await bus.publish(
        Topic.STATES, sample_state_change(machine_id="CNV-01", state=MachineState.STALE, ts=at(2))
    )
    assert tracker.drain() == 2
    assert len(seen) == 2
    assert tracker.state.machines["CNV-01"].state is MachineState.STALE
    assert tracker.state.as_of == at(2)
    tracker.start()
    await bus.publish(Topic.READINGS, reading("CNC-01", 2, 3, state=MachineState.FAULT))
    await tracker.stop()
    assert tracker.state.machines["CNC-01"].state is MachineState.FAULT
