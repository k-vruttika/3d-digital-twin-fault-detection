"""Snapshots and reconstruct on every backend: aligned snapshot times, snapshot == replay,
invalidation when older data arrives later, LATE readings ignored."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.storage.base import EventStore, Stored, Stream
from twin_backend.storage.replay import iter_states, reconstruct
from twin_backend.storage.service import StorageService
from twin_backend.storage.snapshots import next_boundary
from twin_backend.testing import storage_config
from twin_common.enums import MachineState, Quality
from twin_common.testing import sample_reading, sample_state_change

T0 = datetime(2026, 1, 1, 6, 0, 0, tzinfo=UTC)
MACHINES = ["CNV-01", "CNC-01"]
ING = datetime(2026, 11, 12, tzinfo=UTC)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def run_rows(seconds: int) -> list[Stored[Any]]:
    rows: list[Stored[Any]] = []
    for s in range(seconds):
        for m in MACHINES:
            state = MachineState.WARNING if m == "CNC-01" and s >= 100 else MachineState.RUNNING
            rows.append(Stored(sample_reading(machine_id=m, seq=s, ts=at(s), state=state), ING))
    return rows


def test_next_boundary() -> None:
    assert next_boundary(at(0), 60, strictly_after=False) == at(0)
    assert next_boundary(at(0), 60, strictly_after=True) == at(60)
    assert next_boundary(at(0.5), 60, strictly_after=False) == at(60)


def service(store: EventStore, **snap: Any) -> StorageService:
    cfg = storage_config()
    cfg = cfg.model_copy(update={"snapshots": cfg.snapshots.model_copy(update=snap)})
    return StorageService(store, cfg, InMemoryEventBus(1000), MACHINES, clock=FakeClock())


async def test_snapshots_are_aligned_and_equal_replay(event_store: EventStore) -> None:
    await event_store.insert(Stream.READINGS, run_rows(250))
    svc = service(event_store, interval_s=60, settle_s=5)
    # watermark 249 s - settle 5 s = 244 s: snapshots at 0, 60, 120, 180, 240
    assert await svc.snapshotter.run_once() == 5
    assert await event_store.snapshot_times() == [at(0), at(60), at(120), at(180), at(240)]
    assert await svc.snapshotter.run_once() == 0  # nothing new is due
    for t in (at(0), at(59.5), at(60), at(130), at(249), at(400)):
        fast = await reconstruct(event_store, MACHINES, t)
        slow = await reconstruct(event_store, MACHINES, t, use_snapshots=False)
        assert fast == slow
    late_view = (await reconstruct(event_store, MACHINES, at(130))).machines["CNC-01"]
    assert late_view.state is MachineState.WARNING
    assert late_view.last_reading is not None
    assert late_view.last_reading.seq == 130


async def test_older_data_arriving_later_invalidates_snapshots(event_store: EventStore) -> None:
    rows = run_rows(200)
    held_back = [r for r in rows if r.item.machine_id == "CNV-01" and 50 <= r.item.seq < 70]
    await event_store.insert(Stream.READINGS, [r for r in rows if r not in held_back])
    svc = service(event_store, interval_s=60, settle_s=0)
    await svc.snapshotter.run_once()
    assert await event_store.snapshot_times() == [at(0), at(60), at(120), at(180)]
    # A backlog (in order, not LATE) for t = 50..69 is committed now.
    await event_store.insert(Stream.READINGS, held_back)
    await svc._on_committed(Stream.READINGS, held_back)
    assert await event_store.snapshot_times() == [at(0)]
    assert svc.metrics.value("snapshot_invalidations") == 3
    await svc.snapshotter.run_once(settle_s=0)
    for t in (at(65), at(150)):
        assert await reconstruct(event_store, MACHINES, t) == await reconstruct(
            event_store, MACHINES, t, use_snapshots=False
        )


async def test_late_readings_do_not_invalidate_and_are_not_applied(
    event_store: EventStore,
) -> None:
    await event_store.insert(Stream.READINGS, run_rows(130))
    svc = service(event_store, interval_s=60, settle_s=0)
    await svc.snapshotter.run_once()
    late = [
        Stored(
            sample_reading(
                machine_id="CNV-01",
                seq=10_000,
                ts=at(30),
                quality=Quality.LATE,
                state=MachineState.FAULT,
            ),
            ING,
        )
    ]
    await event_store.insert(Stream.READINGS, late)
    await svc._on_committed(Stream.READINGS, late)
    assert len(await event_store.snapshot_times()) == 3
    state = await reconstruct(event_store, MACHINES, at(30), use_snapshots=False)
    assert state.machines["CNV-01"].state is MachineState.RUNNING


async def test_state_changes_are_replayed(event_store: EventStore) -> None:
    await event_store.insert(Stream.READINGS, run_rows(20))
    await event_store.insert(
        Stream.STATE_CHANGES,
        [
            Stored(
                sample_state_change(machine_id="CNC-01", state=MachineState.STALE, ts=at(19.5)), ING
            )
        ],
    )
    assert (await reconstruct(event_store, MACHINES, at(19))).machines["CNC-01"].state is (
        MachineState.RUNNING
    )
    assert (await reconstruct(event_store, MACHINES, at(19.5))).machines["CNC-01"].state is (
        MachineState.STALE
    )


async def test_iter_states_matches_reconstruct(event_store: EventStore) -> None:
    await event_store.insert(Stream.READINGS, run_rows(120))
    states = [
        s async for s in iter_states(event_store, MACHINES, at(10), at(100), timedelta(seconds=15))
    ]
    assert [s.as_of for s in states] == [at(10 + 15 * k) for k in range(7)]
    for s in states:
        assert s.as_of is not None
        assert s == await reconstruct(event_store, MACHINES, s.as_of)


async def test_service_lifecycle_persists_bus_events(tmp_path: Path) -> None:
    from twin_backend.storage.sqlite import SqliteEventStore

    bus = InMemoryEventBus(1000)
    store = SqliteEventStore(tmp_path / "svc.sqlite")
    svc = StorageService(store, storage_config(), bus, MACHINES)
    await svc.start()
    for s in range(5):
        await bus.publish(Topic.READINGS, sample_reading(machine_id="CNC-01", seq=s, ts=at(s)))
    await svc.stop()  # drains the writer
    reopened = SqliteEventStore(tmp_path / "svc.sqlite")
    await reopened.connect()
    assert await reopened.count(Stream.READINGS) == 5
    await reopened.close()
