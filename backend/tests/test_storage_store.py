"""Event store contract, run on every backend in STORE_BACKENDS: exact round trips (including
partial readings and every quality flag), idempotent inserts, time ranges, ordering, snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from twin_backend.ingest.quarantine import QuarantineRecord
from twin_backend.storage.base import (
    STATE_STREAMS,
    EventStore,
    Snapshot,
    Stored,
    Stream,
    bucket_bounds,
    from_us,
    to_us,
)
from twin_common.enums import (
    AckStatus,
    AdapterKind,
    CommandType,
    MachineState,
    Quality,
    QuarantineReason,
    SensorName,
)
from twin_common.testing import (
    SAMPLE_TS,
    sample_ack,
    sample_alert,
    sample_command,
    sample_prediction,
    sample_reading,
    sample_state_change,
)

INGESTED = datetime(2026, 11, 12, 9, 0, 0, 654321, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return SAMPLE_TS + timedelta(seconds=seconds)


def stored(item: Any, ingested: datetime = INGESTED) -> Stored[Any]:
    return Stored(item, ingested)


def readings_with_every_flag() -> list[Stored[Any]]:
    rows: list[Stored[Any]] = []
    seq = 0
    for quality in Quality:
        rows.append(stored(sample_reading(seq=seq, ts=at(seq), quality=quality)))
        seq += 1
    for sensor in SensorName:  # one missing sensor each
        rows.append(
            stored(
                sample_reading(seq=seq, ts=at(seq), quality=Quality.SUSPECT, **{sensor.value: None})
            )
        )
        seq += 1
    # several missing, odd floats, a LATE partial reading, microsecond timestamps
    rows.append(
        stored(
            sample_reading(
                seq=seq,
                ts=at(seq) + timedelta(microseconds=7),
                temp=-3.0000000000000004,
                vib=None,
                current=None,
                load=1e-300,
                quality=Quality.LATE,
            )
        )
    )
    return rows


async def test_partial_readings_and_quality_flags_round_trip(event_store: EventStore) -> None:
    rows = readings_with_every_flag()
    assert await event_store.insert(Stream.READINGS, rows) == len(rows)
    back = await event_store.fetch(Stream.READINGS)
    assert back == rows  # items and ingested_at, exactly
    assert {r.item.quality for r in back} == set(Quality)
    assert any(r.item.temp is None for r in back)


QUARANTINE = QuarantineRecord(
    ts=INGESTED,
    source=AdapterKind.FILE,
    reason=QuarantineReason.NO_SENSOR_DATA,
    stage="validate",
    detail="every sensor value of CNC-01 is missing",
    machine_id="CNC-01",
    payload={"machineId": "CNC-01", "temp": None, "nested": {"a": [1, 2]}},
    meta={"file": "run.csv", "line": "7"},
)

EVERY_STREAM: dict[Stream, list[Any]] = {
    Stream.STATE_CHANGES: [
        sample_state_change(),
        sample_state_change(state=MachineState.RUNNING, previous_state=None, reason=None, ts=at(1)),
    ],
    Stream.PREDICTIONS: [
        sample_prediction(),
        sample_prediction(horizon_s=10.0, ttf_s=None, ttf_interval=None, attributions=[]),
    ],
    Stream.COMMANDS: [
        sample_command(),
        sample_command(command_id="cmd-2", type=CommandType.STOP, params={}),
    ],
    Stream.COMMAND_ACKS: [sample_ack(), sample_ack(status=AckStatus.REJECTED, reason="no")],
    Stream.ALERTS: [
        sample_alert(),
        sample_alert(alert_id="alert-2", ts=at(5)),
        sample_alert(
            alert_id="alert-2",
            ts=at(6),
            active=False,
            acknowledged=True,
            acked_by="alice",
            acked_at=at(100),
            rev=1,
        ),
    ],
    Stream.QUARANTINE: [
        QUARANTINE,
        QUARANTINE.model_copy(update={"payload": "\x00garbage", "machine_id": None}),
    ],
}


@pytest.mark.parametrize("stream", list(EVERY_STREAM), ids=str)
async def test_every_stream_round_trips(event_store: EventStore, stream: Stream) -> None:
    rows = [stored(item) for item in EVERY_STREAM[stream]]
    assert await event_store.insert(stream, rows) == len(rows)
    back = await event_store.fetch(stream)
    assert sorted(back, key=repr) == sorted(rows, key=repr)
    assert await event_store.count(stream) == len(rows)


async def test_inserts_are_idempotent(event_store: EventStore) -> None:
    rows = readings_with_every_flag()
    await event_store.insert(Stream.READINGS, rows)
    # Same events again (a retried batch or a replayed export): nothing added.
    assert await event_store.insert(Stream.READINGS, rows) == 0
    for stream, items in EVERY_STREAM.items():
        batch = [stored(i) for i in items]
        await event_store.insert(stream, batch)
        expected = len(items) if stream is Stream.QUARANTINE else 0  # no natural key
        assert await event_store.insert(stream, batch) == expected
    assert await event_store.count(Stream.READINGS) == len(rows)


async def test_same_seq_after_a_source_restart_is_a_new_event(event_store: EventStore) -> None:
    a = stored(sample_reading(seq=0, ts=at(0)))
    b = stored(sample_reading(seq=0, ts=at(100)))  # seq restarted later
    assert await event_store.insert(Stream.READINGS, [a, b]) == 2


async def test_time_ranges_machine_filter_and_order(event_store: EventStore) -> None:
    rows = [
        stored(sample_reading(machine_id=m, seq=s, ts=at(s)))
        for s in range(10)
        for m in ("INSP-01", "CNC-01")
    ]
    await event_store.insert(Stream.READINGS, list(reversed(rows)))
    window = await event_store.fetch(Stream.READINGS, at(2), at(5))  # [2, 5)
    assert [(r.item.seq, r.item.machine_id) for r in window] == [
        (s, m) for s in (2, 3, 4) for m in ("CNC-01", "INSP-01")
    ]
    only = await event_store.fetch(Stream.READINGS, machine_ids=["INSP-01"])
    assert {r.item.machine_id for r in only} == {"INSP-01"}
    assert len(only) == 10


async def test_fetch_replay_is_half_open_and_merged(event_store: EventStore) -> None:
    await event_store.insert(
        Stream.READINGS, [stored(sample_reading(seq=s, ts=at(s))) for s in range(5)]
    )
    await event_store.insert(Stream.STATE_CHANGES, [stored(sample_state_change(ts=at(2)))])
    await event_store.insert(Stream.PREDICTIONS, [stored(sample_prediction(ts=at(2)))])
    await event_store.insert(Stream.ALERTS, [stored(sample_alert(ts=at(2)))])
    got = await event_store.fetch_replay(STATE_STREAMS, at(1), at(3))  # (1, 3]
    kinds = [(type(r.item).__name__, r.item.ts) for r in got]
    assert kinds == [
        ("Reading", at(2)),
        ("StateChange", at(2)),
        ("Prediction", at(2)),
        ("Reading", at(3)),
    ]
    everything = await event_store.fetch_replay(STATE_STREAMS, None, at(10))
    assert len(everything) == 7
    assert await event_store.min_max_ts(STATE_STREAMS) == (at(0), at(4))


async def test_empty_store(event_store: EventStore) -> None:
    assert await event_store.min_max_ts(list(Stream)) is None
    assert await event_store.fetch(Stream.READINGS) == []
    assert await event_store.latest_snapshot() is None
    assert await event_store.insert(Stream.READINGS, []) == 0


async def test_snapshots(event_store: EventStore) -> None:
    for s in (0, 60, 120):
        await event_store.save_snapshot(Snapshot(at(s), f'{{"t": {s}}}', INGESTED))
    latest = await event_store.latest_snapshot(at(119))
    assert latest is not None
    assert latest.ts == at(60)
    assert latest.state_json == '{"t": 60}'
    assert latest.ingested_at == INGESTED
    newest = await event_store.latest_snapshot()
    assert newest is not None
    assert newest.ts == at(120)
    assert await event_store.invalidate_snapshots_from(at(60)) == 2
    assert await event_store.snapshot_times() == [at(0)]


async def test_delete_all(event_store: EventStore) -> None:
    await event_store.insert(Stream.READINGS, readings_with_every_flag())
    await event_store.save_snapshot(Snapshot(at(0), "{}", INGESTED))
    await event_store.delete_all()
    assert await event_store.count(Stream.READINGS) == 0
    assert await event_store.snapshot_times() == []


def test_microsecond_conversion_is_exact() -> None:
    for ts in (SAMPLE_TS, at(0.000001), datetime(1970, 1, 1, tzinfo=UTC), at(10**8)):
        assert from_us(to_us(ts)) == ts


def test_bucket_bounds_cover_buckets_starting_in_range() -> None:
    t0 = datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 0, 3, 0, tzinfo=UTC)
    lo, hi = bucket_bounds(t0, t1, 60)
    assert (lo, hi) == (
        datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
    )
