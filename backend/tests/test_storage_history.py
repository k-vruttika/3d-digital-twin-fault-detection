"""History downsampling (1 s / 1 min / 10 min) on every backend against a numpy reference on
the same seeded data, including missing (null) sensor values and LATE / SUSPECT readings."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from twin_backend.storage.base import (
    RESOLUTION_SECONDS,
    EventStore,
    HistoryBucket,
    Resolution,
    Stored,
    Stream,
    bucket_bounds,
    from_us,
    to_us,
)
from twin_common.enums import MachineState, Quality, SensorName
from twin_common.testing import sample_reading

T0 = datetime(2026, 1, 1, 6, 0, 0, tzinfo=UTC)
ING = datetime(2026, 11, 12, tzinfo=UTC)
MACHINES = ("CNV-01", "CNC-01", "INSP-01")
SENSORS = [s.value for s in SensorName]
DURATION_S = 1500


def dataset(seed: int = 42) -> list[Stored[Any]]:
    """1 Hz per machine with sub-second jitter, ~20 % missing values, LATE and SUSPECT rows."""
    rng = np.random.default_rng(seed)
    rows: list[Stored[Any]] = []
    for m_index, machine in enumerate(MACHINES):
        for seq in range(DURATION_S):
            if machine == "INSP-01" and 300 <= seq < 1000:
                continue  # a long gap: buckets with no data at all
            jitter_us = int(rng.integers(0, 999_999))
            values: dict[str, float | None] = {
                "speed": float(rng.normal(1000 + 100 * m_index, 50)),
                "temp": float(rng.normal(50, 5)),
                "vib": float(abs(rng.normal(2, 0.5))),
                "load": float(rng.uniform(30, 90)),
                "current": float(abs(rng.normal(10, 2))),
            }
            missing = rng.random(len(SENSORS)) < 0.2
            if machine == "CNC-01" and 600 <= seq < 700:
                missing[:] = True  # one sensor left: a run of nearly empty readings
                missing[0] = False
            for sensor, gone in zip(SENSORS, missing, strict=True):
                if gone:
                    values[sensor] = None
            u = rng.random()
            quality = (
                Quality.LATE if u < 0.05 else (Quality.SUSPECT if missing.any() else Quality.OK)
            )
            reading = sample_reading(
                machine_id=machine,
                seq=seq,
                ts=T0 + timedelta(seconds=seq, microseconds=jitter_us),
                state=MachineState.RUNNING,
                quality=quality,
                **values,
            )
            rows.append(Stored(reading, ING))
    return rows


def reference(
    rows: list[Stored[Any]],
    t0: datetime,
    t1: datetime,
    width_s: int,
    machines: tuple[str, ...] | None,
) -> list[HistoryBucket]:
    from twin_backend.storage.base import SensorStats

    lo, hi = bucket_bounds(t0, t1, width_s)
    w = width_s * 1_000_000
    groups: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for r in rows:
        item = r.item
        if not (lo <= item.ts < hi) or (machines is not None and item.machine_id not in machines):
            continue
        groups[(to_us(item.ts) // w * w, item.machine_id)].append(item)
    out: list[HistoryBucket] = []
    for (bucket, machine), items in sorted(groups.items()):
        stats = {}
        for sensor in SensorName:
            arr = np.array(
                [
                    np.nan if getattr(i, sensor.value) is None else getattr(i, sensor.value)
                    for i in items
                ],
                dtype=float,
            )
            count = int(np.count_nonzero(~np.isnan(arr)))
            stats[sensor] = SensorStats(
                count=count,
                mean=float(np.nanmean(arr)) if count else None,
                min=float(np.nanmin(arr)) if count else None,
                max=float(np.nanmax(arr)) if count else None,
            )
        out.append(
            HistoryBucket(
                bucket=from_us(bucket),
                machine_id=machine,
                n=len(items),
                n_late=sum(i.quality is Quality.LATE for i in items),
                n_suspect=sum(i.quality is Quality.SUSPECT for i in items),
                sensors=stats,
            )
        )
    return out


def assert_buckets_match(got: list[HistoryBucket], want: list[HistoryBucket]) -> None:
    assert [(b.bucket, b.machine_id) for b in got] == [(b.bucket, b.machine_id) for b in want]
    for g, w in zip(got, want, strict=True):
        assert (g.n, g.n_late, g.n_suspect) == (w.n, w.n_late, w.n_suspect), g.bucket
        for sensor in SensorName:
            gs, ws = g.sensors[sensor], w.sensors[sensor]
            assert gs.count == ws.count, (g.bucket, g.machine_id, sensor)
            if ws.count == 0:
                assert (gs.mean, gs.min, gs.max) == (None, None, None)
            else:
                assert gs.mean == pytest.approx(ws.mean, rel=1e-12)
                assert (gs.min, gs.max) == (ws.min, ws.max)


@pytest.fixture
async def filled_store(event_store: EventStore) -> tuple[EventStore, list[Stored[Any]]]:
    rows = dataset()
    await event_store.insert(Stream.READINGS, rows)
    return event_store, rows


@pytest.mark.parametrize("resolution", list(RESOLUTION_SECONDS))
async def test_downsampling_matches_numpy_reference(
    filled_store: tuple[EventStore, list[Stored[Any]]], resolution: Resolution
) -> None:
    store, rows = filled_store
    t0, t1 = T0, T0 + timedelta(seconds=DURATION_S + 1)
    if resolution == "1s":
        t1 = T0 + timedelta(seconds=200)  # keep the 1 s case small
    got = await store.history(t0, t1, resolution)
    want = reference(rows, t0, t1, RESOLUTION_SECONDS[resolution], None)
    assert len(want) > 0
    assert_buckets_match(got, want)


async def test_unaligned_range_and_machine_filter(
    filled_store: tuple[EventStore, list[Stored[Any]]],
) -> None:
    store, rows = filled_store
    t0 = T0 + timedelta(seconds=95, milliseconds=500)
    t1 = T0 + timedelta(seconds=1234)
    got = await store.history(t0, t1, "1m", machine_ids=["CNC-01", "INSP-01"])
    want = reference(rows, t0, t1, 60, ("CNC-01", "INSP-01"))
    assert_buckets_match(got, want)
    assert got[0].bucket == T0 + timedelta(minutes=2)  # first bucket starting at or after t0


async def test_nulls_ignored_and_counted(
    filled_store: tuple[EventStore, list[Stored[Any]]],
) -> None:
    store, _rows = filled_store
    got = await store.history(
        T0 + timedelta(seconds=600), T0 + timedelta(seconds=660), "1m", machine_ids=["CNC-01"]
    )
    [bucket] = got
    # Only speed was reported in this minute.
    assert bucket.sensors[SensorName.SPEED].count == bucket.n
    for sensor in (SensorName.TEMP, SensorName.VIB, SensorName.LOAD, SensorName.CURRENT):
        stats = bucket.sensors[sensor]
        assert (stats.count, stats.mean, stats.min, stats.max) == (0, None, None, None)
    assert bucket.n_suspect + bucket.n_late == bucket.n


async def test_gap_has_no_buckets(filled_store: tuple[EventStore, list[Stored[Any]]]) -> None:
    store, _ = filled_store
    got = await store.history(
        T0 + timedelta(seconds=400), T0 + timedelta(seconds=900), "1m", machine_ids=["INSP-01"]
    )
    assert got == []
