"""TimescaleDB-specific behaviour: continuous aggregates and late arrivals, settings from config,
migration statement splitting, and the throw-away test database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from twin_backend.storage.base import Stored, Stream
from twin_backend.storage.timescale import TimescaleEventStore, split_statements
from twin_backend.testing import TEST_DB_PREFIX, PgService, test_timescale_config
from twin_common.config.settings import TwinSettings
from twin_common.enums import Quality, SensorName
from twin_common.testing import sample_reading

pytestmark = pytest.mark.services

ING = datetime(2026, 11, 12, tzinfo=UTC)


def floor_10m(ts: datetime) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % 10, second=0, microsecond=0)


def minute_of_readings(start: datetime, n: int, machine: str = "CNC-01") -> list[Stored[Any]]:
    return [
        Stored(
            sample_reading(
                machine_id=machine,
                seq=i,
                ts=start + timedelta(seconds=i),
                temp=40.0 + (i % 7),
                vib=None if i % 5 == 0 else 2.0,
            ),
            ING,
        )
        for i in range(n)
    ]


@pytest.fixture
async def ts_store(timescale_test_db: PgService) -> Any:
    store = TimescaleEventStore(timescale_test_db.dsn, test_timescale_config())
    await store.connect()
    await store.migrate()
    await store.delete_all()
    try:
        yield store
    finally:
        await store.close()


async def run_refresh_policies(store: TimescaleEventStore) -> None:
    jobs = await store.refresh_policy_jobs()
    await store.run_job(jobs["readings_1m"])  # hierarchical: 1 min before 10 min
    await store.run_job(jobs["readings_10m"])


async def test_late_reading_in_materialized_bucket_reaches_aggregates(
    ts_store: TimescaleEventStore,
) -> None:
    """A LATE reading lands in a 1-min bucket that is already materialized. The refresh policy's
    window (start_offset >= max_late_arrival_s) reaches back far enough to re-aggregate it."""
    base = floor_10m(datetime.now(UTC) - timedelta(minutes=30))  # inside the policy window
    rows = minute_of_readings(base, 600)  # one 10-minute bucket, 1 Hz
    await ts_store.insert(Stream.READINGS, rows)
    await run_refresh_policies(ts_store)

    bucket = base + timedelta(minutes=5)
    [before] = await ts_store.history(bucket, bucket + timedelta(minutes=1), "1m")
    assert (before.n, before.n_late) == (60, 0)

    late = Stored(
        sample_reading(
            machine_id="CNC-01",
            seq=10_000,
            ts=bucket + timedelta(seconds=30, milliseconds=500),
            temp=1000.0,
            vib=None,
            quality=Quality.LATE,
        ),
        ING,
    )
    await ts_store.insert(Stream.READINGS, [late])
    # The bucket is materialized: until the policy runs again the aggregate is unchanged.
    [stale] = await ts_store.history(bucket, bucket + timedelta(minutes=1), "1m")
    assert stale.n == 60

    await run_refresh_policies(ts_store)
    [after] = await ts_store.history(bucket, bucket + timedelta(minutes=1), "1m")
    temps = [r.item.temp for r in rows if bucket <= r.item.ts < bucket + timedelta(minutes=1)]
    temps.append(1000.0)
    temp = after.sensors[SensorName.TEMP]
    assert (after.n, after.n_late) == (61, 1)
    assert (temp.count, temp.max) == (61, 1000.0)
    assert temp.mean == pytest.approx(float(np.mean(temps)), rel=1e-12)
    assert after.sensors[SensorName.VIB].count == 48  # nulls ignored, the late one has none

    [ten] = await ts_store.history(base, base + timedelta(minutes=10), "10m")
    assert (ten.n, ten.n_late) == (601, 1)
    assert ten.sensors[SensorName.TEMP].max == 1000.0


async def test_backfill_below_the_watermark_is_refreshed_on_read(
    ts_store: TimescaleEventStore,
) -> None:
    """Data older than the policy window (a replayed recording) is materialized before the
    history query that needs it, also when it lands in an already materialized bucket."""
    old = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    await ts_store.insert(Stream.READINGS, minute_of_readings(old, 120))
    [first, _second] = await ts_store.history(old, old + timedelta(minutes=2), "1m")
    assert first.n == 60
    extra = [
        Stored(
            sample_reading(
                machine_id="CNC-01", seq=50_000 + i, ts=old + timedelta(seconds=i, milliseconds=250)
            ),
            ING,
        )
        for i in range(10)
    ]
    await ts_store.insert(Stream.READINGS, extra)
    [first, _second] = await ts_store.history(old, old + timedelta(minutes=2), "1m")
    assert first.n == 70
    [ten] = await ts_store.history(old, old + timedelta(minutes=10), "10m")
    assert ten.n == 130
    assert await ts_store.pending_refreshes() == 0


async def test_old_data_reaches_aggregates_across_restarts_and_processes(
    ts_store: TimescaleEventStore, timescale_test_db: PgService
) -> None:
    """One store instance writes old sim-time data and closes (a restart, or another process);
    a new instance's 1 min / 10 min history still returns it, via the persistent queue."""
    # Move the aggregates' materialization watermark to the present, so old buckets are read
    # from the materialization only (real-time aggregation will not compute them).
    recent = floor_10m(datetime.now(UTC) - timedelta(minutes=30))
    await ts_store.insert(Stream.READINGS, minute_of_readings(recent, 60, machine="CNV-02"))
    await run_refresh_policies(ts_store)

    # Control: an old reading written without a queue entry stays invisible to the aggregates.
    hidden = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    await ts_store.pool.execute(
        "INSERT INTO readings (ts, machine_id, seq, schema_version, state, temp, run_time, "
        "quality, ingested_at) VALUES ($1, 'CNC-01', 1, 2, 'RUNNING', 50.0, 1.0, 'OK', $2)",
        hidden,
        ING,
    )
    assert await ts_store.history(hidden, hidden + timedelta(minutes=1), "1m") == []

    # "Process" A: writes 20 minutes of old sim-time readings, then goes away.
    old = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    writer = TimescaleEventStore(timescale_test_db.dsn, test_timescale_config())
    await writer.connect()
    await writer.migrate()
    rows = minute_of_readings(old, 1200)
    for i in range(0, len(rows), 500):  # several batches -> several queue entries
        await writer.insert(Stream.READINGS, rows[i : i + 500])
    await writer.close()

    # "Process" B: a fresh instance with no memory of what A wrote.
    reader = TimescaleEventStore(timescale_test_db.dsn, test_timescale_config())
    await reader.connect()
    try:
        assert await reader.pending_refreshes() == 3
        minutes = await reader.history(old, old + timedelta(minutes=20), "1m")
        assert [b.n for b in minutes] == [60] * 20
        temps = [r.item.temp for r in rows[:60]]
        assert minutes[0].sensors[SensorName.TEMP].mean == pytest.approx(
            float(np.mean(temps)), rel=1e-12
        )
        tens = await reader.history(old, old + timedelta(minutes=20), "10m")
        assert [b.n for b in tens] == [600, 600]
        assert await reader.pending_refreshes() == 0
        # The hidden row's range was never queued, so it is still not aggregated.
        assert await reader.history(hidden, hidden + timedelta(minutes=1), "1m") == []
    finally:
        await reader.close()


async def test_recent_data_is_not_queued(ts_store: TimescaleEventStore) -> None:
    recent = floor_10m(datetime.now(UTC) - timedelta(minutes=30))
    await ts_store.insert(Stream.READINGS, minute_of_readings(recent, 60))
    assert await ts_store.pending_refreshes() == 0  # the refresh policies cover it


async def test_schema_and_policies_follow_config(ts_store: TimescaleEventStore) -> None:
    pool = ts_store.pool
    hypertables = {
        r["hypertable_name"]
        for r in await pool.fetch("SELECT hypertable_name FROM timescaledb_information.hypertables")
    }
    assert {s.value for s in Stream} <= hypertables
    caggs = {
        r["view_name"]: r["materialized_only"]
        for r in await pool.fetch(
            "SELECT view_name, materialized_only FROM timescaledb_information.continuous_aggregates"
        )
    }
    assert caggs == {"readings_1m": False, "readings_10m": False}  # real-time aggregation
    cfg = ts_store.cfg
    jobs = await pool.fetch(
        "SELECT ca.view_name, j.config, j.schedule_interval FROM timescaledb_information.jobs j "
        "JOIN timescaledb_information.continuous_aggregates ca "
        "ON j.hypertable_name IN (ca.view_name, ca.materialization_hypertable_name) "
        "WHERE j.proc_name = 'policy_refresh_continuous_aggregate'"
    )
    by_view = {r["view_name"]: r for r in jobs}
    assert by_view["readings_1m"]["schedule_interval"] == timedelta(
        seconds=cfg.aggregates.m1.schedule_s
    )
    assert '"start_offset": "02:00:00"' in by_view["readings_1m"]["config"]
    assert '"start_offset": "04:00:00"' in by_view["readings_10m"]["config"]
    chunk = await pool.fetchval(
        "SELECT time_interval FROM timescaledb_information.dimensions "
        "WHERE hypertable_name = 'readings'"
    )
    assert chunk == timedelta(seconds=cfg.chunk_interval_s.readings)

    async def policy_count(proc: str) -> int:
        n: int = await pool.fetchval(
            "SELECT count(*) FROM timescaledb_information.jobs WHERE proc_name = $1", proc
        )
        return n

    assert await policy_count("policy_compression") == 0  # disabled in the shipped config
    assert await policy_count("policy_retention") == 0

    enabled = cfg.model_copy(
        update={
            "compression": cfg.compression.model_copy(update={"enabled": True}),
            "retention": cfg.retention.model_copy(update={"enabled": True}),
        }
    )
    other = TimescaleEventStore(ts_store._dsn, enabled)
    await other.connect()
    try:
        await other.migrate()
        assert await policy_count("policy_compression") == len(Stream)
        assert await policy_count("policy_retention") == len(Stream)
    finally:
        await other.close()
    # Back to the test settings: the policies are removed again.
    await ts_store.migrate()
    assert await policy_count("policy_compression") == 0
    assert await policy_count("policy_retention") == 0


async def test_unchanged_settings_are_not_reapplied(ts_store: TimescaleEventStore) -> None:
    jobs = await ts_store.refresh_policy_jobs()
    assert await ts_store.migrate() == []
    assert await ts_store.refresh_policy_jobs() == jobs  # same job ids: nothing rescheduled


def test_split_statements() -> None:
    sql = (
        "-- linetwin: no-transaction\n-- comment;\nCREATE VIEW a AS\nSELECT 1;\n\n"
        "-- another\nCREATE VIEW b AS SELECT 2;\n"
    )
    assert split_statements(sql) == [
        "CREATE VIEW a AS\nSELECT 1;",
        "-- another\nCREATE VIEW b AS SELECT 2;",
    ]


def test_tests_use_a_separate_database(timescale_test_db: PgService) -> None:
    configured = TwinSettings().pg_db
    name = timescale_test_db.display.rsplit("/", 1)[-1]
    assert name.startswith(TEST_DB_PREFIX)
    assert name != configured
