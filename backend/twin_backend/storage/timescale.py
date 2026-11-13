"""TimescaleDB event store (PostgreSQL 16 + timescaledb, via asyncpg). See ADR 0009.

Imported only when ``storage.backend: timescale`` is selected, so minimal mode never loads
asyncpg.

- **Hypertables** for the seven event tables (``migrations/timescale/0001_init.sql``).
- **Batch inserts:** ``COPY`` into a temporary staging table, then ``INSERT ... SELECT ... ON
  CONFLICT DO NOTHING``, one transaction per batch (idempotent like the SQLite store).
- **History:** 1 s buckets are computed from raw readings; 1 min / 10 min come from the
  continuous aggregates ``readings_1m`` / ``readings_10m`` (real-time aggregation on).
- **Late data and the aggregates:**
  - The refresh policies re-materialize the last ``start_offset_s`` of wall-clock time on every
    run, so LATE readings up to ``max_late_arrival_s`` old reach the aggregates by the next run.
  - Readings older than the policy window (e.g. a replayed recording, or a backfill) are
    recorded in the table ``aggregate_refresh_queue``, in the same transaction as the readings.
    Before a 1 min / 10 min history query, the pending ranges that overlap it are refreshed and
    then deleted. The queue lives in the database, so this works across processes and restarts.
- **Settings from config** (chunk intervals, refresh / compression / retention policies) are
  applied by ``migrate()``; a fingerprint in ``linetwin_settings`` skips re-applying unchanged
  settings, so restarts do not reschedule the background jobs.
- Text values are compared with ``COLLATE "C"`` so ordering matches the SQLite store.
- PostgreSQL text cannot contain NUL; NUL characters in quarantine ``detail`` / ``stage`` (and
  other plain text fields) are stored as U+FFFD. JSON payloads keep them (escaped as \\u0000).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from twin_backend.storage.base import (
    RESOLUTION_SECONDS,
    HistoryBucket,
    Resolution,
    SensorStats,
    Snapshot,
    Stored,
    Stream,
    bucket_bounds,
    sort_key,
)
from twin_backend.storage.migrator import load_migrations, pending
from twin_backend.storage.rows import COLUMNS, ORDER_BY, UNIQUE_KEYS, decode, encode
from twin_common.config.storage import TimescaleStoreConfig
from twin_common.enums import Quality, SensorName
from twin_common.logging import get_logger

log = get_logger(__name__)

AGGREGATES: dict[str, str] = {"1m": "readings_1m", "10m": "readings_10m"}
_NUMERIC_COLUMNS = {"ts", "ingested_at", "seq", "horizon_s", "schema_version", "rev"}
_TEN_MIN_US = 600 * 1_000_000


def _identity(ts: datetime) -> datetime:
    return ts


def _from_db(ts: datetime) -> datetime:
    return ts.astimezone(UTC)


def _order_by(stream: Stream) -> str:
    return ", ".join(
        col if col in _NUMERIC_COLUMNS else f'{col} COLLATE "C"' for col in ORDER_BY[stream]
    )


def _clean(value: Any) -> Any:
    return value.replace("\x00", "�") if isinstance(value, str) else value


def split_statements(sql: str) -> list[str]:
    """Split a migration into statements (for files run outside a transaction).

    Statements end with ``;`` at the end of a line; the migration files contain no semicolons
    inside string literals.
    """
    out: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        current.append(line)
        if line.rstrip().endswith(";"):
            stmt = "\n".join(current).strip()
            body = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            if "".join(body).strip():
                out.append(stmt)
            current = []
    tail = "\n".join(ln for ln in current if not ln.strip().startswith("--")).strip()
    if tail:
        out.append(tail)
    return out


def _interval(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)


class TimescaleEventStore:
    backend = "timescale"

    def __init__(self, dsn: str, cfg: TimescaleStoreConfig) -> None:
        self._dsn = dsn  # contains the password: never log it
        self.cfg = cfg
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("TimescaleEventStore is not connected")
        return self._pool

    # -- lifecycle ---------------------------------------------------------------------------
    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self.cfg.pool_min,
                max_size=self.cfg.pool_max,
                command_timeout=self.cfg.command_timeout_s,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> list[str]:
        done: list[str] = []
        async with self.pool.acquire() as conn:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version integer PRIMARY KEY, name text NOT NULL, checksum text NOT NULL,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
            # One migrator at a time (e.g. two processes starting together).
            await conn.execute("SELECT pg_advisory_lock(hashtext('linetwin_migrate'))")
            try:
                rows = await conn.fetch("SELECT version, name, checksum FROM schema_migrations")
                applied = {r["version"]: (r["name"], r["checksum"]) for r in rows}
                for mig in pending(load_migrations("timescale"), applied):
                    record = (
                        "INSERT INTO schema_migrations (version, name, checksum) "
                        "VALUES ($1, $2, $3)"
                    )
                    if mig.transactional:
                        async with conn.transaction():
                            await conn.execute(mig.sql)
                            await conn.execute(record, mig.version, mig.name, mig.checksum)
                    else:
                        for stmt in split_statements(mig.sql):
                            await conn.execute(stmt)
                        await conn.execute(record, mig.version, mig.name, mig.checksum)
                    done.append(f"{mig.version:04d}_{mig.name}")
                    log.info("storage.migrated", backend=self.backend, migration=done[-1])
                await self._apply_settings(conn)
            finally:
                await conn.execute("SELECT pg_advisory_unlock(hashtext('linetwin_migrate'))")
        return done

    def settings_fingerprint(self) -> str:
        # The stream list is part of it: a table added by a migration gets the settings too.
        text = self.cfg.model_dump_json() + "|" + ",".join(s.value for s in Stream)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _apply_settings(self, conn: asyncpg.Connection) -> None:
        """Chunk intervals and policies from config (only when the config changed)."""
        fingerprint = self.settings_fingerprint()
        current = await conn.fetchval(
            "SELECT value FROM linetwin_settings WHERE key = 'timescale_config'"
        )
        if current == fingerprint:
            return
        cfg = self.cfg
        async with conn.transaction():
            for stream in Stream:
                seconds = (
                    cfg.chunk_interval_s.readings
                    if stream is Stream.READINGS
                    else cfg.chunk_interval_s.default
                )
                await conn.execute(
                    "SELECT set_chunk_time_interval($1::regclass, $2::interval)",
                    stream.value,
                    _interval(seconds),
                )
            for name, policy in (
                ("readings_1m", cfg.aggregates.m1),
                ("readings_10m", cfg.aggregates.m10),
            ):
                await conn.execute(
                    "SELECT remove_continuous_aggregate_policy($1::regclass, if_exists => true)",
                    name,
                )
                await conn.execute(
                    "SELECT add_continuous_aggregate_policy($1::regclass, "
                    "start_offset => $2::interval, end_offset => $3::interval, "
                    "schedule_interval => $4::interval, "
                    # First run one interval from now, not at once (no refresh burst on start).
                    "initial_start => now() + $4::interval)",
                    name,
                    _interval(policy.start_offset_s),
                    _interval(policy.end_offset_s),
                    _interval(policy.schedule_s),
                )
            for stream in Stream:
                table = stream.value
                await conn.execute(
                    "SELECT remove_compression_policy($1::regclass, if_exists => true)", table
                )
                await conn.execute(
                    "SELECT remove_retention_policy($1::regclass, if_exists => true)", table
                )
                if cfg.compression.enabled:
                    segment = "machine_id" if "machine_id" in COLUMNS[stream] else ""
                    await conn.execute(
                        f"ALTER TABLE {table} SET (timescaledb.compress, "
                        f"timescaledb.compress_segmentby = '{segment}', "
                        "timescaledb.compress_orderby = 'ts DESC')"
                    )
                    await conn.execute(
                        "SELECT add_compression_policy($1::regclass, $2::interval)",
                        table,
                        _interval(cfg.compression.after_s),
                    )
                if cfg.retention.enabled:
                    await conn.execute(
                        "SELECT add_retention_policy($1::regclass, $2::interval)",
                        table,
                        _interval(cfg.retention.drop_after_s),
                    )
            await conn.execute(
                "INSERT INTO linetwin_settings (key, value) VALUES ('timescale_config', $1) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                fingerprint,
            )
        log.info(
            "storage.settings_applied",
            compression=cfg.compression.enabled,
            retention=cfg.retention.enabled,
        )

    # -- writes ------------------------------------------------------------------------------
    async def insert(self, stream: Stream, rows: Sequence[Stored[Any]]) -> int:
        if not rows:
            return 0
        cols = COLUMNS[stream]
        records = [tuple(_clean(v) for v in encode(stream, r, _identity)) for r in rows]
        table = stream.value
        async with self.pool.acquire() as conn, conn.transaction():
            if stream not in UNIQUE_KEYS:
                await conn.copy_records_to_table(table, records=records, columns=list(cols))
                added = len(records)
            else:
                stage = f"stage_{table}"
                await conn.execute(
                    f"CREATE TEMP TABLE IF NOT EXISTS {stage} "
                    f"(LIKE {table} INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
                )
                await conn.copy_records_to_table(stage, records=records, columns=list(cols))
                col_list = ", ".join(cols)
                status = await conn.execute(
                    f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {stage} "
                    "ON CONFLICT DO NOTHING"
                )
                added = int(status.split()[-1])
            if stream is Stream.READINGS and added:
                old = self._outside_policy_window(rows)
                if old is not None:
                    # Same transaction: if the readings are committed, so is the refresh job.
                    await conn.execute(
                        "INSERT INTO aggregate_refresh_queue (lo, hi) VALUES ($1, $2)", *old
                    )
        return added

    def _outside_policy_window(
        self, rows: Sequence[Stored[Any]]
    ) -> tuple[datetime, datetime] | None:
        """Event-time span of the readings the refresh policies will never reach.

        Policies refresh ``[now - start_offset, now - end_offset]``; readings older than the
        shorter start offset (less one 10-minute bucket of margin) need an explicit refresh.
        """
        agg = self.cfg.aggregates
        horizon = datetime.now(UTC) - timedelta(
            seconds=min(agg.m1.start_offset_s, agg.m10.start_offset_s) - _TEN_MIN_US / 1e6
        )
        old = [r.item.ts for r in rows if r.item.ts < horizon]
        return (min(old), max(old)) if old else None

    async def refresh_aggregates(self, t0: datetime | None, t1: datetime | None) -> None:
        """Materialize both aggregates for ``[t0, t1)`` (aligned outward to 10 min)."""
        async with self.pool.acquire() as conn:
            await self._refresh(conn, t0, t1)

    async def _refresh(
        self, conn: asyncpg.Connection, t0: datetime | None, t1: datetime | None
    ) -> None:
        lo = hi = None
        if t0 is not None:
            us = int((t0 - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
            lo = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                microseconds=us // _TEN_MIN_US * _TEN_MIN_US
            )
        if t1 is not None:
            us = int((t1 - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
            hi = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                microseconds=(us // _TEN_MIN_US + 1) * _TEN_MIN_US
            )
        for view in ("readings_1m", "readings_10m"):  # hierarchical: 1 min first
            await conn.execute(
                "CALL refresh_continuous_aggregate($1::regclass, $2::timestamptz, $3::timestamptz)",
                view,
                lo,
                hi,
            )

    async def refresh_pending(self, t0: datetime | None = None, t1: datetime | None = None) -> int:
        """Refresh the queued ranges overlapping ``[t0, t1)`` (all when unbounded), then delete
        them. Returns how many queue entries were consumed.

        An advisory lock serializes consumers across processes; a second consumer waits and
        then finds the entries gone. If a background policy job is refreshing the same window
        right now, the entries stay queued and the next query retries.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT pg_advisory_lock(hashtext('linetwin_agg_refresh'))")
            try:
                rows = await conn.fetch(
                    "SELECT id, lo, hi FROM aggregate_refresh_queue "
                    "WHERE ($1::timestamptz IS NULL OR hi >= $1) "
                    "AND ($2::timestamptz IS NULL OR lo < $2)",
                    t0,
                    t1,
                )
                if not rows:
                    return 0
                lo = min(r["lo"] for r in rows)
                hi = max(r["hi"] for r in rows)
                try:
                    await self._refresh(conn, lo, hi)
                except asyncpg.LockNotAvailableError as exc:
                    log.warning("storage.refresh_deferred", reason=str(exc))
                    return 0
                await conn.execute(
                    "DELETE FROM aggregate_refresh_queue WHERE id = ANY($1::bigint[])",
                    [r["id"] for r in rows],
                )
                return len(rows)
            finally:
                await conn.execute("SELECT pg_advisory_unlock(hashtext('linetwin_agg_refresh'))")

    async def pending_refreshes(self) -> int:
        n: int = await self.pool.fetchval("SELECT count(*) FROM aggregate_refresh_queue")
        return n

    async def refresh_policy_jobs(self) -> dict[str, int]:
        """Job ids of the refresh policies, by aggregate name."""
        rows = await self.pool.fetch(
            "SELECT ca.view_name, j.job_id FROM timescaledb_information.jobs j "
            "JOIN timescaledb_information.continuous_aggregates ca "
            # Recent TimescaleDB versions report the view here, older ones the mat. table.
            "ON j.hypertable_name IN (ca.view_name, ca.materialization_hypertable_name) "
            "WHERE j.proc_name = 'policy_refresh_continuous_aggregate'"
        )
        return {r["view_name"]: r["job_id"] for r in rows}

    async def run_job(self, job_id: int) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("CALL run_job($1)", job_id)

    # -- reads -------------------------------------------------------------------------------
    async def _select(
        self,
        stream: Stream,
        lo: datetime | None,
        lo_incl: bool,
        hi: datetime | None,
        hi_incl: bool,
        machine_ids: Sequence[str] | None,
    ) -> list[Stored[Any]]:
        where: list[str] = []
        args: list[Any] = []
        if lo is not None:
            args.append(lo)
            where.append(f"ts {'>=' if lo_incl else '>'} ${len(args)}")
        if hi is not None:
            args.append(hi)
            where.append(f"ts {'<=' if hi_incl else '<'} ${len(args)}")
        if machine_ids is not None and "machine_id" in COLUMNS[stream]:
            args.append(list(machine_ids))
            where.append(f"machine_id = ANY(${len(args)}::text[])")
        sql = f"SELECT {', '.join(COLUMNS[stream])} FROM {stream.value}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + _order_by(stream)
        rows = await self.pool.fetch(sql, *args)
        return [decode(stream, row, _from_db) for row in rows]

    async def fetch(
        self,
        stream: Stream,
        t0: datetime | None = None,
        t1: datetime | None = None,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[Stored[Any]]:
        return await self._select(stream, t0, True, t1, False, machine_ids)

    async def fetch_replay(
        self, streams: Sequence[Stream], after: datetime | None, until: datetime
    ) -> list[Stored[Any]]:
        out: list[Stored[Any]] = []
        for stream in streams:
            out.extend(await self._select(stream, after, False, until, True, None))
        out.sort(key=sort_key)
        return out

    async def count(self, stream: Stream) -> int:
        value = await self.pool.fetchval(f"SELECT count(*) FROM {stream.value}")
        return int(value)

    async def min_max_ts(self, streams: Sequence[Stream]) -> tuple[datetime, datetime] | None:
        parts = " UNION ALL ".join(
            f"SELECT min(ts) AS lo, max(ts) AS hi FROM {s.value}" for s in streams
        )
        row = await self.pool.fetchrow(f"SELECT min(lo) AS lo, max(hi) AS hi FROM ({parts}) x")
        if row is None or row["lo"] is None:
            return None
        return _from_db(row["lo"]), _from_db(row["hi"])

    # -- history -----------------------------------------------------------------------------
    async def history(
        self,
        t0: datetime,
        t1: datetime,
        resolution: Resolution,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[HistoryBucket]:
        lo, hi = bucket_bounds(t0, t1, RESOLUTION_SECONDS[resolution])
        sensors = [s.value for s in SensorName]
        args: list[Any] = [lo, hi]
        if resolution == "1s":
            stats = ", ".join(
                f"avg({s}) AS {s}_mean, min({s}) AS {s}_min, max({s}) AS {s}_max, "
                f"count({s}) AS {s}_count"
                for s in sensors
            )
            sql = (
                "SELECT time_bucket(INTERVAL '1 second', ts) AS bucket, machine_id, "
                "count(*) AS n, "
                f"count(*) FILTER (WHERE quality = '{Quality.LATE.value}') AS n_late, "
                f"count(*) FILTER (WHERE quality = '{Quality.SUSPECT.value}') AS n_suspect, "
                f"{stats} FROM readings WHERE ts >= $1 AND ts < $2"
            )
            if machine_ids is not None:
                args.append(list(machine_ids))
                sql += " AND machine_id = ANY($3::text[])"
            sql += ' GROUP BY 1, 2 ORDER BY 1, machine_id COLLATE "C"'
        else:
            await self.refresh_pending(lo, hi)
            stats = ", ".join(
                f"{s}_sum / NULLIF({s}_count, 0) AS {s}_mean, {s}_min, {s}_max, {s}_count"
                for s in sensors
            )
            sql = (
                f"SELECT bucket, machine_id, n, n_late, n_suspect, {stats} "
                f"FROM {AGGREGATES[resolution]} WHERE bucket >= $1 AND bucket < $2"
            )
            if machine_ids is not None:
                args.append(list(machine_ids))
                sql += " AND machine_id = ANY($3::text[])"
            sql += ' ORDER BY bucket, machine_id COLLATE "C"'
        out: list[HistoryBucket] = []
        for row in await self.pool.fetch(sql, *args):
            out.append(
                HistoryBucket(
                    bucket=_from_db(row["bucket"]),
                    machine_id=row["machine_id"],
                    n=int(row["n"]),
                    n_late=int(row["n_late"]),
                    n_suspect=int(row["n_suspect"]),
                    sensors={
                        s: SensorStats(
                            count=int(row[f"{s.value}_count"]),
                            mean=row[f"{s.value}_mean"],
                            min=row[f"{s.value}_min"],
                            max=row[f"{s.value}_max"],
                        )
                        for s in SensorName
                    },
                )
            )
        return out

    # -- snapshots ---------------------------------------------------------------------------
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        await self.pool.execute(
            "INSERT INTO snapshots (ts, state, ingested_at) VALUES ($1, $2, $3) "
            "ON CONFLICT (ts) DO UPDATE SET state = EXCLUDED.state, "
            "ingested_at = EXCLUDED.ingested_at",
            snapshot.ts,
            snapshot.state_json,
            snapshot.ingested_at,
        )

    async def latest_snapshot(self, at_or_before: datetime | None = None) -> Snapshot | None:
        if at_or_before is None:
            row = await self.pool.fetchrow(
                "SELECT ts, state, ingested_at FROM snapshots ORDER BY ts DESC LIMIT 1"
            )
        else:
            row = await self.pool.fetchrow(
                "SELECT ts, state, ingested_at FROM snapshots WHERE ts <= $1 "
                "ORDER BY ts DESC LIMIT 1",
                at_or_before,
            )
        if row is None:
            return None
        return Snapshot(_from_db(row["ts"]), row["state"], _from_db(row["ingested_at"]))

    async def snapshot_times(self) -> list[datetime]:
        rows = await self.pool.fetch("SELECT ts FROM snapshots ORDER BY ts")
        return [_from_db(r["ts"]) for r in rows]

    async def invalidate_snapshots_from(self, ts: datetime) -> int:
        status = await self.pool.execute("DELETE FROM snapshots WHERE ts >= $1", ts)
        return int(status.split()[-1])

    async def delete_all(self) -> None:
        tables = ", ".join([*(s.value for s in Stream), "snapshots", "aggregate_refresh_queue"])
        async with self.pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {tables}")
            await conn.execute("TRUNCATE readings_1m")
            await conn.execute("TRUNCATE readings_10m")


async def create_database(admin_dsn: str, name: str) -> None:
    """Create an empty database ``name`` (tests and benchmarks use throw-away databases)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe database name {name!r}")
    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()


async def drop_database(admin_dsn: str, name: str) -> None:
    """Drop database ``name``, disconnecting any remaining sessions."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe database name {name!r}")
    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()
