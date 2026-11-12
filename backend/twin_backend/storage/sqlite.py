"""SQLite event store: minimal mode and tests (standard library only, no service).

All database work runs on one dedicated worker thread (the connection never leaves it), so the
event loop never blocks on disk I/O. WAL journal mode lets readers work alongside the writer.
Timestamps are integer microseconds since the epoch; downsampling is computed in SQL, where
``AVG``/``MIN``/``MAX``/``COUNT(col)`` ignore NULL (missing) sensor values.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from twin_backend.storage.base import (
    RESOLUTION_SECONDS,
    HistoryBucket,
    Resolution,
    SensorStats,
    Snapshot,
    Stored,
    Stream,
    bucket_bounds,
    from_us,
    sort_key,
    to_us,
)
from twin_backend.storage.migrator import load_migrations, pending
from twin_backend.storage.rows import COLUMNS, ORDER_BY, UNIQUE_KEYS, decode, encode
from twin_common.enums import Quality, SensorName
from twin_common.logging import get_logger

log = get_logger(__name__)

R = TypeVar("R")
MEMORY = ":memory:"


class SqliteEventStore:
    backend = "sqlite"

    def __init__(self, path: Path | str, *, busy_timeout_s: float = 5.0) -> None:
        self.path = path if path == MEMORY else Path(path)
        self.busy_timeout_s = busy_timeout_s
        self._executor: ThreadPoolExecutor | None = None
        self._conn: sqlite3.Connection | None = None

    # -- plumbing ----------------------------------------------------------------------------
    async def _run(self, fn: Callable[..., R], *args: Any) -> R:
        if self._executor is None:
            raise RuntimeError("SqliteEventStore is not connected")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteEventStore is not connected")
        return self._conn

    def _open(self) -> None:
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.busy_timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        if isinstance(self.path, Path):
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- lifecycle ---------------------------------------------------------------------------
    async def connect(self) -> None:
        if self._executor is not None:
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-store")
        await self._run(self._open)

    async def close(self) -> None:
        if self._executor is None:
            return
        await self._run(self._close)
        self._executor.shutdown(wait=False)
        self._executor = None

    def _migrate(self) -> list[str]:
        conn = self.conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL,"
            " applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
        )
        applied = {
            row["version"]: (row["name"], row["checksum"])
            for row in conn.execute("SELECT version, name, checksum FROM schema_migrations")
        }
        done: list[str] = []
        for mig in pending(load_migrations("sqlite"), applied):
            # Version, name and checksum are validated (digits, [a-z0-9_], hex): safe to inline.
            record = (
                "INSERT INTO schema_migrations (version, name, checksum) VALUES "
                f"({mig.version}, '{mig.name}', '{mig.checksum}');"
            )
            try:
                if mig.transactional:
                    conn.executescript(f"BEGIN;\n{mig.sql}\n{record}\nCOMMIT;")
                else:
                    conn.executescript(f"{mig.sql}\n{record}")
            except sqlite3.Error:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            done.append(f"{mig.version:04d}_{mig.name}")
            log.info("storage.migrated", backend=self.backend, migration=done[-1])
        return done

    async def migrate(self) -> list[str]:
        return await self._run(self._migrate)

    # -- writes ------------------------------------------------------------------------------
    def _insert(self, stream: Stream, rows: list[tuple[Any, ...]]) -> int:
        cols = COLUMNS[stream]
        verb = "INSERT OR IGNORE" if stream in UNIQUE_KEYS else "INSERT"
        sql = (
            f"{verb} INTO {stream.value} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})"
        )
        conn = self.conn
        before = conn.total_changes
        conn.execute("BEGIN")
        try:
            conn.executemany(sql, rows)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return conn.total_changes - before

    async def insert(self, stream: Stream, rows: Sequence[Stored[Any]]) -> int:
        if not rows:
            return 0
        encoded = [encode(stream, r, to_us) for r in rows]
        return await self._run(self._insert, stream, encoded)

    # -- reads -------------------------------------------------------------------------------
    def _select(
        self,
        stream: Stream,
        lo: int | None,
        lo_incl: bool,
        hi: int | None,
        hi_incl: bool,
        machine_ids: Sequence[str] | None,
    ) -> list[Stored[Any]]:
        where: list[str] = []
        args: list[Any] = []
        if lo is not None:
            where.append("ts >= ?" if lo_incl else "ts > ?")
            args.append(lo)
        if hi is not None:
            where.append("ts <= ?" if hi_incl else "ts < ?")
            args.append(hi)
        if machine_ids is not None and "machine_id" in COLUMNS[stream]:
            where.append(f"machine_id IN ({', '.join('?' for _ in machine_ids)})")
            args.extend(machine_ids)
        sql = f"SELECT * FROM {stream.value}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + ", ".join(ORDER_BY[stream])
        return [decode(stream, row, from_us) for row in self.conn.execute(sql, args)]

    async def fetch(
        self,
        stream: Stream,
        t0: datetime | None = None,
        t1: datetime | None = None,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[Stored[Any]]:
        lo = to_us(t0) if t0 is not None else None
        hi = to_us(t1) if t1 is not None else None
        return await self._run(self._select, stream, lo, True, hi, False, machine_ids)

    def _replay(
        self, streams: Sequence[Stream], after: int | None, until: int
    ) -> list[Stored[Any]]:
        out: list[Stored[Any]] = []
        for stream in streams:
            out.extend(self._select(stream, after, False, until, True, None))
        out.sort(key=sort_key)
        return out

    async def fetch_replay(
        self, streams: Sequence[Stream], after: datetime | None, until: datetime
    ) -> list[Stored[Any]]:
        lo = to_us(after) if after is not None else None
        return await self._run(self._replay, streams, lo, to_us(until))

    def _count(self, stream: Stream) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {stream.value}").fetchone()
        return int(row[0])

    async def count(self, stream: Stream) -> int:
        return await self._run(self._count, stream)

    def _min_max(self, streams: Sequence[Stream]) -> tuple[int, int] | None:
        parts = " UNION ALL ".join(
            f"SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM {s.value}" for s in streams
        )
        row = self.conn.execute(f"SELECT MIN(lo), MAX(hi) FROM ({parts})").fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0]), int(row[1])

    async def min_max_ts(self, streams: Sequence[Stream]) -> tuple[datetime, datetime] | None:
        found = await self._run(self._min_max, streams)
        return None if found is None else (from_us(found[0]), from_us(found[1]))

    # -- history -----------------------------------------------------------------------------
    def _history(
        self, lo: int, hi: int, width_us: int, machine_ids: Sequence[str] | None
    ) -> list[HistoryBucket]:
        stats = ", ".join(
            f"AVG({s}) AS {s}_mean, MIN({s}) AS {s}_min, MAX({s}) AS {s}_max, "
            f"COUNT({s}) AS {s}_count"
            for s in (sensor.value for sensor in SensorName)
        )
        args: list[Any] = [width_us, width_us, Quality.LATE.value, Quality.SUSPECT.value, lo, hi]
        sql = (
            "SELECT machine_id, (ts / ?) * ? AS bucket, COUNT(*) AS n, "
            "SUM(quality = ?) AS n_late, SUM(quality = ?) AS n_suspect, "
            f"{stats} FROM readings WHERE ts >= ? AND ts < ?"
        )
        if machine_ids is not None:
            sql += f" AND machine_id IN ({', '.join('?' for _ in machine_ids)})"
            args.extend(machine_ids)
        sql += " GROUP BY machine_id, bucket ORDER BY bucket, machine_id"
        out: list[HistoryBucket] = []
        for row in self.conn.execute(sql, args):
            sensors = {
                s: SensorStats(
                    count=int(row[f"{s.value}_count"]),
                    mean=row[f"{s.value}_mean"],
                    min=row[f"{s.value}_min"],
                    max=row[f"{s.value}_max"],
                )
                for s in SensorName
            }
            out.append(
                HistoryBucket(
                    bucket=from_us(int(row["bucket"])),
                    machine_id=row["machine_id"],
                    n=int(row["n"]),
                    n_late=int(row["n_late"]),
                    n_suspect=int(row["n_suspect"]),
                    sensors=sensors,
                )
            )
        return out

    async def history(
        self,
        t0: datetime,
        t1: datetime,
        resolution: Resolution,
        *,
        machine_ids: Sequence[str] | None = None,
    ) -> list[HistoryBucket]:
        width = RESOLUTION_SECONDS[resolution]
        lo, hi = bucket_bounds(t0, t1, width)
        return await self._run(self._history, to_us(lo), to_us(hi), width * 1_000_000, machine_ids)

    # -- snapshots ---------------------------------------------------------------------------
    def _save_snapshot(self, ts: int, state: str, ingested_at: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (ts, state, ingested_at) VALUES (?, ?, ?)",
            (ts, state, ingested_at),
        )

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        await self._run(
            self._save_snapshot,
            to_us(snapshot.ts),
            snapshot.state_json,
            to_us(snapshot.ingested_at),
        )

    def _latest_snapshot(self, at_or_before: int | None) -> Snapshot | None:
        if at_or_before is None:
            row = self.conn.execute(
                "SELECT ts, state, ingested_at FROM snapshots ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT ts, state, ingested_at FROM snapshots WHERE ts <= ? "
                "ORDER BY ts DESC LIMIT 1",
                (at_or_before,),
            ).fetchone()
        if row is None:
            return None
        return Snapshot(from_us(row["ts"]), row["state"], from_us(row["ingested_at"]))

    async def latest_snapshot(self, at_or_before: datetime | None = None) -> Snapshot | None:
        bound = to_us(at_or_before) if at_or_before is not None else None
        return await self._run(self._latest_snapshot, bound)

    def _snapshot_times(self) -> list[datetime]:
        return [
            from_us(row[0]) for row in self.conn.execute("SELECT ts FROM snapshots ORDER BY ts")
        ]

    async def snapshot_times(self) -> list[datetime]:
        return await self._run(self._snapshot_times)

    def _invalidate(self, ts: int) -> int:
        return self.conn.execute("DELETE FROM snapshots WHERE ts >= ?", (ts,)).rowcount

    async def invalidate_snapshots_from(self, ts: datetime) -> int:
        return await self._run(self._invalidate, to_us(ts))

    def _delete_all(self) -> None:
        conn = self.conn
        conn.execute("BEGIN")
        for stream in Stream:
            conn.execute(f"DELETE FROM {stream.value}")
        conn.execute("DELETE FROM snapshots")
        conn.execute("COMMIT")

    async def delete_all(self) -> None:
        await self._run(self._delete_all)
