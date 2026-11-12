"""Event store write benchmark: ``python -m twin_backend.storage.bench`` (poe: storage-bench).

For each backend, on seeded synthetic readings (4 machines, ~20 % with a missing sensor value,
~1 % LATE):

1. **Sustained throughput through the StorageWriter** (shipped ``writer:`` settings): a paced
   producer publishes readings on the event bus at a fixed target rate for ``--duration``
   seconds, for each rate in ``--rates``. Reported per rate: achieved offered rate, rows written
   per second during the window, rows dropped by the writer's bounded queue, and p50 / p95 batch
   insert latency. The **sustained throughput** is the highest offered rate with no drops. The
   final drain after a window is not counted. Producer and writer share one event loop and one
   Python process, as in the backend.
2. **Direct batch inserts:** ``--rows`` readings inserted with ``EventStore.insert`` at batch sizes
   100 / 1000 / 5000 (one transaction per batch), without the bus. For TimescaleDB also a plain
   ``executemany`` INSERT at batch size 1000, for comparison with the COPY-based path the store
   uses.

SQLite writes to a file in a temporary directory (WAL mode, as in minimal mode). TimescaleDB runs
in a throw-away database ``linetwin_bench_<hex>`` that is dropped afterwards; the configured
database is never touched. A backend that cannot run is reported as not run, with the reason.
Results go to ``experiments/results/storage_bench.md`` (``--out``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import random
import secrets
import sqlite3
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import SystemClock
from twin_backend.storage.base import Stored, Stream
from twin_backend.storage.metrics import StorageMetrics
from twin_backend.storage.writer import StorageWriter
from twin_common import aio
from twin_common.config import load_storage_config
from twin_common.config.settings import TwinSettings
from twin_common.config.storage import StorageConfig
from twin_common.contracts import Reading
from twin_common.enums import MachineState, Quality, SensorName
from twin_common.logging import configure_logging

MACHINES = ("CNV-01", "CNC-01", "INSP-01", "CNV-02")
BATCH_SIZES = (100, 1000, 5000)
DEFAULT_RATES = (1_000, 2_000, 5_000, 10_000, 20_000, 40_000)
T0 = datetime(2026, 1, 1, tzinfo=UTC)
TICK_S = 0.01  # producer pacing step


class ReadingFactory:
    """Seeded synthetic readings; seq and ts continue across calls (no duplicates)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.n = 0

    def next(self) -> Reading:
        i = self.n
        self.n += 1
        rng = self.rng
        values: dict[str, float | None] = {
            "speed": rng.gauss(1200, 40),
            "temp": rng.gauss(55, 3),
            "vib": abs(rng.gauss(2.0, 0.3)),
            "load": rng.uniform(40, 80),
            "current": abs(rng.gauss(12, 1)),
        }
        quality = Quality.OK
        if rng.random() < 0.2:
            values[rng.choice([s.value for s in SensorName])] = None
            quality = Quality.SUSPECT
        if rng.random() < 0.01:
            quality = Quality.LATE
        step = i // len(MACHINES)
        return Reading.model_validate(
            {
                "machine_id": MACHINES[i % len(MACHINES)],
                "seq": step,
                "ts": T0 + timedelta(seconds=step),
                "state": MachineState.RUNNING,
                "run_time": float(step),
                "quality": quality,
                **values,
            }
        )

    def batch(self, n: int, ingested: datetime) -> list[Stored[Any]]:
        return [Stored(self.next(), ingested) for _ in range(n)]


class TimedStore:
    """Wraps a store and records how long each insert takes."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.backend = inner.backend
        self.durations: list[float] = []

    async def insert(self, stream: Stream, rows: Sequence[Stored[Any]]) -> int:
        start = time.perf_counter()
        added: int = await self.inner.insert(stream, rows)
        self.durations.append(time.perf_counter() - start)
        return added

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class RateResult:
    target_per_s: int
    flush_interval_s: float
    duration_s: float
    offered: int
    written: int
    dropped: int
    batches: int
    p50_ms: float
    p95_ms: float

    @property
    def offered_per_s(self) -> float:
        return self.offered / self.duration_s

    @property
    def written_per_s(self) -> float:
        return self.written / self.duration_s

    @property
    def backlog(self) -> int:
        """Rows still queued (or in flight) when the window ended."""
        return self.offered - self.written - self.dropped

    @property
    def sustainable(self) -> bool:
        """No drops, and the writer kept up: the backlog at the end is at most about one flush
        interval plus 1 s of data (a growing queue would drop data later)."""
        limit = self.offered_per_s * (self.flush_interval_s + 1.0)
        return self.dropped == 0 and self.backlog <= limit


@dataclass
class BackendResult:
    backend: str
    version: str = ""
    rates: list[RateResult] = field(default_factory=list)
    direct: dict[str, float] = field(default_factory=dict)  # label -> rows/s
    not_run: str | None = None

    def sustained(self) -> RateResult | None:
        clean = [r for r in self.rates if r.sustainable]
        return max(clean, key=lambda r: r.offered_per_s) if clean else None


def _percentile(values: list[float], q: int) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


async def at_rate(
    store: Any, cfg: StorageConfig, target: int, duration_s: float, factory: ReadingFactory
) -> RateResult:
    timed = TimedStore(store)
    bus = InMemoryEventBus(cfg.writer.queue_size)
    metrics = StorageMetrics()
    writer = StorageWriter(timed, bus, cfg.writer, clock=SystemClock(), metrics=metrics)
    writer.start()
    offered = 0
    start = time.perf_counter()
    deadline = start + duration_s
    while (now := time.perf_counter()) < deadline:
        due = int((now - start) * target) - offered  # catch up to the target schedule
        for _ in range(max(0, due)):
            await bus.publish(Topic.READINGS, factory.next())
        offered += max(0, due)
        await asyncio.sleep(TICK_S)
    elapsed = time.perf_counter() - start
    written = int(metrics.value("written", stream="readings"))
    durations = list(timed.durations)
    writer._sync_drop_metrics()
    dropped = int(metrics.dropped_total())
    await writer.stop()  # final drain: not part of the measurement
    return RateResult(
        target_per_s=target,
        flush_interval_s=cfg.writer.flush_interval_s,
        duration_s=elapsed,
        offered=offered,
        written=written,
        dropped=dropped,
        batches=len(durations),
        p50_ms=_percentile(durations, 50) * 1000,
        p95_ms=_percentile(durations, 95) * 1000,
    )


async def direct(store: Any, rows: int, seed: int) -> dict[str, float]:
    out: dict[str, float] = {}
    factory = ReadingFactory(seed + 1)
    now = datetime.now(UTC)
    for size in BATCH_SIZES:
        batches = [factory.batch(size, now) for _ in range(max(1, rows // size))]
        start = time.perf_counter()
        for batch in batches:
            await store.insert(Stream.READINGS, batch)
        out[f"insert, batch {size}"] = sum(len(b) for b in batches) / (time.perf_counter() - start)
    return out


async def timescale_executemany(store: Any, rows: int, seed: int) -> float:
    """Plain INSERT ... ON CONFLICT via executemany, batch 1000 (comparison only)."""
    from twin_backend.storage.rows import COLUMNS, encode

    factory = ReadingFactory(seed + 2)
    now = datetime.now(UTC)
    cols = COLUMNS[Stream.READINGS]
    sql = (
        f"INSERT INTO readings ({', '.join(cols)}) VALUES "
        f"({', '.join(f'${i + 1}' for i in range(len(cols)))}) ON CONFLICT DO NOTHING"
    )
    batches = [
        [encode(Stream.READINGS, r, lambda ts: ts) for r in factory.batch(1000, now)]
        for _ in range(max(1, rows // 1000))
    ]
    start = time.perf_counter()
    for batch in batches:
        async with store.pool.acquire() as conn, conn.transaction():
            await conn.executemany(sql, batch)
    return sum(len(b) for b in batches) / (time.perf_counter() - start)


async def ladder(store: Any, cfg: StorageConfig, args: argparse.Namespace) -> list[RateResult]:
    factory = ReadingFactory(args.seed)
    out: list[RateResult] = []
    for target in args.rates:
        print(f"  {target}/s ...", file=sys.stderr, flush=True)
        out.append(await at_rate(store, cfg, target, args.duration, factory))
    return out


async def bench_sqlite(cfg: StorageConfig, args: argparse.Namespace) -> BackendResult:
    from twin_backend.storage.sqlite import SqliteEventStore

    result = BackendResult("sqlite", f"SQLite {sqlite3.sqlite_version}")
    with tempfile.TemporaryDirectory(prefix="linetwin_bench_") as tmp:
        for phase in ("ladder", "direct"):
            store = SqliteEventStore(Path(tmp) / f"{phase}.sqlite")
            await store.connect()
            await store.migrate()
            try:
                if phase == "ladder":
                    result.rates = await ladder(store, cfg, args)
                else:
                    result.direct = await direct(store, args.rows, args.seed)
            finally:
                await store.close()
    return result


async def bench_timescale(cfg: StorageConfig, args: argparse.Namespace) -> BackendResult:
    from twin_backend.storage.timescale import TimescaleEventStore, create_database, drop_database

    result = BackendResult("timescale")
    if cfg.timescale is None:
        result.not_run = "storage.yaml has no timescale section"
        return result
    settings = TwinSettings()
    name = f"linetwin_bench_{secrets.token_hex(6)}"
    bench_dsn = settings.model_copy(update={"pg_db": name}).pg_dsn
    try:
        await create_database(settings.pg_dsn, name)
    except Exception as exc:
        result.not_run = (
            f"TimescaleDB not reachable at {settings.pg_host}:{settings.pg_port} "
            f"({type(exc).__name__}) - start it with `uv run poe up`"
        )
        return result
    try:
        for phase in ("ladder", "direct", "executemany"):
            store = TimescaleEventStore(bench_dsn, cfg.timescale)
            await store.connect()
            await store.migrate()
            await store.delete_all()
            try:
                if not result.version:
                    pg = await store.pool.fetchval("SHOW server_version")
                    ts = await store.pool.fetchval(
                        "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                    )
                    result.version = f"PostgreSQL {pg}, TimescaleDB {ts}"
                if phase == "ladder":
                    result.rates = await ladder(store, cfg, args)
                elif phase == "direct":
                    result.direct = await direct(store, args.rows, args.seed)
                else:
                    result.direct[
                        "executemany INSERT, batch 1000 (comparison, not used)"
                    ] = await timescale_executemany(store, args.rows, args.seed)
            finally:
                await store.close()
    finally:
        await drop_database(settings.pg_dsn, name)
    return result


def render(results: list[BackendResult], cfg: StorageConfig, args: argparse.Namespace) -> str:
    w = cfg.writer
    lines = [
        "# Storage write benchmark",
        "",
        "Generated by `uv run poe storage-bench` (`python -m twin_backend.storage.bench`) on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. Do not edit by hand.",
        "",
        "## Setup",
        "",
        f"- Machine: {platform.platform()}, {os.cpu_count()} logical CPUs, "
        f"processor `{platform.processor() or 'unknown'}`",
        f"- Python {platform.python_version()}",
        f"- Data: seeded synthetic readings (seed {args.seed}), 4 machines, about 20 % with one "
        "missing sensor value, about 1 % LATE",
        f"- Writer settings (config/storage.yaml): batch_size {w.batch_size}, flush_interval_s "
        f"{w.flush_interval_s:g}, queue_size {w.queue_size}",
        f"- Sustained: {args.duration:g} s per target rate; producer and writer in one process "
        f"and event loop. Direct inserts: {args.rows} rows per batch size.",
        "- SQLite: file on the local disk in a temporary directory, WAL mode. TimescaleDB: "
        "Docker Compose service on the same machine, throw-away database, shipped settings.",
        "",
        "## Sustained throughput",
        "",
        "Highest offered rate at which the StorageWriter kept up: nothing dropped, and at the "
        "end of the window at most about one flush interval plus 1 s of data still queued.",
        "",
        "| Backend | Version | Sustained rows/s |",
        "|---|---|---:|",
    ]
    for r in results:
        if r.not_run:
            lines.append(f"| {r.backend} | - | not run: {r.not_run} |")
            continue
        best = r.sustained()
        value = f"{best.offered_per_s:,.0f}" if best else "none of the tested rates"
        top = max((x.target_per_s for x in r.rates), default=0)
        if best is not None and best.target_per_s == top:
            value += " (highest rate tested; the limit is higher)"
        lines.append(f"| {r.backend} | {r.version} | {value} |")
    lines += [
        "",
        "Per target rate. Written rows/s counts rows committed during the window; the backlog "
        "is written in the final drain and not counted.",
        "",
        "| Backend | Target rows/s | Offered rows/s | Written rows/s | Dropped rows | "
        "Backlog at end | Kept up | Batches | Batch p50 ms | Batch p95 ms |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        for x in r.rates:
            lines.append(
                f"| {r.backend} | {x.target_per_s:,} | {x.offered_per_s:,.0f} | "
                f"{x.written_per_s:,.0f} | {x.dropped:,} | {x.backlog:,} | "
                f"{'yes' if x.sustainable else 'no'} | {x.batches} | {x.p50_ms:.1f} | "
                f"{x.p95_ms:.1f} |"
            )
    lines += [
        "",
        "## Direct batch inserts (`EventStore.insert`, one transaction per batch, no bus)",
        "",
        "| Backend | Path | Rows/s |",
        "|---|---|---:|",
    ]
    for r in results:
        if r.not_run:
            lines.append(f"| {r.backend} | not run: {r.not_run} | |")
            continue
        for label, rate in r.direct.items():
            lines.append(f"| {r.backend} | {label} | {rate:,.0f} |")
    lines += [
        "",
        "For scale: the simulated line produces 4 readings/s (4 machines at 1 Hz).",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> list[BackendResult]:
    cfg = load_storage_config(args.config_dir / "storage.yaml")
    results: list[BackendResult] = []
    for backend in args.backends:
        print(f"benchmarking {backend} ...", file=sys.stderr, flush=True)
        if backend == "sqlite":
            results.append(await bench_sqlite(cfg, args))
        else:
            results.append(await bench_timescale(cfg, args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results, cfg, args), encoding="utf-8")
    return results


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m twin_backend.storage.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=10.0, help="seconds per target rate")
    p.add_argument("--rates", type=_ints, default=list(DEFAULT_RATES), help="rows/s, comma list")
    p.add_argument("--rows", type=int, default=50_000, help="rows per direct batch size")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--backends", type=lambda s: s.split(","), default=["sqlite", "timescale"])
    p.add_argument("--config-dir", type=Path, default=Path("config"))
    p.add_argument("--out", type=Path, default=Path("experiments/results/storage_bench.md"))
    args = p.parse_args(argv)
    configure_logging("WARNING", stream=sys.stderr)
    aio.run(run(args))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
