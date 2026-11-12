"""Phase 3 integration: storage and event sourcing, on every backend in STORE_BACKENDS.

1. A seeded simulator run, edited to contain a 16 s outage of CNC-01 (-> STALE, OFFLINE,
   recovery) and a reading that arrives too late (-> LATE), goes through the file adapter, the
   ingest pipeline and the storage writer. A LiveStateTracker records the live line state.
   ``reconstruct(t)`` equals the recorded live state at random t, with and without snapshots.
2. Export -> FileReplayAdapter -> pipeline -> a fresh store gives the same readings (CSV, JSONL).
3. ``build_runtime`` wires storage into the SourceManager: live ingestion (including quarantined
   messages) persists automatically, and the adapter's replay speed reaches the staleness monitor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from twin_backend.adapters.file import FileReplayAdapter
from twin_backend.bus import InMemoryEventBus
from twin_backend.clock import FakeClock
from twin_backend.engine.line_state import LineState, LiveStateTracker
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import MemoryQuarantine
from twin_backend.runtime import build_runtime
from twin_backend.storage.base import EventStore, Stream
from twin_backend.storage.export import ExportFormat, export_range
from twin_backend.storage.replay import reconstruct
from twin_backend.storage.service import StorageService
from twin_backend.storage.sqlite import SqliteEventStore
from twin_backend.testing import (
    REPO_ROOT,
    SimExport,
    load_jsonl_readings,
    storage_config,
)
from twin_common.config import load_line_config
from twin_common.config.settings import TwinSettings
from twin_common.config.sources import FieldMapping, SourcesConfig
from twin_common.contracts import Reading
from twin_common.enums import AdapterKind, MachineState, Quality, QuarantineReason

LATENESS_S = 0.5
OUTAGE = ("CNC-01", 40, 56)  # machine, first missing second, first second back
LATE = ("INSP-01", 80, 86)  # machine, second of the reading, second after which it arrives
RANDOM_TIMES = 25


def edited_run(export: SimExport) -> list[Reading]:
    """The run with an outage (no seq gap, so nothing waits in the reorder buffer) and one
    reading moved later than the lateness window."""
    readings = load_jsonl_readings(export.jsonl)
    t0 = readings[0].ts
    machine, lo, hi = OUTAGE
    out: list[Reading] = []
    for r in readings:
        sec = (r.ts - t0).total_seconds()
        if r.machine_id == machine and lo <= sec < hi:
            continue
        if r.machine_id == machine and sec >= hi:
            r = r.model_copy(update={"seq": r.seq - (hi - lo)})
        out.append(r)
    late_machine, late_sec, arrive_after = LATE
    late = next(
        r for r in out if r.machine_id == late_machine and (r.ts - t0).total_seconds() == late_sec
    )
    out.remove(late)
    pos = max(i for i, r in enumerate(out) if (r.ts - t0).total_seconds() <= arrive_after)
    out.insert(pos + 1, late)
    return out


def write_jsonl(readings: list[Reading], path: Path) -> Path:
    path.write_text("".join(r.model_dump_json() + "\n" for r in readings), encoding="utf-8")
    return path


def file_sources(sources: SourcesConfig, path: Path, fmt: str, speed: float) -> SourcesConfig:
    assert sources.file is not None
    file_cfg = sources.file.model_copy(
        update={
            "path": str(path),
            "format": fmt,
            "loop": False,
            "replay_speed": speed,
            "mapping": FieldMapping(),
        }
    )
    ingest = sources.ingest.model_copy(update={"lateness_window_s": LATENESS_S})
    return sources.model_copy(update={"file": file_cfg, "ingest": ingest})


@dataclass
class Record:
    arrival_ts: datetime  # event ts of the message just processed
    settled: bool  # nothing held in the reorder buffer
    state: LineState


@dataclass
class Run:
    store: EventStore
    service: StorageService
    records: list[Record]
    machines: list[str]


async def ingest_file(
    path: Path, fmt: str, sources: SourcesConfig, store: EventStore, *, record: bool
) -> Run:
    """File -> adapter -> pipeline -> bus -> (writer, live tracker), driven step by step."""
    line = load_line_config(REPO_ROOT / "config" / "line.yaml")
    machines = [m.id for m in line.machines]
    srcs = file_sources(sources, path, fmt, speed=1.0)
    clock = FakeClock(auto_advance=True)
    bus = InMemoryEventBus(1_000_000, clock=clock)
    cfg = storage_config()
    cfg = cfg.model_copy(
        update={"snapshots": cfg.snapshots.model_copy(update={"interval_s": 10.0})}
    )
    service = StorageService(store, cfg, bus, machines, clock=clock)
    service.writer.subscribe()
    tracker = LiveStateTracker(bus, machines)
    pipeline = IngestPipeline(
        line,
        srcs.ingest,
        bus,
        service.quarantine_sink(MemoryQuarantine(10_000)),
        clock,
        {AdapterKind.FILE: FieldMapping()},
    )
    assert srcs.file is not None
    adapter = FileReplayAdapter(srcs.file, line, root=path.parent, clock=clock)
    pipeline.set_time_scale(adapter.time_scale)
    records: list[Record] = []
    await adapter.start()
    try:
        async for raw in adapter.stream():
            await pipeline.process(raw)
            await pipeline.tick()
            tracker.drain()
            if record:
                assert isinstance(raw.payload, dict)
                records.append(
                    Record(
                        datetime.fromisoformat(raw.payload["ts"]),
                        not any(pipeline.reorder.held(m) for m in machines),
                        tracker.snapshot(),
                    )
                )
        await pipeline.flush()
    finally:
        await adapter.stop()
    tracker.drain()
    await service.writer.flush()
    return Run(store, service, records, machines)


def settled_times(
    records: list[Record], rng: random.Random, n: int
) -> list[tuple[datetime, Record]]:
    """Times t = arrival + 0.5 s at which every event <= t had arrived and none > t had."""
    candidates: list[tuple[datetime, Record]] = []
    suffix_min: list[datetime] = [records[-1].arrival_ts] * len(records)
    for i in range(len(records) - 2, -1, -1):
        suffix_min[i] = min(records[i + 1].arrival_ts, suffix_min[i + 1])
    prefix_max = records[0].arrival_ts
    for i, rec in enumerate(records[:-1]):
        prefix_max = max(prefix_max, rec.arrival_ts)
        t = rec.arrival_ts + timedelta(seconds=0.5)
        if (
            rec.settled
            and prefix_max <= t < suffix_min[i]
            and (records[i + 1].arrival_ts != rec.arrival_ts)
        ):
            candidates.append((t, rec))
    assert len(candidates) >= n
    return rng.sample(candidates, n)


@pytest.fixture
async def recorded_run(
    sim_export: SimExport, sources_cfg: SourcesConfig, store_factory: Any, tmp_path: Path
) -> Any:
    path = write_jsonl(edited_run(sim_export), tmp_path / "edited.jsonl")
    store = await store_factory(tmp_path)
    try:
        yield await ingest_file(path, "jsonl", sources_cfg, store, record=True)
    finally:
        await store.close()


async def test_reconstruct_equals_live_state_at_random_times(recorded_run: Run) -> None:
    run = recorded_run
    store = run.store
    changes = [c.item for c in await store.fetch(Stream.STATE_CHANGES)]
    states = {(c.machine_id, c.state) for c in changes}
    assert ("CNC-01", MachineState.STALE) in states
    assert ("CNC-01", MachineState.OFFLINE) in states
    assert any(c.reason == "data resumed" for c in changes)
    t0 = run.records[0].arrival_ts
    late = [r.item for r in await store.fetch(Stream.READINGS) if r.item.quality is Quality.LATE]
    late_ts = t0 + timedelta(seconds=LATE[1])
    assert [(r.machine_id, r.ts) for r in late] == [(LATE[0], late_ts)]

    picks = settled_times(run.records, random.Random(2026), RANDOM_TIMES)
    # The outage and the late reading lie inside the sampled range.
    assert min(t for t, _ in picks) < t0 + timedelta(seconds=40)
    assert max(t for t, _ in picks) > t0 + timedelta(seconds=86)
    for t, rec in picks:
        replayed = await reconstruct(store, run.machines, t, use_snapshots=False)
        assert replayed.body() == rec.state.body(), t

    saved = await run.service.snapshotter.run_once(settle_s=0)
    assert saved >= 10
    for t, rec in picks:
        fast = await reconstruct(store, run.machines, t)
        assert fast.body() == rec.state.body(), t

    # During the outage the twin shows CNC-01 OFFLINE; the late INSP-01 reading never applies.
    t_out = t0 + timedelta(seconds=50.5)
    during = await reconstruct(store, run.machines, t_out)
    assert during.machines["CNC-01"].state is MachineState.OFFLINE
    after_late = await reconstruct(store, run.machines, t0 + timedelta(seconds=90.5))
    inspected = after_late.machines["INSP-01"].last_reading
    assert inspected is not None
    assert inspected.ts == t0 + timedelta(seconds=90)


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
async def test_export_replay_store_gives_the_same_events(
    recorded_run: Run,
    sources_cfg: SourcesConfig,
    store_factory: Any,
    tmp_path: Path,
    fmt: ExportFormat,
) -> None:
    original = await recorded_run.store.fetch(Stream.READINGS)
    out = tmp_path / f"export.{fmt}"
    await export_range(
        recorded_run.store, None, None, out, fmt, machine_order=recorded_run.machines
    )
    fresh_dir = tmp_path / f"fresh_{fmt}"
    fresh_dir.mkdir()
    fresh = await store_factory(fresh_dir)
    try:
        await ingest_file(out, fmt, sources_cfg, fresh, record=False)
        replayed = await fresh.fetch(Stream.READINGS)
    finally:
        await fresh.close()
    assert [r.item for r in replayed] == [r.item for r in original]
    assert {r.item.quality for r in replayed} >= {Quality.LATE}


async def test_runtime_persists_live_ingestion_automatically(
    sim_export: SimExport, sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    """Shipped sim CSV mapping, SQLite from storage.yaml: readings and quarantine persist."""
    assert sources_cfg.file is not None
    file_cfg = sources_cfg.file.model_copy(
        update={"path": str(sim_export.csv), "loop": False, "replay_speed": 1e6}
    )
    ingest = sources_cfg.ingest.model_copy(update={"tick_s": 0.01, "lateness_window_s": 0.05})
    sources = sources_cfg.model_copy(update={"file": file_cfg, "ingest": ingest})
    base = storage_config()
    storage_cfg = base.model_copy(
        update={
            "sqlite": base.sqlite.model_copy(update={"path": str(tmp_path / "rt.sqlite")}),
            "snapshots": base.snapshots.model_copy(update={"settle_s": 0.6}),
        }
    )
    runtime = build_runtime(
        TwinSettings(config_dir=REPO_ROOT / "config"),
        root=REPO_ROOT,
        sources=sources,
        storage_cfg=storage_cfg,
    )
    assert runtime.storage is not None
    await runtime.start()
    assert runtime.manager.pipeline.staleness.time_scale == 1e6
    await runtime.manager.wait_idle()
    await runtime.stop()

    expected = len(load_jsonl_readings(sim_export.jsonl))
    check = SqliteEventStore(tmp_path / "rt.sqlite")
    await check.connect()
    try:
        assert await check.count(Stream.READINGS) == expected
        quarantined = [q.item for q in await check.fetch(Stream.QUARANTINE)]
        assert len(quarantined) == runtime.manager.quarantine.total  # type: ignore[attr-defined]
        assert {q.reason for q in quarantined} <= {QuarantineReason.NO_SENSOR_DATA}
    finally:
        await check.close()
    assert runtime.storage.metrics.dropped_total() == 0
