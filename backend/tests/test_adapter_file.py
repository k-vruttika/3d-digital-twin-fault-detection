"""FileReplayAdapter: CSV / JSONL / JSON replay, pacing, offset, loop, missing-file message.

These tests need nothing but files: no broker, database, network or ML (minimal mode).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from twin_backend.adapters.base import AdapterError
from twin_backend.adapters.file import FileReplayAdapter
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import MemoryQuarantine
from twin_backend.testing import (
    SimExport,
    canonical_order,
    drain_readings,
    load_jsonl_readings,
    replay_into,
)
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping, FileSourceConfig, SourcesConfig
from twin_common.enums import AdapterKind, AdapterStatus, Quality, QuarantineReason


def file_cfg(sources: SourcesConfig, path: Path, fmt: str, **kw: Any) -> FileSourceConfig:
    assert sources.file is not None
    base = sources.file.model_copy(update={"path": str(path), "format": fmt, "loop": False})
    if fmt != "csv":
        base = base.model_copy(update={"mapping": FieldMapping()})
    return base.model_copy(update=kw)


class Rig:
    def __init__(self, line: LineConfig, sources: SourcesConfig, cfg: FileSourceConfig) -> None:
        self.clock = FakeClock(auto_advance=True)
        self.bus = InMemoryEventBus(queue_size=1_000_000)
        self.sub = self.bus.subscribe(Topic.READINGS)
        self.quarantine = MemoryQuarantine(10_000)
        self.adapter = FileReplayAdapter(cfg, line, root=Path("."), clock=self.clock)
        self.pipeline = IngestPipeline(
            line,
            sources.ingest,
            self.bus,
            self.quarantine,
            self.clock,
            {AdapterKind.FILE: cfg.mapping},
        )

    async def run(self) -> list[Any]:
        await replay_into(self.adapter, self.pipeline)
        return drain_readings(self.sub)


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
async def test_replay_reproduces_the_simulated_run(
    fmt: str, sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    path = sim_export.jsonl if fmt == "jsonl" else sim_export.csv
    rig = Rig(line_cfg, sources_cfg, file_cfg(sources_cfg, path, fmt))
    out = await rig.run()
    expected = load_jsonl_readings(sim_export.jsonl)
    machines = [m.id for m in line_cfg.machines]
    assert canonical_order(out, machines) == canonical_order(expected, machines)
    assert all(r.quality is Quality.OK for r in out)
    assert rig.adapter.health().status is AdapterStatus.EOF
    # Real-time pacing: the replay spans the recording's duration on the (fake) clock.
    span = (expected[-1].ts - expected[0].ts).total_seconds()
    assert sum(rig.clock.slept) == pytest.approx(span)


async def test_speed_multiplier(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = Rig(
        line_cfg, sources_cfg, file_cfg(sources_cfg, sim_export.jsonl, "jsonl", replay_speed=10.0)
    )
    out = await rig.run()
    span = (out[-1].ts - out[0].ts).total_seconds()
    assert sum(rig.clock.slept) == pytest.approx(span / 10.0)


async def test_start_offset(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = Rig(
        line_cfg, sources_cfg, file_cfg(sources_cfg, sim_export.csv, "csv", start_offset_s=60.0)
    )
    out = await rig.run()
    first = load_jsonl_readings(sim_export.jsonl)[0].ts
    assert min(r.ts for r in out) == first + timedelta(seconds=60)
    assert {r.machine_id for r in out} == {m.id for m in line_cfg.machines}


async def test_loop_continues_seq_and_ts(
    sim_export_single: SimExport, single_line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    cfg = file_cfg(sources_cfg, sim_export_single.jsonl, "jsonl", loop=True, replay_speed=1e6)
    rig = Rig(single_line_cfg, sources_cfg, cfg)
    await rig.adapter.start()
    n_file = len(load_jsonl_readings(sim_export_single.jsonl))
    count = 0
    async for raw in rig.adapter.stream():
        await rig.pipeline.process(raw)
        await rig.pipeline.tick()
        count += 1
        if count == 2 * n_file + 5:
            break
    await rig.adapter.stop()
    await rig.pipeline.flush()
    out = drain_readings(rig.sub)
    seqs = [r.seq for r in out]
    assert seqs == list(range(1, len(seqs) + 1))  # continuous across passes, nothing deduped
    ts = [r.ts for r in out]
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)
    assert rig.pipeline.metrics.value("duplicates", source="file") == 0


async def test_single_line_replay_of_full_line_file_quarantines_other_machines(
    sim_export: SimExport, single_line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = Rig(single_line_cfg, sources_cfg, file_cfg(sources_cfg, sim_export.csv, "csv"))
    out = await rig.run()
    assert {r.machine_id for r in out} == {"CNV-01"}
    reasons = {q.reason for q in rig.quarantine.records}
    assert reasons == {QuarantineReason.UNKNOWN_MACHINE}
    assert {q.machine_id for q in rig.quarantine.records} == {"CNC-01", "INSP-01", "CNV-02"}


async def test_json_array_and_bad_lines(
    tmp_path: Path, line_cfg: LineConfig, sources_cfg: SourcesConfig, sim_export: SimExport
) -> None:
    readings = load_jsonl_readings(sim_export.jsonl)[:8]
    arr = tmp_path / "readings.json"
    arr.write_text(json.dumps([r.model_dump(mode="json") for r in readings]), encoding="utf-8")
    rig = Rig(line_cfg, sources_cfg, file_cfg(sources_cfg, arr, "json"))
    assert len(await rig.run()) == 8

    jsonl = tmp_path / "messy.jsonl"
    lines = [r.model_dump_json() for r in readings[:4]]
    jsonl.write_text(
        "\n".join([lines[0], "{oops", lines[1], "", "[1,2]", *lines[2:]]), encoding="utf-8"
    )
    rig = Rig(line_cfg, sources_cfg, file_cfg(sources_cfg, jsonl, "jsonl"))
    assert len(await rig.run()) == 4
    assert [q.reason for q in rig.quarantine.records] == [QuarantineReason.DECODE_ERROR] * 2
    assert rig.quarantine.records[0].meta["line"] == "2"


async def test_missing_file_has_helpful_message(
    line_cfg: LineConfig, sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    cfg = file_cfg(sources_cfg, tmp_path / "nope.csv", "csv")
    adapter = FileReplayAdapter(cfg, line_cfg, root=tmp_path, clock=FakeClock())
    with pytest.raises(AdapterError, match=r"replay file not found.*uv run poe sim-quick"):
        await adapter.start()


async def test_file_adapter_rejects_commands(
    line_cfg: LineConfig, sources_cfg: SourcesConfig, sim_export: SimExport
) -> None:
    from twin_common.testing import sample_command

    adapter = FileReplayAdapter(
        file_cfg(sources_cfg, sim_export.jsonl, "jsonl"),
        line_cfg,
        root=Path("."),
        clock=FakeClock(),
    )
    ack = await adapter.send_command(sample_command())
    assert ack.status.value == "REJECTED"
    assert "does not support commands" in (ack.reason or "")
    assert adapter.capabilities == frozenset()
