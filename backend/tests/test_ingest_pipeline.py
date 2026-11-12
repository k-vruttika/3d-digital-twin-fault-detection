"""IngestPipeline end to end on hand-made messages: quarantine, dedup, reorder, LATE, staleness,
metrics and the quarantine sinks."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from twin_backend.adapters.base import RawMessage
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import JsonlQuarantine, MemoryQuarantine, build_quarantine
from twin_common.config.line import LineConfig
from twin_common.config.sources import QuarantineConfig, SourcesConfig
from twin_common.contracts import Reading, StateChange
from twin_common.enums import AdapterKind, MachineState, Quality, QuarantineReason
from twin_common.testing import SAMPLE_TS, sample_reading


class Harness:
    def __init__(self, line: LineConfig, sources: SourcesConfig, **ingest: Any) -> None:
        self.clock = FakeClock()
        self.bus = InMemoryEventBus(queue_size=100_000)
        self.readings = self.bus.subscribe(Topic.READINGS)
        self.states = self.bus.subscribe(Topic.STATES)
        self.quarantine = MemoryQuarantine(1000)
        cfg = sources.ingest.model_copy(update=ingest)
        self.pipeline = IngestPipeline(
            line,
            cfg,
            self.bus,
            self.quarantine,
            self.clock,
            {AdapterKind.FILE: sources.file.mapping},  # type: ignore[union-attr]
        )
        self.window = cfg.lateness_window_s

    def raw(self, payload: Any) -> RawMessage:
        return RawMessage(AdapterKind.MOCK, payload, self.clock.monotonic(), self.clock.now())

    async def send(self, machine: str, seq: int, **overrides: Any) -> None:
        r = sample_reading(machine_id=machine, seq=seq, ts=SAMPLE_TS + timedelta(seconds=seq))
        data = r.model_dump(mode="json")
        data.update(overrides)
        await self.pipeline.process(self.raw(data))

    def out(self) -> list[Reading]:
        got = []
        while (m := self.readings.get_nowait()) is not None:
            assert isinstance(m, Reading)
            got.append(m)
        return got

    def state_changes(self) -> list[StateChange]:
        got = []
        while (m := self.states.get_nowait()) is not None:
            assert isinstance(m, StateChange)
            got.append(m)
        return got

    def metric(self, name: str, **labels: str) -> float:
        return self.pipeline.metrics.value(name, source="mock", **labels)

    async def warm_up(self, machine: str = "CNC-01") -> None:
        """Seq 1 through the initial hold."""
        await self.send(machine, 1)
        self.clock.advance(self.window)
        await self.pipeline.tick()
        assert [r.seq for r in self.out()] == [1]


@pytest.fixture
def h(line_cfg: LineConfig, sources_cfg: SourcesConfig) -> Harness:
    return Harness(line_cfg, sources_cfg)


async def test_in_order_flow_and_metrics(h: Harness) -> None:
    await h.warm_up()
    for seq in range(2, 6):
        await h.send("CNC-01", seq)
    assert [r.seq for r in h.out()] == [2, 3, 4, 5]
    assert h.metric("messages_in") == 5
    assert h.metric("published") == 5
    hist = h.pipeline.metrics.registry.get_sample_value(
        "linetwin_ingest_e2e_latency_seconds_count", {"source": "mock", "basis": "receive"}
    )
    assert hist == 5


async def test_duplicates_dropped(h: Harness) -> None:
    await h.warm_up()
    await h.send("CNC-01", 2)
    await h.send("CNC-01", 2)
    await h.send("CNC-01", 1)
    assert [r.seq for r in h.out()] == [2]
    assert h.metric("duplicates") == 2


async def test_reorder_within_window(h: Harness) -> None:
    await h.warm_up()
    await h.send("CNC-01", 3)
    await h.send("CNC-01", 4)
    assert h.out() == []
    h.clock.advance(h.window / 2)
    await h.send("CNC-01", 2)
    out = h.out()
    assert [r.seq for r in out] == [2, 3, 4]
    assert all(r.quality is Quality.OK for r in out)
    assert h.metric("reordered") == 2


async def test_late_message_flagged_and_gap_counted(h: Harness) -> None:
    await h.warm_up()
    await h.send("CNC-01", 4)
    h.clock.advance(h.window)
    await h.pipeline.tick()
    assert [r.seq for r in h.out()] == [4]
    assert h.metric("gap_seqs") == 2
    await h.send("CNC-01", 2)
    [late] = h.out()
    assert (late.seq, late.quality) == (2, Quality.LATE)
    assert h.metric("late") == 1


async def test_quarantine_reasons(h: Harness) -> None:
    await h.pipeline.process(h.raw(b"\x00garbage"))
    await h.send("CNC-01", 1, temp=float("nan"))
    bad_seq = sample_reading(seq=2).model_dump(mode="json") | {"seq": "x"}
    await h.pipeline.process(h.raw(bad_seq))
    await h.send("PRESS-9", 3)
    reasons = [q.reason for q in h.quarantine.records]
    assert reasons == [
        QuarantineReason.DECODE_ERROR,
        QuarantineReason.NON_FINITE,
        QuarantineReason.SCHEMA_ERROR,
        QuarantineReason.UNKNOWN_MACHINE,
    ]
    rec = h.quarantine.records[3]
    assert rec.machine_id == "PRESS-9"
    assert rec.stage == "validate"
    assert isinstance(rec.payload, dict)
    assert rec.payload["machineId"] == "PRESS-9"
    assert h.metric("quarantined", reason="NON_FINITE") == 1
    assert h.out() == []


async def test_out_of_range_suspect_published(h: Harness) -> None:
    await h.warm_up()
    await h.send("CNC-01", 2, temp=500.0)
    [r] = h.out()
    assert r.quality is Quality.SUSPECT
    assert h.metric("suspect") == 1


async def test_out_of_range_quarantine_mode(
    line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    hq = Harness(line_cfg, sources_cfg, out_of_range="quarantine")
    await hq.send("CNC-01", 1, temp=500.0)
    assert [q.reason for q in hq.quarantine.records] == [QuarantineReason.OUT_OF_RANGE]


async def test_staleness_states_on_bus(h: Harness) -> None:
    await h.warm_up("CNV-01")  # CNV-01 last seen at t=0, clock now at t=window
    h.state_changes()
    stale_s, offline_s = h.pipeline.cfg.stale_after_s, h.pipeline.cfg.offline_after_s
    h.clock.advance(stale_s - h.window)
    await h.pipeline.tick()
    assert {(c.machine_id, c.state) for c in h.state_changes()} >= {("CNV-01", MachineState.STALE)}
    h.clock.advance(offline_s - stale_s)
    await h.pipeline.tick()
    assert ("CNV-01", MachineState.OFFLINE) in {(c.machine_id, c.state) for c in h.state_changes()}
    await h.send("CNV-01", 2)
    [back] = [c for c in h.state_changes() if c.machine_id == "CNV-01"]
    assert (back.state, back.previous_state) == (MachineState.RUNNING, MachineState.OFFLINE)


async def test_late_data_does_not_refresh_staleness(h: Harness) -> None:
    await h.warm_up()
    t_live = h.clock.monotonic()
    await h.send("CNC-01", 5)  # last live data
    h.clock.advance(h.window)
    await h.pipeline.tick()  # gap 2-4 given up
    h.out()
    h.clock.advance(t_live + h.pipeline.cfg.stale_after_s - h.clock.monotonic())
    await h.send("CNC-01", 3)  # late: must not count as a sign of life
    await h.pipeline.tick()
    changes = [c for c in h.state_changes() if c.machine_id == "CNC-01"]
    assert [c.state for c in changes] == [MachineState.STALE]
    assert h.out()[0].quality is Quality.LATE


async def test_source_reset_restarts_sequence(h: Harness) -> None:
    await h.warm_up()
    threshold = h.pipeline.cfg.seq_reset_threshold
    big = threshold + 100
    await h.send("CNC-01", big)
    h.clock.advance(h.window)
    await h.pipeline.tick()
    assert [r.seq for r in h.out()] == [big]
    await h.send("CNC-01", 1)  # restarted source: new, not a duplicate and not late
    h.clock.advance(h.window)
    await h.pipeline.tick()
    out = h.out()
    assert [(r.seq, r.quality) for r in out] == [(1, Quality.OK)]
    assert h.metric("source_resets") == 1


async def test_sent_at_basis(h: Harness) -> None:
    raw = RawMessage(
        AdapterKind.MOCK,
        sample_reading(seq=1).model_dump(mode="json"),
        h.clock.monotonic(),
        h.clock.now(),
        sent_at=h.clock.now() - timedelta(seconds=0.25),
    )
    await h.pipeline.process(raw)
    h.clock.advance(h.window)
    await h.pipeline.tick()
    total = h.pipeline.metrics.registry.get_sample_value(
        "linetwin_ingest_e2e_latency_seconds_sum", {"source": "mock", "basis": "source"}
    )
    assert total == pytest.approx(0.25 + h.window)


async def test_jsonl_quarantine_sink(tmp_path: Path, line_cfg: LineConfig) -> None:
    cfg = QuarantineConfig(kind="jsonl", path="q/quarantine.jsonl", max_records=1)
    sink = build_quarantine(cfg, tmp_path)
    assert isinstance(sink, JsonlQuarantine)
    from twin_backend.ingest.quarantine import QuarantineRecord

    rec = QuarantineRecord(
        ts=SAMPLE_TS,
        source=AdapterKind.FILE,
        reason=QuarantineReason.DECODE_ERROR,
        stage="normalize",
        detail="bad",
        machine_id=None,
        payload="garbage",
        meta={"line": "3"},
    )
    await sink.put(rec)
    await sink.put(rec)
    lines = (tmp_path / "q" / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "DECODE_ERROR"


async def test_memory_quarantine_bounded() -> None:
    sink = MemoryQuarantine(max_records=2)
    from twin_backend.ingest.quarantine import QuarantineRecord

    for i in range(3):
        await sink.put(
            QuarantineRecord(
                ts=SAMPLE_TS,
                source=AdapterKind.FILE,
                reason=QuarantineReason.SCHEMA_ERROR,
                stage="validate",
                detail=str(i),
                machine_id=None,
                payload={},
                meta={},
            )
        )
    assert [r.detail for r in sink.records] == ["1", "2"]
    assert sink.total == 3
