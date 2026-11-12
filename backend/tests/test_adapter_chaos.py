"""ChaosAdapter: determinism, fault rates, and the pipeline surviving it.

Key requirement: with 10 % duplicates and reordering inside the lateness window, the pipeline
output is complete and correctly ordered; corrupt messages are quarantined with a reason.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from twin_backend.adapters.base import AdapterHealth, BaseAdapter, RawMessage
from twin_backend.adapters.chaos import CORRUPTIONS, ChaosAdapter, corrupt_payload
from twin_backend.adapters.file import FileReplayAdapter
from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.ingest.errors import RejectedMessageError
from twin_backend.ingest.normalizer import Normalizer
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import MemoryQuarantine
from twin_backend.ingest.validator import Validator
from twin_backend.testing import SimExport, drain_readings, load_jsonl_readings
from twin_common.config.line import LineConfig
from twin_common.config.sources import ChaosConfig, FieldMapping, SourcesConfig
from twin_common.contracts import Reading
from twin_common.enums import AdapterKind, Quality, QuarantineReason
from twin_common.testing import sample_reading


def chaos_cfg(sources: SourcesConfig, **rates: float) -> ChaosConfig:
    zero = {"drop": 0.0, "duplicate": 0.0, "delay": 0.0, "reorder": 0.0, "corrupt": 0.0}
    return sources.chaos.model_copy(update={**zero, "enabled": True, **rates})


class ListAdapter(BaseAdapter):
    """Emits a fixed list of payloads, then EOF."""

    kind = AdapterKind.MOCK

    def __init__(self, payloads: list[Any], clock: FakeClock) -> None:
        super().__init__(clock)
        self.payloads = payloads

    async def _run(self) -> None:
        for p in self.payloads:
            await self._emit(p)


async def collect(adapter: ChaosAdapter) -> list[RawMessage]:
    await adapter.start()
    out = [m async for m in adapter.stream()]
    await adapter.stop()
    return out


def seq_payloads(n: int) -> list[dict[str, Any]]:
    return [sample_reading(seq=i).model_dump(mode="json") for i in range(1, n + 1)]


async def test_same_seed_same_output(sources_cfg: SourcesConfig) -> None:
    cfg = chaos_cfg(sources_cfg, drop=0.1, duplicate=0.1, reorder=0.2, corrupt=0.05)
    runs = []
    for _ in range(2):
        clock = FakeClock()
        out = await collect(ChaosAdapter(ListAdapter(seq_payloads(300), clock), cfg, clock))
        runs.append([m.payload for m in out])
    assert runs[0] == runs[1]
    clock = FakeClock()
    other = await collect(
        ChaosAdapter(
            ListAdapter(seq_payloads(300), clock), cfg.model_copy(update={"seed": 99}), clock
        )
    )
    assert [m.payload for m in other] != runs[0]


async def test_rates_roughly_match(sources_cfg: SourcesConfig) -> None:
    n = 5000
    clock = FakeClock()
    chaos = ChaosAdapter(
        ListAdapter(seq_payloads(n), clock),
        chaos_cfg(sources_cfg, drop=0.1, duplicate=0.1, corrupt=0.05),
        clock,
    )
    out = await collect(chaos)
    s = chaos.stats
    assert s["in"] == n
    for key, rate in (("dropped", 0.1), ("duplicated", 0.09), ("corrupted", 0.045)):
        assert abs(s[key] / n - rate) < 0.02, (key, s[key])
    assert len(out) == n - s["dropped"] + s["duplicated"]


async def test_reorder_bounded_displacement(sources_cfg: SourcesConfig) -> None:
    clock = FakeClock()
    cfg = chaos_cfg(sources_cfg, reorder=0.3)
    out = await collect(ChaosAdapter(ListAdapter(seq_payloads(500), clock), cfg, clock))
    seqs = [int(dict(m.payload)["seq"]) for m in out]  # type: ignore[arg-type]
    assert sorted(seqs) == list(range(1, 501))
    assert seqs != sorted(seqs)
    late_by = [max(0, i - (s - 1)) for i, s in enumerate(seqs)]
    assert max(late_by) <= 10  # a message is overtaken by only a few others


async def test_delay_releases_later(sources_cfg: SourcesConfig) -> None:
    clock = FakeClock()

    class Ticking(ListAdapter):
        async def _run(self) -> None:
            for p in self.payloads:
                await self._emit(p)
                clock.advance(1.0)

    cfg = chaos_cfg(sources_cfg, delay=0.2, delay_s=3.0)
    chaos = ChaosAdapter(Ticking(seq_payloads(200), clock), cfg, clock)
    out = await collect(chaos)
    seqs = [int(dict(m.payload)["seq"]) for m in out]  # type: ignore[arg-type]
    assert sorted(seqs) == list(range(1, 201))
    assert chaos.stats["delayed"] > 0
    assert seqs != sorted(seqs)


@pytest.mark.parametrize("variant", CORRUPTIONS)
def test_every_corruption_is_rejected(variant: str, line_cfg: LineConfig) -> None:
    clock = FakeClock()
    good = sample_reading().model_dump(mode="json")
    for payload in (good, {k: str(v) for k, v in good.items() if k != "schemaVersion"}):
        bad = corrupt_payload(payload, variant)
        raw = RawMessage(AdapterKind.FILE, bad, clock.monotonic(), clock.now())
        with pytest.raises(RejectedMessageError) as info:
            Validator(line_cfg, "suspect").validate(
                Normalizer(FieldMapping(), line_cfg).normalize(raw)
            )
        assert info.value.reason in {
            QuarantineReason.DECODE_ERROR,
            QuarantineReason.NON_FINITE,
            QuarantineReason.SCHEMA_ERROR,
        }


def by_machine(readings: list[Reading]) -> dict[str, list[Reading]]:
    out: dict[str, list[Reading]] = defaultdict(list)
    for r in readings:
        out[r.machine_id].append(r)
    return dict(out)


class ChaosRig:
    def __init__(
        self, line: LineConfig, sources: SourcesConfig, export: SimExport, chaos: ChaosConfig
    ) -> None:
        self.clock = FakeClock()  # arrival time stands still: the window never expires
        self.bus = InMemoryEventBus(queue_size=1_000_000)
        self.sub = self.bus.subscribe(Topic.READINGS)
        self.quarantine = MemoryQuarantine(100_000)
        assert sources.file is not None
        file_cfg = sources.file.model_copy(
            update={
                "path": str(export.jsonl),
                "format": "jsonl",
                "loop": False,
                "replay_speed": 1e9,
                "mapping": FieldMapping(),
            }
        )
        self.inner = FileReplayAdapter(file_cfg, line, root=Path("."), clock=self.clock)
        self.chaos = ChaosAdapter(self.inner, chaos, self.clock)
        self.pipeline = IngestPipeline(
            line,
            sources.ingest,
            self.bus,
            self.quarantine,
            self.clock,
            {AdapterKind.FILE: FieldMapping()},
        )

    async def run(self) -> list[Reading]:
        """Arrival time advances 10 ms per message: the chaos displacement (a few messages)
        stays far inside the 2 s lateness window, while the initial hold does expire."""
        await self.chaos.start()
        async for raw in self.chaos.stream():
            self.clock.advance(ARRIVAL_STEP_S)
            await self.pipeline.process(raw)
            await self.pipeline.tick()
        await self.chaos.stop()
        await self.pipeline.flush()
        return drain_readings(self.sub)


ARRIVAL_STEP_S = 0.01


async def test_duplicates_and_reordering_give_complete_ordered_output(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = ChaosRig(
        line_cfg, sources_cfg, sim_export, chaos_cfg(sources_cfg, duplicate=0.10, reorder=0.10)
    )
    out = await rig.run()
    expected = by_machine(load_jsonl_readings(sim_export.jsonl))
    got = by_machine(out)
    assert got == expected  # complete, no duplicates, seq order per machine, values intact
    assert all(r.quality is Quality.OK for r in out)
    m = rig.pipeline.metrics
    assert rig.chaos.stats["duplicated"] > 0
    assert rig.chaos.stats["reordered"] > 0
    assert m.value("duplicates", source="file") == rig.chaos.stats["duplicated"]
    assert m.value("reordered", source="file") > 0
    assert m.value("late", source="file") == 0
    assert m.value("gap_seqs", source="file") == 0
    assert not rig.quarantine.records


async def test_corrupt_messages_are_quarantined_with_reason(
    sim_export: SimExport, line_cfg: LineConfig, sources_cfg: SourcesConfig
) -> None:
    rig = ChaosRig(
        line_cfg,
        sources_cfg,
        sim_export,
        chaos_cfg(sources_cfg, corrupt=0.05, duplicate=0.10, reorder=0.10),
    )
    out = await rig.run()
    expected = load_jsonl_readings(sim_export.jsonl)
    corrupted = rig.chaos.stats["corrupted"]
    assert corrupted > 0
    # Every message is either delivered once or quarantined; nothing else is lost.
    assert len(out) == len(expected) - corrupted
    expected_keys = {(r.machine_id, r.seq): r for r in expected}
    assert all(expected_keys[(r.machine_id, r.seq)] == r for r in out)
    for machine, readings in by_machine(out).items():
        seqs = [r.seq for r in readings]
        assert seqs == sorted(seqs), machine
    records = list(rig.quarantine.records)
    assert len(records) >= corrupted  # duplicated corrupt copies are quarantined too
    assert {r.reason for r in records} <= {
        QuarantineReason.DECODE_ERROR,
        QuarantineReason.NON_FINITE,
        QuarantineReason.SCHEMA_ERROR,
    }
    assert all(r.detail for r in records)
    assert sum(rig.pipeline.metrics.quarantined_by_reason(AdapterKind.FILE).values()) == len(
        records
    )


async def test_chaos_passes_through_commands_and_health(sources_cfg: SourcesConfig) -> None:
    from twin_common.testing import sample_command

    clock = FakeClock()
    inner = ListAdapter([], clock)
    chaos = ChaosAdapter(inner, chaos_cfg(sources_cfg), clock)
    assert chaos.kind is inner.kind
    assert isinstance(chaos.health(), AdapterHealth)
    ack = await chaos.send_command(sample_command())
    assert ack.status.value == "REJECTED"
