"""TwinEngine on the event bus: engine StateChanges with root-cause reasons, alerts, data
quality, acknowledgement, restart restore, and live state == reconstruct with the engine on."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.engine.twin import TwinEngine
from twin_backend.ingest.quarantine import QuarantineRecord
from twin_backend.storage.base import Stream
from twin_backend.testing import REPO_ROOT, SimExport, drain_readings, file_runtime
from twin_common.config import load_engine_config, load_line_config
from twin_common.config.engine import AlertKind
from twin_common.contracts import Alert, Reading, StateChange
from twin_common.enums import AdapterKind, DataQuality, MachineState, QuarantineReason
from twin_common.testing import SAMPLE_TS, sample_reading

R, IDLE, S, B = (
    MachineState.RUNNING,
    MachineState.IDLE,
    MachineState.STARVED,
    MachineState.BLOCKED,
)
LINE = load_line_config(REPO_ROOT / "config" / "line.yaml")
ENGINE = load_engine_config(REPO_ROOT / "config" / "engine.yaml")
ORDER = ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]


class Rig:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.bus = InMemoryEventBus(100_000, clock=self.clock)
        self.states = self.bus.subscribe(Topic.STATES)
        self.alerts = self.bus.subscribe(Topic.ALERTS)
        self.engine = TwinEngine(LINE, ENGINE, self.bus, self.clock)
        self.seq = 0
        self.t = 0

    async def second(self, *states: MachineState) -> None:
        """One reading per machine at the next second, in line order."""
        self.seq += 1
        for m, st in zip(ORDER, states, strict=True):
            r = sample_reading(
                machine_id=m, seq=self.seq, ts=SAMPLE_TS + timedelta(seconds=self.t), state=st
            )
            await self.bus.publish(Topic.READINGS, r)
        self.t += 1
        await self.engine.drain()

    def state_changes(self) -> list[StateChange]:
        return [m for m in drain_readings(self.states) if isinstance(m, StateChange)]

    def alert_events(self) -> list[Alert]:
        return [m for m in drain_readings(self.alerts) if isinstance(m, Alert)]


async def test_cnc_stop_propagation_reasons_and_line_stoppage_alert() -> None:
    rig = Rig()
    await rig.second(R, R, R, R)
    first = rig.state_changes()
    assert {c.machine_id: (c.state, c.reason) for c in first} == dict.fromkeys(
        ORDER, (R, "reported by source")
    )
    await rig.second(R, IDLE, R, R)  # operator stops CNC-01
    await rig.second(B, IDLE, S, S)
    changes = {c.machine_id: c for c in rig.state_changes()}
    assert changes["CNC-01"].state is IDLE
    assert changes["CNC-01"].previous_state is R
    assert changes["CNV-01"].reason == "BLOCKED by CNC-01"
    assert changes["INSP-01"].reason == "STARVED by CNC-01"
    assert changes["CNV-02"].reason == "STARVED by CNC-01"
    assert rig.engine.line_view().impacted == {"CNC-01": ["CNV-01", "INSP-01", "CNV-02"]}
    assert rig.alert_events() == []  # debounce not over yet

    debounce = ENGINE.alerts.kinds[AlertKind.LINE_STOPPAGE].raise_after_s
    for _ in range(int(debounce)):
        await rig.second(B, IDLE, S, S)
    [raised] = rig.alert_events()
    assert raised.kind == "line_stoppage"
    assert raised.machine_id == "CNC-01"
    assert raised.active
    assert "stops CNV-01 (BLOCKED), INSP-01 (STARVED), CNV-02 (STARVED)" in raised.message
    await rig.second(B, IDLE, S, S)
    assert rig.alert_events() == []  # deduplicated

    await rig.second(B, R, S, S)  # restart: CNC-01 runs again; the others still wait
    await rig.second(R, R, R, R)
    [cleared] = rig.alert_events()
    assert (cleared.alert_id, cleared.active, cleared.acknowledged) == (
        raised.alert_id,
        False,
        False,
    )
    assert rig.engine.alerts.current() == [cleared]  # still visible until acknowledged
    res = rig.engine.acknowledge(raised.alert_id, "alice")
    await rig.engine.flush()
    assert res.alert is not None
    assert res.alert.resolved
    [acked] = rig.alert_events()
    assert acked.acked_by == "alice"
    assert acked.acked_at == rig.clock.now()
    assert acked.ts >= cleared.ts  # event-time axis
    assert rig.engine.alerts.current() == []
    assert rig.engine.state_machines.invalid_count() == 0


async def test_fault_warning_and_offline_alerts_follow_state() -> None:
    rig = Rig()
    await rig.second(R, R, R, R)
    await rig.second(R, MachineState.WARNING, R, R)
    await rig.second(R, MachineState.FAULT, R, R)
    events = rig.alert_events()
    assert [(a.kind, a.active) for a in events] == [
        ("machine_warning", True),
        ("machine_warning", False),
        ("machine_fault", True),
    ]
    assert events[2].severity.value == "CRITICAL"
    # the pipeline reports the link down (StateChange, not a reading)
    offline_ts = SAMPLE_TS + timedelta(seconds=10)
    await rig.bus.publish(
        Topic.STATES,
        StateChange(
            machine_id="CNV-02",
            state=MachineState.OFFLINE,
            previous_state=MachineState.STALE,
            ts=offline_ts,
            reason="no data",
        ),
    )
    await rig.engine.drain()
    [offline] = rig.alert_events()
    assert offline.kind == "machine_offline"
    assert offline.ts == offline_ts
    offline_changes = [c for c in rig.state_changes() if c.state is MachineState.OFFLINE]
    assert [c.reason for c in offline_changes] == ["no data"]  # not re-emitted by the engine
    rig.t = 11
    await rig.second(R, MachineState.FAULT, R, R)
    assert [(a.kind, a.active) for a in rig.alert_events()] == [("machine_offline", False)]


async def test_data_quality_degrades_on_quarantine_and_raises_alert() -> None:
    rig = Rig()
    await rig.second(R, R, R, R)
    for _ in range(ENGINE.data_quality.min_samples):
        await rig.bus.publish(
            Topic.QUARANTINE,
            QuarantineRecord(
                ts=rig.clock.now(),
                source=AdapterKind.FILE,
                reason=QuarantineReason.OUT_OF_RANGE,
                stage="validate",
                detail="x",
                machine_id="INSP-01",
                payload={},
                meta={},
            ),
        )
    await rig.engine.drain()
    rig.engine.tick()
    await rig.engine.flush()
    assert rig.engine.data_quality("INSP-01").status is DataQuality.DEGRADED
    assert rig.engine.data_quality("CNC-01").status is DataQuality.OK
    [a] = rig.alert_events()
    assert a.kind == "data_quality"
    assert a.machine_id == "INSP-01"
    assert "quarantined" in a.message
    rig.clock.advance(ENGINE.data_quality.window_s + ENGINE.data_quality.recover_after_s)
    rig.engine.tick()
    rig.engine.tick()
    await rig.engine.flush()
    assert rig.engine.data_quality("INSP-01").status is DataQuality.OK
    assert [x.active for x in rig.alert_events()] == [False]


async def test_listeners_get_line_and_data_quality_updates() -> None:
    rig = Rig()
    seen: list[str] = []
    rig.engine.add_listener(lambda channel, _payload: seen.append(channel))
    await rig.second(R, R, R, R)
    rig.engine.tick()
    assert "line" in seen
    assert "dataQuality" in seen


# ---------------------------------------------------------------- with the runtime + store
async def test_live_engine_state_equals_reconstruct_and_alerts_survive_restart(
    sim_export: SimExport, store_backend: str, runtime_dsn: str | None, tmp_path: Path
) -> None:
    dsn = runtime_dsn
    rt = file_runtime(sim_export, tmp_path, store_backend, dsn)
    await rt.start()
    await rt.manager.wait_idle()
    await rt.stop()
    live = rt.engine.snapshot()
    unresolved = rt.engine.alerts.current()

    rt2 = file_runtime(sim_export, tmp_path, store_backend, dsn)
    await rt2.storage.start()
    try:
        store = rt2.storage.store
        watermark = live.watermark()
        assert watermark is not None
        rebuilt = await rt2.storage.reconstruct(watermark)
        assert rebuilt.body() == live.body()
        engine_changes = [
            s.item
            for s in await store.fetch(Stream.STATE_CHANGES)
            if s.item.reason and s.item.reason != "no data"
        ]
        assert engine_changes, "engine StateChanges are stored"
        stored_alerts = [s.item for s in await store.fetch(Stream.ALERTS)]
    finally:
        await rt2.storage.stop()
    # restart restores the unresolved alerts
    rt2.engine.load_alerts(stored_alerts)
    assert rt2.engine.alerts.current() == unresolved
    assert all(isinstance(r, Reading) for r in [live.machines["CNC-01"].last_reading])
