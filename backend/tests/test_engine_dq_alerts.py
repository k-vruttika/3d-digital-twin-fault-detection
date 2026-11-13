"""Data-quality status (thresholds, staleness, hysteresis) and the alert lifecycle (ISA-18.2)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from twin_backend.clock import FakeClock
from twin_backend.engine.alerts import AlertManager
from twin_backend.engine.data_quality import DataQualityMonitor
from twin_backend.storage.base import Stored, Stream
from twin_backend.testing import REPO_ROOT
from twin_common.config import load_engine_config
from twin_common.config.engine import AlertKind, DataQualityConfig
from twin_common.contracts import Alert
from twin_common.enums import AlertSeverity, DataQuality, Quality
from twin_common.testing import SAMPLE_TS, sample_reading

ENGINE = load_engine_config(REPO_ROOT / "config" / "engine.yaml")
DQ = DataQualityConfig(
    window_s=10.0,
    min_samples=5,
    max_late_pct=10.0,
    max_quarantine_pct=10.0,
    max_partial_pct=20.0,
    stale_is_degraded=True,
    recover_after_s=3.0,
    evaluate_every_s=1.0,
)


def _monitor() -> tuple[DataQualityMonitor, FakeClock]:
    clock = FakeClock()
    return DataQualityMonitor(["M"], DQ, clock), clock


def _feed(dq: DataQualityMonitor, clock: FakeClock, n: int, **kw: Any) -> None:
    for i in range(n):
        dq.add_reading(sample_reading(machine_id="M", seq=i, **kw))
        clock.advance(0.1)


def test_dq_ok_then_degraded_on_late_rate_then_recovers_after_hysteresis() -> None:
    dq, clock = _monitor()
    _feed(dq, clock, 20)
    assert dq.evaluate()["M"].status is DataQuality.OK
    _feed(dq, clock, 5, quality=Quality.LATE)  # 5 / 25 = 20 % late
    report = dq.evaluate()["M"]
    assert report.status is DataQuality.DEGRADED
    assert report.late_pct == pytest.approx(20.0)
    assert any("late" in r for r in report.reasons)
    _feed(dq, clock, 10)  # 5 / 35 = 14 %: still a violation (0.1 s per reading)
    assert dq.evaluate()["M"].status is DataQuality.DEGRADED
    _feed(dq, clock, 20)  # 5 / 55 = 9 %: no violation any more...
    report = dq.evaluate()["M"]
    assert report.reasons == []
    assert report.status is DataQuality.DEGRADED
    clock.advance(DQ.recover_after_s)
    assert dq.evaluate()["M"].status is DataQuality.OK


def test_dq_partial_and_quarantine_rates() -> None:
    dq, clock = _monitor()
    _feed(dq, clock, 6, vib=None)  # partial readings
    _feed(dq, clock, 4)
    report = dq.evaluate()["M"]
    assert report.partial_pct == pytest.approx(60.0)
    assert report.status is DataQuality.DEGRADED
    dq2, clock2 = _monitor()
    _feed(dq2, clock2, 9)
    dq2.add_quarantined("M")
    dq2.add_quarantined(None)  # unattributable: not counted against any machine
    dq2.add_quarantined("other")
    report2 = dq2.evaluate()["M"]
    assert report2.samples == 10
    assert report2.quarantine_pct == pytest.approx(10.0)
    assert report2.status is DataQuality.OK  # not above the 10 % threshold
    dq2.add_quarantined("M")
    assert dq2.evaluate()["M"].status is DataQuality.DEGRADED


def test_dq_needs_min_samples_but_link_down_always_degrades() -> None:
    dq, clock = _monitor()
    _feed(dq, clock, 2, quality=Quality.LATE)  # 100 % late, but only 2 samples
    assert dq.evaluate()["M"].status is DataQuality.OK
    dq.set_link_down("M", True)
    report = dq.evaluate()["M"]
    assert report.status is DataQuality.DEGRADED
    assert report.link_down
    dq.set_link_down("M", False)
    clock.advance(DQ.recover_after_s)
    assert dq.evaluate()["M"].status is DataQuality.OK


# ---------------------------------------------------------------- alert lifecycle
def at(s: float) -> Any:
    return SAMPLE_TS + timedelta(seconds=s)


def _mgr() -> AlertManager:
    return AlertManager(ENGINE.alerts)


def test_raise_is_deduplicated_and_clear_keeps_unacked_alert_visible() -> None:
    m = _mgr()
    [a] = m.update("CNC-01", AlertKind.MACHINE_FAULT, True, at(0), "CNC-01 is FAULT")
    assert (a.active, a.acknowledged, a.rev) == (True, False, 0)
    assert a.severity is AlertSeverity.CRITICAL
    assert a.source == "engine"
    assert a.alert_id == f"CNC-01:machine_fault:{int(at(0).timestamp() * 1e6)}"
    assert m.update("CNC-01", AlertKind.MACHINE_FAULT, True, at(1), "again") == []  # dedup
    [c] = m.update("CNC-01", AlertKind.MACHINE_FAULT, False, at(2), "")
    assert (c.alert_id, c.active, c.acknowledged, c.rev) == (a.alert_id, False, False, 1)
    assert m.current() == [c]  # cleared but not acknowledged: still listed
    res = m.acknowledge(a.alert_id, "op", at(3), at(100))
    assert res.changed
    assert res.alert is not None
    assert res.alert.resolved
    assert res.alert.acked_by == "op"
    assert res.alert.rev == 2
    assert m.current() == []
    # a new occurrence gets a new id
    [n] = m.update("CNC-01", AlertKind.MACHINE_FAULT, True, at(10), "CNC-01 is FAULT")
    assert n.alert_id != a.alert_id
    assert n.rev == 0


def test_ack_while_active_then_clear_resolves() -> None:
    m = _mgr()
    [a] = m.update("CNC-01", AlertKind.MACHINE_WARNING, True, at(0), "w")
    res = m.acknowledge(a.alert_id, "op", at(1), at(1))
    assert res.alert is not None
    assert res.alert.active
    assert res.alert.acknowledged
    again = m.acknowledge(a.alert_id, "op2", at(2), at(2))
    assert not again.changed
    assert again.alert == res.alert
    [c] = m.update("CNC-01", AlertKind.MACHINE_WARNING, False, at(3), "")
    assert c.resolved
    assert c.rev == 2
    assert c.acked_by == "op"
    assert m.current() == []
    assert m.acknowledge(a.alert_id, "op", at(4), at(4)).alert is None


def test_reraise_after_clear_before_ack_reuses_the_alert() -> None:
    m = _mgr()
    [a] = m.update("CNC-01", AlertKind.MACHINE_OFFLINE, True, at(0), "offline")
    m.update("CNC-01", AlertKind.MACHINE_OFFLINE, False, at(5), "")
    [r] = m.update("CNC-01", AlertKind.MACHINE_OFFLINE, True, at(9), "offline again")
    assert (r.alert_id, r.active, r.rev, r.message) == (a.alert_id, True, 2, "offline again")


def test_line_stoppage_is_debounced_on_event_time() -> None:
    m = _mgr()
    debounce = ENGINE.alerts.kinds[AlertKind.LINE_STOPPAGE].raise_after_s
    assert debounce > 0
    assert m.update("CNC-01", AlertKind.LINE_STOPPAGE, True, at(0), "s") == []
    assert m.update("CNC-01", AlertKind.LINE_STOPPAGE, False, at(1), "") == []  # blip: reset
    assert m.update("CNC-01", AlertKind.LINE_STOPPAGE, True, at(2), "s") == []
    assert m.update("CNC-01", AlertKind.LINE_STOPPAGE, True, at(2 + debounce - 0.5), "s") == []
    [a] = m.update("CNC-01", AlertKind.LINE_STOPPAGE, True, at(2 + debounce), "s")
    assert a.ts == at(2 + debounce)


def test_disabled_kind_never_raises() -> None:
    kinds = dict(ENGINE.alerts.kinds)
    kinds[AlertKind.MACHINE_WARNING] = kinds[AlertKind.MACHINE_WARNING].model_copy(
        update={"enabled": False}
    )
    m = AlertManager(ENGINE.alerts.model_copy(update={"kinds": kinds}))
    assert m.update("CNC-01", AlertKind.MACHINE_WARNING, True, at(0), "w") == []


def test_alert_contract_requires_consistent_ack_fields() -> None:
    with pytest.raises(ValueError, match="ackedBy and ackedAt"):
        Alert(
            alert_id="a",
            machine_id="M",
            ts=at(0),
            severity=AlertSeverity.INFO,
            kind="k",
            message="",
            source="s",
            acknowledged=True,
        )
    with pytest.raises(ValueError, match="only set on an acknowledged"):
        Alert(
            alert_id="a",
            machine_id="M",
            ts=at(0),
            severity=AlertSeverity.INFO,
            kind="k",
            message="",
            source="s",
            acked_by="x",
        )


async def test_lifecycle_round_trips_through_the_store_and_restores(
    store_factory: Any, tmp_path: Path
) -> None:
    m = _mgr()
    events: list[Alert] = []
    events += m.update("CNC-01", AlertKind.MACHINE_FAULT, True, at(0), "f")
    [w] = m.update("INSP-01", AlertKind.MACHINE_WARNING, True, at(1), "w")
    events.append(w)
    events += m.update("CNC-01", AlertKind.MACHINE_FAULT, False, at(2), "")
    events.append(m.acknowledge(w.alert_id, "alice", at(3), at(50)).alert)  # type: ignore[arg-type]
    [g] = m.update("CNV-01", AlertKind.MACHINE_OFFLINE, True, at(4), "o")
    events.append(g)
    events += m.update("CNV-01", AlertKind.MACHINE_OFFLINE, False, at(5), "")
    events.append(m.acknowledge(g.alert_id, "bob", at(6), at(60)).alert)  # type: ignore[arg-type]

    store = await store_factory(tmp_path)
    try:
        assert await store.insert(Stream.ALERTS, [Stored(e, at(99)) for e in events]) == len(events)
        assert await store.insert(Stream.ALERTS, [Stored(e, at(99)) for e in events]) == 0
        stored = [s.item for s in await store.fetch(Stream.ALERTS)]
    finally:
        await store.close()
    assert sorted(stored, key=lambda a: (a.alert_id, a.rev)) == sorted(
        events, key=lambda a: (a.alert_id, a.rev)
    )
    restored = _mgr()
    assert restored.load(reversed(stored)) == 2  # CNC fault (cleared, unacked) + INSP warning
    assert {(a.machine_id, a.active, a.acknowledged) for a in restored.current()} == {
        ("CNC-01", False, False),
        ("INSP-01", True, True),
    }
    # the restored manager continues the lifecycle: no duplicate raise, clear resolves
    assert restored.update("INSP-01", AlertKind.MACHINE_WARNING, True, at(7), "w") == []
    [c] = restored.update("INSP-01", AlertKind.MACHINE_WARNING, False, at(8), "")
    assert c.resolved
    assert c.rev == 2
