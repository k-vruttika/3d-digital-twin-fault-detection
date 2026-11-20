"""Inference service (Phase 11, ADR 0030) with a fake worker pool: predictions, gate, alarm
rule, alert lifecycle (raise, escalate, clear, unusual behaviour), reliability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from prometheus_client import CollectorRegistry

from twin_backend.bus import InMemoryEventBus
from twin_backend.bus.base import Topic
from twin_backend.clock import FakeClock
from twin_backend.engine.alerts import AlertManager
from twin_backend.engine.data_quality import DataQualityReport
from twin_backend.engine.twin import AlertCondition, TwinEngine
from twin_backend.inference.service import InferenceService
from twin_backend.inference.workers import PoolInfo, WorkerPool
from twin_common.config import load_engine_config, load_line_config
from twin_common.config.engine import AlertKind
from twin_common.config.inference import load_inference_config
from twin_common.contracts import Alert, Prediction
from twin_common.enums import DataQuality, HealthClass, Reliability

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
M = "CNC-01"
NORMAL = [0.9, 0.06, 0.04]
WARN = [0.2, 0.7, 0.1]
FAULT = [0.1, 0.3, 0.6]


class FakePool:
    def __init__(self) -> None:
        self.info = PoolInfo(
            bundles={
                "supervised": {
                    "label": "test-gru v1 (gru seed 0)",
                    "sustainS": 3,
                    "horizons": [10, 30],
                },
                "anomaly": {"label": "test-autoencoder v1 (autoencoder seed 0)"},
            },
            window=30,
        )
        self.closed = 0

    async def start(self) -> PoolInfo:
        return self.info

    async def close(self) -> None:
        self.closed += 1

    alive = True


def out(raw: list[float], cal: list[float] | None = None, flag: bool = False) -> dict[str, Any]:
    cal = cal or raw
    return {
        "probs": {"10": raw, "30": raw},
        "cal": {"10": cal, "30": cal},
        "set": {"10": ["NORMAL"], "30": ["NORMAL", "WARNING"]},
        "ttf": 120.0,
        "ttfInterval": [60.0, 180.0],
        "alarmClass": raw is not NORMAL,
        "anomalyScore": 0.5 if flag else 0.01,
        "anomalyThreshold": 0.2,
        "anomalyFlag": flag,
    }


@pytest.fixture
async def svc() -> Any:
    line = load_line_config(CONFIG / "line.yaml")
    bus = InMemoryEventBus(10_000)
    clock = FakeClock(T0, auto_advance=True)
    engine = TwinEngine(line, load_engine_config(CONFIG / "engine.yaml"), bus, clock)
    cfg = load_inference_config(CONFIG / "inference.yaml")
    pool = cast(WorkerPool, FakePool())
    s: Any = InferenceService(
        cfg, line, bus, engine, clock, ROOT, registry=CollectorRegistry(), pool=pool
    )
    assert await s._start_pool("starting")
    s.preds = bus.subscribe(Topic.PREDICTIONS)
    s.alerts_sub = bus.subscribe(Topic.ALERTS)
    s.seq = 0
    return s


def drift_report(drifted: bool, load_psi: float = 9.2) -> dict[str, Any]:
    return {
        "drifted": drifted,
        "ready": True,
        "psi": {"speed": 0.02, "load": load_psi},
        "alert": {"speed": 0.2, "load": 2.0},
        "adwinTs": {"speed": None, "load": (T0 + timedelta(seconds=5)).timestamp()},
        "since": (T0 + timedelta(seconds=5)).timestamp() if drifted else None,
    }


async def feed(
    s: Any, *entries: tuple[str, dict[str, Any] | None], drift: dict[str, Any] | None = None
) -> list[Alert]:
    rows = []
    for status, o in entries:
        s.seq += 1
        s._ts[M][s.seq] = T0 + timedelta(seconds=s.seq)
        e: dict[str, Any] = {"seq": s.seq, "ts": 0.0, "status": status, "runLen": 30}
        if o is not None:
            e["out"] = o
        if drift is not None:
            e["drift"] = drift
        rows.append(e)
    await s._apply({M: rows})
    await s.engine.flush()
    alerts = []
    while (a := s.alerts_sub.get_nowait()) is not None:
        alerts.append(a)
    return alerts


def drain_predictions(s: Any) -> list[Prediction]:
    out_: list[Prediction] = []
    while (p := s.preds.get_nowait()) is not None:
        out_.append(p)
    return out_


async def test_prediction_events_carry_calibrated_confidence(svc: Any) -> None:
    await feed(svc, ("predict", out(NORMAL, cal=[0.8, 0.15, 0.05])))
    preds = drain_predictions(svc)
    assert [p.horizon_s for p in preds] == [10, 30]
    p30 = preds[1]
    assert p30.calibrated
    assert p30.probs[HealthClass.NORMAL] == pytest.approx(0.8)
    assert p30.conformal_set == [HealthClass.NORMAL, HealthClass.WARNING]
    assert p30.ttf_interval == (60.0, 180.0)
    assert (
        p30.model_version == "test-gru v1 (gru seed 0) + test-autoencoder v1 (autoencoder seed 0)"
    )
    assert p30.alarm is False
    assert preds[0].alarm is None  # the alarm belongs to the primary horizon
    assert p30.attributions == []


def explanation(s: Any, seq: int) -> dict[str, Any]:
    return {
        "machine": M,
        "seq": seq,
        "ts": (T0 + timedelta(seconds=seq)).timestamp(),
        "attributions": [{"sensor": "vib", "part": "spindle_bearing", "weight": 0.8}],
        "computeMs": 400.0,
        "ageMs": 450.0,
        "batch": 1,
        "error": None,
    }


async def test_predicted_warning_after_the_sustain_time_then_escalates_and_clears(svc: Any) -> None:
    assert await feed(svc, ("predict", out(WARN)), ("predict", out(WARN))) == []
    svc._explanations([explanation(svc, 1)])  # explains window 1, delivered later
    raised = await feed(svc, ("predict", out(WARN)))
    assert [(a.kind, a.active) for a in raised] == [("predicted_warning", True)]
    assert raised[0].source == "inference"
    assert "predicted WARNING within 30 s" in raised[0].message
    assert "spindle_bearing" in raised[0].message
    preds = drain_predictions(svc)
    assert preds[-1].alarm is True
    assert preds[-1].attributions[0].part == "spindle_bearing"
    assert preds[-1].attributions_ts == T0 + timedelta(seconds=1)
    assert preds[-1].ts == T0 + timedelta(seconds=3)
    # argmax FAULT for 3 windows: escalate (the alarm itself never drops)
    events = await feed(
        svc, ("predict", out(FAULT)), ("predict", out(FAULT)), ("predict", out(FAULT))
    )
    assert [(a.kind, a.active) for a in events] == [
        ("predicted_warning", False),
        ("predicted_fault", True),
    ]
    events = await feed(svc, ("predict", out(NORMAL)))
    assert [(a.kind, a.active) for a in events] == [("predicted_fault", False)]
    assert drain_predictions(svc)[-1].attributions == []  # no highlight on NORMAL


async def test_unusual_behaviour_only_without_a_supervised_alarm(svc: Any) -> None:
    events = await feed(svc, *[("predict", out(NORMAL, flag=True))] * 3)
    assert [(a.kind, a.active) for a in events] == [("unusual_behaviour", True)]
    assert "cause unknown" in events[0].message
    assert drain_predictions(svc)[-1].attributions == []  # never a part from the autoencoder
    events = await feed(svc, *[("predict", out(WARN, flag=True))] * 3)
    assert ("unusual_behaviour", False) in [(a.kind, a.active) for a in events]
    assert ("predicted_warning", True) in [(a.kind, a.active) for a in events]


async def test_closed_gate_clears_alarms_and_restarts_the_count(svc: Any) -> None:
    await feed(svc, *[("predict", out(WARN))] * 3)
    assert svc.machine_status(M)["alarm"] is True
    events = await feed(svc, ("stopped", None))
    assert [(a.kind, a.active) for a in events] == [("predicted_warning", False)]
    drain_predictions(svc)
    status = svc.machine_status(M)
    assert status["status"] == "stopped"
    # the GET /predict gate view clears with the alerts (it kept alarm: true before)
    assert (status["alarm"], status["predictedFault"], status["unusual"]) == (False, False, False)
    assert (
        await feed(svc, ("warming_up", None), ("predict", out(WARN)), ("predict", out(WARN))) == []
    )
    assert drain_predictions(svc)  # predicted, but the sustain count started again
    events = await feed(svc, ("predict", out(WARN)))
    assert [(a.kind, a.active) for a in events] == [("predicted_warning", True)]


async def test_no_prediction_while_the_gate_is_closed(svc: Any) -> None:
    await feed(svc, ("stopped", None), ("waiting", None), ("warming_up", None))
    assert drain_predictions(svc) == []
    assert svc.machine_status(M)["status"] == "warming_up"


async def test_reliability_degraded_with_data_quality(svc: Any, monkeypatch: Any) -> None:
    report = DataQualityReport(machine_id=M, status=DataQuality.DEGRADED)
    monkeypatch.setattr(svc.engine, "data_quality", lambda m: report)
    await feed(svc, ("predict", out(NORMAL)))
    assert {p.reliability for p in drain_predictions(svc)} == {Reliability.DEGRADED}


async def test_engine_refuses_engine_alert_kinds(svc: Any) -> None:
    with pytest.raises(ValueError, match="engine alert kind"):
        svc.engine.update_conditions([AlertCondition(M, AlertKind.MACHINE_FAULT, True, T0, "x")])


def test_alert_manager_restores_inference_alerts() -> None:
    cfg = load_engine_config(CONFIG / "engine.yaml").alerts
    a = AlertManager(cfg)
    [raised] = a.update(M, AlertKind.UNUSUAL_BEHAVIOUR, True, T0, "unusual")
    assert raised.source == "inference"
    b = AlertManager(cfg)
    assert b.load([raised]) == 1
    assert b.open_alert(M, AlertKind.UNUSUAL_BEHAVIOUR) is not None
    other = raised.model_copy(update={"alert_id": "x", "source": "elsewhere"})
    assert AlertManager(cfg).load([other]) == 0


def test_worker_plans_keep_torch_and_xgboost_apart() -> None:
    cfg = load_inference_config(CONFIG / "inference.yaml")
    [nn] = cfg.with_model("gru").worker_plans()
    assert nn.roles == ("supervised", "monitor")
    xgb, monitor = cfg.with_model("xgb").worker_plans()
    assert xgb.roles == ("supervised",)
    assert xgb.anomaly is None
    assert monitor.roles == ("monitor",)
    assert monitor.supervised is None
    cmd = xgb.command(cfg, ROOT, "python")
    assert cmd[:6] == ["python", "-X", "faulthandler", "-m", "twin_ml.serve", "worker"]
    assert "--anomaly" not in cmd


def test_backend_inference_modules_import_no_ml() -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import twin_backend.inference.service, twin_backend.inference.workers\n"
        "import twin_backend.inference.alarms, twin_backend.api.routes.predict\n"
        "import twin_common.jsonl_process, twin_common.config.inference\n"
        "mods = ('torch', 'xgboost', 'sklearn', 'mlflow', 'twin_ml', 'twin_sim')\n"
        "print([m for m in mods if m in sys.modules])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------- drift (ADR 0031)
async def test_drift_marks_predictions_degraded_and_raises_the_alert(svc: Any) -> None:
    await feed(svc, ("predict", out(NORMAL)), drift=drift_report(False))
    assert {p.reliability for p in drain_predictions(svc)} == {Reliability.OK}
    alerts = await feed(svc, ("predict", out(NORMAL)), drift=drift_report(True))
    drift = [a for a in alerts if a.kind == AlertKind.MODEL_DRIFT.value]
    assert [(a.active, a.source) for a in drift] == [(True, "inference")]
    assert "load PSI 9.20 vs alert level 2.00" in drift[0].message
    assert "DEGRADED" in drift[0].message
    assert "since 00:00:05 UTC" in drift[0].message
    assert {p.reliability for p in drain_predictions(svc)} == {Reliability.DEGRADED}
    # an unchanged report raises nothing again
    assert await feed(svc, ("predict", out(NORMAL)), drift=drift_report(True)) == []
    # the alarm rule is untouched by drift: still the evaluated rule on the same probabilities
    drain_predictions(svc)
    await feed(svc, *[("predict", out(WARN))] * 3, drift=drift_report(True))
    assert svc.gates[M].alarms.alarm is True
    drain_predictions(svc)
    cleared = await feed(svc, ("predict", out(NORMAL)), drift=drift_report(False))
    assert [(a.kind, a.active) for a in cleared if a.kind == "model_drift"] == [
        ("model_drift", False)
    ]
    assert {p.reliability for p in drain_predictions(svc)} == {Reliability.OK}


async def test_drift_survives_a_closed_gate_but_not_a_lost_link(svc: Any) -> None:
    await feed(svc, ("predict", out(NORMAL)), drift=drift_report(True))
    assert svc.gates[M].drifted
    await feed(svc, ("stopped", None))  # a stop does not end drift: the conditions come back
    assert svc.gates[M].drifted
    await svc._close_all_gates()  # workers lost: nothing is known any more
    assert not svc.gates[M].drifted


async def test_a_worker_without_a_drift_reference_changes_nothing(svc: Any) -> None:
    alerts = await feed(svc, ("predict", out(NORMAL)))
    assert [a for a in alerts if a.kind == "model_drift"] == []
    assert {p.reliability for p in drain_predictions(svc)} == {Reliability.OK}
