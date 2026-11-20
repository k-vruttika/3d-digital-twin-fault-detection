"""Phase 11 Part C: the four demo scenarios as end-to-end tests (ADR 0031, docs/demo_script.md).

Each scenario runs against the real thing: the mock machine server (the simulator, **in real
time**), the backend in dev mode (TimescaleDB, a throw-away database per scenario) with live
inference on the deployed bundles of ``config/inference.yaml``, and the REST API as an operator
would use it. Nothing is faked; the assertions are on what the running system reports.

* **(a) Healthy line + what-if:** no alarms of any kind, predictions NORMAL with reliability OK;
  a what-if "stop CNC-01" projects the stoppage and starving without sending anything.
* **(b) Unseen fault (lubrication_loss):** the supervised model is blind; the autoencoder raises
  "unusual behaviour, cause unknown", with no part and no predicted alarm.
* **(c) Known fault (bearing_wear), the ordering check:** the alerts come in the order
  unusual behaviour -> predicted WARNING -> the simulator's own WARNING, the predicted alarm
  precedes the simulator's WARNING, and the explanation names the spindle bearing.
* **(d) Operating-condition drift:** load 120 % (above anything trained) raises ``model_drift``,
  marks the predictions DEGRADED, and both clear once the load is back in range.

Real time is the point: explanations are delayed under CPU load and the alarm needs its sustain
time (ADR 0030). A scenario takes 3-15 minutes. Needs the Docker Compose services and the deployed
bundles (``uv run poe up``, ``uv run poe deploy-models``); skipped without the bundles.
``LINETWIN_DEMO_RESULTS_DIR`` makes the scenarios write their measured facts there (JSON), which
is how ``experiments/results/demo_scenarios.json`` is generated.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest

from twin_backend.e2e_stack import (
    PREDICT_PROGRESSION_S,
    PREDICT_SIM_SPEED,
    Stack,
    bundles_missing,
    dev_stack,
    throwaway_db,
)
from twin_backend.testing import PgService

pytestmark = [pytest.mark.e2e, pytest.mark.services]

MACHINE = "CNC-01"
BEARING = "spindle_bearing"
INFERENCE_KINDS = ("predicted_warning", "predicted_fault", "unusual_behaviour", "model_drift")
POLL_S = 2.0
WARM_TIMEOUT_S = (
    240.0  # inference ready is awaited by the stack; this is the 30 s window + a margin
)
FAULT_TIMEOUT_S = 1200.0
DRIFT_TIMEOUT_S = 600.0
HEALTHY_WATCH_S = 120.0
WHATIF_HORIZON_S = 60


class Api:
    def __init__(self, stack: Stack, key: str) -> None:
        self.http = httpx.Client(base_url=stack.api_url, headers={"X-API-Key": key}, timeout=60)

    def get(self, path: str, **params: Any) -> Any:
        r = self.http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return self.http.post(path, json=body)

    def machines(self) -> dict[str, dict[str, Any]]:
        return {m["machineId"]: m for m in self.get("/machines")["machines"]}

    def predict(self, machine: str = MACHINE) -> dict[str, Any]:
        m: dict[str, Any] = self.get("/predict")["machines"].get(machine, {})
        return m

    def primary(self, machine: str = MACHINE) -> dict[str, Any] | None:
        preds = self.predict(machine).get("predictions", {})
        p: dict[str, Any] | None = preds.get("30")
        return p

    def alert_events(self, t0: datetime, machine: str = MACHINE) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = self.get(
            "/alerts/history", **{"from": t0.isoformat(), "machine": machine}
        )["alerts"]
        return [e for e in events if e["kind"] in INFERENCE_KINDS]


def when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def raised(events: list[dict[str, Any]], kind: str) -> datetime | None:
    ts = [when(e["ts"]) for e in events if e["kind"] == kind and e["active"]]
    return min(ts) if ts else None


def cleared(events: list[dict[str, Any]], kind: str) -> datetime | None:
    ts = [when(e["ts"]) for e in events if e["kind"] == kind and not e["active"]]
    return max(ts) if ts else None


def wait_until(what: str, fn: Callable[[], Any], timeout_s: float, poll_s: float = POLL_S) -> Any:
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(poll_s)
    pytest.fail(f"timed out after {timeout_s:.0f} s waiting for {what} (last: {last!r})")


def write_facts(name: str, tmp_path: Path, facts: dict[str, Any]) -> None:
    text = json.dumps(facts, indent=2, default=str) + "\n"
    (tmp_path / f"{name}.json").write_text(text, encoding="utf-8")
    out = os.environ.get("LINETWIN_DEMO_RESULTS_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{name}.json").write_text(text, encoding="utf-8")
    print(f"\n{name}: {text}")


@contextmanager
def demo(tmp_path: Path) -> Iterator[tuple[Api, Stack]]:
    if (missing := bundles_missing()) is not None:
        pytest.skip(missing)
    key = secrets.token_hex(16)
    with throwaway_db() as dsn:
        if dsn is None:
            pytest.skip("TimescaleDB not reachable; start services with: uv run poe up")
        with dev_stack(tmp_path, dsn, key, inference=True, sim_speed=PREDICT_SIM_SPEED) as stack:
            yield Api(stack, key), stack


def warm(api: Api) -> datetime:
    """Every machine RUNNING and predicted (past its warm-up); returns the event time now."""

    def ready() -> bool:
        states = api.machines()
        if not all(m["state"] == "RUNNING" for m in states.values()):
            return False
        gates = api.get("/predict")["machines"]
        return bool(gates) and all(g["gate"]["status"] == "predict" for g in gates.values())

    wait_until("a running, predicted line", ready, WARM_TIMEOUT_S)
    return when(api.machines()[MACHINE]["lastReading"]["ts"])


def inject(api: Api, fault: str) -> datetime:
    r = api.post(
        "/scenario",
        {
            "action": "INJECT_FAULT",
            "machineId": MACHINE,
            "faultType": fault,
            "progressionS": PREDICT_PROGRESSION_S,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["acks"][0]["status"] == "EXECUTED", r.text
    return when(r.json()["request"]["ts"])


def command(api: Api, ctype: str, **params: float) -> dict[str, Any]:
    r = api.post(
        "/commands",
        {
            "machineId": MACHINE,
            "type": ctype,
            "params": params,
            "idempotencyKey": secrets.token_hex(8),
        },
    )
    assert r.status_code == 200, r.text
    ack: dict[str, Any] = r.json()
    assert ack["status"] == "EXECUTED", ack
    return ack


# ------------------------------------------------------------------------------------------- (a)
def test_demo_a_healthy_line_and_what_if(tmp_path: Path, timescale_test_db: PgService) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        time.sleep(HEALTHY_WATCH_S)  # a healthy line, watched in real time
        facts: dict[str, Any] = {"scenario": "a_healthy_line", "watched_s": HEALTHY_WATCH_S}
        for m in ("CNV-01", MACHINE, "INSP-01", "CNV-02"):
            events = api.alert_events(t0, m)
            assert events == [], f"{m}: unexpected inference alerts on a healthy line: {events}"
            p = api.primary(m)
            assert p is not None, f"{m} has no prediction"
            assert p["reliability"] == "OK", p
            assert p["alarm"] is False, p
            top = max(p["probs"], key=lambda c: p["probs"][c])
            assert top == "NORMAL", p
            assert not p.get("attributions"), "NORMAL predictions have no attributions"
        facts["alerts"] = 0
        cnc = api.primary() or {}
        facts["primary_cnc"] = {k: cnc.get(k) for k in ("probs", "conformalSet", "reliability")}

        before = api.get("/commands")["commands"]
        r = api.post(
            "/whatif",
            {"machineId": MACHINE, "type": "STOP", "horizonS": WHATIF_HORIZON_S},
        )
        assert r.status_code == 200, r.text
        w = r.json()
        assert w["model"], "scored by the deployed model"
        base, scen = w["baseline"]["machines"], w["scenario"]["machines"]
        assert base[MACHINE]["ghost"]["end"] == "RUNNING"
        assert scen[MACHINE]["ghost"]["end"] == "IDLE"
        assert scen["INSP-01"]["ghost"]["end"] in ("STARVED", "IDLE")
        assert base["INSP-01"]["ghost"]["end"] == "RUNNING"
        stopped = [p for p in scen[MACHINE]["points"] if p["state"] == "IDLE"]
        assert stopped
        assert all(p["gate"] != "predict" for p in stopped[1:]), "no prediction while stopped"
        # running a what-if changes nothing on the real line
        assert api.get("/commands")["commands"] == before
        assert api.machines()[MACHINE]["state"] == "RUNNING"
        facts["what_if"] = {
            "elapsed_ms": w["elapsedMs"],
            "ghost_cnc": scen[MACHINE]["ghost"],
            "ghost_insp": scen["INSP-01"]["ghost"],
            "model": w["model"],
        }
        write_facts("demo_a_healthy_line", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (b)
def test_demo_b_unseen_fault_is_unusual_behaviour_only(
    tmp_path: Path, timescale_test_db: PgService
) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        injected = inject(api, "lubrication_loss")  # held out: never seen in training
        deadline = time.monotonic() + FAULT_TIMEOUT_S

        def unusual_on() -> list[dict[str, Any]] | None:
            ev = api.alert_events(t0)
            return ev if raised(ev, "unusual_behaviour") else None

        events = wait_until("unusual_behaviour", unusual_on, deadline - time.monotonic())
        at_alarm = api.primary()
        assert at_alarm is not None
        top = max(at_alarm["probs"], key=lambda c: at_alarm["probs"][c])
        assert top == "NORMAL", (
            f"the supervised model should be blind to an unseen fault: {at_alarm}"
        )
        assert at_alarm["alarm"] is False
        assert not at_alarm.get("attributions"), "unusual behaviour never names a part"
        assert api.machines()[MACHINE]["state"] in ("RUNNING", "WARNING")
        unusual = next(e for e in events if e["kind"] == "unusual_behaviour" and e["active"])
        assert "cause unknown" in unusual["message"]
        assert BEARING not in unusual["message"]
        assert raised(events, "predicted_warning") is None, (
            "no supervised alarm for an unseen fault"
        )
        assert raised(events, "predicted_fault") is None
        # watch a while longer in real time: how often does it switch on and off (flapping)?
        time.sleep(180)
        events = api.alert_events(t0)
        switches = sorted(
            (when(e["ts"]), e["active"]) for e in events if e["kind"] == "unusual_behaviour"
        )
        flips = sum(1 for (_, a), (_, b) in pairwise(switches) if a != b)
        facts = {
            "scenario": "b_unseen_fault_lubrication_loss",
            "injected_at": injected,
            "unusual_raised_at": raised(events, "unusual_behaviour"),
            "unusual_after_injection_s": (
                raised(events, "unusual_behaviour") - injected  # type: ignore[operator]
            ).total_seconds(),
            "unusual_events": len(switches),
            "on_off_switches": flips,
            "supervised_alarm_raised": raised(events, "predicted_warning") is not None,
            "supervised_top_class_at_alarm": top,
            "probs_at_alarm": at_alarm["probs"],
        }
        write_facts("demo_b_unseen_fault", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (c)
def test_demo_c_known_fault_alert_order(tmp_path: Path, timescale_test_db: PgService) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        injected = inject(api, "bearing_wear")
        state_warning: dict[str, datetime] = {}
        top_part: dict[str, Any] = {}

        def observe() -> bool:
            if "ts" not in state_warning:
                m = api.machines()[MACHINE]
                if m["state"] == "WARNING":
                    state_warning["ts"] = when(m["stateTs"])
            p = api.primary()
            if (
                p is not None
                and p.get("alarm")
                and p.get("attributions")
                and "part" not in top_part
            ):
                att = sorted(p["attributions"], key=lambda a: -a["weight"])
                top_part["part"] = att[0]["part"]
                top_part["attributions"] = att[:3]
                top_part["attributionsTs"] = p.get("attributionsTs")
            ev = api.alert_events(t0)
            return (
                raised(ev, "predicted_warning") is not None
                and "ts" in state_warning
                and "part" in top_part
            )

        wait_until(
            "the predicted warning, the simulator WARNING and an explanation",
            observe,
            FAULT_TIMEOUT_S,
        )
        events = api.alert_events(t0)
        unusual_at = raised(events, "unusual_behaviour")
        predicted_at = raised(events, "predicted_warning")
        warning_at = state_warning["ts"]
        assert predicted_at is not None
        # the ordering check: injection < unusual behaviour <= predicted WARNING < simulator WARNING
        assert injected < predicted_at, "the alarm follows the injection"
        assert predicted_at < warning_at, (
            "the predicted warning must come before the simulator's own WARNING "
            f"(predicted {predicted_at}, simulator {warning_at})"
        )
        assert unusual_at is not None, "the autoencoder should have flagged the build-up"
        assert unusual_at <= predicted_at, (
            f"unusual behaviour {unusual_at} should not come after the predicted warning "
            f"{predicted_at}"
        )
        assert top_part["part"] == BEARING, top_part
        facts = {
            "scenario": "c_known_fault_bearing_wear",
            "injected_at": injected,
            "unusual_after_injection_s": (unusual_at - injected).total_seconds(),
            "predicted_warning_after_injection_s": (predicted_at - injected).total_seconds(),
            "simulator_warning_after_injection_s": (warning_at - injected).total_seconds(),
            "predicted_lead_over_simulator_warning_s": (warning_at - predicted_at).total_seconds(),
            "unusual_lead_over_predicted_s": (predicted_at - unusual_at).total_seconds(),
            "top_part_at_alarm": top_part["part"],
            "attributions_at_alarm": top_part["attributions"],
        }
        write_facts("demo_c_known_fault", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (d)
def test_demo_d_load_drift_degrades_and_clears(
    tmp_path: Path, timescale_test_db: PgService
) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        assert api.primary()["reliability"] == "OK"  # type: ignore[index]
        info = api.get("/predict")["inference"]["anomaly"]["drift"]
        assert info["available"], "the autoencoder bundle must carry the drift reference (v2)"
        sent_at = when(api.machines()[MACHINE]["lastReading"]["ts"])
        command(api, "SET_LOAD", load=120.0)  # above every trained load (support ends near 96 %)

        def drifting() -> list[dict[str, Any]] | None:
            ev = api.alert_events(t0)
            return ev if raised(ev, "model_drift") else None

        events = wait_until("model_drift", drifting, DRIFT_TIMEOUT_S, poll_s=5.0)
        drift_alert = next(e for e in events if e["kind"] == "model_drift" and e["active"])
        assert "load" in drift_alert["message"]
        assert "DEGRADED" in drift_alert["message"]
        raised_at = raised(events, "model_drift")
        assert raised_at is not None
        assert raised_at > sent_at

        def degraded() -> bool:
            p = api.primary()
            return p is not None and p["reliability"] == "DEGRADED"

        wait_until("DEGRADED reliability", degraded, 60)
        alarm_free = [e for e in events if e["kind"] in ("predicted_warning", "predicted_fault")]
        facts: dict[str, Any] = {
            "scenario": "d_load_drift",
            "load_set_at": sent_at,
            "drift_raised_after_load_change_s": (raised_at - sent_at).total_seconds(),
            "drift_message": drift_alert["message"],
            "predicted_alarms_during_drift": len(alarm_free),
        }
        back_at = when(api.machines()[MACHINE]["lastReading"]["ts"])
        command(api, "SET_LOAD", load=55.0)  # back inside the trained range

        def clear() -> datetime | None:
            return cleared(api.alert_events(t0), "model_drift")

        cleared_at = wait_until("drift to clear", clear, DRIFT_TIMEOUT_S, poll_s=5.0)
        assert cleared_at > back_at
        wait_until("OK reliability", lambda: not degraded(), 60)
        facts["drift_cleared_after_load_back_s"] = (cleared_at - back_at).total_seconds()
        write_facts("demo_d_load_drift", tmp_path, facts)


def test_demo_results_timestamps_are_event_time() -> None:
    """The scenarios compare event times (alert ts, state ts), never wall-clock arrival."""
    assert when("2026-01-01T00:00:05Z") == datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    assert when("2026-01-01T00:00:05Z") - timedelta(seconds=5) == datetime(2026, 1, 1, tzinfo=UTC)
