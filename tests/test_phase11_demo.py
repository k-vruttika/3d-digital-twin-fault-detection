"""Phase 11 Part C: the demo scenarios as end-to-end tests (ADR 0031, docs/demo_script.md).

Each scenario runs against the real thing: the mock machine server (the simulator, **in real
time**), the backend in dev mode (TimescaleDB, a throw-away database per scenario) with live
inference on the deployed bundles of ``config/inference.yaml``, and the REST API as an operator
would use it. Nothing is faked; the assertions are on what the running system reports.

The labels follow the Part C plan (C3):

* **(a) Known fault (bearing_wear):** the predicted WARNING comes before the simulator's own
  WARNING and the explanation names the spindle bearing (with the alert order unusual behaviour
  <= predicted WARNING < simulator WARNING).
* **(b) Unseen fault (lubrication_loss):** the supervised model is blind; the autoencoder raises
  "unusual behaviour, cause unknown", with no part and no predicted alarm.
* **(c) Operating-condition drift:** load 120 % (above anything trained) raises ``model_drift``
  and marks the predictions DEGRADED, **before** any overload-related simulator WARNING or
  FAULT (the ordering check); both clear once the load is back in range.
* **(d) What-if during (a):** while the bearing_wear alarm is on, a what-if with a lower load is
  compared with the unchanged projection, then applied as a real command.
* **(e) Healthy line + stop what-if** (beyond the plan): no alarms of any kind, predictions
  NORMAL with reliability OK; a what-if "stop CNC-01" projects the stoppage and starving.

Real time is the point: explanations are delayed under CPU load and the alarm needs its sustain
time (ADR 0030). A scenario takes 3-15 minutes, so the scenarios are **opt-in** (marker ``demo``,
``--run-demo``; ``uv run poe demo-tests``). They need the Docker Compose services and the deployed
bundles (``uv run poe up``, ``uv run poe deploy-models``) and skip with a message without them.
``LINETWIN_DEMO_RESULTS_DIR`` (set by the poe task) makes them write their measured facts there
(JSON), which is how ``experiments/results/demo_scenarios/`` is generated.

One fast test is not opt-in: the simulator-only check that the drift step of (c) stays below
every simulator limit of CNC-01 (``test_demo_c_load_step_stays_below_the_simulator_limits``).
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
from statistics import fmean
from typing import Any

import httpx
import pytest

from twin_backend.e2e_stack import (
    DEV_SCENARIO,
    PREDICT_PROGRESSION_S,
    PREDICT_SIM_SPEED,
    Stack,
    bundles_missing,
    dev_stack,
    throwaway_db,
)
from twin_common.config.line import LineConfig
from twin_common.contracts import Command
from twin_common.enums import CommandType
from twin_sim.config import SimConfig, load_scenario
from twin_sim.publishers.session import build_simulator

MACHINE = "CNC-01"
BEARING = "spindle_bearing"
INFERENCE_KINDS = ("predicted_warning", "predicted_fault", "unusual_behaviour", "model_drift")
FAULT_STATES = ("WARNING", "FAULT")  # simulator states a fault or an overload would cause
POLL_S = 2.0
WARM_TIMEOUT_S = (
    240.0  # inference ready is awaited by the stack; this is the 30 s window + a margin
)
FAULT_TIMEOUT_S = 1200.0
DRIFT_TIMEOUT_S = 600.0
HEALTHY_WATCH_S = 120.0
WHATIF_HORIZON_S = 60
# (c): above every trained load (support ends near 96 %), at the simulator's max_load_pct.
DRIFT_LOAD_PCT = 120.0
BACK_LOAD_PCT = 55.0  # inside the trained range
# (d): the lower load tried during the bearing fault, inside the trained range (no drift) with
# the load band (about +-12 % around it) above the trained minimum of 30 %.
WHATIF_LOW_LOAD_PCT = 40.0
WHATIF_FAULT_HORIZON_S = 300  # whatif.yaml horizon_max_s
LOAD_SETTLE_S = 15  # SET_LOAD lag (load_lag_tau_s 3 s) plus a margin before comparing loads


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

    def now(self) -> datetime:
        """Event time of the machine's latest reading."""
        return when(self.machines()[MACHINE]["lastReading"]["ts"])

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

    def readings(self, t0: datetime, t1: datetime, machine: str = MACHINE) -> list[dict[str, Any]]:
        """Stored raw readings of one machine in [t0, t1), oldest first."""
        rows: list[dict[str, Any]] = self.get(
            "/history",
            **{"from": t0.isoformat(), "to": t1.isoformat(), "resolution": "raw"},
            machines=machine,
        )["readings"]
        return sorted(rows, key=lambda r: when(r["ts"]))


def when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def raised(events: list[dict[str, Any]], kind: str) -> datetime | None:
    ts = [when(e["ts"]) for e in events if e["kind"] == kind and e["active"]]
    return min(ts) if ts else None


def cleared(events: list[dict[str, Any]], kind: str) -> datetime | None:
    ts = [when(e["ts"]) for e in events if e["kind"] == kind and not e["active"]]
    return max(ts) if ts else None


def first_in_state(rows: list[dict[str, Any]], states: tuple[str, ...]) -> dict[str, Any] | None:
    return next((r for r in rows if r["state"] in states), None)


def mean_load(rows: list[dict[str, Any]]) -> float | None:
    loads = [float(r["load"]) for r in rows if r.get("load") is not None]
    return fmean(loads) if loads else None


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
    return api.now()


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


def wait_for_predicted_warning(
    api: Api, t0: datetime
) -> tuple[datetime, dict[str, Any], list[dict[str, Any]]]:
    """Wait for the predicted WARNING alert and an explained alarm: (raised, top, events)."""
    top: dict[str, Any] = {}

    def observe() -> bool:
        p = api.primary()
        if p is not None and p.get("alarm") and p.get("attributions") and "part" not in top:
            att = sorted(p["attributions"], key=lambda a: -a["weight"])
            top["part"] = att[0]["part"]
            top["attributions"] = att[:3]
            top["attributionsTs"] = p.get("attributionsTs")
        return raised(api.alert_events(t0), "predicted_warning") is not None and "part" in top

    wait_until("the predicted warning and its explanation", observe, FAULT_TIMEOUT_S)
    events = api.alert_events(t0)
    at = raised(events, "predicted_warning")
    assert at is not None
    return at, top, events


# ------------------------------------------------------------------------------------------- (a)
@pytest.mark.usefixtures("timescale_test_db")
@pytest.mark.demo
@pytest.mark.e2e
@pytest.mark.services
def test_demo_a_known_fault_warns_before_the_simulator(tmp_path: Path) -> None:
    with demo(tmp_path) as (api, _stack):
        t0 = warm(api)
        injected = inject(api, "bearing_wear")
        state_warning: dict[str, datetime] = {}

        def simulator_warning() -> bool:
            m = api.machines()[MACHINE]
            if m["state"] in FAULT_STATES and "ts" not in state_warning:
                state_warning["ts"] = when(m["stateTs"])
            return "ts" in state_warning

        predicted_at, top_part, _events = wait_for_predicted_warning(api, t0)
        wait_until("the simulator's WARNING", simulator_warning, FAULT_TIMEOUT_S)
        events = api.alert_events(t0)
        unusual_at = raised(events, "unusual_behaviour")
        # the stored readings date the simulator's WARNING to the second (event time)
        rows = api.readings(injected, api.now() + timedelta(seconds=1))
        first = first_in_state(rows, FAULT_STATES)
        assert first is not None
        warning_at = when(first["ts"])
        # injection < unusual behaviour <= predicted WARNING < simulator WARNING
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
            "scenario": "a_known_fault_bearing_wear",
            "injected_at": injected,
            "unusual_after_injection_s": (unusual_at - injected).total_seconds(),
            "predicted_warning_after_injection_s": (predicted_at - injected).total_seconds(),
            "simulator_warning_after_injection_s": (warning_at - injected).total_seconds(),
            "simulator_first_state": first["state"],
            "predicted_lead_over_simulator_warning_s": (warning_at - predicted_at).total_seconds(),
            "unusual_lead_over_predicted_s": (predicted_at - unusual_at).total_seconds(),
            "top_part_at_alarm": top_part["part"],
            "attributions_at_alarm": top_part["attributions"],
        }
        write_facts("demo_a_known_fault", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (b)
@pytest.mark.usefixtures("timescale_test_db")
@pytest.mark.demo
@pytest.mark.e2e
@pytest.mark.services
def test_demo_b_unseen_fault_is_unusual_behaviour_only(tmp_path: Path) -> None:
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
def test_demo_c_load_step_stays_below_the_simulator_limits(
    line_cfg: LineConfig, sim_cfg: SimConfig
) -> None:
    """The physics side of the ordering check in (c), without a model or a server.

    The dev-stack simulator (same line, physics and seeded scenario as the mock server in the
    demos) held at the drift load for longer than (c) holds it must never reach WARNING or
    FAULT on CNC-01; otherwise the step would cause an overload fault the drift has to beat.
    """
    sim = build_simulator(line_cfg, sim_cfg, load_scenario(DEV_SCENARIO))
    sim.fast_forward(30.0)
    ack = sim.apply_command(
        Command(
            command_id="demo-c",
            idempotency_key="demo-c",
            machine_id=MACHINE,
            type=CommandType.SET_LOAD,
            params={"load": DRIFT_LOAD_PCT},
            issued_by="test",
            ts=datetime.now(UTC),
        )
    )
    assert ack.status.value == "EXECUTED", ack
    m = sim.machine(MACHINE)
    hold_s = int(2 * DRIFT_TIMEOUT_S + 60)  # longer than (c) can keep the load high
    states = set()
    for _ in range(hold_s):
        sim.step(1.0)
        states.add(m.state.value)
    assert not states & set(FAULT_STATES), (
        f"load {DRIFT_LOAD_PCT} % drives {MACHINE} into {states}; pick a load step below the "
        "simulator's limits (docs/demo_script.md, scenario (c))"
    )


@pytest.mark.usefixtures("timescale_test_db")
@pytest.mark.demo
@pytest.mark.e2e
@pytest.mark.services
def test_demo_c_load_drift_comes_before_any_overload_fault(tmp_path: Path) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        assert api.primary()["reliability"] == "OK"  # type: ignore[index]
        info = api.get("/predict")["inference"]["anomaly"]["drift"]
        assert info["available"], "the autoencoder bundle must carry the drift reference (v2)"
        sent_at = api.now()
        command(api, "SET_LOAD", load=DRIFT_LOAD_PCT)

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
        degraded_seen_at = api.now()
        alarm_free = [e for e in events if e["kind"] in ("predicted_warning", "predicted_fault")]
        back_at = api.now()
        command(api, "SET_LOAD", load=BACK_LOAD_PCT)

        def clear() -> datetime | None:
            return cleared(api.alert_events(t0), "model_drift")

        cleared_at = wait_until("drift to clear", clear, DRIFT_TIMEOUT_S, poll_s=5.0)
        assert cleared_at > back_at
        wait_until("OK reliability", lambda: not degraded(), 60)

        # The ordering check: the drift flag (alert, banner, DEGRADED) comes before any
        # overload-related simulator WARNING or FAULT, judged on the stored readings (event time).
        high = api.readings(sent_at, back_at + timedelta(seconds=1))
        assert high, "no stored readings while the load was high"
        overload = first_in_state(high, FAULT_STATES)
        overload_at = when(overload["ts"]) if overload is not None else None
        assert overload_at is None or overload_at > raised_at, (
            f"an overload-related {overload['state'] if overload else ''} at {overload_at} came "
            f"before the drift flag at {raised_at}"
        )
        facts: dict[str, Any] = {
            "scenario": "c_load_drift",
            "load_step_pct": DRIFT_LOAD_PCT,
            "load_set_at": sent_at,
            "drift_raised_after_load_change_s": (raised_at - sent_at).total_seconds(),
            "degraded_seen_after_load_change_s": (degraded_seen_at - sent_at).total_seconds(),
            "drift_message": drift_alert["message"],
            "predicted_alarms_during_drift": len(alarm_free),
            "high_load_held_s": (back_at - sent_at).total_seconds(),
            "high_load_max_temp_c": max(r["temp"] for r in high if r.get("temp") is not None),
            "high_load_max_current_a": max(
                r["current"] for r in high if r.get("current") is not None
            ),
            "first_overload_related_event": (
                None if overload is None else {"state": overload["state"], "ts": overload_at}
            ),
            "drift_cleared_after_load_back_s": (cleared_at - back_at).total_seconds(),
        }
        write_facts("demo_c_load_drift", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (d)
@pytest.mark.usefixtures("timescale_test_db")
@pytest.mark.demo
@pytest.mark.e2e
@pytest.mark.services
def test_demo_d_what_if_lower_load_during_the_bearing_fault(tmp_path: Path) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        injected = inject(api, "bearing_wear")
        predicted_at, _top, _events = wait_for_predicted_warning(api, t0)
        state_at_whatif = api.machines()[MACHINE]["state"]

        # what-if: lower the load now, compared with the unchanged projection
        r = api.post(
            "/whatif",
            {
                "machineId": MACHINE,
                "type": "SET_LOAD",
                "params": {"load": WHATIF_LOW_LOAD_PCT},
                "horizonS": WHATIF_FAULT_HORIZON_S,
            },
        )
        assert r.status_code == 200, r.text
        w = r.json()
        assert w["model"], "scored by the deployed model"
        base, scen = w["baseline"]["machines"][MACHINE], w["scenario"]["machines"][MACHINE]
        settled = [p for p in base["points"] if p["tS"] > LOAD_SETTLE_S]
        base_load = mean_load(settled)
        scen_load = mean_load([p for p in scen["points"] if p["tS"] > LOAD_SETTLE_S])
        assert base_load is not None
        assert scen_load is not None
        assert scen_load < base_load, "the change lowers the projected load"
        assert any(p["gate"] == "predict" for p in scen["points"]), "the projection is scored"

        def later_or_never(changed: int | None, unchanged: int | None) -> bool:
            # bearing wear progresses with load^1.5 and vibration rises with load, so a lower
            # load can delay a simulator WARNING / FAULT but must not bring it forward
            return unchanged is None or changed is None or changed >= unchanged

        assert later_or_never(scen["firstWarningS"], base["firstWarningS"]), (base, scen)
        assert later_or_never(scen["firstFaultS"], base["firstFaultS"]), (base, scen)

        # "apply as command": the same change through the normal command path (applyBody)
        before = api.get("/commands")["commands"]
        assert api.machines()[MACHINE]["state"] not in ("FAULT", "MAINTENANCE"), (
            "the machine faulted before the change could be applied"
        )
        applied_at = api.now()
        ack = command(api, w["type"], **w["params"])
        after = api.get("/commands")["commands"]
        assert len(after) == len(before) + 1, "the applied change is in the command audit"
        fork_at = when(w["asOf"])
        end_at = fork_at + timedelta(seconds=WHATIF_FAULT_HORIZON_S)

        # watch the live line until the end of the projection, then compare
        wait_until(
            "the end of the projection on the live line",
            lambda: api.now() >= end_at,
            WHATIF_FAULT_HORIZON_S + 120.0,
            poll_s=5.0,
        )
        live = api.readings(applied_at, end_at)
        live_load = mean_load(
            [r for r in live if when(r["ts"]) > applied_at + timedelta(seconds=LOAD_SETTLE_S)]
        )
        assert live_load is not None
        assert live_load < base_load, (
            f"the applied command lowered the live load ({live_load} vs {base_load})"
        )
        live_warning = first_in_state(live, ("WARNING",))
        live_fault = first_in_state(live, ("FAULT",))

        def after_fork(row: dict[str, Any] | None) -> float | None:
            return None if row is None else (when(row["ts"]) - fork_at).total_seconds()

        facts = {
            "scenario": "d_whatif_lower_load_during_bearing_wear",
            "injected_at": injected,
            "predicted_warning_after_injection_s": (predicted_at - injected).total_seconds(),
            "state_at_whatif": state_at_whatif,
            "whatif": {
                "fork_after_injection_s": (fork_at - injected).total_seconds(),
                "change": {"type": w["type"], "params": w["params"]},
                "horizon_s": w["horizonS"],
                "elapsed_ms": w["elapsedMs"],
                "model": w["model"],
            },
            "projection": {
                name: {
                    "mean_load_pct": mean_load([p for p in v["points"] if p["tS"] > LOAD_SETTLE_S]),
                    "first_warning_s": v["firstWarningS"],
                    "first_fault_s": v["firstFaultS"],
                    "first_alarm_s": v["firstAlarmS"],
                    "max_p_fault": v["maxPFault"],
                    "end_state": v["ghost"]["end"],
                }
                for name, v in (("baseline", base), ("lower_load", scen))
            },
            "applied": {
                "after_fork_s": (applied_at - fork_at).total_seconds(),
                "ack": ack["status"],
                "live_mean_load_pct": live_load,
                "live_first_warning_after_fork_s": after_fork(live_warning),
                "live_first_fault_after_fork_s": after_fork(live_fault),
                "live_state_at_projection_end": live[-1]["state"] if live else None,
            },
        }
        write_facts("demo_d_whatif_during_fault", tmp_path, facts)


# ------------------------------------------------------------------------------------------- (e)
@pytest.mark.usefixtures("timescale_test_db")
@pytest.mark.demo
@pytest.mark.e2e
@pytest.mark.services
def test_demo_e_healthy_line_and_stop_what_if(tmp_path: Path) -> None:
    with demo(tmp_path) as (api, _):
        t0 = warm(api)
        time.sleep(HEALTHY_WATCH_S)  # a healthy line, watched in real time
        facts: dict[str, Any] = {"scenario": "e_healthy_line", "watched_s": HEALTHY_WATCH_S}
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
        write_facts("demo_e_healthy_line", tmp_path, facts)


def test_demo_results_timestamps_are_event_time() -> None:
    """The scenarios compare event times (alert ts, state ts), never wall-clock arrival."""
    assert when("2026-01-01T00:00:05Z") == datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    assert when("2026-01-01T00:00:05Z") - timedelta(seconds=5) == datetime(2026, 1, 1, tzinfo=UTC)
