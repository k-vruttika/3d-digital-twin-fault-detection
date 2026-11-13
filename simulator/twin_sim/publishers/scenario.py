"""Scenario injection for the publishers (simulation only, ADR 0015).

Real equipment would not offer this: it exists so a demo or a test can make a fault happen now
(``INJECT_FAULT``) or play a scenario file's events from now on (``LOAD_SCENARIO``). The mock
server, the MQTT publisher and the OPC UA server all call :func:`apply_scenario` at a tick
boundary through ``SimSession.submit_scenario``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from twin_common.config.loader import ConfigError
from twin_common.contracts import ScenarioAck, ScenarioRequest
from twin_common.enums import AckStatus, ScenarioAction
from twin_sim.config import FaultType, check_scenario_compat, load_scenario
from twin_sim.line import LineSimulator
from twin_sim.scenario import make_run_plan


def _ack(
    req: ScenarioRequest, status: AckStatus, reason: str | None, ts: datetime | None = None
) -> ScenarioAck:
    return ScenarioAck(
        request_id=req.request_id,
        status=status,
        reason=reason,
        ts=ts if ts is not None else datetime.now(UTC),
    )


def apply_scenario(sim: LineSimulator, req: ScenarioRequest, scenario_dir: Path) -> ScenarioAck:
    """Apply ``req`` now; the ack carries simulation time, like the simulator's command acks."""
    now = sim.timestamp(sim.t)
    if req.action is ScenarioAction.INJECT_FAULT:
        # both set: guaranteed by the contract validator
        assert req.machine_id is not None
        assert req.fault_type is not None
        try:
            fault = FaultType(req.fault_type)
        except ValueError:
            known = ", ".join(f.value for f in FaultType)
            return _ack(req, AckStatus.REJECTED, f"unknown fault '{req.fault_type}' ({known})", now)
        status, reason = sim.inject_fault(req.machine_id, fault, req.progression_s)
        return _ack(req, status, reason, now)
    assert req.scenario is not None
    path = scenario_dir / f"{req.scenario}.yaml"
    if not path.is_file():
        return _ack(req, AckStatus.REJECTED, f"no scenario '{req.scenario}' in {scenario_dir}", now)
    try:
        scenario = load_scenario(path)
        check_scenario_compat(scenario, sim.cfg, sim.line)
        plan = make_run_plan(
            scenario, sim.cfg, sim.line, req.seed if req.seed is not None else scenario.seed, 0
        )
    except (ConfigError, ValueError) as exc:
        reason = f"scenario '{req.scenario}' cannot run here: {exc}"
        return _ack(req, AckStatus.REJECTED, reason, now)
    n = sim.schedule_plan(plan)
    return _ack(req, AckStatus.EXECUTED, f"{n} events scheduled from now", now)


def timeout_ack(req: ScenarioRequest) -> ScenarioAck:
    return _ack(
        req, AckStatus.TIMEOUT, "the simulator is not running or did not apply the request in time"
    )
