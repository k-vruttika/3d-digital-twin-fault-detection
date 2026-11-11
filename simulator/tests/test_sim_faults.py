"""Fault behaviour: healthy margin, reachability, minimum lead time, jam, sensor faults."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from twin_common.config.line import LineConfig
from twin_common.enums import CommandType, MachineState, SensorName
from twin_sim.config import (
    GRADUAL_FAULTS,
    FaultSchedule,
    FaultType,
    RandomFaults,
    RecoveryParams,
    ScenarioConfig,
    ScheduledFault,
    ScheduledSensorFault,
    ScriptedCommand,
    SensorFaultSchedule,
    SensorFaultType,
    SimConfig,
    load_scenario,
)
from twin_sim.line import LineSimulator, StepFrame
from twin_sim.physics import fault_load_exponent, load_factor
from twin_sim.scenario import make_run_plan

SimFactory = Callable[..., LineSimulator]
ONSET = 300.0
BLOCK_S = 60
PRIMARY = {
    FaultType.BEARING_WEAR: "true_vib",
    FaultType.SPINDLE_IMBALANCE: "true_vib",
    FaultType.OVERHEATING: "true_temp",
    FaultType.LUBRICATION_LOSS: "true_current",
}


def _cases(line_cfg: LineConfig, sim_cfg: SimConfig) -> list[tuple[FaultType, str, float]]:
    """(fault, machine, speed setpoint): both ends of each machine's operating speed range."""
    out: list[tuple[FaultType, str, float]] = []
    for f in GRADUAL_FAULTS:
        for m in line_cfg.machines:
            if sim_cfg.faults.applicable(f, m.kind) and m.id != "CNV-02":
                for sp in sorted(set(sim_cfg.kinds[m.kind].operating_speeds)):
                    out.append((f, m.id, sp))
    return out


def _set_speed(machine: str, speed: float) -> list[ScriptedCommand]:
    cmd = ScriptedCommand(
        at_s=0.0, machine=machine, type=CommandType.SET_SPEED, params={"speed": speed}
    )
    return [cmd]


def _states(frames: list[StepFrame], i: int) -> list[MachineState]:
    return [f.samples[i].state for f in frames]


def _first(states: list[MachineState], target: MachineState) -> int | None:
    return next((k for k, s in enumerate(states) if s is target), None)


def _scenario(
    fault: FaultType,
    machine: str,
    load: str,
    ambient: float,
    d: float,
    dur: float,
    speed: float,
) -> ScenarioConfig:
    return ScenarioConfig(
        name="t",
        commands=_set_speed(machine, speed),
        duration_s=dur,
        load_ranges=[load],
        ambient_c=(ambient, ambient),
        recovery=RecoveryParams(auto=False),
        faults=FaultSchedule(
            scheduled=[ScheduledFault(type=fault, machine=machine, onset_s=ONSET, progression_s=d)]
        ),
    )


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("cnc_speed", [7500.0, 10000.0])
def test_healthy_line_never_warns_at_worst_case(
    make_sim: SimFactory, seed: int, cnc_speed: float
) -> None:
    sc = ScenarioConfig(
        name="t",
        duration_s=1800,
        load_ranges=["high"],
        ambient_c=(28.0, 28.0),
        commands=_set_speed("CNC-01", cnc_speed),
    )
    frames = make_sim(sc, seed=seed, thr_scale=0.9).run()
    bad = {s.state for f in frames for s in f.samples} & {MachineState.WARNING, MachineState.FAULT}
    assert not bad


def test_every_gradual_fault_reachable_at_best_case(
    make_sim: SimFactory, line_cfg: LineConfig, sim_cfg: SimConfig
) -> None:
    """Low load, cold ambient, thresholds +10 %, D = D_max: FAULT within progression time."""
    d_max = sim_cfg.faults.progression_s[1]
    for fault, mid, sp in _cases(line_cfg, sim_cfg):
        g = load_factor(
            line_cfg.load_ranges.low.min, fault_load_exponent(fault, sim_cfg.faults), sim_cfg.faults
        )
        budget = d_max / g
        for seed in (11, 12):
            sim = make_sim(
                _scenario(fault, mid, "low", 18.0, d_max, ONSET + budget + 60, sp), seed, 0, 1.1
            )
            states = _states(sim.run(), sim.ids.index(mid))
            k = _first(states, MachineState.FAULT)
            assert k is not None, f"{fault.value} on {mid} at {sp} never reached FAULT"
            assert k + 1 - ONSET <= budget, f"{fault.value} on {mid} at {sp}: {k + 1 - ONSET}"


def test_every_gradual_fault_has_min_lead_time_and_rising_precursor(
    make_sim: SimFactory, line_cfg: LineConfig, sim_cfg: SimConfig
) -> None:
    """High load, hot ambient, thresholds -10 %, D = D_min (worst case for lead time)."""
    d_min = sim_cfg.faults.progression_s[0]
    min_lead = sim_cfg.validation.min_lead_s
    for fault, mid, sp in _cases(line_cfg, sim_cfg):
        for seed in (21, 22, 23):
            sim = make_sim(_scenario(fault, mid, "high", 28.0, d_min, 1500, sp), seed, 0, 0.9)
            frames = sim.run()
            i = sim.ids.index(mid)
            states = _states(frames, i)
            k_fault = _first(states, MachineState.FAULT)
            k_warn = _first(states, MachineState.WARNING)
            assert k_fault is not None
            assert k_warn is not None
            assert k_warn < k_fault
            t_fault = frames[k_fault].t_s
            assert t_fault - ONSET >= min_lead, (
                f"{fault.value} on {mid} at {sp}: lead {t_fault - ONSET}"
            )
            # Primary precursor: 60 s block means over the lead window rise overall and never
            # fall by more than 2 % of the total rise.
            start = k_fault - int(min_lead)
            sig = np.array([getattr(f.samples[i], PRIMARY[fault]) for f in frames[start:k_fault]])
            blocks = sig[: len(sig) // BLOCK_S * BLOCK_S].reshape(-1, BLOCK_S).mean(axis=1)
            rise = blocks[-1] - blocks[0]
            assert rise > 0
            assert np.all(np.diff(blocks) >= -0.02 * rise), f"{fault.value} on {mid}: {blocks}"


def test_belt_jam_speed_drops_and_current_spikes(make_sim: SimFactory, sim_cfg: SimConfig) -> None:
    sc = ScenarioConfig(
        name="t",
        duration_s=400,
        load_ranges=["medium"],
        recovery=RecoveryParams(auto=False),
        faults=FaultSchedule(
            scheduled=[ScheduledFault(type=FaultType.BELT_JAM, machine="CNV-01", onset_s=200)]
        ),
    )
    sim = make_sim(sc)
    frames = sim.run()
    rated = sim.kp[0].current_rated_a
    before = frames[198].samples[0]
    after = [frames[k].samples[0] for k in range(200, 204)]  # frame k is t = k + 1
    assert before.state is MachineState.RUNNING
    assert before.speed > 1000
    assert all(s.state is MachineState.FAULT for s in after)
    assert after[0].fault_type == "belt_jam"
    assert max(s.current for s in after[:2]) > 2.5 * rated
    assert after[1].speed < 0.05 * before.speed
    assert frames[260].samples[0].current == pytest.approx(0.0, abs=0.2)
    assert sim_cfg.faults.belt_jam.trip_after_s <= 2.0


def test_sensor_faults_change_observations_not_states(make_sim: SimFactory) -> None:
    base = ScenarioConfig(
        name="t",
        duration_s=900,
        load_ranges=["medium"],
        faults=FaultSchedule(
            scheduled=[ScheduledFault(type=FaultType.BEARING_WEAR, machine="INSP-01", onset_s=100)]
        ),
    )
    faulty = base.model_copy(
        update={
            "sensor_faults": SensorFaultSchedule(
                scheduled=[
                    ScheduledSensorFault(
                        type=SensorFaultType.STUCK,
                        machine="INSP-01",
                        sensor=SensorName.TEMP,
                        onset_s=100,
                        duration_s=200,
                    ),
                    ScheduledSensorFault(
                        type=SensorFaultType.DRIFT,
                        machine="INSP-01",
                        sensor=SensorName.VIB,
                        onset_s=400,
                        duration_s=200,
                    ),
                    ScheduledSensorFault(
                        type=SensorFaultType.DROPOUT, machine="INSP-01", onset_s=700, duration_s=50
                    ),
                ]
            )
        }
    )
    clean, dirty = make_sim(base, seed=5), make_sim(faulty, seed=5)
    fc, fd = clean.run(), dirty.run()
    i = clean.ids.index("INSP-01")
    assert [f.samples[i].state for f in fc] == [f.samples[i].state for f in fd]
    assert [f.samples[i].true_vib for f in fc] == [f.samples[i].true_vib for f in fd]
    stuck = {fd[k].samples[i].temp for k in range(110, 290)}
    assert len(stuck) == 1
    assert fd[200].samples[i].sensor_fault == "stuck:temp"
    # Drift: 10 %/h of the 0-50 mm/s range, 191 s after onset at t = 400 (frame 590 is t = 591).
    expected = 50.0 * 0.1 / 3600.0 * (591 - 400)
    drift_gap = fd[590].samples[i].vib - fc[590].samples[i].vib
    assert drift_gap == pytest.approx(expected, abs=1e-9)
    assert np.isnan(fd[720].samples[i].temp)
    assert fd[720].samples[i].dropout
    assert fd[720].samples[i].sensor_fault == "dropout"
    assert len(dirty.readings(fd[720])) == 3  # dropout: INSP-01 emits no Reading
    assert fd[850].samples[i].sensor_fault == "none"


def test_held_out_fault_excluded_by_default(line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    random = RandomFaults(
        rate_per_machine_per_hour=6.0, types=list(FaultType), include_held_out=False
    )
    sc = ScenarioConfig(name="t", faults=FaultSchedule(random=random))
    types = {
        e.type for i in range(30) for e in make_run_plan(sc, sim_cfg, line_cfg, 9, i).fault_events
    }
    assert FaultType.LUBRICATION_LOSS not in types
    assert FaultType.BEARING_WEAR in types

    sc_in = sc.model_copy(
        update={
            "faults": FaultSchedule(random=random.model_copy(update={"include_held_out": True}))
        }
    )
    types_in = {
        e.type
        for i in range(30)
        for e in make_run_plan(sc_in, sim_cfg, line_cfg, 9, i).fault_events
    }
    assert FaultType.LUBRICATION_LOSS in types_in


def test_spindle_imbalance_only_on_cnc(line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    sc = ScenarioConfig(
        name="t",
        faults=FaultSchedule(
            random=RandomFaults(rate_per_machine_per_hour=6.0, types=[FaultType.SPINDLE_IMBALANCE])
        ),
    )
    plan = make_run_plan(sc, sim_cfg, line_cfg, 1, 0)
    assert plan.fault_events
    assert {e.machine for e in plan.fault_events} == {"CNC-01"}


def test_spindle_speed_changes_are_planned_and_applied(
    make_sim: SimFactory, line_cfg: LineConfig, sim_cfg: SimConfig, config_dir: Path
) -> None:
    sc = load_scenario(config_dir / "scenarios" / "default.yaml")
    plan = make_run_plan(sc, sim_cfg, line_cfg, 42, 0)
    changes = [c for c in plan.commands if c.type is CommandType.SET_SPEED]
    assert len(changes) >= 3
    gaps = np.diff([0.0] + [c.at_s for c in changes])
    assert np.all((gaps >= 300.0) & (gaps <= 900.0))
    assert all(c.machine == "CNC-01" and 7500 <= c.params["speed"] <= 10000 for c in changes)
    # Independent stream: adding speed changes does not alter the rest of the plan.
    plain = make_run_plan(sc.model_copy(update={"speed_changes": []}), sim_cfg, line_cfg, 42, 0)
    assert plain.fault_events == plan.fault_events
    assert plain.machines == plan.machines

    no_faults = sc.model_copy(update={"faults": FaultSchedule(), "duration_s": 2400})
    sim = make_sim(no_faults, seed=42)
    frames = sim.run()
    i = sim.ids.index("CNC-01")
    first = next(c for c in sim.plan.commands if c.type is CommandType.SET_SPEED)
    k = round(first.at_s) + 30
    slip = sim.kp[i].slip_at_full_load * frames[k].samples[i].load / 100.0
    assert frames[k].samples[i].speed == pytest.approx(first.params["speed"] * (1 - slip), rel=0.02)
    assert all(s.state is MachineState.RUNNING for f in frames for s in f.samples)
