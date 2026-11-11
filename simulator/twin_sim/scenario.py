"""RunPlan: the concrete realisation of a scenario for one seeded run."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from twin_common.config.line import LineConfig
from twin_common.enums import CommandType, SensorName
from twin_sim.config import (
    FaultType,
    LoadProcess,
    RecoveryParams,
    ScenarioConfig,
    SensorFaultType,
    SimConfig,
    SourceParams,
)
from twin_sim.rng import make_streams
from twin_sim.thresholds import Thresholds, sample_thresholds

_SECONDS_PER_HOUR = 3600.0


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FaultEvent(PlanModel):
    type: FaultType
    machine: str
    onset_s: float
    progression_s: float | None = None  # None for sudden faults


class SensorFaultEvent(PlanModel):
    type: SensorFaultType
    machine: str
    sensor: SensorName | None
    onset_s: float
    duration_s: float
    drift_rate_per_s: float = 0.0


class PlanCommand(PlanModel):
    at_s: float
    machine: str
    type: CommandType
    params: dict[str, float] = Field(default_factory=dict)


class MachinePlan(PlanModel):
    machine_id: str
    load_mu: float
    thresholds: Thresholds


class RunPlan(PlanModel):
    run_id: str
    run_index: int
    base_seed: int
    scenario_name: str
    duration_s: float
    start_ts: AwareDatetime
    load_range: str
    load_band: tuple[float, float]
    ambient_c: float
    load_process: LoadProcess
    source: SourceParams
    recovery: RecoveryParams
    machines: list[MachinePlan]
    fault_events: list[FaultEvent]
    sensor_fault_events: list[SensorFaultEvent]
    commands: list[PlanCommand]
    eligible_fault_types: list[FaultType]

    def machine_plan(self, machine_id: str) -> MachinePlan:
        for m in self.machines:
            if m.machine_id == machine_id:
                return m
        raise KeyError(machine_id)


def _poisson_times(
    rng: np.random.Generator, rate_per_hour: float, start: float, end: float
) -> list[float]:
    if rate_per_hour <= 0 or start >= end:
        return []
    mean_gap = _SECONDS_PER_HOUR / rate_per_hour
    times: list[float] = []
    t = start
    while True:
        t += float(rng.exponential(mean_gap))
        if t >= end:
            return times
        times.append(round(t, 3))


def eligible_fault_types(scenario: ScenarioConfig, sim: SimConfig) -> list[FaultType]:
    """Random-sampling fault types: held-out types only with ``include_held_out``."""
    rf = scenario.faults.random
    return [f for f in rf.types if rf.include_held_out or not sim.faults.spec(f).held_out]


def make_run_plan(
    scenario: ScenarioConfig,
    sim: SimConfig,
    line: LineConfig,
    base_seed: int,
    run_index: int,
) -> RunPlan:
    streams = make_streams(base_seed, run_index)
    rng, srng = streams["plan"], streams["sensor_plan"]
    duration = scenario.duration_s

    range_name = scenario.load_ranges[int(rng.integers(len(scenario.load_ranges)))]
    band_cfg = getattr(line.load_ranges, range_name)
    band = (band_cfg.min, band_cfg.max)
    ambient = float(rng.uniform(*scenario.ambient_c))

    machines = [
        MachinePlan(
            machine_id=m.id,
            load_mu=float(rng.uniform(*band)),
            thresholds=sample_thresholds(
                sim.kinds[m.kind], scenario.threshold_perturbation_pct, rng
            ),
        )
        for m in line.machines
    ]

    d_lo, d_hi = sim.faults.progression_s
    eligible = eligible_fault_types(scenario, sim)
    events: list[FaultEvent] = []

    def progression(fault: FaultType) -> float | None:
        if fault is FaultType.BELT_JAM:
            return None
        return round(float(rng.uniform(d_lo, d_hi)), 3)

    for f in scenario.faults.scheduled:
        events.append(
            FaultEvent(
                type=f.type,
                machine=f.machine,
                onset_s=f.onset_s,
                progression_s=f.progression_s if f.progression_s else progression(f.type),
            )
        )
    rf = scenario.faults.random
    for m in line.machines:
        types = [f for f in eligible if sim.faults.applicable(f, m.kind)]
        if not types:
            continue
        for t in _poisson_times(rng, rf.rate_per_machine_per_hour, rf.earliest_onset_s, duration):
            fault = types[int(rng.integers(len(types)))]
            events.append(
                FaultEvent(type=fault, machine=m.id, onset_s=t, progression_s=progression(fault))
            )
    events.sort(key=lambda e: (e.onset_s, e.machine))

    sensor_events = _plan_sensor_faults(scenario, sim, line, srng, duration)
    commands = [
        PlanCommand(at_s=c.at_s, machine=c.machine, type=c.type, params=c.params)
        for c in scenario.commands
    ]
    commands += _plan_speed_changes(scenario, streams["speed_plan"], duration)
    commands.sort(key=lambda c: c.at_s)
    return RunPlan(
        run_id=f"run_{run_index:04d}",
        run_index=run_index,
        base_seed=base_seed,
        scenario_name=scenario.name,
        duration_s=duration,
        start_ts=scenario.start_ts,
        load_range=range_name,
        load_band=band,
        ambient_c=ambient,
        load_process=scenario.load_process,
        source=scenario.source,
        recovery=scenario.recovery,
        machines=machines,
        fault_events=events,
        sensor_fault_events=sensor_events,
        commands=commands,
        eligible_fault_types=eligible,
    )


def _plan_speed_changes(
    scenario: ScenarioConfig, rng: np.random.Generator, duration: float
) -> list[PlanCommand]:
    """Job-to-job spindle speed changes, issued as SET_SPEED commands."""
    out: list[PlanCommand] = []
    for sc in scenario.speed_changes:
        t = 0.0
        while True:
            t += round(float(rng.uniform(*sc.interval_s)), 3)
            if t >= duration:
                break
            speed = round(float(rng.uniform(*sc.speed_rpm)))
            out.append(
                PlanCommand(
                    at_s=t,
                    machine=sc.machine,
                    type=CommandType.SET_SPEED,
                    params={"speed": float(speed)},
                )
            )
    return out


def _plan_sensor_faults(
    scenario: ScenarioConfig,
    sim: SimConfig,
    line: LineConfig,
    rng: np.random.Generator,
    duration: float,
) -> list[SensorFaultEvent]:
    sfp = sim.sensor_faults

    def drift_rate(sensor: SensorName | None) -> float:
        if sensor is None:
            return 0.0
        spec = line.sensors[sensor.value]
        return (spec.max - spec.min) * sfp.drift_rate_frac_per_hour / _SECONDS_PER_HOUR

    out: list[SensorFaultEvent] = []
    for s in scenario.sensor_faults.scheduled:
        out.append(
            SensorFaultEvent(
                type=s.type,
                machine=s.machine,
                sensor=None if s.type is SensorFaultType.DROPOUT else s.sensor,
                onset_s=s.onset_s,
                duration_s=s.duration_s,
                drift_rate_per_s=drift_rate(s.sensor) if s.type is SensorFaultType.DRIFT else 0.0,
            )
        )
    rs = scenario.sensor_faults.random
    if rs.types:
        for m in line.machines:
            for t in _poisson_times(
                rng, rs.rate_per_machine_per_hour, rs.earliest_onset_s, duration
            ):
                kind = rs.types[int(rng.integers(len(rs.types)))]
                sensor: SensorName | None = None
                if kind is not SensorFaultType.DROPOUT:
                    sensor = sfp.sensors[int(rng.integers(len(sfp.sensors)))]
                out.append(
                    SensorFaultEvent(
                        type=kind,
                        machine=m.id,
                        sensor=sensor,
                        onset_s=t,
                        duration_s=round(float(rng.uniform(*sfp.duration_s)), 3),
                        drift_rate_per_s=drift_rate(sensor)
                        if kind is SensorFaultType.DRIFT
                        else 0.0,
                    )
                )
    out.sort(key=lambda e: (e.onset_s, e.machine))
    return out


def plan_summary(plan: RunPlan) -> dict[str, Any]:
    """Compact JSON-able description for the manifest."""
    return plan.model_dump(mode="json")
