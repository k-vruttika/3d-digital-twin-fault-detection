"""Simulator configuration: physics/fault parameters (``config/simulator.yaml``) and scenarios."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from twin_common.config.line import LineConfig
from twin_common.config.loader import ConfigError, load_model
from twin_common.enums import CommandType, MachineKind, SensorName


class FaultType(StrEnum):
    BEARING_WEAR = "bearing_wear"
    OVERHEATING = "overheating"
    SPINDLE_IMBALANCE = "spindle_imbalance"
    BELT_JAM = "belt_jam"
    LUBRICATION_LOSS = "lubrication_loss"


GRADUAL_FAULTS: tuple[FaultType, ...] = (
    FaultType.BEARING_WEAR,
    FaultType.OVERHEATING,
    FaultType.SPINDLE_IMBALANCE,
    FaultType.LUBRICATION_LOSS,
)

# Label used when a machine trips without an injected fault (e.g. SET_LOAD into overload).
OVERLOAD_LABEL = "overload"
NO_FAULT_LABEL = "none"


class SensorFaultType(StrEnum):
    STUCK = "stuck"
    DRIFT = "drift"
    DROPOUT = "dropout"


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Range = tuple[float, float]


def _check_range(r: Range, name: str, lo: float = 0.0) -> None:
    if not lo <= r[0] <= r[1]:
        raise ValueError(f"{name} must satisfy {lo} <= lo <= hi (got {list(r)})")


# --------------------------------------------------------------------------- physics


class KindThresholds(ConfigModel):
    vib_warning: float = Field(gt=0)
    vib_fault: float = Field(gt=0)
    temp_warning: float
    temp_fault: float
    current_warning_x: float = Field(gt=0)
    current_fault_x: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> KindThresholds:
        for name in ("vib", "temp", "current"):
            w = getattr(self, f"{name}_warning" + ("_x" if name == "current" else ""))
            f = getattr(self, f"{name}_fault" + ("_x" if name == "current" else ""))
            if w >= f:
                raise ValueError(f"{name}: warning ({w}) must be below fault ({f})")
        return self


class KindPhysics(ConfigModel):
    speed_set_rpm: float = Field(gt=0)
    speed_max_rpm: float = Field(gt=0)
    # Setpoint range scenarios may use (job-to-job speed changes); the feasibility validator
    # checks both ends. Defaults to the nominal setpoint only.
    speed_operating_rpm: Range | None = None
    slip_at_full_load: float = Field(ge=0, lt=1)
    overload_speed_drop: float = Field(ge=0)
    speed_noise_frac: float = Field(ge=0)
    heat_capacity_j_per_k: float = Field(gt=0)
    cooling_w_per_k: float = Field(gt=0)
    heat_noload_w: float = Field(ge=0)
    heat_load_w: float = Field(ge=0)
    heat_friction_w: float = Field(ge=0)
    current_noload_a: float = Field(ge=0)
    current_load_a: float = Field(gt=0)
    current_friction_a: float = Field(ge=0)
    vib_base_mm_s: float = Field(gt=0)
    vib_load_coef: float = Field(ge=0)
    cycle_time_s: float = Field(gt=0)
    thresholds: KindThresholds

    @property
    def current_rated_a(self) -> float:
        return self.current_noload_a + self.current_load_a

    @model_validator(mode="after")
    def _speeds(self) -> KindPhysics:
        if self.speed_set_rpm > self.speed_max_rpm:
            raise ValueError("speed_set_rpm must not exceed speed_max_rpm")
        lo, hi = self.operating_speeds
        if not 0 < lo <= self.speed_set_rpm <= hi <= self.speed_max_rpm:
            raise ValueError(
                "speed_operating_rpm must satisfy 0 < lo <= speed_set_rpm <= hi <= speed_max_rpm"
            )
        return self

    @property
    def operating_speeds(self) -> Range:
        if self.speed_operating_rpm is None:
            return (self.speed_set_rpm, self.speed_set_rpm)
        return self.speed_operating_rpm


class MeasurementNoise(ConfigModel):
    temp_sigma_c: float = Field(ge=0)
    vib_mult: float = Field(ge=0)
    vib_add_mm_s: float = Field(ge=0)
    current_mult: float = Field(ge=0)
    current_add_a: float = Field(ge=0)
    load_sigma_pct: float = Field(ge=0)


class FaultSpec(ConfigModel):
    kinds: list[MachineKind] = Field(min_length=1)
    held_out: bool = False


class GradualFaultSpec(FaultSpec):
    load_exponent: float = Field(ge=0)


class BearingWearSpec(GradualFaultSpec):
    vib_amp_x_fault: float = Field(gt=0)
    gamma: float = Field(gt=0)
    friction_coef: float = Field(ge=0)


class OverheatingSpec(GradualFaultSpec):
    cooling_loss_rate: float = Field(ge=0)
    cooling_loss_max: float = Field(ge=0, lt=1)
    fault_heat_dt_k: float = Field(ge=0)


class ImbalanceSpec(GradualFaultSpec):
    vib_amp_x_fault: float = Field(gt=0)


class LubricationSpec(GradualFaultSpec):
    friction_exponent: float = Field(gt=0)
    current_x_rated: float = Field(gt=0)
    speed_sag: float = Field(ge=0, lt=1)
    speed_jitter_gain: float = Field(ge=0)
    burst_prob_gain: float = Field(ge=0)
    burst_amp_x_fault: float = Field(ge=0)
    burst_amp_range: Range

    @model_validator(mode="after")
    def _r(self) -> LubricationSpec:
        _check_range(self.burst_amp_range, "burst_amp_range")
        return self


class JamSpec(FaultSpec):
    speed_tau_s: float = Field(gt=0)
    stall_current_x_rated: float = Field(gt=0)
    trip_after_s: float = Field(ge=0)


class FaultParams(ConfigModel):
    progression_s: Range
    progress_noise_sigma: float = Field(ge=0)
    load_ref_pct: float = Field(gt=0)
    load_factor_clip: Range
    bearing_wear: BearingWearSpec
    overheating: OverheatingSpec
    spindle_imbalance: ImbalanceSpec
    lubrication_loss: LubricationSpec
    belt_jam: JamSpec

    @model_validator(mode="after")
    def _r(self) -> FaultParams:
        _check_range(self.progression_s, "progression_s", lo=1.0)
        _check_range(self.load_factor_clip, "load_factor_clip")
        if self.load_factor_clip[0] <= 0:
            raise ValueError("load_factor_clip lower bound must be > 0")
        return self

    def spec(self, fault: FaultType) -> FaultSpec:
        spec: FaultSpec = getattr(self, fault.value)
        return spec

    def applicable(self, fault: FaultType, kind: MachineKind) -> bool:
        return kind in self.spec(fault).kinds

    def held_out_types(self) -> list[FaultType]:
        return [f for f in FaultType if self.spec(f).held_out]


class OperationParams(ConfigModel):
    hysteresis_frac: float = Field(ge=0, lt=1)
    starved_after_s: float = Field(ge=0)
    blocked_after_s: float = Field(ge=0)
    idle_load_pct: float = Field(ge=0)
    load_lag_tau_s: float = Field(gt=0)
    max_load_pct: float = Field(gt=0)
    maintenance_duration_s: float = Field(gt=0)
    max_substep_s: float = Field(gt=0, le=1.0)
    initial_buffer_level: int = Field(ge=0)


class SensorFaultParams(ConfigModel):
    duration_s: Range
    drift_rate_frac_per_hour: float
    sensors: list[SensorName] = Field(min_length=1)

    @model_validator(mode="after")
    def _r(self) -> SensorFaultParams:
        _check_range(self.duration_s, "duration_s", lo=1.0)
        return self


class LabelParams(ConfigModel):
    ttf_cap_s: float = Field(gt=0)


class OperatingCase(ConfigModel):
    load_pct: float = Field(gt=0)
    ambient_c: float


class ValidationParams(ConfigModel):
    min_lead_s: float = Field(ge=0)
    reach_max_progress: float = Field(gt=0)
    best_case: OperatingCase
    worst_case: OperatingCase
    threshold_perturbation_pct: float = Field(ge=0, lt=100)


class SimConfig(ConfigModel):
    kinds: dict[MachineKind, KindPhysics]
    measurement: MeasurementNoise
    faults: FaultParams
    operation: OperationParams
    sensor_faults: SensorFaultParams
    labels: LabelParams
    validation: ValidationParams

    @model_validator(mode="after")
    def _all_kinds(self) -> SimConfig:
        missing = sorted(set(MachineKind) - set(self.kinds))
        if missing:
            raise ValueError(f"kinds: missing parameters for {missing}")
        return self


# --------------------------------------------------------------------------- scenarios


class ScheduledFault(ConfigModel):
    type: FaultType
    machine: str = Field(min_length=1)
    onset_s: float = Field(ge=0)
    progression_s: float | None = Field(default=None, gt=0)


class RandomFaults(ConfigModel):
    rate_per_machine_per_hour: float = Field(default=0.0, ge=0)
    types: list[FaultType] = Field(default_factory=list)
    include_held_out: bool = False
    earliest_onset_s: float = Field(default=120.0, ge=0)


class FaultSchedule(ConfigModel):
    random: RandomFaults = RandomFaults()
    scheduled: list[ScheduledFault] = Field(default_factory=list)


class ScheduledSensorFault(ConfigModel):
    type: SensorFaultType
    machine: str = Field(min_length=1)
    sensor: SensorName | None = None  # required for stuck/drift, ignored for dropout
    onset_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _sensor(self) -> ScheduledSensorFault:
        if self.type is not SensorFaultType.DROPOUT and self.sensor is None:
            raise ValueError(f"sensor fault '{self.type.value}' requires 'sensor'")
        return self


class RandomSensorFaults(ConfigModel):
    rate_per_machine_per_hour: float = Field(default=0.0, ge=0)
    types: list[SensorFaultType] = Field(default_factory=lambda: list(SensorFaultType))
    earliest_onset_s: float = Field(default=60.0, ge=0)


class SensorFaultSchedule(ConfigModel):
    random: RandomSensorFaults = RandomSensorFaults()
    scheduled: list[ScheduledSensorFault] = Field(default_factory=list)


class LoadProcess(ConfigModel):
    tau_s: float = Field(default=300.0, gt=0)
    sigma_frac: float = Field(default=0.15, ge=0)


class SourceParams(ConfigModel):
    arrival_interval_s: float = Field(default=6.5, gt=0)
    jitter_frac: float = Field(default=0.1, ge=0, lt=1)


class RecoveryParams(ConfigModel):
    auto: bool = True
    jam_clear_s: Range = (30.0, 120.0)
    repair_delay_s: Range = (120.0, 600.0)
    maintenance_duration_s: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def _r(self) -> RecoveryParams:
        _check_range(self.jam_clear_s, "jam_clear_s")
        _check_range(self.repair_delay_s, "repair_delay_s")
        return self


class ScriptedCommand(ConfigModel):
    at_s: float = Field(ge=0)
    machine: str = Field(min_length=1)
    type: CommandType
    params: dict[str, float] = Field(default_factory=dict)


class SpeedChangeSchedule(ConfigModel):
    """Job-to-job setpoint changes: every U(interval_s) a SET_SPEED to U(speed_rpm)."""

    machine: str = Field(min_length=1)
    interval_s: Range
    speed_rpm: Range

    @model_validator(mode="after")
    def _r(self) -> SpeedChangeSchedule:
        _check_range(self.interval_s, "interval_s", lo=1.0)
        _check_range(self.speed_rpm, "speed_rpm", lo=1.0)
        return self


LoadRangeName = Annotated[str, Field(pattern="^(low|medium|high)$")]


class ScenarioConfig(ConfigModel):
    name: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    duration_s: float = Field(default=3600.0, gt=0)
    start_ts: AwareDatetime = datetime(2026, 1, 1, tzinfo=UTC)
    load_ranges: list[LoadRangeName] = Field(default_factory=lambda: ["low", "medium", "high"])
    load_process: LoadProcess = LoadProcess()
    ambient_c: Range = (18.0, 28.0)
    threshold_perturbation_pct: float = Field(default=10.0, ge=0, lt=100)
    source: SourceParams = SourceParams()
    faults: FaultSchedule = FaultSchedule()
    sensor_faults: SensorFaultSchedule = SensorFaultSchedule()
    recovery: RecoveryParams = RecoveryParams()
    commands: list[ScriptedCommand] = Field(default_factory=list)
    speed_changes: list[SpeedChangeSchedule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _r(self) -> ScenarioConfig:
        if not self.load_ranges:
            raise ValueError("load_ranges must not be empty")
        if self.ambient_c[0] > self.ambient_c[1]:
            raise ValueError("ambient_c must satisfy lo <= hi")
        return self


# --------------------------------------------------------------------------- loaders


def load_sim_config(path: str | Path, *, check_feasibility: bool = True) -> SimConfig:
    """Load ``simulator.yaml``; by default also run the fault-feasibility validator."""
    cfg = load_model(path, SimConfig)
    if check_feasibility:
        from twin_sim.feasibility import check_fault_feasibility

        problems = check_fault_feasibility(cfg)
        if problems:
            raise ConfigError(
                f"{path}: fault parameters are infeasible ({len(problems)} problem(s))\n"
                + "\n".join(f"  - {p}" for p in problems)
            )
    return cfg


def load_scenario(path: str | Path) -> ScenarioConfig:
    return load_model(path, ScenarioConfig)


def check_scenario_compat(scenario: ScenarioConfig, sim: SimConfig, line: LineConfig) -> None:
    """Cross-checks between a scenario, the sim config and the line. Raises ConfigError."""
    errors: list[str] = []
    if scenario.threshold_perturbation_pct > sim.validation.threshold_perturbation_pct:
        errors.append(
            f"threshold_perturbation_pct {scenario.threshold_perturbation_pct} exceeds the "
            f"{sim.validation.threshold_perturbation_pct} % the fault validator guarantees"
        )
    known = {m.id for m in line.machines}
    refs = [("faults.scheduled", f.machine) for f in scenario.faults.scheduled]
    refs += [("sensor_faults.scheduled", f.machine) for f in scenario.sensor_faults.scheduled]
    refs += [("commands", c.machine) for c in scenario.commands]
    refs += [("speed_changes", s.machine) for s in scenario.speed_changes]
    for where, mid in refs:
        if mid not in known:
            errors.append(f"{where}: unknown machine '{mid}' (not on line '{line.line_id}')")
    for sc in scenario.speed_changes:
        if sc.machine not in known:
            continue
        kp = sim.kinds[line.machine(sc.machine).kind]
        lo, hi = kp.operating_speeds
        if not lo <= sc.speed_rpm[0] <= sc.speed_rpm[1] <= hi:
            errors.append(
                f"speed_changes ({sc.machine}): speed_rpm {list(sc.speed_rpm)} outside the "
                f"validated operating range [{lo}, {hi}] RPM (simulator.yaml)"
            )
        slowest_cycle = kp.cycle_time_s * kp.speed_set_rpm / sc.speed_rpm[0]
        if slowest_cycle >= scenario.source.arrival_interval_s:
            errors.append(
                f"speed_changes ({sc.machine}): at {sc.speed_rpm[0]} RPM the cycle time "
                f"{slowest_cycle:.2f} s is not below the arrival interval "
                f"{scenario.source.arrival_interval_s} s; the line would back up"
            )
    if errors:
        raise ConfigError(f"scenario '{scenario.name}':\n" + "\n".join(f"  - {e}" for e in errors))
