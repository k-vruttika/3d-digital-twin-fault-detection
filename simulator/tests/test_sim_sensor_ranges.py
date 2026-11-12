"""Every value the simulator can legitimately produce lies inside the line.yaml sensor ranges.

The ingest validator flags values outside the catalog range as SUSPECT (or quarantines them),
so legitimate simulator output must never be outside it. Two checks:

- **Analytic:** worst-case bounds derived from config/simulator.yaml (protection trips with
  perturbed thresholds, belt-jam stall current, max commanded speed/load, measurement noise at
  5 sigma, sensor drift, stick-slip bursts).
- **Empirical:** fault-dense runs of several scenarios on both lines, plus a commanded
  overload, with every observed sample checked.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from twin_common.config.line import LineConfig
from twin_common.contracts import Command
from twin_common.enums import CommandType, MachineKind
from twin_sim.config import SimConfig, load_scenario
from twin_sim.line import LineSimulator, MachineSample
from twin_sim.publishers.session import build_simulator

NOISE_SIGMAS = 5.0  # Gaussian noise bound used for "can legitimately produce"
MAX_FAULT_PROGRESS = 2.0  # conservative cap on gradual-fault progress for speed jitter
SENSORS = ("speed", "temp", "vib", "load", "current")


def analytic_bounds(sim: SimConfig, kind: MachineKind) -> dict[str, tuple[float, float]]:
    kp = sim.kinds[kind]
    thr = kp.thresholds
    noise = sim.measurement
    pert = 1.0 + sim.validation.threshold_perturbation_pct / 100.0
    lub = sim.faults.lubrication_loss
    max_load = sim.operation.max_load_pct
    k = NOISE_SIGMAS

    jitter = 1.0 + lub.speed_jitter_gain * MAX_FAULT_PROGRESS
    speed_hi = kp.speed_max_rpm * (1.0 + k * kp.speed_noise_frac * jitter)

    # The machine trips on the deterministic signal, so it cannot run beyond the perturbed
    # fault threshold (one 1 s step of extra rise is covered by the trip margin below).
    temp_hi = thr.temp_fault * pert + k * noise.temp_sigma_c
    vib_true_hi = thr.vib_fault * pert
    burst_hi = max(lub.burst_amp_range) * lub.burst_amp_x_fault * thr.vib_fault
    vib_hi = vib_true_hi * (1.0 + k * noise.vib_mult) + k * noise.vib_add_mm_s + burst_hi
    rated = kp.current_rated_a
    current_true_hi = max(
        thr.current_fault_x * pert * rated,
        kp.current_noload_a + kp.current_load_a * max_load / 100.0,
        sim.faults.belt_jam.stall_current_x_rated * rated
        if kind in sim.faults.belt_jam.kinds
        else 0.0,
    )
    current_hi = current_true_hi * (1.0 + k * noise.current_mult) + k * noise.current_add_a
    load_hi = max_load + k * noise.load_sigma_pct
    temp_lo = min(sim.validation.best_case.ambient_c, sim.validation.worst_case.ambient_c)
    return {
        "speed": (0.0, speed_hi),
        "temp": (temp_lo - k * noise.temp_sigma_c, temp_hi),
        "vib": (0.0, vib_hi),
        "load": (0.0, load_hi),
        "current": (0.0, current_hi),
    }


def max_drift_frac(sim: SimConfig) -> float:
    """Largest sensor drift, as a fraction of the catalog range (rate x longest fault)."""
    sf = sim.sensor_faults
    return sf.drift_rate_frac_per_hour * max(sf.duration_s) / 3600.0


@pytest.mark.parametrize("kind", list(MachineKind))
def test_analytic_worst_case_inside_catalog(
    kind: MachineKind, sim_cfg: SimConfig, line_cfg: LineConfig
) -> None:
    bounds = analytic_bounds(sim_cfg, kind)
    drift_frac = max_drift_frac(sim_cfg)
    problems = []
    for sensor in SENSORS:
        spec = line_cfg.sensors[sensor]
        lo, hi = bounds[sensor]
        if sensor in sim_cfg.sensor_faults.sensors:
            # Drift can go either way; the simulator clamps every drifted value except
            # temperature at 0, so only temperature can drift below its noise-free floor.
            hi += drift_frac * (spec.max - spec.min)
            if sensor == "temp":
                lo -= drift_frac * (spec.max - spec.min)
        if lo < spec.min or hi > spec.max:
            problems.append(
                f"{kind.value} {sensor}: worst case [{lo:.2f}, {hi:.2f}] {spec.unit} "
                f"exceeds catalog [{spec.min}, {spec.max}]"
            )
    assert not problems, "\n".join(problems)


# -- empirical sweep -------------------------------------------------------------------------
def samples(sim: LineSimulator, seconds: int) -> Iterator[MachineSample]:
    for _ in range(seconds):
        yield from (s for s in sim.step(1.0).samples if not s.dropout)


def check(
    line: LineConfig, sim: LineSimulator, seconds: int, observed: dict[str, list[float]]
) -> list[str]:
    problems: list[str] = []
    for s in samples(sim, seconds):
        for sensor in SENSORS:
            value = getattr(s, sensor)
            assert math.isfinite(value)
            observed[sensor].append(value)
            spec = line.sensors[sensor]
            if not spec.min <= value <= spec.max and len(problems) < 10:
                problems.append(f"{s.machine_id} seq {s.seq}: {sensor}={value:.3f} {spec.unit}")
    return problems


SWEEP = [
    ("line.yaml", "quick.yaml"),
    ("line.yaml", "sensor_faults.yaml"),
    ("line.yaml", "e5_lubrication_loss.yaml"),
    ("line.yaml", "cnc_stop.yaml"),
    ("line_single.yaml", "quick_single.yaml"),
]
SWEEP_SEEDS = (1, 2, 3)
SWEEP_DURATION_S = 1800.0


@pytest.mark.parametrize(("line_name", "scenario_name"), SWEEP)
def test_empirical_sweep_inside_catalog(
    line_name: str,
    scenario_name: str,
    config_dir: object,
    sim_cfg: SimConfig,
    line_cfg: LineConfig,
    single_line_cfg: LineConfig,
) -> None:
    from pathlib import Path

    line = line_cfg if line_name == "line.yaml" else single_line_cfg
    scenario = load_scenario(Path(str(config_dir)) / "scenarios" / scenario_name).model_copy(
        update={"duration_s": SWEEP_DURATION_S}
    )
    observed: dict[str, list[float]] = {s: [] for s in SENSORS}
    problems: list[str] = []
    for seed in SWEEP_SEEDS:
        sim = build_simulator(line, sim_cfg, scenario, seed)
        problems += check(line, sim, round(SWEEP_DURATION_S), observed)
    assert not problems, "values outside the catalog range:\n" + "\n".join(problems)
    assert all(observed[s] for s in SENSORS)


def test_commanded_overload_inside_catalog(
    config_dir: object, sim_cfg: SimConfig, line_cfg: LineConfig
) -> None:
    """SET_LOAD to the simulator's maximum (max_load_pct) on every machine."""
    from pathlib import Path

    scenario = load_scenario(Path(str(config_dir)) / "scenarios" / "cnc_stop.yaml")
    sim = build_simulator(line_cfg, sim_cfg, scenario, 5)
    max_load = sim_cfg.operation.max_load_pct
    for i, m in enumerate(line_cfg.machines):
        cmd = Command(
            command_id=f"c{i}",
            idempotency_key=f"k{i}",
            machine_id=m.id,
            type=CommandType.SET_LOAD,
            params={"load": max_load},
            issued_by="test",
            ts=datetime.now(UTC),
        )
        sim.apply_command(cmd)
    observed: dict[str, list[float]] = {s: [] for s in SENSORS}
    problems = check(line_cfg, sim, 900, observed)
    assert not problems, "values outside the catalog range:\n" + "\n".join(problems)
