from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from twin_common.enums import MachineKind
from twin_sim.config import FaultType, RecoveryParams, ScenarioConfig, SimConfig
from twin_sim.line import LineSimulator
from twin_sim.physics import (
    NO_EFFECTS,
    current_true,
    fault_effects,
    heat_w,
    load_factor,
    steady_temp,
    thermal_step,
    vib_true,
)

SimFactory = Callable[..., LineSimulator]


def test_thermal_step_is_exact_and_converges(sim_cfg: SimConfig) -> None:
    kp = sim_cfg.kinds[MachineKind.CNC]
    heat, amb = 800.0, 20.0
    t_ss = steady_temp(kp, heat, 0.0, amb)
    # One 100 s step equals one hundred 1 s steps (exact exponential integration).
    big = thermal_step(30.0, heat, kp, 0.0, amb, 100.0)
    small = 30.0
    for _ in range(100):
        small = thermal_step(small, heat, kp, 0.0, amb, 1.0)
    assert big == pytest.approx(small, rel=1e-12)
    tau = kp.heat_capacity_j_per_k / kp.cooling_w_per_k
    assert thermal_step(30.0, heat, kp, 0.0, amb, 20 * tau) == pytest.approx(t_ss, abs=1e-6)


@pytest.mark.parametrize("kind", list(MachineKind))
def test_heat_and_current_increase_with_load(sim_cfg: SimConfig, kind: MachineKind) -> None:
    kp = sim_cfg.kinds[kind]
    loads = [30.0, 60.0, 95.0]
    temps = [steady_temp(kp, heat_w(kp, lo, 1.0, NO_EFFECTS), 0.0, 20.0) for lo in loads]
    currents = [current_true(kp, lo, NO_EFFECTS) for lo in loads]
    assert temps == sorted(temps)
    assert len(set(temps)) == 3
    assert currents == sorted(currents)
    assert len(set(currents)) == 3


def test_simulated_steady_state_higher_at_higher_load(make_sim: SimFactory) -> None:
    def mean_last(load_range: str) -> tuple[float, float]:
        sc = ScenarioConfig(
            name="t",
            duration_s=1500,
            load_ranges=[load_range],
            ambient_c=(22.0, 22.0),
            recovery=RecoveryParams(auto=False),
        )
        frames = make_sim(sc, seed=3).run()
        i = 1  # CNC-01
        tail = frames[-300:]
        return (
            sum(f.samples[i].temp for f in tail) / len(tail),
            sum(f.samples[i].current for f in tail) / len(tail),
        )

    t_lo, i_lo = mean_last("low")
    t_hi, i_hi = mean_last("high")
    assert t_hi > t_lo + 5.0
    assert i_hi > i_lo + 3.0


@pytest.mark.parametrize(
    "fault", [FaultType.BEARING_WEAR, FaultType.SPINDLE_IMBALANCE, FaultType.LUBRICATION_LOSS]
)
def test_degradation_monotone_in_progress(sim_cfg: SimConfig, fault: FaultType) -> None:
    kp = sim_cfg.kinds[MachineKind.CNC]
    ws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    vib = [vib_true(kp, 60.0, 1.0, fault_effects(fault, w, kp, sim_cfg.faults)) for w in ws]
    cur = [current_true(kp, 60.0, fault_effects(fault, w, kp, sim_cfg.faults)) for w in ws]
    assert vib == sorted(vib)
    assert cur == sorted(cur)
    assert vib[-1] > vib[0] or cur[-1] > cur[0]


def test_overheating_cooling_loss_is_capped(sim_cfg: SimConfig) -> None:
    kp = sim_cfg.kinds[MachineKind.CONVEYOR]
    o = sim_cfg.faults.overheating
    eff = fault_effects(FaultType.OVERHEATING, 5.0, kp, sim_cfg.faults)
    assert eff.cooling_loss == o.cooling_loss_max
    assert eff.fault_heat_w == pytest.approx(o.fault_heat_dt_k * kp.cooling_w_per_k * 5.0)


def test_load_factor_clipped(sim_cfg: SimConfig) -> None:
    fp = sim_cfg.faults
    lo, hi = fp.load_factor_clip
    assert load_factor(1.0, 1.5, fp) == lo
    assert load_factor(200.0, 1.5, fp) == hi
    assert load_factor(fp.load_ref_pct, 1.5, fp) == pytest.approx(1.0)
    assert load_factor(10.0, 0.0, fp) == 1.0
    assert math.isfinite(load_factor(0.0, 1.5, fp))


def test_imbalance_scales_with_speed_squared_bearing_linearly(sim_cfg: SimConfig) -> None:
    kp = sim_cfg.kinds[MachineKind.CNC]
    fp = sim_cfg.faults

    def fault_part(fault: FaultType, ratio: float) -> float:
        eff = fault_effects(fault, 0.8, kp, fp)
        return vib_true(kp, 60.0, ratio, eff) - vib_true(kp, 60.0, ratio, NO_EFFECTS)

    lo, hi = 7500.0 / kp.speed_set_rpm, 10000.0 / kp.speed_set_rpm
    imb = fault_part(FaultType.SPINDLE_IMBALANCE, hi) / fault_part(FaultType.SPINDLE_IMBALANCE, lo)
    brg = fault_part(FaultType.BEARING_WEAR, hi) / fault_part(FaultType.BEARING_WEAR, lo)
    assert imb == pytest.approx((hi / lo) ** 2)
    assert brg == pytest.approx(hi / lo)
