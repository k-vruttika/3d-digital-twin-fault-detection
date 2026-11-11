"""Pure physics functions for one machine. Equations: docs/simulator.md section 2-3.

All functions are side-effect free and take plain floats so they are cheap in the per-step loop
and can be reused by the noise-free feasibility validator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from twin_sim.config import FaultParams, FaultType, KindPhysics


@dataclass(frozen=True, slots=True)
class FaultEffects:
    """How the active fault (at progress w) modifies the healthy machine model."""

    friction: float = 0.0  # friction index f: extra heat qf * f
    cooling_loss: float = 0.0  # c: cooling coefficient is k * (1 - c)
    fault_heat_w: float = 0.0  # extra heat from the failing component
    vib_bearing: float = 0.0  # mm/s at nominal speed, scales with speed ratio
    vib_imbalance: float = 0.0  # mm/s at nominal speed, scales with speed ratio^2
    current_extra_a: float = 0.0
    speed_factor: float = 1.0
    speed_jitter: float = 1.0  # multiplies the speed noise
    burst_prob: float = 0.0  # stick-slip vibration bursts per second (observed only)


NO_EFFECTS = FaultEffects()


def load_factor(load_pct: float, exponent: float, fp: FaultParams) -> float:
    """g(L) = clip((L / L_ref)^m, lo, hi): degradation speed-up with load."""
    if exponent == 0.0:
        return 1.0
    lo, hi = fp.load_factor_clip
    g = float((max(load_pct, 0.0) / fp.load_ref_pct) ** exponent)
    return min(max(g, lo), hi)


def fault_load_exponent(fault: FaultType, fp: FaultParams) -> float:
    spec = fp.spec(fault)
    return float(getattr(spec, "load_exponent", 0.0))


def fault_effects(
    fault: FaultType | None, w: float, kp: KindPhysics, fp: FaultParams
) -> FaultEffects:
    """Effects of a gradual fault at progress ``w`` (uncapped, >= 0)."""
    if fault is None or w <= 0.0:
        return NO_EFFECTS
    if fault is FaultType.BEARING_WEAR:
        b = fp.bearing_wear
        amp = b.vib_amp_x_fault * kp.thresholds.vib_fault
        friction = b.friction_coef * w
        return FaultEffects(
            friction=friction,
            vib_bearing=amp * (math.exp(b.gamma * w) - 1.0) / (math.exp(b.gamma) - 1.0),
            current_extra_a=kp.current_friction_a * friction,
        )
    if fault is FaultType.OVERHEATING:
        o = fp.overheating
        return FaultEffects(
            cooling_loss=min(o.cooling_loss_rate * w, o.cooling_loss_max),
            fault_heat_w=o.fault_heat_dt_k * kp.cooling_w_per_k * w,
        )
    if fault is FaultType.SPINDLE_IMBALANCE:
        amp = fp.spindle_imbalance.vib_amp_x_fault * kp.thresholds.vib_fault
        return FaultEffects(vib_imbalance=amp * w * w)
    if fault is FaultType.LUBRICATION_LOSS:
        lub = fp.lubrication_loss
        f_l = w**lub.friction_exponent
        return FaultEffects(
            friction=f_l,
            current_extra_a=lub.current_x_rated * kp.current_rated_a * f_l,
            speed_factor=max(1.0 - lub.speed_sag * w * w, 0.0),
            speed_jitter=1.0 + lub.speed_jitter_gain * w,
            burst_prob=min(lub.burst_prob_gain * w, 1.0),
        )
    return NO_EFFECTS  # belt_jam is handled as an event, not through progress


def speed_rpm(kp: KindPhysics, speed_set: float, load_pct: float, eff: FaultEffects) -> float:
    """Deterministic speed: setpoint minus slip under load and drop under overload."""
    lf = load_pct / 100.0
    factor = 1.0 - kp.slip_at_full_load * lf - kp.overload_speed_drop * max(0.0, lf - 1.0)
    return max(speed_set * factor * eff.speed_factor, 0.0)


def heat_w(kp: KindPhysics, load_pct: float, speed_ratio: float, eff: FaultEffects) -> float:
    """Heat input of a running machine: q0 + qL*l^2 + qf*f*(w/w_set) + fault heat."""
    lf = load_pct / 100.0
    return (
        kp.heat_noload_w
        + kp.heat_load_w * lf * lf
        + kp.heat_friction_w * eff.friction * speed_ratio
        + eff.fault_heat_w
    )


def thermal_step(
    temp: float, heat: float, kp: KindPhysics, cooling_loss: float, ambient: float, dt: float
) -> float:
    """Exact solution of C dT/dt = heat - k(1-c)(T - T_amb) over dt (stable for any dt)."""
    k_eff = kp.cooling_w_per_k * (1.0 - cooling_loss)
    t_ss = ambient + heat / k_eff
    return t_ss + (temp - t_ss) * math.exp(-dt * k_eff / kp.heat_capacity_j_per_k)


def steady_temp(kp: KindPhysics, heat: float, cooling_loss: float, ambient: float) -> float:
    return ambient + heat / (kp.cooling_w_per_k * (1.0 - cooling_loss))


def vib_true(kp: KindPhysics, load_pct: float, speed_ratio: float, eff: FaultEffects) -> float:
    """Deterministic RMS vibration (bursts are added in the observation layer)."""
    lf = load_pct / 100.0
    base = kp.vib_base_mm_s * (1.0 + kp.vib_load_coef * lf)
    return (base + eff.vib_bearing) * speed_ratio + eff.vib_imbalance * speed_ratio * speed_ratio


def current_true(kp: KindPhysics, load_pct: float, eff: FaultEffects) -> float:
    """I = I0 + IL*l + fault friction current."""
    return kp.current_noload_a + kp.current_load_a * load_pct / 100.0 + eff.current_extra_a
