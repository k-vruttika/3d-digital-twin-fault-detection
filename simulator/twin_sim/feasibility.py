"""Fault-feasibility validator: guarantees reachability and a minimum lead time.

For every gradual fault and applicable machine kind, the noise-free single-machine model is
simulated at constant load and ambient:

* best case for reaching FAULT (low load, cold ambient, all thresholds at +p): FAULT must be
  crossed before the progress variable exceeds ``reach_max_progress`` (i.e. within the
  progression time D/g), for both progression bounds;
* worst case for lead time (high load, hot ambient, thresholds at -p, D = D_min): FAULT must
  be crossed no earlier than ``min_lead_s`` after onset;
* healthy machine at the worst case must stay below every WARNING threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from twin_sim.config import GRADUAL_FAULTS, FaultType, KindPhysics, SimConfig
from twin_sim.physics import (
    NO_EFFECTS,
    current_true,
    fault_effects,
    fault_load_exponent,
    heat_w,
    load_factor,
    speed_rpm,
    steady_temp,
    thermal_step,
    vib_true,
)
from twin_sim.thresholds import Thresholds, scaled_thresholds

# Stop the noise-free simulation once progress reaches this multiple of the allowed maximum.
_SEARCH_PROGRESS_LIMIT = 3.0


@dataclass(frozen=True)
class Crossing:
    t_s: float
    progress: float
    signal: str


def simulate_noise_free(
    cfg: SimConfig,
    kp: KindPhysics,
    fault: FaultType,
    load_pct: float,
    ambient_c: float,
    thresholds: Thresholds,
    progression_s: float,
    speed_set: float | None = None,
    dt: float = 1.0,
) -> Crossing | None:
    """First FAULT crossing of the deterministic model, starting from healthy steady state.

    ``speed_set`` is the spindle/motor setpoint (default: nominal).
    """
    fp = cfg.faults
    setpoint = kp.speed_set_rpm if speed_set is None else speed_set
    g = load_factor(load_pct, fault_load_exponent(fault, fp), fp)
    speed0 = speed_rpm(kp, setpoint, load_pct, NO_EFFECTS)
    temp = steady_temp(
        kp, heat_w(kp, load_pct, speed0 / kp.speed_set_rpm, NO_EFFECTS), 0.0, ambient_c
    )
    w, t = 0.0, 0.0
    while w < _SEARCH_PROGRESS_LIMIT * cfg.validation.reach_max_progress:
        t += dt
        w += g * dt / progression_s
        eff = fault_effects(fault, w, kp, fp)
        ratio = speed_rpm(kp, setpoint, load_pct, eff) / kp.speed_set_rpm
        temp = thermal_step(
            temp, heat_w(kp, load_pct, ratio, eff), kp, eff.cooling_loss, ambient_c, dt
        )
        signal = thresholds.fault_signal(
            vib_true(kp, load_pct, ratio, eff), temp, current_true(kp, load_pct, eff)
        )
        if signal is not None:
            return Crossing(t, w, signal)
    return None


def _speeds(kp: KindPhysics) -> list[float]:
    """Setpoints to check: both ends of the operating range (effects are monotone in speed)."""
    lo, hi = kp.operating_speeds
    return [lo] if lo == hi else [lo, hi]


def check_fault_feasibility(cfg: SimConfig) -> list[str]:
    """Return a list of human-readable problems (empty if the config is feasible).

    Every check runs at both ends of each kind's operating speed range.
    """
    v = cfg.validation
    p = v.threshold_perturbation_pct / 100.0
    pct = v.threshold_perturbation_pct
    d_min, d_max = cfg.faults.progression_s
    bc, wc = v.best_case, v.worst_case
    problems: list[str] = []
    for kind, kp in cfg.kinds.items():
        thr_lo = scaled_thresholds(kp, 1.0 - p)
        thr_hi = scaled_thresholds(kp, 1.0 + p)
        for sp in _speeds(kp):
            at = f"{sp:.0f} RPM"
            # Healthy margin at the worst case.
            ratio = speed_rpm(kp, sp, wc.load_pct, NO_EFFECTS) / kp.speed_set_rpm
            temp = steady_temp(kp, heat_w(kp, wc.load_pct, ratio, NO_EFFECTS), 0.0, wc.ambient_c)
            if thr_lo.warning_state(
                False,
                vib_true(kp, wc.load_pct, ratio, NO_EFFECTS),
                temp,
                current_true(kp, wc.load_pct, NO_EFFECTS),
                0.0,
            ):
                problems.append(
                    f"{kind.value}: healthy machine reaches WARNING at the worst case ({at})"
                )

            for fault in GRADUAL_FAULTS:
                if not cfg.faults.applicable(fault, kind):
                    continue
                where = f"{fault.value} on {kind.value} at {at}"
                for d in (d_min, d_max):
                    c = simulate_noise_free(
                        cfg, kp, fault, bc.load_pct, bc.ambient_c, thr_hi, d, sp
                    )
                    if c is None or c.progress > v.reach_max_progress:
                        got = "never" if c is None else f"at progress {c.progress:.2f}"
                        problems.append(
                            f"{where}: best case (load {bc.load_pct} %, ambient {bc.ambient_c} C, "
                            f"thresholds +{pct} %, D={d:.0f} s) reaches FAULT {got}; "
                            f"must be <= {v.reach_max_progress}"
                        )
                c = simulate_noise_free(
                    cfg, kp, fault, wc.load_pct, wc.ambient_c, thr_lo, d_min, sp
                )
                if c is not None and c.t_s < v.min_lead_s:
                    problems.append(
                        f"{where}: worst case (load {wc.load_pct} %, ambient {wc.ambient_c} C, "
                        f"thresholds -{pct} %, D={d_min:.0f} s) reaches FAULT after "
                        f"{c.t_s:.0f} s; minimum lead time is {v.min_lead_s:.0f} s"
                    )
    return problems
