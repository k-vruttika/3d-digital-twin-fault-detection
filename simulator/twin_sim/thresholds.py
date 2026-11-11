"""Per-machine WARNING/FAULT thresholds, per-run perturbation and the hysteresis classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twin_sim.config import KindPhysics


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Absolute thresholds for one machine in one run (current in A)."""

    vib_warning: float
    vib_fault: float
    temp_warning: float
    temp_fault: float
    current_warning: float
    current_fault: float

    def fault_signal(self, vib: float, temp: float, current: float) -> str | None:
        """Name of the first signal above its FAULT threshold, or None."""
        if vib >= self.vib_fault:
            return "vib"
        if temp >= self.temp_fault:
            return "temp"
        if current >= self.current_fault:
            return "current"
        return None

    def warning_state(
        self, was_warning: bool, vib: float, temp: float, current: float, hysteresis: float
    ) -> bool:
        """Enter WARNING above any warning threshold; leave once all are (1-h) below."""
        if vib >= self.vib_warning or temp >= self.temp_warning or current >= self.current_warning:
            return True
        if not was_warning:
            return False
        k = 1.0 - hysteresis
        return (
            vib >= self.vib_warning * k
            or temp >= self.temp_warning * k
            or current >= self.current_warning * k
        )


def scaled_thresholds(kp: KindPhysics, scale: float) -> Thresholds:
    """All thresholds multiplied by the same factor (used by the feasibility validator)."""
    return perturbed_thresholds(kp, (scale,) * 6)


def perturbed_thresholds(kp: KindPhysics, factors: tuple[float, ...]) -> Thresholds:
    t = kp.thresholds
    rated = kp.current_rated_a
    return Thresholds(
        vib_warning=t.vib_warning * factors[0],
        vib_fault=t.vib_fault * factors[1],
        temp_warning=t.temp_warning * factors[2],
        temp_fault=t.temp_fault * factors[3],
        current_warning=t.current_warning_x * rated * factors[4],
        current_fault=t.current_fault_x * rated * factors[5],
    )


def sample_thresholds(kp: KindPhysics, pct: float, rng: np.random.Generator) -> Thresholds:
    """Each threshold multiplied by an independent U(1 - p, 1 + p) factor.

    Warning is kept strictly below fault for every signal.
    """
    p = pct / 100.0
    f = rng.uniform(1.0 - p, 1.0 + p, size=6).tolist()
    thr = perturbed_thresholds(kp, tuple(f))
    return Thresholds(
        vib_warning=min(thr.vib_warning, thr.vib_fault * 0.95),
        vib_fault=thr.vib_fault,
        temp_warning=min(thr.temp_warning, thr.temp_fault - 1.0),
        temp_fault=thr.temp_fault,
        current_warning=min(thr.current_warning, thr.current_fault * 0.95),
        current_fault=thr.current_fault,
    )
