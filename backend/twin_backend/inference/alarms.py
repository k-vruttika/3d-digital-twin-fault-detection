"""The live alarm rule (= the evaluated rule, ADR 0022 / 0030) and its alert conditions.

Evaluated rule (``twin_ml.evaluate.early_warning``, ``config/models/eval.yaml``): an alarm is
raised when the *uncalibrated* argmax at the primary horizon (+30 s) is WARNING or FAULT for
``k = ceil(sustain_s / step_s)`` consecutive windows (stride 1 s, so 3 windows for 3 s).
Calibrated probabilities and conformal sets never trigger anything.

Live, the counters run over gated windows only (all window rows operating); when the gate
closes they restart. The combined policy of Phase 9 (ADR 0025) adds the autoencoder: its
per-window flag (score above the per-kind p99.9 threshold) sustained the same way.

Alert conditions (``engine.yaml`` kinds, source ``inference``):

* ``predicted_fault``: the alarm, and the argmax has been FAULT for the sustain time;
* ``predicted_warning``: the alarm otherwise (so the union of the two is the evaluated alarm);
* ``unusual_behaviour``: the autoencoder alarm while the supervised alarm is off ("cause
  unknown": no fault type and never a part highlight).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from twin_common.enums import HealthClass

CLASS_ORDER = (HealthClass.NORMAL, HealthClass.WARNING, HealthClass.FAULT)
WARNING_CODE = CLASS_ORDER.index(HealthClass.WARNING)
FAULT_CODE = CLASS_ORDER.index(HealthClass.FAULT)


def sustain_windows(sustain_s: float, step_s: float = 1.0) -> int:
    """Windows that must alarm in a row (``early_warning.sustain_windows`` at stride 1)."""
    return max(1, math.ceil(sustain_s / step_s))


def argmax(probs: Sequence[float]) -> int:
    """Index of the largest value; ties go to the first (as ``numpy.argmax``)."""
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    return best


class Sustain:
    """True once the input has been True for ``k`` consecutive updates."""

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.run = 0

    def update(self, value: bool) -> bool:
        self.run = self.run + 1 if value else 0
        return self.run >= self.k

    def reset(self) -> None:
        self.run = 0

    @property
    def on(self) -> bool:
        return self.run >= self.k


@dataclass(frozen=True, slots=True)
class AlarmState:
    alarm: bool  # supervised alarm, the evaluated rule
    fault: bool  # argmax FAULT sustained (only meaningful with the alarm)
    anomaly: bool  # autoencoder alarm (sustained flag)

    @property
    def predicted_fault(self) -> bool:
        return self.alarm and self.fault

    @property
    def predicted_warning(self) -> bool:
        return self.alarm and not self.fault

    @property
    def unusual(self) -> bool:
        return self.anomaly and not self.alarm


OFF = AlarmState(False, False, False)


class MachineAlarms:
    """The per-machine sustain counters."""

    def __init__(self, k: int) -> None:
        self.alarm = Sustain(k)
        self.fault = Sustain(k)
        self.anomaly = Sustain(k)

    def update(self, raw_probs: Sequence[float], anomaly_flag: bool | None) -> AlarmState:
        top = argmax(raw_probs)
        return AlarmState(
            alarm=self.alarm.update(top >= WARNING_CODE),
            fault=self.fault.update(top == FAULT_CODE),
            anomaly=self.anomaly.update(bool(anomaly_flag)),
        )

    def reset(self) -> None:
        for s in (self.alarm, self.fault, self.anomaly):
            s.reset()

    @property
    def state(self) -> AlarmState:
        return AlarmState(self.alarm.on, self.fault.on, self.anomaly.on)
