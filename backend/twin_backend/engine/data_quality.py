"""Per-machine data-quality status (OK / DEGRADED) over a rolling wall-clock window.

Inputs per machine: readings (LATE? partial: some sensor value missing?), quarantined messages
attributed to the machine, and whether its data link is down (STALE / OFFLINE). A machine is
DEGRADED as soon as a rate exceeds its threshold (with at least ``min_samples`` messages in the
window) or its link is down; it returns to OK only after ``recover_after_s`` without a
violation. Thresholds come from ``config/engine.yaml``.

Quarantined messages that could not be attributed to a machine (e.g. undecodable payloads) are
not counted against any machine; they show up in the ingest metrics.

The window is wall-clock time (the injected ``Clock``): data quality describes the live feed,
and a replay at 10x speed should not look 10 times worse.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pydantic import Field

from twin_backend.clock import Clock
from twin_backend.engine.line_state import StateModel
from twin_common.config.engine import DataQualityConfig
from twin_common.contracts import Reading
from twin_common.enums import DataQuality, Quality


class DataQualityReport(StateModel):
    """What the API and WS stream show per machine."""

    machine_id: str
    status: DataQuality = DataQuality.OK
    samples: int = 0
    late_pct: float = 0.0
    quarantine_pct: float = 0.0
    partial_pct: float = 0.0
    link_down: bool = False
    reasons: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class _Sample:
    t: float
    reading: bool  # False = quarantined message
    late: bool = False
    partial: bool = False


@dataclass
class _MachineWindow:
    samples: deque[_Sample] = field(default_factory=deque)
    link_down: bool = False
    degraded: bool = False
    seen: int = 0  # messages since startup
    last_violation: float | None = None


class DataQualityMonitor:
    def __init__(self, machine_ids: list[str], cfg: DataQualityConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._w = {m: _MachineWindow() for m in machine_ids}

    def add_reading(self, reading: Reading) -> None:
        w = self._w.get(reading.machine_id)
        if w is not None:
            w.seen += 1
            w.samples.append(
                _Sample(
                    self.clock.monotonic(),
                    True,
                    late=reading.quality is Quality.LATE,
                    partial=bool(reading.missing_sensors()),
                )
            )

    def add_quarantined(self, machine_id: str | None) -> None:
        w = self._w.get(machine_id) if machine_id is not None else None
        if w is not None:
            w.seen += 1
            w.samples.append(_Sample(self.clock.monotonic(), False))

    def set_link_down(self, machine_id: str, down: bool) -> None:
        w = self._w.get(machine_id)
        if w is not None:
            w.link_down = down

    def seen(self, machine_id: str) -> int:
        """Messages (readings + quarantined) attributed to the machine since startup."""
        w = self._w.get(machine_id)
        return w.seen if w is not None else 0

    def evaluate(self) -> dict[str, DataQualityReport]:
        """Current report per machine (also advances the OK/DEGRADED hysteresis)."""
        return {m: self._evaluate(m, w) for m, w in self._w.items()}

    def _evaluate(self, machine_id: str, w: _MachineWindow) -> DataQualityReport:
        cfg, now = self.cfg, self.clock.monotonic()
        while w.samples and w.samples[0].t < now - cfg.window_s:
            w.samples.popleft()
        readings = [s for s in w.samples if s.reading]
        n_read, n_all = len(readings), len(w.samples)
        late = _pct(sum(s.late for s in readings), n_read)
        partial = _pct(sum(s.partial for s in readings), n_read)
        quarantine = _pct(n_all - n_read, n_all)
        reasons: list[str] = []
        if cfg.stale_is_degraded and w.link_down:
            reasons.append("data link down (STALE/OFFLINE)")
        if n_all >= cfg.min_samples:
            if late > cfg.max_late_pct:
                reasons.append(f"late readings {late:.1f} % > {cfg.max_late_pct:g} %")
            if quarantine > cfg.max_quarantine_pct:
                reasons.append(
                    f"quarantined messages {quarantine:.1f} % > {cfg.max_quarantine_pct:g} %"
                )
            if partial > cfg.max_partial_pct:
                reasons.append(f"partial readings {partial:.1f} % > {cfg.max_partial_pct:g} %")
        if reasons:
            w.degraded, w.last_violation = True, now
        elif w.degraded and (
            w.last_violation is None or now - w.last_violation >= cfg.recover_after_s
        ):
            w.degraded = False
        return DataQualityReport(
            machine_id=machine_id,
            status=DataQuality.DEGRADED if w.degraded else DataQuality.OK,
            samples=n_all,
            late_pct=late,
            quarantine_pct=quarantine,
            partial_pct=partial,
            link_down=w.link_down,
            reasons=reasons,
        )


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0
