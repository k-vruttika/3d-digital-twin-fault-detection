"""Live drift detection on the operating-condition inputs (Phase 11 Part C, ADR 0031).

"Drift" here means: the machine runs under conditions the models were not trained on. It is
measured on the inputs that describe the operating condition (default: speed and load), never
on the health signals (vibration, temperature, current), which a developing fault changes by
design. Two detectors run per machine and channel:

* **PSI** (population stability index, bin counts additively smoothed) of the last
  ``window_s`` operating rows against the training distribution of the machine's kind
  (quantile bins of the training rows plus overflow bins, stored in the monitor bundle). A
  machine sits in one load band for long stretches, so even
  in-distribution windows differ from the pooled training distribution; the alert level is
  therefore not the textbook 0.25 but the ``alert_quantile`` of the PSI that *clean validation
  windows* reach (calibrated per kind and channel at build time, ``calibrate_thresholds``).
  Drift = PSI above that level on any channel for ``sustain_s`` in a row; it clears below
  ``clear_ratio`` x the level for ``clear_s`` (hysteresis).
* **ADWIN** (``river``) on the standardised value of each channel. It does not decide; it dates
  the change ("the inputs changed at ...") so an operator can see when the shift began.

Pure numpy plus ``river``: no torch, no scikit-learn, so the offline report and the live
worker run the same class.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Numerical floor of a bin proportion (the log of an empty bin is undefined). Not a threshold.
PSI_EPS = 1e-4
# Additive smoothing of the live bin counts (a finite window has empty bins by chance; without
# smoothing every empty bin contributes the same large term and the statistic saturates).
SMOOTHING = 0.5


@dataclass(frozen=True)
class DriftSettings:
    """Live settings: ``config/inference.yaml`` ``drift`` plus the window of the bundle."""

    window_s: int  # from the bundle: the window the thresholds were calibrated on
    margin: float  # multiplies the calibrated alert level
    clear_ratio: float
    sustain_s: float
    clear_s: float
    adwin_delta: float
    step_s: float = 1.0

    @classmethod
    def of(cls, data: Mapping[str, Any], window_s: int, step_s: float = 1.0) -> DriftSettings:
        return cls(
            window_s=int(window_s),
            margin=float(data["margin"]),
            clear_ratio=float(data["clearRatio"]),
            sustain_s=float(data["sustainS"]),
            clear_s=float(data["clearS"]),
            adwin_delta=float(data["adwinDelta"]),
            step_s=step_s,
        )

    @property
    def sustain_rows(self) -> int:
        return max(1, math.ceil(self.sustain_s / self.step_s))

    @property
    def clear_rows(self) -> int:
        return max(1, math.ceil(self.clear_s / self.step_s))


def psi(live: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]) -> float:
    """Population stability index of two bin-proportion vectors (floored at ``PSI_EPS``)."""
    p = np.maximum(live, PSI_EPS)
    q = np.maximum(ref, PSI_EPS)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_counts(counts: npt.NDArray[np.int64], ref: npt.NDArray[np.float64]) -> float:
    """PSI of live bin counts (additively smoothed) against the reference proportions."""
    smoothed = (counts + SMOOTHING) / (float(counts.sum()) + SMOOTHING * counts.size)
    return psi(smoothed.astype(np.float64), ref)


# ---------------------------------------------------------------------------- reference
def build_reference(
    raw: npt.NDArray[np.float64],
    kind_of_row: npt.NDArray[np.int64],
    operating: npt.NDArray[np.bool_],
    sensors: Sequence[str],
    kinds: Sequence[str],
    channels: Sequence[str],
    bins: int,
    support_quantile: float,
) -> dict[str, Any]:
    """Training reference per kind and channel: quantile bin edges, proportions, mean, std.

    ``raw`` is rows x sensors of the training split; only ``operating`` rows count. The
    support is ``[q, 1 - q]`` of the training rows of that kind (``q = support_quantile``); it
    is cut into ``bins`` quantile bins (each holds about ``1 / bins`` of the rows), and a
    value outside the support falls in an overflow bin below or above it, whose training mass
    is only ``q``. Without overflow bins a load far above the trained range would land in the
    top bin and look like "the top decile".
    """
    out: dict[str, Any] = {}
    for ki, kind in enumerate(kinds):
        sel = operating & (kind_of_row == ki)
        per_channel: dict[str, Any] = {}
        for ch in channels:
            x = raw[sel, list(sensors).index(ch)]
            x = x[np.isfinite(x)]
            if x.size < bins * 10:
                continue  # a kind without (enough) training rows has no reference
            lo, hi = np.quantile(x, [support_quantile, 1.0 - support_quantile])
            inner = x[(x >= lo) & (x <= hi)]
            interior = np.quantile(inner, np.linspace(0.0, 1.0, bins + 1)[1:-1])
            edges = np.unique(np.concatenate([[lo], interior, [np.nextafter(hi, np.inf)]]))
            counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=edges.size + 1)
            std = float(np.std(x))
            per_channel[ch] = {
                "edges": [float(e) for e in edges],
                "ref": [float(c) / x.size for c in counts],
                "mean": float(np.mean(x)),
                "std": std if std > 1e-9 else 1.0,
                "n": int(x.size),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
        if per_channel:
            out[kind] = per_channel
    return out


def operating_series(
    split: Any, states: Sequence[Any]
) -> list[tuple[npt.NDArray[np.float64], str]]:
    """The operating rows of every machine run of a dataset split: ``(raw rows, kind name)``.
    Non-operating rows are dropped, as the live monitor only sees operating rows."""
    from twin_ml.data.labels import STATES

    codes = [STATES.index(s) for s in states]
    a = split.arrays
    out: list[tuple[npt.NDArray[np.float64], str]] = []
    for sid in np.unique(a["series"]):
        rows = np.flatnonzero((a["series"] == sid) & np.isin(a["state"], codes))
        if rows.size:
            kind = split.schema.kinds[int(a["kind"][rows[0]])]
            out.append((np.asarray(a["raw"][rows], dtype=np.float64), kind))
    return out


def window_psi_array(
    values: npt.NDArray[np.float64], ref: Mapping[str, Any], window: int
) -> npt.NDArray[np.float64]:
    """PSI of every full ``window`` of one operating series (entry ``i`` ends at row
    ``i + window - 1``), vectorised; the same statistic as the live ``ChannelDrift``."""
    edges = np.asarray(ref["edges"], dtype=np.float64)
    q = np.maximum(np.asarray(ref["ref"], dtype=np.float64), PSI_EPS)
    if values.size < window:
        return np.empty(0)
    b = np.searchsorted(edges, values, side="right")
    onehot = np.zeros((values.size + 1, q.size), dtype=np.float64)
    onehot[np.arange(1, values.size + 1), b] = 1.0
    cum = np.cumsum(onehot, axis=0)
    counts = cum[window:] - cum[:-window]
    p = np.maximum((counts + SMOOTHING) / (window + SMOOTHING * q.size), PSI_EPS)
    out: npt.NDArray[np.float64] = np.sum((p - q) * np.log(p / q), axis=1)
    return out


def window_psis(
    values: npt.NDArray[np.float64], ref: Mapping[str, Any], window: int, stride: int
) -> list[float]:
    """PSI of every ``window``-long stretch (every ``stride`` rows) of one operating series."""
    return [float(x) for x in window_psi_array(values, ref, window)[::stride]]


def calibrate_thresholds(
    reference: dict[str, Any],
    series: Sequence[tuple[npt.NDArray[np.float64], str]],
    sensors: Sequence[str],
    window: int,
    stride: int,
    quantile: float,
) -> dict[str, Any]:
    """Add ``psi_threshold`` to every kind / channel: the ``quantile`` of the PSI of clean
    windows. ``series`` are ``(raw rows of the operating rows of one machine run, kind)``
    from the validation split (never the training reference's own rows)."""
    for kind, per_channel in reference.items():
        for ch, ref in per_channel.items():
            col = list(sensors).index(ch)
            psis: list[float] = []
            for raw, k in series:
                if k != kind:
                    continue
                x = raw[:, col]
                x = x[np.isfinite(x)]
                if x.size >= window:
                    psis += window_psis(x, ref, window, stride)
            if not psis:
                raise ValueError(f"no validation windows for {kind} / {ch}")
            ref["psi_threshold"] = float(np.quantile(psis, quantile))
            ref["calibration_windows"] = len(psis)
    return reference


# ---------------------------------------------------------------------------- live
class ChannelDrift:
    """PSI over a rolling window of operating rows and ADWIN on one channel."""

    def __init__(self, ref: Mapping[str, Any], cfg: DriftSettings) -> None:
        from river.drift import ADWIN

        self.cfg = cfg
        self.edges = np.asarray(ref["edges"], dtype=np.float64)
        self.ref = np.asarray(ref["ref"], dtype=np.float64)
        self.mean = float(ref["mean"])
        self.std = float(ref["std"])
        self.alert = float(ref["psi_threshold"]) * cfg.margin
        self.clear = self.alert * cfg.clear_ratio
        self.bins: deque[int] = deque()
        self.counts = np.zeros(self.ref.size, dtype=np.int64)
        self.adwin = ADWIN(delta=cfg.adwin_delta)
        self.changed_ts: float | None = None  # when ADWIN last reported a change

    def update(self, value: float, ts: float) -> None:
        b = int(np.searchsorted(self.edges, value, side="right"))
        self.bins.append(b)
        self.counts[b] += 1
        if len(self.bins) > self.cfg.window_s:
            self.counts[self.bins.popleft()] -= 1
        self.adwin.update((value - self.mean) / self.std)
        if self.adwin.drift_detected:
            self.changed_ts = ts

    @property
    def ready(self) -> bool:
        return (
            len(self.bins) >= self.cfg.window_s
        )  # full windows only: the alert level is calibrated on them

    def psi(self) -> float | None:
        if not self.ready:
            return None
        return psi_counts(self.counts, self.ref)


class Decider:
    """The sustain / clear rule over PSI values: drift is raised when any channel is at or
    above its alert level for ``sustain_rows`` in a row, and cleared when every known channel
    is below its clear level for ``clear_rows`` in a row. Shared by the live monitor and the
    offline report, so both apply the same rule."""

    def __init__(
        self, alert: Mapping[str, float], clear: Mapping[str, float], cfg: DriftSettings
    ) -> None:
        self.alert = dict(alert)
        self.clear = dict(clear)
        self.cfg = cfg
        self.above = 0
        self.below = 0
        self.drifted = False

    def step(self, psis: Mapping[str, float | None]) -> bool:
        """One operating row; returns True on the row that raises drift."""
        known = {ch: p for ch, p in psis.items() if p is not None}
        over = any(p >= self.alert[ch] for ch, p in known.items())
        calm = bool(known) and all(p < self.clear[ch] for ch, p in known.items())
        self.above = self.above + 1 if over else 0
        self.below = self.below + 1 if calm else 0
        if not self.drifted and self.above >= self.cfg.sustain_rows:
            self.drifted = True
            return True
        if self.drifted and self.below >= self.cfg.clear_rows:
            self.drifted = False
        return False


class MachineDrift:
    """Drift state of one machine: the channels of its kind and the decision rule."""

    def __init__(self, kind: str, reference: Mapping[str, Any], cfg: DriftSettings) -> None:
        self.cfg = cfg
        per_kind = reference.get(kind, {})
        self.channels = {ch: ChannelDrift(r, cfg) for ch, r in per_kind.items()}
        self.decider = Decider(
            {ch: c.alert for ch, c in self.channels.items()},
            {ch: c.clear for ch, c in self.channels.items()},
            cfg,
        )
        self.since: float | None = None

    @property
    def drifted(self) -> bool:
        return self.decider.drifted

    @property
    def available(self) -> bool:
        return bool(self.channels)

    def update(self, values: Mapping[str, float], ts: float) -> dict[str, Any]:
        """One operating row (channel -> raw value; missing channels are skipped)."""
        for ch, c in self.channels.items():
            v = values.get(ch)
            if v is not None and math.isfinite(v):
                c.update(float(v), ts)
        psis = {ch: c.psi() for ch, c in self.channels.items()}
        was = self.decider.drifted
        if self.decider.step(psis):
            self.since = self._change_time(ts)
        elif was and not self.decider.drifted:
            self.since = None
        return self.report(psis)

    def _change_time(self, ts: float) -> float:
        """ADWIN's change time on the channel(s) over the alert level, else the alert time."""
        times = [
            c.changed_ts
            for c in self.channels.values()
            if c.changed_ts is not None and (c.psi() or 0.0) >= c.alert
        ]
        return min(times) if times else ts

    def report(self, psis: Mapping[str, float | None] | None = None) -> dict[str, Any]:
        psis = psis if psis is not None else {ch: c.psi() for ch, c in self.channels.items()}
        return {
            "drifted": self.drifted,
            "ready": any(p is not None for p in psis.values()),
            "psi": dict(psis),
            "alert": {ch: c.alert for ch, c in self.channels.items()},
            "adwinTs": {ch: c.changed_ts for ch, c in self.channels.items()},
            "since": self.since,
        }


def drift_public(ref: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the backend and the UI may know about a bundle's drift reference."""
    if not ref:
        return {"available": False}
    return {
        "available": True,
        "channels": list(ref["channels"]),
        "bins": int(ref["bins"]),
        "windowS": int(ref["window_s"]),
        "alertQuantile": float(ref["alert_quantile"]),
        "kinds": sorted(ref["kinds"]),
    }
