"""Test-time perturbations of raw sensor series (Phase 8, ADR 0021). Pure numpy.

Perturbations act on a machine series' raw ``sensors`` array (T x 5, NaN = missing) *before*
windowing, forward fill and normalisation, so the missing masks and the forward fill behave as
they would on live data:

* ``noise_<p>``: additive Gaussian noise, sigma = p % of the sensor's normal-operation std
  (per machine kind, from training data; :func:`normal_std`). Missing values stay missing;
  values are not clipped (a noisy sensor can read slightly below zero).
* ``mcar_<p>``: every sensor value is removed (NaN) independently with probability p %.
* ``dropout_<p>``: single-sensor gaps (NaN) of ``duration_s`` placed until p % of the
  series' sensor-time is covered.
* ``stuck_<p>``: a single sensor repeats its last value for ``duration_s``, until p % of the
  sensor-time is covered.

Every draw comes from ``SeedSequence(seed, (condition, run, machine))`` with CRC-32 keys, so
a condition is identical for every model (which makes paired comparisons valid), and
independent of the order in which series are processed.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind
from twin_ml.config import PERTURBATION_KINDS, PerturbationKind, PerturbationsConfig
from twin_ml.data.labels import STATES
from twin_ml.features.inputs import KINDS, SENSORS, FloatArray

CLEAN = "clean"
PERTURBATIONS = "perturbations"  # in a study's condition list: the whole grid
SIGNAL_NOISE = "signal_noise"  # in a study's condition list: the signal-relative noise grid
SIGNAL_NOISE_PREFIX = "snoise"
STD_NOISE = "std_noise"  # in a study's condition list: the uniform noise grid in std units
STD_NOISE_PREFIX = "stdnoise"


@dataclass(frozen=True)
class Condition:
    """One synthetic test condition; ``kind`` None is the clean data."""

    name: str
    kind: PerturbationKind | None
    level: float  # fraction (0.2 = 20 %)

    @property
    def level_pct(self) -> int:
        return round(self.level * 100)


def condition_name(kind: str, level_pct: int) -> str:
    return f"{kind}_{level_pct}"


def grid_conditions(cfg: PerturbationsConfig) -> list[Condition]:
    """Every perturbation of the grid, type by type, ascending level."""
    return [
        Condition(condition_name(kind, lv), kind, lv / 100.0)
        for kind in PERTURBATION_KINDS
        for lv in cfg.grid(kind).levels_pct
    ]


def parse_condition(name: str, cfg: PerturbationsConfig) -> Condition:
    if name == CLEAN:
        return Condition(CLEAN, None, 0.0)
    for c in grid_conditions(cfg):
        if c.name == name:
            return c
    raise KeyError(f"unknown perturbation condition {name!r}")


def _key(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def series_rng(seed: int, condition: str, run_id: str, machine_id: str) -> np.random.Generator:
    ss = np.random.SeedSequence(seed, spawn_key=(_key(condition), _key(run_id), _key(machine_id)))
    return np.random.default_rng(ss)


# ---------------------------------------------------------------------- primitives


def add_noise(
    raw: FloatArray, sigma: npt.NDArray[np.floating[Any]], rng: np.random.Generator
) -> FloatArray:
    """Gaussian noise with per-sensor ``sigma``; NaN stays NaN."""
    noise = rng.standard_normal(raw.shape) * np.asarray(sigma, dtype=np.float64)[None, :]
    return (raw.astype(np.float64) + noise).astype(np.float32)


def drop_mcar(raw: FloatArray, p: float, rng: np.random.Generator) -> FloatArray:
    out = raw.copy()
    out[rng.random(raw.shape) < p] = np.nan
    return out


def _segments(
    shape: tuple[int, int],
    fraction: float,
    duration_s: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Non-identical single-sensor segments (sensor, start, stop) covering ``fraction`` of the
    cells. Overlaps are allowed; only newly covered cells count towards the budget."""
    n, c = shape
    lo, hi = duration_s
    budget = round(fraction * n * c)
    covered = np.zeros(shape, dtype=bool)
    out: list[tuple[int, int, int]] = []
    count = 0
    max_tries = 100 * max(1, budget // max(lo, 1) + 1)
    for _ in range(max_tries):
        if count >= budget:
            break
        d = int(rng.integers(lo, hi + 1))
        d = min(d, n)
        sensor = int(rng.integers(c))
        start = int(rng.integers(0, n - d + 1))
        new = int((~covered[start : start + d, sensor]).sum())
        if new == 0:
            continue
        covered[start : start + d, sensor] = True
        out.append((sensor, start, start + d))
        count += new
    return out


def drop_segments(
    raw: FloatArray, fraction: float, duration_s: tuple[int, int], rng: np.random.Generator
) -> FloatArray:
    out = raw.copy()
    for sensor, start, stop in _segments(raw.shape, fraction, duration_s, rng):
        out[start:stop, sensor] = np.nan
    return out


def stick_segments(
    raw: FloatArray, fraction: float, duration_s: tuple[int, int], rng: np.random.Generator
) -> FloatArray:
    """The sensor repeats the last value observed before the segment (the first observed
    value of the segment when it starts the series). Segments are applied in order, so a
    later segment can freeze an already stuck value, as a real stuck sensor would."""
    out = raw.copy()
    for sensor, start, stop in _segments(raw.shape, fraction, duration_s, rng):
        col = out[:, sensor]
        before = col[:start][~np.isnan(col[:start])]
        if before.size:
            value = before[-1]
        else:
            seg = col[start:stop][~np.isnan(col[start:stop])]
            if seg.size == 0:
                continue
            value = seg[0]
        out[start:stop, sensor] = value
    return out


def perturb_array(
    raw: FloatArray,
    condition: Condition,
    cfg: PerturbationsConfig,
    sigma: npt.NDArray[np.floating[Any]],
    rng: np.random.Generator,
) -> FloatArray:
    """Apply one condition to one raw series (T x 5)."""
    if condition.kind is None:
        return raw.copy()
    if condition.kind == "noise":
        return add_noise(raw, condition.level * np.asarray(sigma), rng)
    if condition.kind == "mcar":
        return drop_mcar(raw, condition.level, rng)
    grid = cfg.grid(condition.kind)
    assert grid.duration_s is not None  # validated in the config
    if condition.kind == "dropout":
        return drop_segments(raw, condition.level, grid.duration_s, rng)
    return stick_segments(raw, condition.level, grid.duration_s, rng)


# ---------------------------------------------------------------------- series


class _Series(Protocol):
    @property
    def run_id(self) -> str: ...
    @property
    def machine_id(self) -> str: ...
    @property
    def kind(self) -> MachineKind: ...
    @property
    def sensors(self) -> FloatArray: ...


S = TypeVar("S", bound=_Series)


def perturb_series(
    series: Iterable[S],
    condition: Condition,
    cfg: PerturbationsConfig,
    sigma: npt.NDArray[np.floating[Any]],
    seed: int,
    kinds: Sequence[str] = tuple(KINDS),
) -> list[S]:
    """Perturbed copies of machine series (dataclasses); labels are never touched.

    ``sigma``: (len(kinds), sensors) normal-operation std per machine kind and sensor.
    """
    out: list[S] = []
    for s in series:
        rng = series_rng(seed, condition.name, s.run_id, s.machine_id)
        raw = perturb_array(s.sensors, condition, cfg, sigma[list(kinds).index(s.kind)], rng)
        out.append(dataclasses.replace(s, sensors=raw))  # type: ignore[type-var]
    return out


def normal_std(
    raw: FloatArray,
    state: npt.NDArray[np.integer[Any]],
    kind: npt.NDArray[np.integer[Any]],
    fault_type: npt.NDArray[np.integer[Any]],
    no_fault_code: int,
    normal_states: Sequence[Any],
    n_kinds: int = len(KINDS),
) -> FloatArray:
    """Std of each sensor per machine kind over normal-operation training rows.

    Rows: state in ``normal_states`` and no active hidden fault. A kind without such rows (or
    a constant sensor) falls back to the std over all kinds' normal rows, then to 1.
    """
    codes = [STATES.index(s) for s in normal_states]
    normal = np.isin(state, codes) & (fault_type == no_fault_code)
    pooled = np.nanstd(raw[normal].astype(np.float64), axis=0) if normal.any() else None
    out = np.ones((n_kinds, raw.shape[1]), dtype=np.float64)
    for k in range(n_kinds):
        sel = normal & (kind == k)
        if sel.any():
            std = np.nanstd(raw[sel].astype(np.float64), axis=0)
        elif pooled is not None:
            std = pooled
        else:
            continue
        if pooled is not None:
            std = np.where(np.isfinite(std) & (std > 1e-9), std, pooled)
        out[k] = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------- fault signal size


@dataclass(frozen=True)
class EpisodeRise:
    kind: int  # index into KINDS
    fault_type: int  # vocabulary code
    rise: FloatArray  # (5,) mean of the last window before FAULT minus the first after start
    precursor_s: float


def episode_rises(
    raw: FloatArray,
    kind: npt.NDArray[np.integer[Any]],
    t_s: npt.NDArray[np.floating[Any]],
    ep_start: npt.NDArray[np.integer[Any]],
    ep_fault: npt.NDArray[np.integer[Any]],
    ep_fault_type: npt.NDArray[np.integer[Any]],
    window: int,
) -> list[EpisodeRise]:
    """Per gradual episode: how much each sensor rises from degradation start to FAULT.

    Window means (``window`` rows, at most half the precursor) at both ends of the precursor.
    """
    out: list[EpisodeRise] = []
    for e in range(ep_fault.shape[0]):
        i0, i1 = int(ep_start[e]), int(ep_fault[e])
        span = min(window, (i1 - i0) // 2)
        if span < 1:
            continue
        rise = np.nanmean(raw[i1 - span : i1], axis=0) - np.nanmean(raw[i0 : i0 + span], axis=0)
        out.append(
            EpisodeRise(
                int(kind[i1]),
                int(ep_fault_type[e]),
                rise.astype(np.float32),
                float(t_s[i1] - t_s[i0]),
            )
        )
    return out


def leading_sensors(
    rises: Sequence[EpisodeRise], sigma: npt.NDArray[np.floating[Any]]
) -> dict[tuple[int, int], int]:
    """(kind, fault type) -> sensor with the largest median |rise| / normal-operation std."""
    groups: dict[tuple[int, int], list[FloatArray]] = {}
    for r in rises:
        groups.setdefault((r.kind, r.fault_type), []).append(r.rise)
    out = {}
    for (k, ft), rs in groups.items():
        med = np.median(np.stack(rs), axis=0)
        out[(k, ft)] = int(np.argmax(np.abs(med) / np.asarray(sigma)[k]))
    return out


def signal_scale(rises: Sequence[EpisodeRise], sigma: npt.NDArray[np.floating[Any]]) -> FloatArray:
    """Typical rise before FAULT per machine kind and sensor (KINDS x 5).

    For a sensor that leads some fault types of a kind (largest median |rise| / std): the
    median |rise| over that kind's episodes of those fault types. For a sensor that leads
    none: its median |rise| over all of the kind's episodes. A kind without episodes falls
    back to the normal-operation std. Used to scale the signal-relative noise grid.
    """
    lead = leading_sensors(rises, sigma)
    out = np.asarray(sigma, dtype=np.float64).copy()
    for k in range(len(KINDS)):
        mine = [r for r in rises if r.kind == k]
        if not mine:
            continue
        for s in range(len(SENSORS)):
            led = [r for r in mine if lead[(r.kind, r.fault_type)] == s]
            pool = led or mine
            out[k, s] = float(np.median([abs(float(r.rise[s])) for r in pool]))
    return np.where(out > 1e-9, out, np.asarray(sigma, dtype=np.float64)).astype(np.float32)


def signal_noise_conditions(levels_pct: Sequence[int]) -> list[Condition]:
    """Signal-relative noise: kind "noise" with sigma = level x signal_scale."""
    return [
        Condition(condition_name(SIGNAL_NOISE_PREFIX, lv), "noise", lv / 100.0) for lv in levels_pct
    ]


def std_noise_conditions(multiples: Sequence[float]) -> list[Condition]:
    """Uniform noise in normal-operation std units: kind "noise" with sigma = m x normal std,
    the same multiple for every sensor (``stdnoise_1`` = 1 x std, ``stdnoise_0.5`` = 0.5 x)."""
    return [Condition(f"{STD_NOISE_PREFIX}_{m:g}", "noise", float(m)) for m in multiples]
