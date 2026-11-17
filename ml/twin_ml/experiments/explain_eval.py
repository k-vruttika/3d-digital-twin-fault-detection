"""Explanation sanity checks (Phase 9, ADR 0024 / 0026; no MLflow, no simulator).

* :func:`explained_windows`: per gradual episode, the alarmed precursor windows (sustained
  alarm, end time in [episode start, fault)), thinned to one per ``window_stride_s`` and
  capped at ``max_windows_per_episode`` (evenly spaced);
* :func:`agreement_sums`: per run and fault type, episodes explained and episodes whose top
  sensor (largest mean signed attribution over the episode's windows) is the fault type's
  leading physical signal (``explain.expected_sensor``);
* :func:`deletion_inputs` / :func:`deletion_sums`: the deletion test. The top-attributed
  sensor's value channels (plus ``vib_norm`` for vib) are replaced by the baseline (0 after
  normalisation), and so are one random other sensor's (seeded per window); the change in
  p(WARNING) + p(FAULT) must be larger for the top sensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.data.dataset import SplitData
from twin_ml.features.inputs import SENSORS, VALUE_CHANNELS

DELETION_FIELDS = ("n", "d_top", "d_rand", "top_gt_rand")
COMPLETENESS_FIELDS = ("n", "abs_err", "abs_delta")


@dataclass(frozen=True)
class EpisodeWindows:
    episode: int  # index into the split's episode arrays
    run: int  # index into split.run_ids
    fault_type: int  # code in the (extended) vocabulary
    windows: npt.NDArray[np.int64]


def explained_windows(
    split: SplitData,
    sustained_alarm: npt.NDArray[np.bool_],
    stride_s: int,
    max_windows: int,
) -> list[EpisodeWindows]:
    a = split.arrays
    t_end = split.at_end("t_s").astype(np.float64)
    win_series = split.at_end("series")
    rw = split.window_run()
    out: list[EpisodeWindows] = []
    for e in range(a["ep_series"].shape[0]):
        sid = int(a["ep_series"][e])
        t0, tf = float(a["t_s"][a["ep_start"][e]]), float(a["t_s"][a["ep_fault"][e]])
        cand = np.flatnonzero((win_series == sid) & (t_end >= t0) & (t_end < tf) & sustained_alarm)
        if not cand.size:
            continue
        kept: list[int] = []
        last = -np.inf
        for w in cand[np.argsort(t_end[cand], kind="stable")]:
            if t_end[w] >= last + stride_s:
                kept.append(int(w))
                last = t_end[w]
        sel = np.array(kept, dtype=np.int64)
        if sel.size > max_windows:
            sel = sel[np.round(np.linspace(0, sel.size - 1, max_windows)).astype(np.int64)]
        out.append(EpisodeWindows(e, int(rw[sel[0]]), int(a["ep_fault_type"][e]), sel))
    return out


def agreement_sums(
    episodes: list[EpisodeWindows],
    sensor_weights: Mapping[int, npt.NDArray[np.float64]],
    expected: Mapping[int, int],
    n_runs: int,
    n_types: int,
) -> tuple[npt.NDArray[np.float64], list[dict[str, Any]]]:
    """(runs, types, 2) [episodes, agreeing] and a per-episode record.

    ``sensor_weights[e]`` holds the (windows, sensors) normalised signed weights of episode
    ``e``; ``expected`` maps a fault-type code to the expected sensor index.
    """
    sums = np.zeros((n_runs, n_types, 2))
    records: list[dict[str, Any]] = []
    for ep in episodes:
        if ep.fault_type not in expected:
            continue
        mean = sensor_weights[ep.episode].mean(axis=0)
        top = int(np.argmax(mean))
        agree = top == expected[ep.fault_type]
        sums[ep.run, ep.fault_type, 0] += 1
        sums[ep.run, ep.fault_type, 1] += float(agree)
        records.append(
            {
                "episode": ep.episode,
                "run": ep.run,
                "fault_type": ep.fault_type,
                "windows": int(ep.windows.size),
                "top": SENSORS[top],
                "agree": bool(agree),
                "mean_weight": mean.tolist(),
            }
        )
    return sums, records


def sensor_channels(sensor: int) -> list[int]:
    """Value channels replaced when a sensor is deleted (vib also drops vib_norm)."""
    name = SENSORS[sensor]
    out = [VALUE_CHANNELS.index(name)]
    if name == "vib":
        out.append(VALUE_CHANNELS.index("vib_norm"))
    return out


def random_other(
    top: npt.NDArray[np.int64], window_ids: npt.NDArray[np.int64], seed: int
) -> npt.NDArray[np.int64]:
    """One random sensor other than ``top`` per window, seeded by (seed, window id)."""
    out = np.empty_like(top)
    for i, (t, w) in enumerate(zip(top, window_ids, strict=True)):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(w),)))
        others = [s for s in range(len(SENSORS)) if s != int(t)]
        out[i] = others[int(rng.integers(len(others)))]
    return out


def deletion_inputs(values: npt.NDArray[np.float32], sensor: npt.NDArray[np.int64]) -> Any:
    """Copy of (B, W, 6) values with each window's ``sensor`` set to the baseline (0)."""
    out = values.copy()
    for s in np.unique(sensor):
        rows = sensor == s
        for c in sensor_channels(int(s)):
            out[rows, :, c] = 0.0
    return out


def deletion_sums(
    runs: npt.NDArray[np.int64],
    p0: npt.NDArray[np.float64],
    p_top: npt.NDArray[np.float64],
    p_rand: npt.NDArray[np.float64],
    n_runs: int,
) -> npt.NDArray[np.float64]:
    """(runs, 4) [windows, sum |dp| top, sum |dp| random, windows with top > random]."""
    d_top, d_rand = np.abs(p0 - p_top), np.abs(p0 - p_rand)
    out = np.zeros((n_runs, len(DELETION_FIELDS)))
    np.add.at(out[:, 0], runs, 1.0)
    np.add.at(out[:, 1], runs, d_top)
    np.add.at(out[:, 2], runs, d_rand)
    np.add.at(out[:, 3], runs, (d_top > d_rand).astype(np.float64))
    return out


def completeness_sums(
    runs: npt.NDArray[np.int64],
    err: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    n_runs: int,
) -> npt.NDArray[np.float64]:
    out = np.zeros((n_runs, len(COMPLETENESS_FIELDS)))
    np.add.at(out[:, 0], runs, 1.0)
    np.add.at(out[:, 1], runs, err)
    np.add.at(out[:, 2], runs, np.abs(delta))
    return out
