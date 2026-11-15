"""Label helpers (pure numpy, no simulator import).

Codes used in the built dataset:

* state: index into ``STATES`` (``twin_common.enums.MachineState`` order);
* health: index into ``HEALTH`` (NORMAL, WARNING, FAULT), CLAUDE.md §6 mapping;
* episode type: ``EPISODE_NONE`` / ``EPISODE_GRADUAL`` / ``EPISODE_JAM`` / ``EPISODE_OTHER``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twin_common.enums import HealthClass, MachineState, is_labelable, to_health_class

STATES: tuple[MachineState, ...] = tuple(MachineState)
HEALTH: tuple[HealthClass, ...] = (HealthClass.NORMAL, HealthClass.WARNING, HealthClass.FAULT)
NORMAL, WARNING, FAULT = 0, 1, 2

EPISODE_NONE, EPISODE_GRADUAL, EPISODE_JAM, EPISODE_OTHER = 0, 1, 2, 3
EPISODE_NAMES = ("none", "gradual", "jam", "other")

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

FAULT_STATE = STATES.index(MachineState.FAULT)
WARNING_STATE = STATES.index(MachineState.WARNING)
# state code -> health code (-1 for states without a health class: OFFLINE, STALE)
STATE_TO_HEALTH = np.array(
    [HEALTH.index(to_health_class(s)) if is_labelable(s) else -1 for s in STATES], dtype=np.int8
)


def state_codes(states: Sequence[str]) -> npt.NDArray[np.int8]:
    lookup = {s.value: i for i, s in enumerate(STATES)}
    try:
        return np.array([lookup[s] for s in states], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"unknown machine state {exc.args[0]!r}") from exc


def health_codes(state: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    health = STATE_TO_HEALTH[state]
    if (health < 0).any():
        raise ValueError("OFFLINE/STALE rows have no health class; drop them before labelling")
    return health


def censor_ttf(
    ttf: FloatArray, no_fault_ahead: BoolArray, cap_s: float
) -> tuple[FloatArray, BoolArray]:
    """Apply the time-to-fault cap and the censoring rule (ADR 0019).

    ``ttf`` is the time to the next FAULT onset, or the time to the end of the run where
    ``no_fault_ahead``. The target ``min(ttf, cap)`` is known when a fault lies ahead, or when
    at least ``cap`` of the run is left without one (then the true value is >= cap, so the
    capped target is exactly cap). Otherwise the row is right-censored: only ``ttf_true >= ttf``
    is known.
    """
    capped = np.minimum(ttf, cap_s)
    censored = no_fault_ahead & (ttf < cap_s)
    return capped.astype(np.float64), censored


def fault_onsets(state: npt.NDArray[np.int8]) -> IntArray:
    """Row indices where FAULT starts after a non-FAULT row (or at row 0)."""
    is_fault = state == FAULT_STATE
    prev = np.concatenate(([False], is_fault[:-1]))
    return np.flatnonzero(is_fault & ~prev)


def episode_types(
    state: npt.NDArray[np.int8],
    fault_type: Sequence[str],
    t_s: FloatArray,
    cap_s: float,
    gradual: Collection[str],
    jam: Collection[str],
) -> npt.NDArray[np.int8]:
    """Type of the current (in FAULT) or next (within ``cap_s``) fault episode of each row."""
    n = state.shape[0]
    out = np.full(n, EPISODE_NONE, dtype=np.int8)
    onsets = fault_onsets(state)
    if onsets.size == 0:
        return out
    ft = np.asarray(fault_type, dtype=object)
    onset_cls = np.array(
        [
            EPISODE_GRADUAL if ft[i] in gradual else EPISODE_JAM if ft[i] in jam else EPISODE_OTHER
            for i in onsets
        ],
        dtype=np.int8,
    )
    rows = np.arange(n)
    in_fault = state == FAULT_STATE
    # current episode: last onset <= row (rows in FAULT)
    last = np.searchsorted(onsets, rows, side="right") - 1
    cur = np.where(in_fault & (last >= 0), onset_cls[np.maximum(last, 0)], EPISODE_NONE)
    # next episode: first onset > row, within the cap
    nxt = np.searchsorted(onsets, rows, side="right")
    has_next = nxt < onsets.size
    nxt_c = np.minimum(nxt, onsets.size - 1)
    near = has_next & ((t_s[onsets[nxt_c]] - t_s) <= cap_s)
    out = np.where(in_fault, cur, np.where(near, onset_cls[nxt_c], EPISODE_NONE))
    return out.astype(np.int8)


@dataclass(frozen=True)
class Episode:
    """One gradual fault episode of one machine series."""

    fault_type: str
    i_start: int  # first row of the degradation (fault_type set)
    i_warn: int | None  # first simulator WARNING row before the fault, if any
    i_fault: int  # FAULT onset row


def gradual_episodes(
    state: npt.NDArray[np.int8], fault_type: Sequence[str], gradual: Collection[str]
) -> list[Episode]:
    """Gradual fault episodes: the contiguous ``fault_type`` stretch that ends in a FAULT onset."""
    ft = list(fault_type)
    out: list[Episode] = []
    for i_fault in fault_onsets(state):
        kind = ft[i_fault]
        if kind not in gradual:
            continue
        i_start = int(i_fault)
        while i_start > 0 and ft[i_start - 1] == kind and state[i_start - 1] != FAULT_STATE:
            i_start -= 1
        warn = np.flatnonzero(state[i_start:i_fault] == WARNING_STATE)
        out.append(
            Episode(
                fault_type=kind,
                i_start=i_start,
                i_warn=int(i_start + warn[0]) if warn.size else None,
                i_fault=int(i_fault),
            )
        )
    return out
