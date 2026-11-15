"""Windowing, horizon labels, censoring, episodes, forward fill and the training subsample."""

from __future__ import annotations

import numpy as np
import pytest

from twin_common.enums import MachineState as S
from twin_ml.config import ClassWeightConfig
from twin_ml.data.labels import (
    EPISODE_GRADUAL,
    EPISODE_JAM,
    EPISODE_NONE,
    FAULT,
    NORMAL,
    WARNING,
    censor_ttf,
    episode_types,
    fault_onsets,
    gradual_episodes,
    health_codes,
    state_codes,
)
from twin_ml.data.windows import far_normal, subsample_far_normal, window_ends
from twin_ml.features.inputs import forward_fill
from twin_ml.models.base import class_weights, effective_weights
from twin_ml.testing import make_split


def test_window_ends_bounds_and_stride() -> None:
    assert window_ends(10, 3, 2, 1).tolist() == [2, 3, 4, 5, 6, 7]
    assert window_ends(10, 3, 2, 2).tolist() == [2, 4, 6]
    assert window_ends(10, 3, 0, 4).tolist() == [2, 6]
    assert window_ends(4, 3, 2, 1).tolist() == []  # too short for window + horizon
    with pytest.raises(ValueError, match="window"):
        window_ends(10, 0, 1, 1)


def test_horizon_target_is_state_h_rows_ahead() -> None:
    states = [S.RUNNING] * 4 + [S.WARNING] * 2 + [S.FAULT] * 3 + [S.MAINTENANCE]
    split = make_split(states, window=3, horizons=(1, 2))
    ends = split.arrays["ends"]
    assert ends.tolist() == [2, 3, 4, 5, 6, 7]  # tail windows without a +2 target dropped
    health = health_codes(state_codes([s.value for s in states]))
    assert split.target(1).tolist() == health[ends + 1].tolist()
    assert split.target(2).tolist() == [WARNING, WARNING, FAULT, FAULT, FAULT, NORMAL]
    assert split.at_end("health").tolist() == [NORMAL, NORMAL, WARNING, WARNING, FAULT, FAULT]
    with pytest.raises(KeyError):
        split.target(5)


def test_window_gather_covers_exactly_the_window() -> None:
    values = np.arange(10, dtype=np.float32)[:, None].repeat(6, axis=1)
    split = make_split([S.RUNNING] * 10, values=values, window=3, horizons=(1,))
    v, mask, kind = split.gather(np.array([0, 3]))
    assert v[:, :, 0].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert mask.shape == (2, 3, 5)
    assert kind.tolist() == [[1, 0, 0], [1, 0, 0]]
    seq = split.sequence(np.array([0]))
    assert seq.shape == (1, 3, 14)


def test_health_mapping_matches_claude_md() -> None:
    codes = state_codes(
        ["RUNNING", "IDLE", "STARVED", "BLOCKED", "MAINTENANCE", "WARNING", "FAULT"]
    )
    assert health_codes(codes).tolist() == [NORMAL] * 5 + [WARNING, FAULT]
    with pytest.raises(ValueError, match="OFFLINE"):
        health_codes(state_codes(["OFFLINE"]))
    with pytest.raises(ValueError, match="unknown"):
        state_codes(["BROKEN"])


def test_censoring_rule() -> None:
    # fault ahead (known), far from the end without fault (known at cap), near end (censored)
    ttf = np.array([100.0, 900.0, 700.0, 300.0, 0.0])
    no_fault_ahead = np.array([False, False, True, True, False])
    capped, censored = censor_ttf(ttf, no_fault_ahead, cap_s=600.0)
    assert capped.tolist() == [100.0, 600.0, 600.0, 300.0, 0.0]
    assert censored.tolist() == [False, False, False, True, False]


def test_episodes_and_episode_types() -> None:
    states = (
        [S.RUNNING] * 3
        + [S.WARNING] * 2
        + [S.FAULT] * 2
        + [S.MAINTENANCE] * 2
        + [S.RUNNING] * 2
        + [S.FAULT] * 2
        + [S.RUNNING]
    )
    ft = (
        ["none", "bearing_wear", "bearing_wear", "bearing_wear", "bearing_wear"]
        + ["bearing_wear"] * 2
        + ["none"] * 4
        + ["belt_jam"] * 2
        + ["none"]
    )
    codes = state_codes([s.value for s in states])
    assert fault_onsets(codes).tolist() == [5, 11]
    eps = gradual_episodes(codes, ft, {"bearing_wear"})
    assert len(eps) == 1  # the jam is not a gradual episode
    assert (eps[0].i_start, eps[0].i_warn, eps[0].i_fault) == (1, 3, 5)
    t = np.arange(1, len(states) + 1, dtype=np.float64)
    et = episode_types(codes, ft, t, cap_s=3.0, gradual={"bearing_wear"}, jam={"belt_jam"})
    # rows 2..4 are within 3 s of the gradual onset at row 5, rows 8..10 of the jam at row 11
    assert et.tolist() == [
        EPISODE_NONE,
        EPISODE_NONE,
        EPISODE_GRADUAL,
        EPISODE_GRADUAL,
        EPISODE_GRADUAL,
        EPISODE_GRADUAL,
        EPISODE_GRADUAL,
        EPISODE_NONE,
        EPISODE_JAM,
        EPISODE_JAM,
        EPISODE_JAM,
        EPISODE_JAM,
        EPISODE_JAM,
        EPISODE_NONE,
    ]


def test_forward_fill_with_leading_gap_and_mask() -> None:
    nan = np.nan
    raw = np.array([[nan, 1.0], [2.0, nan], [nan, nan], [4.0, 5.0]], dtype=np.float32)
    filled, mask = forward_fill(raw, fallback=np.array([9.0, 8.0], dtype=np.float32))
    assert filled.tolist() == [[9.0, 1.0], [2.0, 1.0], [2.0, 1.0], [4.0, 5.0]]
    assert mask.tolist() == [[1, 0], [0, 1], [1, 1], [0, 0]]


def test_far_normal_and_subsample_keep_every_fault_near_window() -> None:
    health_now = np.array([NORMAL, NORMAL, NORMAL, WARNING, NORMAL], dtype=np.int8)
    ahead = [np.array([NORMAL, NORMAL, FAULT, FAULT, NORMAL], dtype=np.int8)]
    active = np.array([False, True, False, False, False])
    near = np.array([False, False, False, False, True])
    far = far_normal(health_now, ahead, active, near)
    assert far.tolist() == [True, False, False, False, False]
    keep, weight = subsample_far_normal(np.tile(far, 200), 0.25, np.random.default_rng(0))
    assert keep[~np.tile(far, 200)].all()
    assert set(weight[np.tile(far, 200)].tolist()) == {4.0}


def test_subsample_weights_restore_the_class_prior() -> None:
    """Weighted class distribution after subsampling == distribution without subsampling."""
    rng = np.random.default_rng(1)
    n = 200_000
    y = rng.choice([NORMAL, WARNING, FAULT], size=n, p=[0.94, 0.04, 0.02])
    far = (y == NORMAL) & (rng.random(n) < 0.8)
    keep, weight = subsample_far_normal(far, 0.25, np.random.default_rng(2))
    full = np.bincount(y, minlength=3) / n
    w = np.bincount(y[keep], weights=weight[keep], minlength=3)
    assert keep.sum() < 0.5 * n  # it really subsamples
    np.testing.assert_allclose(w / w.sum(), full, atol=0.003)
    # unweighted counts would inflate the minority classes
    raw = np.bincount(y[keep], minlength=3) / keep.sum()
    assert raw[FAULT] > 2 * full[FAULT]
    # class weights computed from the weighted counts equal those of the full set
    cfg = ClassWeightConfig(power=1.0, max=1000.0)
    cw_sub = class_weights(y[keep], weight[keep], 3, cfg)
    cw_full = class_weights(y, np.ones(n, dtype=np.float32), 3, cfg)
    np.testing.assert_allclose(cw_sub, cw_full, rtol=0.03)


def test_class_weights_rule() -> None:
    y = np.array([0] * 8 + [1] * 2)
    w = np.ones(10, dtype=np.float32)
    cw = class_weights(y, w, 3, ClassWeightConfig(power=1.0, max=50.0))
    np.testing.assert_allclose(cw, [10 / (2 * 8), 10 / (2 * 2), 0.0])
    capped = class_weights(y, w, 3, ClassWeightConfig(power=1.0, max=2.0))
    assert capped[1] == 2.0
    assert class_weights(y, w, 3, ClassWeightConfig(power=0.0, max=50.0)).tolist() == [1, 1, 0]
    eff = effective_weights(y, np.full(10, 2.0, dtype=np.float32), cw)
    assert eff[0] == pytest.approx(2 * cw[0])
    assert eff[-1] == pytest.approx(2 * cw[1])
