"""Metric functions, early-warning logic and the run-level bootstrap against known values."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from twin_ml.evaluate.bootstrap import RunStats, bootstrap_ci, paired_bootstrap
from twin_ml.evaluate.early_warning import (
    EpisodeResult,
    episode_lead,
    false_alarms,
    rising_edges,
    summarize,
    sustain_windows,
    sustained,
)
from twin_ml.evaluate.metrics import (
    average_precision,
    confusion,
    mae_uncensored,
    percentiles_ms,
    scores_from_confusion,
)


def test_confusion_and_scores_by_hand() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 0, 1, 1, 2, 2, 0])
    cm = confusion(y, p, 3)
    assert cm.tolist() == [[3, 1, 0], [0, 1, 1], [1, 0, 1]]
    s = scores_from_confusion(cm)
    assert s["accuracy"] == pytest.approx(5 / 8)
    assert s["precision"] == pytest.approx([3 / 4, 1 / 2, 1 / 2])
    assert s["recall"] == pytest.approx([3 / 4, 1 / 2, 1 / 2])
    assert s["f1"] == pytest.approx([3 / 4, 1 / 2, 1 / 2])
    assert s["macro_f1"] == pytest.approx((0.75 + 0.5 + 0.5) / 3)
    assert s["support"] == [4, 2, 2]


def test_macro_f1_ignores_classes_without_support() -> None:
    cm = confusion(np.array([0, 0, 1]), np.array([0, 2, 1]), 3)
    s = scores_from_confusion(cm)
    assert s["f1"][2] == 0.0
    assert s["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)  # class 2 absent from truth


def test_average_precision_known_values() -> None:
    # scikit-learn's documented example: AP = 0.8333...
    y = np.array([False, False, True, True])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert average_precision(y, s) == pytest.approx(0.8333333333)
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    # all scores tied: one threshold, AP = prevalence
    assert average_precision(y, np.full(4, 0.5)) == pytest.approx(0.5)
    assert np.isnan(average_precision(np.zeros(3, dtype=bool), np.ones(3)))


def test_mae_excludes_censored_and_capped_rows() -> None:
    pred = np.array([10.0, 50.0, 0.0, 600.0])
    ttf = np.array([20.0, 40.0, 100.0, 600.0])
    cens = np.array([False, False, True, False])
    mae, n = mae_uncensored(pred, ttf, cens, cap_s=600.0)
    assert mae == pytest.approx(10.0)
    assert n == 2
    assert mae_uncensored(pred[2:3], ttf[2:3], cens[2:3], 600.0)[1] == 0


def test_sustained_and_rising_edges() -> None:
    a = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert sustained(a, 3).tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    assert sustained(a, 1).tolist() == a.tolist()
    assert sustained(a[:2], 3).tolist() == [False, False]
    assert rising_edges(sustained(a, 3)).tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert sustain_windows(3, 1) == 3
    assert sustain_windows(3, 2) == 2
    assert sustain_windows(1, 5) == 1


def test_episode_lead_and_relative_lead() -> None:
    t = np.arange(0.0, 100.0)
    alarm = np.zeros(100, dtype=bool)
    alarm[10:13] = True  # before the degradation started: ignored
    alarm[40:100] = True
    sus = sustained(alarm, 3)
    detected, lead, rel = episode_lead(t, sus, t_start=20.0, t_fault=90.0, t_warn=60.0)
    assert detected
    assert lead == 90.0 - 42.0
    assert rel == 60.0 - 42.0
    assert episode_lead(t, sus, 20.0, 41.0, None) == (False, None, None)
    late = np.zeros(100, dtype=bool)
    late[85:] = True
    assert episode_lead(t, sustained(late, 3), 20.0, 90.0, 60.0) == (True, 90.0 - 87.0, 60.0 - 87.0)


def test_false_alarms_per_hour() -> None:
    sus = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    neg = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    count, neg_s = false_alarms(sus, neg, stride=2)
    assert (count, neg_s) == (2, 14.0)  # edge at index 4 is in a positive window
    eps = [
        EpisodeResult(0, "CNC-01", "CNC", "bearing_wear", 0, 50, 100, True, 80.0, 30.0),
        EpisodeResult(0, "CNV-01", "CONVEYOR", "overheating", 0, None, 100, True, 40.0, None),
        EpisodeResult(1, "CNC-01", "CNC", "bearing_wear", 0, 60, 100, False, None, None),
    ]
    s = summarize(eps, fa_count=3, negative_s=7200.0)
    assert s["episodes"] == 3
    assert s["detected"] == 2
    assert s["miss_rate"] == pytest.approx(1 / 3)
    assert s["lead_mean_s"] == 60.0
    assert s["rel_lead_mean_s"] == 30.0
    assert s["rel_lead_n"] == 1
    assert s["false_alarms_per_h"] == pytest.approx(1.5)
    assert s["episodes_per_kind"] == {"CNC": 2, "CONVEYOR": 1}


def _stats(n_runs: int) -> RunStats:
    rng = np.random.default_rng(5)
    st = RunStats.empty(n_runs, 3)
    for r in range(n_runs):
        y = rng.choice(3, 300, p=[0.8, 0.12, 0.08])
        p = np.where(rng.random(300) < 0.8, y, rng.choice(3, 300))
        st.confusion[r] = confusion(y, p, 3)
        st.episodes[r] = 2
        st.detected[r] = int(rng.integers(0, 3))
        st.lead_sum[r] = st.detected[r] * rng.uniform(100, 300)
        st.rel_sum[r] = st.detected[r] * rng.uniform(-20, 60)
        st.rel_n[r] = st.detected[r]
        st.false_alarms[r] = int(rng.integers(0, 4))
        st.negative_s[r] = 3000.0
    return st


def test_bootstrap_deterministic_and_brackets_point() -> None:
    st = _stats(20)
    a = bootstrap_ci(st, n_resamples=300, seed=7, ci=0.95)
    b = bootstrap_ci(st, n_resamples=300, seed=7, ci=0.95)
    assert a == b
    assert bootstrap_ci(st, n_resamples=300, seed=8, ci=0.95) != a
    for m, v in a.items():
        assert v["lo"] <= v["point"] <= v["hi"], m
    # point estimate = metrics over all runs once
    assert a["macro_f1"]["point"] == pytest.approx(
        scores_from_confusion(st.confusion.sum(axis=0))["macro_f1"]
    )
    assert a["false_alarms_per_h"]["point"] == pytest.approx(
        st.false_alarms.sum() / (st.negative_s.sum() / 3600)
    )


def test_bootstrap_resamples_whole_runs() -> None:
    """With identical runs every resample equals the point; one odd run creates spread."""
    st = RunStats.empty(10, 3)
    st.confusion[:] = np.array([[50, 0, 0], [0, 5, 5], [0, 0, 10]])
    st.episodes[:] = 1
    st.detected[:] = 1
    st.lead_sum[:] = 200.0
    st.negative_s[:] = 3600.0
    flat = bootstrap_ci(st, 100, 1, 0.9)
    assert flat["lead_mean_s"]["lo"] == flat["lead_mean_s"]["hi"] == 200.0
    st.lead_sum[0] = 1000.0  # one run with a much longer lead
    spread = bootstrap_ci(st, 200, 1, 0.9)
    assert spread["lead_mean_s"]["hi"] > spread["lead_mean_s"]["lo"]
    assert spread["lead_mean_s"]["lo"] >= 200.0
    assert spread["lead_mean_s"]["hi"] <= 1000.0


def test_latency_percentiles() -> None:
    p = percentiles_ms([0.001] * 50 + [0.002] * 50)
    assert p["p50_ms"] == pytest.approx(1.5)
    assert p["p95_ms"] == pytest.approx(2.0)


def test_paired_bootstrap_identical_models_have_zero_difference() -> None:
    st = _stats(20)
    d = paired_bootstrap(st, st, n_resamples=200, seed=3, ci=0.95)
    for m, v in d.items():
        assert (v["point"], v["lo"], v["hi"]) == (0.0, 0.0, 0.0), m
        assert not v["excludes_zero"]


def test_paired_bootstrap_uses_the_same_resamples() -> None:
    a = _stats(20)
    b = copy.deepcopy(a)
    b.lead_sum = a.lead_sum - 30.0 * a.detected  # every detected episode 30 s later
    d = paired_bootstrap(a, b, n_resamples=300, seed=3, ci=0.95)
    # same runs on both sides: the difference is exactly 30 s in every resample
    assert d["lead_mean_s"]["point"] == pytest.approx(30.0)
    assert d["lead_mean_s"]["lo"] == pytest.approx(30.0)
    assert d["lead_mean_s"]["hi"] == pytest.approx(30.0)
    assert d["lead_mean_s"]["excludes_zero"]
    # unpaired CIs of the two sides overlap heavily, the paired one does not
    ua = bootstrap_ci(a, 300, 3, 0.95)["lead_mean_s"]
    ub = bootstrap_ci(b, 300, 3, 0.95)["lead_mean_s"]
    assert ua["lo"] < ub["hi"]
    with pytest.raises(ValueError, match="same test runs"):
        paired_bootstrap(a, _stats(10), 10, 1, 0.95)
