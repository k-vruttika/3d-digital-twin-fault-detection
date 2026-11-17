"""Phase 9 anomaly detectors and E5 evaluation pieces (ADR 0024, 0025)."""

from __future__ import annotations

import numpy as np
import pytest

from twin_common.enums import MachineState
from twin_ml.config import AutoencoderConfig, IForestConfig, load_eval_config
from twin_ml.data.dataset import Dataset
from twin_ml.data.labels import NORMAL, WARNING
from twin_ml.evaluate import run_sums
from twin_ml.evaluate.evaluator import horizon_run_stats, subset_masks
from twin_ml.experiments import e5_eval
from twin_ml.models.anomaly import AnomalyThresholds, normal_windows
from twin_ml.testing import CONFIG_DIR, make_split

R, W, F = MachineState.RUNNING, MachineState.WARNING, MachineState.FAULT
EVAL = load_eval_config(CONFIG_DIR / "models" / "eval.yaml")
EW = ("episodes", "detected", "lead_sum", "rel_sum", "rel_n", "false_alarms", "negative_s")


def test_normal_windows_rule() -> None:
    # window 3, horizon 2: ends 2..(n-3)
    states = [R] * 6 + [W] + [R] * 3 + [MachineState.IDLE] * 4 + [R] * 6
    ft = [0] * 20
    ft[16] = 1  # an active fault on row 16
    split = make_split(states, window=3, horizons=(2,), fault_type=ft)
    idx = normal_windows(split, no_fault_code=0, horizon=2)
    ends = split.arrays["ends"][idx]
    for e in ends:
        assert split.arrays["state"][e] == 0  # RUNNING at the end
        assert (np.array(ft)[e - 2 : e + 1] == 0).all()  # no fault in the window
        assert split.arrays["health"][e + 2] == NORMAL
    # the WARNING at row 6 makes the windows ending at 4 (target row 6) non-normal
    assert 4 not in ends
    assert 3 in ends
    # a fault on row 16 excludes windows ending at 16..18
    assert not ({16, 17} & set(ends.tolist()))
    # IDLE end states are excluded with the default states, included when listed
    assert 11 not in ends
    idle = normal_windows(split, 0, 2, (R, MachineState.IDLE))
    assert 11 in split.arrays["ends"][idle]


def test_thresholds_per_kind_percentiles() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate(
        [rng.normal(0, 1, 1000), rng.normal(10, 1, 1000), rng.normal(5, 2, 1000)]
    )
    kinds = np.repeat([0, 1, 2], 1000)
    thr = AnomalyThresholds([99.0, 99.9], per_kind=True).fit(scores, kinds)
    for k in range(3):
        assert thr.values[99.0][k] == pytest.approx(np.percentile(scores[kinds == k], 99.0))
    assert thr.values[99.9][1] > thr.values[99.0][1]
    a = thr.alarm(scores, kinds, 99.0)
    for k in range(3):
        assert a[kinds == k].mean() == pytest.approx(0.01, abs=0.002)
    glob = AnomalyThresholds([99.0], per_kind=False).fit(scores, kinds)
    assert np.allclose(glob.values[99.0], np.percentile(scores, 99.0))
    assert thr.n_cal == [1000, 1000, 1000]


def test_sustained_alarm_and_alarm_run_sums_match_the_early_warning_code(
    ml_dataset: Dataset,
) -> None:
    split = ml_dataset.split("val")  # the quick test split has no gradual episode
    ft = list(ml_dataset.fault_types)
    rng = np.random.default_rng(3)
    # a noisy alarm that is more likely near faults, so episodes get detected and missed
    p = np.where(split.target(30) > 0, 0.6, 0.02)
    labels = np.where(rng.random(split.n_windows) < p, WARNING, NORMAL).astype(np.int64)
    op = subset_masks(split, EVAL)["operating"]
    _, rs = horizon_run_stats(split, labels, 30, op, ft, EVAL)
    direct = run_sums.from_runstats(rs)
    sus = e5_eval.sustained_alarm(split, labels >= WARNING, EVAL.early_warning.sustain_s)
    via = e5_eval.alarm_run_sums(split, sus, 30, ft, EVAL)
    assert direct["episodes"].sum() > 0
    assert direct["detected"].sum() > 0
    for f in EW:
        assert np.allclose(direct[f], via[f]), f


def test_combine_pair_with_silent_detector_equals_supervised(ml_dataset: Dataset) -> None:
    from twin_ml.experiments.e5_worker import combine_pair

    split = ml_dataset.split("test")
    ft = list(ml_dataset.fault_types)
    rng = np.random.default_rng(5)
    a = rng.random(split.n_windows) < 0.05
    b = rng.random(split.n_windows) < 0.05
    n, dg = split.n_windows, e5_eval.ends_digest(split)

    def art(group: str, x: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"clean|{group}": e5_eval.pack_alarm(x),
            "n:clean": np.array([n]),
            "digest:clean": np.array([dg]),
        }

    silent = combine_pair(
        split, "clean", art("h30", a), art("p99.9", np.zeros(n, bool)), [99.9], 30, ft, EVAL
    )["p99.9"]
    alone = e5_eval.alarm_run_sums(split, a, 30, ft, EVAL)
    for f in EW:
        assert np.allclose(silent[f], alone[f]), f
    both = combine_pair(split, "clean", art("h30", a), art("p99.9", b), [99.9], 30, ft, EVAL)
    expect = e5_eval.alarm_run_sums(split, a | b, 30, ft, EVAL)
    for f in EW:
        assert np.allclose(both["p99.9"][f], expect[f]), f
    assert both["p99.9"]["detected"].sum() >= alone["detected"].sum()
    bad = art("p99.9", b) | {"digest:clean": np.array(["other"])}
    with pytest.raises(ValueError, match="do not match"):
        combine_pair(split, "clean", art("h30", a), bad, [99.9], 30, ft, EVAL)


def test_drift_flag_sums_by_hand() -> None:
    n = 30
    split = make_split([R] * n, window=3, horizons=(2,))
    sf = np.zeros(n, dtype=bool)
    sf[5:10] = True  # segment 1: rows 5..9
    sf[20:25] = True  # segment 2: rows 20..24
    ends = split.arrays["ends"]
    sus = np.zeros(ends.shape[0], dtype=bool)
    sus[ends == 7] = True  # inside segment 1, 2 s after its start
    sus[ends == 15] = True  # outside any segment
    neg = np.ones(ends.shape[0], dtype=bool)
    out = e5_eval.drift_flag_sums(split, sus, neg, sf)
    assert out.tolist() == [[2.0, 1.0, 2.0]]
    m = run_sums.robust_metrics({"drift": out.sum(axis=0)})
    assert m["sf_flag_rate"] == 0.5
    assert m["sf_flag_delay_s"] == 2.0
    # an alarm on a non-negative window (e.g. a real fault ahead) does not count as a flag
    assert e5_eval.drift_flag_sums(split, sus, ~neg, sf).tolist() == [[2.0, 0.0, 0.0]]


def test_conf_sums_and_metrics() -> None:
    p = np.array([[0.95, 0.04, 0.01], [0.2, 0.7, 0.1], [0.92, 0.05, 0.03], [0.1, 0.1, 0.8]])
    y = np.array([0, 0, 2, 2])
    s = e5_eval.conf_sums(p, y, high=0.9)
    # wrong: rows 1 (pred W, conf .7) and 2 (pred N for FAULT, conf .92 -> high, missed)
    assert s.tolist() == pytest.approx([4, 2, 1.62, 1, 1.75, 1, 0.92, 1])
    m = run_sums.robust_metrics({"confw": s})
    assert m["wrong_rate"] == 0.5
    assert m["wrong_conf_mean"] == pytest.approx(0.81)
    assert m["wrong_high_frac"] == 0.5
    assert m["missed_high_frac"] == 1.0
    assert m["correct_conf_mean"] == pytest.approx(0.875)


def test_pack_roundtrip() -> None:
    x = np.random.default_rng(1).random(1001) < 0.3
    assert np.array_equal(e5_eval.unpack_alarm(e5_eval.pack_alarm(x), x.shape[0]), x)


def test_iforest_deterministic_per_seed(ml_dataset: Dataset) -> None:
    from twin_ml.models.anomaly import IForestDetector

    cfg = IForestConfig(type="iforest", n_estimators=20, max_samples=64, max_features=1.0, n_jobs=1)
    no_fault = list(ml_dataset.fault_types).index("none")
    train, val = ml_dataset.split("train"), ml_dataset.split("val")
    scores = []
    for seed in (0, 0, 1):
        det = IForestDetector(cfg, no_fault, 30, (R,))
        det.fit(train, val, seed)
        scores.append(det.score(val))
    assert np.array_equal(scores[0], scores[1])
    assert not np.array_equal(scores[0], scores[2])


def test_autoencoder_deterministic_and_flags_shifted_windows(ml_dataset: Dataset) -> None:
    from twin_ml.models.autoencoder import AutoencoderDetector

    cfg = AutoencoderConfig(
        type="autoencoder",
        hidden=32,
        latent=8,
        epochs=10,
        batch_size=128,
        lr=0.003,
        weight_decay=0.0,
        eval_batch_size=4096,
        num_threads=2,
    )
    no_fault = list(ml_dataset.fault_types).index("none")
    train, val = ml_dataset.split("train"), ml_dataset.split("val")
    dets = []
    for seed in (0, 0):
        det = AutoencoderDetector(cfg, no_fault, 30, (R,))
        info = det.fit(train, val, seed)
        dets.append(det)
    assert 1 <= info["best_epoch"] <= 10
    idx = normal_windows(val, no_fault, 30)
    s0, s1 = dets[0].score(val, idx), dets[1].score(val, idx)
    assert np.array_equal(s0, s1)
    # the same windows with the temperature channel shifted by 3 std reconstruct worse
    shifted = val.arrays | {"values": val.arrays["values"].copy()}
    shifted["values"][:, 1] += 3.0
    from twin_ml.data.dataset import SplitData

    sv = SplitData(val.name, val.window, val.horizons, val.stride, shifted)
    assert np.median(dets[0].score(sv, idx)) > 2 * np.median(s0)
    # per-sensor errors (vib_norm folded into vib) add up to the score (mean over 30 x 6)
    err = dets[0].sensor_errors(sv, idx[:50])
    assert err.shape == (50, 5)
    assert (err >= 0).all()
    assert np.allclose(err.sum(axis=1) / (30 * 6), dets[0].score(sv, idx[:50]), rtol=1e-4)
