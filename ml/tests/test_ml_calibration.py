"""Phase 8 calibration: ECE by hand, bin sums add up over runs, temperature, OvR maps."""

from __future__ import annotations

import numpy as np

from twin_ml.calibrate import fit_head
from twin_ml.calibrate.ece import (
    bin_sums,
    class_sums,
    ece_from_sums,
    reliability,
    top_label_sums,
)
from twin_ml.calibrate.temperature import fit_temperature, nll, softmax
from twin_ml.calibrate.trees import OvRCalibrator, group_folds, mean_nll, select_method
from twin_ml.config import CalibrationConfig

CAL = CalibrationConfig(
    tree_methods=["isotonic", "platt"], cv_folds=3, temperature_bounds=(0.05, 20)
)


def test_ece_by_hand() -> None:
    # bins of width 0.5: [0.3, 0.4] -> bin 0 (acc 0.5, conf 0.35); [0.9, 1.0] -> bin 1
    conf = np.array([0.3, 0.4, 0.9, 1.0])
    hit = np.array([1, 0, 1, 1])
    sums = bin_sums(conf, hit, 2)
    assert sums[:, 0].tolist() == [2, 2]
    expected = 0.5 * abs(0.5 - 0.35) + 0.5 * abs(1.0 - 0.95)
    assert abs(ece_from_sums(sums) - expected) < 1e-12
    rel = reliability(sums)
    assert rel["accuracy"] == [0.5, 1.0]
    assert np.isnan(ece_from_sums(np.zeros((2, 3))))


def test_top_label_and_class_sums() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.4, 0.5, 0.1]])
    y = np.array([0, 1, 1])
    top = top_label_sums(p, y, 10)
    assert top[:, 0].sum() == 3
    assert top[:, 2].sum() == 2
    fault = class_sums(p, y, 2, 10)
    assert abs(fault[:, 1].sum() - 1.0) < 1e-12
    assert fault[:, 2].sum() == 0


def test_bin_sums_add_up_over_runs() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=1000)
    y = rng.integers(0, 3, 1000)
    runs = rng.integers(0, 7, 1000)
    total = top_label_sums(p, y, 15)
    parts = np.sum([top_label_sums(p[runs == r], y[runs == r], 15) for r in range(7)], axis=0)
    np.testing.assert_allclose(parts, total)
    assert abs(ece_from_sums(parts) - ece_from_sums(total)) < 1e-12


def test_temperature_recovers_known_t() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, size=(20_000, 3))
    p_true = softmax(z, 2.5)  # labels drawn from the T=2.5 distribution
    y = np.array([rng.choice(3, p=row) for row in p_true])
    t = fit_temperature(z, y, (0.05, 20.0))
    assert abs(t - 2.5) < 0.15
    assert nll(z, y, t) <= nll(z, y, 1.0)


def test_temperature_keeps_argmax() -> None:
    z = np.random.default_rng(2).normal(size=(100, 3))
    assert np.array_equal(np.argmax(softmax(z, 3.0), 1), np.argmax(z, 1))


def _overconfident(n: int = 6000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    z = rng.normal(0, 1, (n, 3))
    z[np.arange(n), y] += 1.0
    return softmax(z, 0.3), y  # sharpened: overconfident


def test_ovr_maps_valid_and_monotone() -> None:
    p, y = _overconfident()
    for method in ("isotonic", "platt"):
        cal = OvRCalibrator(method).fit(p, y)
        q = cal.transform(p)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-9)
        assert (q >= 0).all()
        assert mean_nll(q, y) < mean_nll(p, y)
        grid = np.linspace(0, 1, 50)
        probe = np.stack([grid, (1 - grid) / 2, (1 - grid) / 2], axis=1)
        raw_col = (
            cal.maps[0].predict(grid)
            if method == "isotonic"
            else cal.maps[0].predict_proba(
                np.log(np.clip(grid, 1e-6, 1 - 1e-6) / np.clip(1 - grid, 1e-6, 1))[:, None]
            )[:, 1]
        )
        assert np.all(np.diff(raw_col) >= -1e-12)
        assert np.isfinite(cal.transform(probe)).all()


def test_absent_class_column_kept() -> None:
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1], [0.3, 0.6, 0.1]])
    y = np.array([0, 1, 0, 1])
    cal = OvRCalibrator("isotonic").fit(p, y)
    assert 2 not in cal.maps


def test_selection_deterministic_and_grouped() -> None:
    p, y = _overconfident()
    groups = np.repeat(np.arange(12), 500)
    folds = group_folds(groups, 3)
    assert sum(f.size for f in folds) == 6000
    for f in folds:  # a group never spans two folds
        assert set(groups[f]).isdisjoint(set(groups[np.setdiff1d(np.arange(6000), f)]))
    a = select_method(p, y, groups, ["isotonic", "platt"], 3)
    b = select_method(p, y, groups, ["isotonic", "platt"], 3)
    assert a == b
    assert a[0] in ("isotonic", "platt")


def test_fit_head_families() -> None:
    p, y = _overconfident(3000)
    groups = np.repeat(np.arange(6), 500)
    logits = np.log(np.clip(p, 1e-12, 1))
    nn = fit_head("nn", p, logits, y, groups, CAL)
    assert nn.method == "temperature"
    assert nn.temperature is not None
    assert nn.temperature > 1
    assert nn.val_nll_after < nn.val_nll_before
    tree = fit_head("tree", p, None, y, groups, CAL)
    assert tree.method in ("isotonic", "platt")
    assert set(tree.cv_nll) == {"isotonic", "platt"}
    base = fit_head("baseline", p, None, y, groups, CAL)
    assert base.method == "none"
    assert np.allclose(base.apply(p), p)
