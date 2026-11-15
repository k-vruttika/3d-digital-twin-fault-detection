"""Phase 8 conformal prediction: quantile, LAC / APS / Mondrian by hand, coverage, TTF."""

from __future__ import annotations

import math

import numpy as np

from twin_ml.conformal.classification import (
    ConformalClassifier,
    aps_scores,
    aps_sets,
    conformal_quantile,
    lac_scores,
    lac_sets,
    mondrian_quantiles,
    set_sums,
)
from twin_ml.conformal.regression import fit_ttf_quantile, interval, interval_sums
from twin_ml.evaluate import run_sums


def test_quantile_rank() -> None:
    s = np.arange(1, 10, dtype=float)  # n = 9, alpha 0.1 -> k = ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s, 0.5) == 5.0  # k = ceil(10 * 0.5) = 5
    assert math.isinf(conformal_quantile(np.arange(3.0), 0.1))  # k = 4 > n
    assert math.isinf(conformal_quantile(np.zeros(0), 0.1))


def test_lac_by_hand() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    assert np.allclose(lac_scores(p, np.array([0, 2])), [0.3, 0.7])
    sets = lac_sets(p, 0.65)  # 1 - p <= 0.65  <=>  p >= 0.35
    assert sets.tolist() == [[True, False, False], [False, True, False]]
    per_class = lac_sets(p, np.array([0.2, 0.9, 0.95]))
    assert per_class.tolist() == [[False, True, True], [False, True, True]]


def test_aps_by_hand() -> None:
    p = np.array([[0.5, 0.3, 0.2]])
    # true class 1: mass above = 0.5, plus u * 0.3
    assert np.allclose(aps_scores(p, np.array([1]), np.array([0.5])), 0.65)
    # u = 0: include classes whose mass above <= q
    assert aps_sets(p, 0.6, np.array([0.0])).tolist() == [[True, True, False]]
    assert aps_sets(p, 0.79, np.array([0.0])).tolist() == [[True, True, False]]
    assert aps_sets(p, 0.8, np.array([0.0])).tolist() == [[True, True, True]]
    # randomised: u = 1 makes the score p_k + mass above
    assert aps_sets(p, 0.4, np.array([1.0])).tolist() == [[False, False, False]]


def _exchangeable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([2, 1, 0.5], size=n)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def test_marginal_and_class_coverage_on_exchangeable_data() -> None:
    p_cal, y_cal = _exchangeable(4000, 0)
    p_te, y_te = _exchangeable(20_000, 1)
    rng = np.random.default_rng(2)
    cc = ConformalClassifier(0.1, ["lac", "aps", "mondrian_lac"], 3).fit(
        p_cal, y_cal, rng.random(4000)
    )
    sets = cc.sets(p_te, rng.random(20_000))
    for m in ("lac", "aps", "mondrian_lac"):
        cov = sets[m][np.arange(20_000), y_te].mean()
        assert cov >= 0.88, (m, cov)  # 1 - alpha minus sampling slack
    mond = sets["mondrian_lac"]
    for k in range(3):
        sel = y_te == k
        assert mond[sel, k].mean() >= 0.87, k
    assert cc.describe()["n_cal"] == 4000


def test_mondrian_quantiles_per_class() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([0, 0, 1, 1])
    q = mondrian_quantiles(scores, y, 3, 0.5)  # n=2 per class, k = ceil(3 * 0.5) = 2
    assert q[0] == 0.2
    assert q[1] == 0.95
    assert math.isinf(q[2])


def test_set_sums_fields() -> None:
    sets = np.array([[True, False, False], [True, True, False], [False, False, False]])
    y = np.array([0, 1, 2])
    s = set_sums(sets, y, 3)
    assert s[:, 0].tolist() == [1, 1, 1]  # n
    assert s[:, 1].tolist() == [1, 1, 0]  # covered
    assert s[:, 2].tolist() == [1, 2, 0]  # size
    assert s[:, 3].tolist() == [1, 0, 0]  # singleton
    assert s[:, 4].tolist() == [0, 0, 1]  # empty


def test_ttf_interval_and_censoring() -> None:
    cap = 600.0
    ttf = np.array([100.0, 200.0, 300.0, 600.0, 50.0, 400.0])
    cens = np.array([False, False, False, False, True, True])
    pred = np.array([110.0, 180.0, 330.0, 590.0, 10.0, 450.0])
    q, n = fit_ttf_quantile(pred, ttf, cens, cap, 0.5)  # rows 0-2: residuals 10, 20, 30
    assert n == 3
    assert q == 20.0
    lo, hi = interval(np.array([5.0, 595.0]), 20.0, cap)
    assert lo.tolist() == [0.0, 575.0]
    assert hi.tolist() == [25.0, 600.0]
    s = interval_sums(pred, ttf, cens, q, cap)
    # known (y < cap): rows 0, 1 covered, row 2 (|30| > 20) not; censored: row 4 hi 30 < 50
    assert s.tolist()[:2] == [3, 2]
    assert s[3] == 2
    assert s[4] == 1


def test_run_sums_metrics_and_bootstraps() -> None:
    rng = np.random.default_rng(0)
    n_runs = 8
    sums = {
        "confusion": rng.integers(0, 50, (n_runs, 3, 3)).astype(float),
        "episodes": np.full(n_runs, 2.0),
        "detected": np.array([2, 2, 1, 2, 2, 0, 2, 2], dtype=float),
        "lead_sum": rng.uniform(300, 600, n_runs),
        "rel_sum": rng.uniform(0, 100, n_runs),
        "rel_n": np.full(n_runs, 1.0),
        "false_alarms": rng.integers(0, 2, n_runs).astype(float),
        "negative_s": np.full(n_runs, 3000.0),
        "conf_lac": rng.integers(1, 20, (n_runs, 3, 5)).astype(float),
    }
    pt = run_sums.point(sums)
    assert abs(pt["miss_rate"] - 3 / 16) < 1e-12
    assert "cov_lac_FAULT" in pt
    assert "size_lac" in pt
    ci = run_sums.bootstrap(sums, 200, 1, 0.95)
    assert ci["miss_rate"]["lo"] <= pt["miss_rate"] <= ci["miss_rate"]["hi"]
    same = run_sums.paired(sums, sums, 100, 1, 0.95, ["miss_rate", "macro_f1"])
    assert same["miss_rate"]["point"] == 0
    assert same["miss_rate"]["lo"] == 0
    assert not same["miss_rate"]["excludes_zero"]
    sub = run_sums.select(sums, [0, 1, 2])
    assert run_sums.n_runs(sub) == 3
    un = run_sums.unpaired(sums, sub, 100, 1, 0.95, ["miss_rate"])
    assert np.isfinite(un["miss_rate"]["point"])
    flat = run_sums.save_npz_dict({"clean|h30": sums})
    assert np.array_equal(run_sums.load_npz_dict(flat)["clean|h30"]["episodes"], sums["episodes"])
