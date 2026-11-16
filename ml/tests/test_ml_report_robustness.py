"""Phase 8 report helpers: seen / unseen classification and seed-wise comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from twin_ml.config import load_robustness_config
from twin_ml.experiments.report_robustness import compare, seen_status, stat
from twin_ml.testing import CONFIG_DIR, eval_config

RCFG = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")


def test_seen_status() -> None:
    assert seen_status(RCFG, "clean") == "clean"
    assert seen_status(RCFG, "noise_20") == "seen"
    assert seen_status(RCFG, "stuck_30") == "seen"
    assert seen_status(RCFG, "sf_seen") == "seen"
    assert seen_status(RCFG, "sf_drift") == "unseen"
    assert seen_status(RCFG, "sf_all") == "mixed"


def _entry(seed: int, shift: float, runs: list[str]) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(runs)

    def sums(extra: float) -> dict[str, np.ndarray]:
        det = np.full(n, 2.0)
        return {
            "confusion": rng.integers(5, 50, (n, 3, 3)).astype(float),
            "episodes": np.full(n, 2.0),
            "detected": det,
            "lead_sum": det * (300.0 + extra),
            "rel_sum": np.full(n, 50.0 + extra),
            "rel_n": np.ones(n),
            "false_alarms": np.zeros(n),
            "negative_s": np.full(n, 3600.0),
        }

    boot = {"lead_mean_s": {"point": 300.0 + shift, "lo": 290.0 + shift, "hi": 310.0 + shift}}
    return {
        "sums": {"a|h30": sums(0.0), "b|h30": sums(shift)},
        "run_ids": {"a": runs, "b": runs},
        "detail": {"conditions": {"b": {"robust": {"30": {"bootstrap": boot}}}}},
    }


def test_compare_paired_unpaired_and_subsets(tmp_path: Any) -> None:
    ev = eval_config(tmp_path / "mlruns", tmp_path / "out")
    runs = [f"run_{i:04d}" for i in range(8)]
    seeds = {s: _entry(s, 20.0, runs) for s in range(3)}
    d = compare(seeds, "b", seeds, "a", "30", ("lead_mean_s", "rel_lead_mean_s"), ev)
    assert d["lead_mean_s"]["point"] == 20.0
    assert d["lead_mean_s"]["seeds"] == 3
    assert d["lead_mean_s"]["seeds_excluding_zero"] == 3
    assert d["lead_mean_s"]["excludes_zero"]
    u = compare(seeds, "b", seeds, "a", "30", ("lead_mean_s",), ev, mode="unpaired")
    assert u["lead_mean_s"]["point"] == 20.0
    sub = compare(
        seeds, "b", seeds, "a", "30", ("lead_mean_s",), ev, a_runs=runs[:3], b_runs=runs[:3]
    )
    assert sub["lead_mean_s"]["point"] == 20.0
    # a deterministic side (one seed) is reused for every seed of the other side
    one = {0: seeds[0]}
    assert compare(seeds, "b", one, "a", "30", ("lead_mean_s",), ev)["lead_mean_s"]["seeds"] == 3
    st = stat(seeds, "b", "30", "lead_mean_s")
    assert st["point"] == 320.0
    assert st["seeds"] == 3


_PLOT_KEYS = (
    "miss_rate",
    "lead_mean_s",
    "false_alarms_per_h",
    "ece_fault_cal",
    "cov_mondrian_lac_FAULT",
    "macro_f1",
)


def _grid_report(
    key: str, prefix: str, studies: dict[str, list[str]], levels: list[float], base: bool = False
) -> dict[str, Any]:
    conds = ["clean", *[f"{prefix}_{p:g}" for p in levels]]
    v = {"point": 0.5, "lo": 0.4, "hi": 0.6}

    def entry(models: list[str], cs: list[str], xs: list[float]) -> dict[str, Any]:
        return {
            "models": models,
            "conditions": cs,
            "x": dict(zip(cs, xs, strict=True)),
            "values": {m: {c: dict.fromkeys(_PLOT_KEYS, v) for c in cs} for m in models},
        }

    out: dict[str, Any] = {}
    for name, models in studies.items():
        out[name] = entry(models, conds, [0.0, *levels])
        if base:
            out[name]["base"] = entry(models, ["clean", "noise_10", "noise_30"], [0.0, 0.1, 0.3])
    return {"plots": {}, key: {"levels": levels, "studies": out}}


def test_plot_noise_grids_partial_studies(tmp_path: Any) -> None:
    """Regression: persistence alone (never drawn) or only the aug study must not crash;
    the std grid draws its base-study points on a log axis."""
    from twin_ml.experiments.report_robustness import plot_snoise, plot_stdnoise

    rcfg = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")
    assert rcfg.signal_noise is not None
    assert rcfg.std_noise is not None
    cases = (
        ("signal_noise", "snoise", plot_snoise, [float(x) for x in rcfg.signal_noise.levels_pct]),
        ("std_noise", "stdnoise", plot_stdnoise, list(rcfg.std_noise.multiples)),
    )
    for key, prefix, plot, levels in cases:
        e1, aug = f"e1models_{prefix}", f"aug_{prefix}"
        for studies in (
            {e1: ["persistence"]},
            {aug: ["persistence", "xgb"]},
            {e1: ["persistence", "xgb"], aug: ["xgb"]},
        ):
            report = _grid_report(key, prefix, studies, levels, base=key == "std_noise")
            paths = plot(report, rcfg, tmp_path)
            assert [p.name for p in paths] == [f"{rcfg.report_name}_{key}.png"]
            assert paths[0].stat().st_size > 0


def _diff(point: float, lo: float, hi: float, seeds_ex: int) -> dict[str, Any]:
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "excludes_zero": lo > 0 or hi < 0,
        "seeds_excluding_zero": seeds_ex,
        "seeds": 3,
    }


def test_fa_onset_across_base_and_grid() -> None:
    """The onset is the lowest level (base grid included) with a paired false-alarm rise in
    at least 2 of 3 seeds; the level below it is the last quiet one."""
    from twin_ml.experiments.report_robustness import _fa_onset

    quiet = {"false_alarms_per_h": _diff(0.0, 0.0, 0.0, 0)}
    weak = {"false_alarms_per_h": _diff(0.2, 0.05, 0.4, 1)}  # CI excludes 0, only 1 seed
    rise = {"false_alarms_per_h": _diff(3.0, 1.0, 5.0, 3)}
    val = {"false_alarms_per_h": {"point": 3.0}, "macro_f1": {"point": 0.6}}
    entry: dict[str, Any] = {
        "models": ["xgb"],
        "x": {"clean": 0.0, "stdnoise_1": 1.0, "stdnoise_2": 2.0},
        "values": {"xgb": {"stdnoise_1": val, "stdnoise_2": val}},
        "degradation": {"xgb": {"stdnoise_1": weak, "stdnoise_2": rise}},
        "base": {
            "x": {"clean": 0.0, "noise_10": 0.1, "noise_30": 0.3},
            "values": {"xgb": {"noise_10": val, "noise_30": val}},
            "degradation": {"xgb": {"noise_10": quiet, "noise_30": quiet}},
        },
    }
    o = _fa_onset(entry)["xgb"]
    assert o["first_rise_x"] == 2.0
    assert o["first_rise_condition"] == "stdnoise_2"
    assert o["last_quiet_x"] == 1.0
    assert o["fa_per_h"] == 3.0
    assert o["macro_f1"] == 0.6
    entry["degradation"]["xgb"]["stdnoise_2"] = quiet
    o = _fa_onset(entry)["xgb"]
    assert o["first_rise_x"] is None
    assert o["last_quiet_x"] == 2.0
