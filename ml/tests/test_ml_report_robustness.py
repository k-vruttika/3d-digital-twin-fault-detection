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
