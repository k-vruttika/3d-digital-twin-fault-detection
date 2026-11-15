"""Phase 8 perturbations: determinism, levels, segments, labels untouched, rebuilt splits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from twin_common.enums import MachineKind, MachineState
from twin_ml.config import DatasetConfig, PerturbationsConfig, load_robustness_config
from twin_ml.data.dataset import Dataset
from twin_ml.data.labels import STATES
from twin_ml.data.perturb import (
    CLEAN,
    Condition,
    add_noise,
    drop_mcar,
    drop_segments,
    grid_conditions,
    normal_std,
    parse_condition,
    perturb_array,
    perturb_series,
    series_rng,
    stick_segments,
)
from twin_ml.features.inputs import KINDS
from twin_ml.testing import CONFIG_DIR

RCFG = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")
PCFG: PerturbationsConfig = RCFG.perturbations


@dataclass(frozen=True)
class FakeSeries:
    run_id: str
    machine_id: str
    kind: MachineKind
    sensors: np.ndarray
    state: list[str] = field(default_factory=list)


def _raw(n: int = 3600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(50.0, 5.0, size=(n, 5))).astype(np.float32)


def test_grid_names_and_parse() -> None:
    names = [c.name for c in grid_conditions(PCFG)]
    assert names[:3] == ["noise_10", "noise_20", "noise_30"]
    assert "mcar_20" in names
    assert "dropout_30" in names
    assert "stuck_5" in names
    c = parse_condition("mcar_20", PCFG)
    assert (c.kind, c.level, c.level_pct) == ("mcar", 0.2, 20)
    assert parse_condition(CLEAN, PCFG).kind is None
    with pytest.raises(KeyError):
        parse_condition("noise_15", PCFG)


def test_clean_is_identity_and_copy() -> None:
    raw = _raw(100)
    out = perturb_array(
        raw, Condition(CLEAN, None, 0.0), PCFG, np.ones(5), np.random.default_rng(1)
    )
    assert np.array_equal(out, raw)
    assert out is not raw


def test_same_seed_same_output_and_order_independent() -> None:
    sigma = np.ones((len(KINDS), 5), dtype=np.float32)
    series = [FakeSeries(f"run_{i:04d}", "CNC-01", MachineKind.CNC, _raw(500, i)) for i in range(3)]
    for name in ("noise_20", "mcar_10", "dropout_20", "stuck_20"):
        cond = parse_condition(name, PCFG)
        a = perturb_series(series, cond, PCFG, sigma, seed=5)
        b = perturb_series(series[::-1], cond, PCFG, sigma, seed=5)[::-1]
        c = perturb_series(series, cond, PCFG, sigma, seed=6)
        for x, y, z in zip(a, b, c, strict=True):
            assert np.array_equal(x.sensors, y.sensors, equal_nan=True)
            assert not np.array_equal(x.sensors, z.sensors, equal_nan=True)


def test_rng_depends_on_condition_run_and_machine() -> None:
    draws = {
        k: series_rng(1, *k).random()
        for k in [("a", "r1", "m1"), ("b", "r1", "m1"), ("a", "r2", "m1"), ("a", "r1", "m2")]
    }
    assert len(set(draws.values())) == 4


def test_noise_scale_and_nan_kept() -> None:
    raw = np.zeros((200_000, 5), dtype=np.float32)
    raw[0, 0] = np.nan
    sigma = np.array([1.0, 2.0, 0.5, 4.0, 3.0])
    out = add_noise(raw, 0.2 * sigma, np.random.default_rng(0))
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[1:]).all()
    np.testing.assert_allclose(np.nanstd(out, axis=0), 0.2 * sigma, rtol=0.02)


def test_mcar_rate() -> None:
    out = drop_mcar(_raw(100_000), 0.2, np.random.default_rng(0))
    assert abs(np.isnan(out).mean() - 0.2) < 0.005


@pytest.mark.parametrize("fraction", [0.05, 0.2, 0.3])
def test_dropout_coverage_and_segment_lengths(fraction: float) -> None:
    raw = _raw()
    out = drop_segments(raw, fraction, (10, 60), np.random.default_rng(3))
    miss = np.isnan(out)
    target = round(fraction * raw.size)
    assert target <= miss.sum() <= target + 60  # the last segment may overshoot
    for s in range(5):  # every gap lies in single-sensor runs; shortest is >= 10 unless merged
        col = miss[:, s].astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], col, [0]])))
        lengths = edges[1::2] - edges[::2]
        assert (lengths >= 10).all()


def test_stuck_holds_previous_value() -> None:
    raw = _raw(3600)
    out = stick_segments(raw, 0.2, (30, 300), np.random.default_rng(4))
    changed = out != raw
    assert 0.15 * raw.size < changed.sum() <= 0.2 * raw.size + 300
    assert not np.isnan(out).any()
    for s in range(5):
        col, orig = out[:, s], raw[:, s]
        # every changed value equals some earlier observed value of the same column
        idx = np.flatnonzero(col != orig)
        for i in idx[:50]:
            assert col[i] in set(orig[:i].tolist()) | set(col[:i].tolist())


def test_labels_untouched() -> None:
    s = FakeSeries("run_0001", "CNV-01", MachineKind.CONVEYOR, _raw(400), ["RUNNING"] * 400)
    (p,) = perturb_series([s], parse_condition("mcar_30", PCFG), PCFG, np.ones((3, 5)), 1)
    assert p.state == s.state
    assert p.run_id == s.run_id
    assert p.kind == s.kind
    assert np.isnan(p.sensors).any()
    assert not np.isnan(s.sensors).any()


def test_normal_std_per_kind_and_state() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    kind = np.repeat([0, 1], n // 2)
    raw = np.where(kind[:, None] == 0, rng.normal(0, 1, (n, 5)), rng.normal(0, 3, (n, 5)))
    running, fault = STATES.index(MachineState.RUNNING), STATES.index(MachineState.FAULT)
    state = np.full(n, running)
    state[:100] = fault  # excluded rows with huge values
    raw[:100] = 1e6
    ft = np.zeros(n, dtype=int)
    out = normal_std(raw.astype(np.float32), state, kind, ft, 0, [MachineState.RUNNING])
    np.testing.assert_allclose(out[0], 1.0, rtol=0.05)
    np.testing.assert_allclose(out[1], 3.0, rtol=0.05)
    assert out.shape == (len(KINDS), 5)
    # kind 2 has no rows: pooled std
    assert np.all((out[2] > 1.0) & (out[2] < 3.0))


def test_augmentation_grid_copies_robustness_grid() -> None:
    """dataset_aug.yaml copies the test grid (ADR 0021); they must stay identical."""
    from twin_ml.config import load_dataset_config

    path = CONFIG_DIR / "models" / "dataset_aug.yaml"
    if not path.is_file():
        pytest.skip("dataset_aug.yaml arrives with Part C")
    cfg = load_dataset_config(path)
    assert cfg.augmentation is not None
    assert cfg.augmentation.perturbations == PCFG


def test_preregistration_matches_adr() -> None:
    """The pre-registered primary outcomes in robustness.yaml equal those named in ADR 0021."""
    adr = (Path(CONFIG_DIR).parent / "docs" / "adr" / "0021-robustness-protocol.md").read_text(
        encoding="utf-8"
    )
    section = adr.split("## Pre-registration", 1)[1].split("## Consequences", 1)[0]
    p = RCFG.primary
    assert p.horizon == 30
    assert "+30 s" in section
    assert p.reference == "threshold"
    assert "threshold baseline" in section
    for name in [*p.early_warning, *p.calibration, *p.conformal, *p.conditions]:
        assert f"`{name}`" in section or name in ("miss_rate", "false_alarms_per_h"), name
    assert "miss rate" in section
    assert "false alarms per hour" in section
    assert p.early_warning == ["miss_rate", "rel_lead_mean_s", "false_alarms_per_h"]
    assert p.calibration == ["ece_cal"]
    assert p.conformal == ["cov_lac_FAULT", "cov_aps_FAULT", "cov_mondrian_lac_FAULT"]
    assert p.conditions == ["noise_20", "mcar_20", "sf_all", "e4_high"]


def test_perturbed_test_split_rebuild(ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig) -> None:
    """Clean rebuild equals the stored test split; MCAR shows up in masks, labels unchanged."""
    from twin_ml.data.test_sets import load_series, perturbed_split, windowed

    cfg = ml_dataset_cfg
    series = load_series(cfg.source_dir, ml_dataset.meta["splits"]["test"], cfg.line_config)
    stored = ml_dataset.split("test")
    rebuilt = windowed(ml_dataset, series)
    for k in ("values", "mask", "ends", "health", "fault_type", "ttf", "ep_fault"):
        assert np.array_equal(stored.arrays[k], rebuilt.arrays[k]), k
    sigma = np.ones((len(KINDS), 5), dtype=np.float32)
    pert = perturbed_split(ml_dataset, series, parse_condition("mcar_20", PCFG), PCFG, sigma, 1)
    assert abs(pert.arrays["mask"].mean() - 0.2) < 0.01
    assert not np.isnan(pert.arrays["values"]).any()  # forward fill as in live use
    for k in ("ends", "health", "state", "fault_type", "ttf", "ttf_censored", "ep_start"):
        assert np.array_equal(stored.arrays[k], pert.arrays[k]), k
    # forward fill: a missing value repeats the last observed (normalised) value
    m = pert.arrays["mask"][:, 1].astype(bool)
    i = int(np.flatnonzero(m[1:])[0]) + 1
    assert pert.arrays["values"][i, 1] == pert.arrays["values"][i - 1, 1]


def test_series_replace_keeps_frozen_dataclass() -> None:
    s = FakeSeries("r", "m", MachineKind.CNC, _raw(10))
    t = replace(s, sensors=s.sensors * 2)
    assert t.run_id == "r"
    assert np.allclose(t.sensors, s.sensors * 2)
