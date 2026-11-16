"""Exploratory signal-relative noise (Phase 8 Part C): fault-signal scale, conditions and
config hashes of extension studies."""

from __future__ import annotations

import numpy as np

from twin_ml.config import load_eval_config, load_model_config, load_robustness_config
from twin_ml.data.perturb import (
    EpisodeRise,
    episode_rises,
    leading_sensors,
    signal_noise_conditions,
    signal_scale,
)
from twin_ml.experiments.robust_worker import expand_conditions, study_config_sha
from twin_ml.testing import CONFIG_DIR

RCFG = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")


def test_episode_rises_window_means() -> None:
    n = 200
    raw = np.zeros((n, 5), dtype=np.float32)
    raw[100:, 2] = np.linspace(0.0, 10.0, 100)  # vib ramps up before the fault at row 199
    t = np.arange(n, dtype=np.float64)
    rises = episode_rises(
        raw, np.zeros(n, dtype=np.int8), t, np.array([50]), np.array([199]), np.array([1]), 30
    )
    (r,) = rises
    expected = raw[169:199, 2].mean() - raw[50:80, 2].mean()
    assert abs(float(r.rise[2]) - expected) < 1e-5
    assert r.precursor_s == 149.0
    # too short a precursor: skipped
    assert (
        episode_rises(raw, np.zeros(n, np.int8), t, np.array([5]), np.array([6]), np.array([1]), 30)
        == []
    )


def test_leading_sensor_and_scale() -> None:
    sigma = np.ones((3, 5))
    sigma[0, 1] = 10.0  # temp has a large normal std
    rises = [
        EpisodeRise(0, 1, np.array([0, 0, 4.0, 0, 0.5], np.float32), 600.0),  # vib leads
        EpisodeRise(0, 1, np.array([0, 0, 6.0, 0, 0.5], np.float32), 700.0),
        EpisodeRise(0, 2, np.array([0, 30.0, 0.5, 0, 1.0], np.float32), 500.0),  # temp leads
    ]
    lead = leading_sensors(rises, sigma)
    assert lead == {(0, 1): 2, (0, 2): 1}
    scale = signal_scale(rises, sigma)
    assert scale[0, 2] == 5.0  # median over the episodes vib leads
    assert scale[0, 1] == 30.0  # the episode temp leads
    assert abs(scale[0, 4] - 0.5) < 1e-6  # leads nothing: median over all episodes
    assert np.allclose(scale[1], sigma[1])  # no episodes for this kind: normal std


def test_signal_noise_conditions_and_expansion() -> None:
    assert RCFG.signal_noise is not None
    conds = signal_noise_conditions(RCFG.signal_noise.levels_pct)
    assert [c.name for c in conds] == ["snoise_10", "snoise_25", "snoise_50", "snoise_100"]
    assert all(c.kind == "noise" for c in conds)
    assert conds[-1].level == 1.0
    names = expand_conditions(RCFG.studies["e1models_snoise"], RCFG)
    assert names == ["clean", "snoise_10", "snoise_25", "snoise_50", "snoise_100"]
    assert RCFG.studies["e1models_snoise"].extension
    assert not RCFG.studies["e1models"].extension


def test_extension_studies_keep_earlier_hashes() -> None:
    """Removing the extension studies and the signal-noise grid leaves every base study's hash
    unchanged; an extension study's hash depends on the grid."""
    ev = load_eval_config(CONFIG_DIR / "models" / "eval.yaml")
    cfg = load_model_config(CONFIG_DIR / "models" / "xgb.yaml")
    base = RCFG.model_copy(
        update={
            "signal_noise": None,
            "studies": {k: v for k, v in RCFG.studies.items() if not v.extension},
        }
    )
    for study in ("e1models", "e4", "aug"):
        assert study_config_sha(study, cfg, RCFG, ev, "k", "src") == study_config_sha(
            study, cfg, base, ev, "k", "src"
        )
    assert RCFG.signal_noise is not None
    grid = RCFG.signal_noise.model_copy(update={"levels_pct": [10]})
    other = RCFG.model_copy(update={"signal_noise": grid})
    assert study_config_sha("e1models_snoise", cfg, RCFG, ev, "k", "src") != study_config_sha(
        "e1models_snoise", cfg, other, ev, "k", "src"
    )


def test_std_noise_conditions_and_expansion() -> None:
    from twin_ml.data.perturb import std_noise_conditions

    assert RCFG.std_noise is not None
    conds = std_noise_conditions(RCFG.std_noise.multiples)
    assert [c.name for c in conds] == ["stdnoise_1", "stdnoise_2", "stdnoise_4", "stdnoise_8"]
    assert all(c.kind == "noise" for c in conds)
    assert [c.level for c in conds] == [1.0, 2.0, 4.0, 8.0]  # sigma = level x normal std
    assert std_noise_conditions([0.5])[0].name == "stdnoise_0.5"
    for study in ("e1models_stdnoise", "aug_stdnoise"):
        assert RCFG.studies[study].extension
        names = expand_conditions(RCFG.studies[study], RCFG)
        assert names == ["clean", "stdnoise_1", "stdnoise_2", "stdnoise_4", "stdnoise_8"]


def test_std_noise_grid_keeps_earlier_hashes() -> None:
    """Adding the std-noise grid and its studies leaves every earlier study's hash unchanged,
    including the signal-noise extension studies; the std-noise studies depend on the grid."""
    ev = load_eval_config(CONFIG_DIR / "models" / "eval.yaml")
    cfg = load_model_config(CONFIG_DIR / "models" / "xgb.yaml")
    std_studies = {"e1models_stdnoise", "aug_stdnoise"}
    before = RCFG.model_copy(
        update={
            "std_noise": None,
            "studies": {k: v for k, v in RCFG.studies.items() if k not in std_studies},
        }
    )
    for study in ("e1models", "e4", "aug", "e1models_snoise", "aug_snoise"):
        assert study_config_sha(study, cfg, RCFG, ev, "k", "src") == study_config_sha(
            study, cfg, before, ev, "k", "src"
        )
    assert RCFG.std_noise is not None
    other = RCFG.model_copy(
        update={"std_noise": RCFG.std_noise.model_copy(update={"multiples": [1.0]})}
    )
    assert study_config_sha("e1models_stdnoise", cfg, RCFG, ev, "k", "src") != study_config_sha(
        "e1models_stdnoise", cfg, other, ev, "k", "src"
    )
