"""Phase 9 pre-registration: e5.yaml matches ADR 0024, and the fixed E5 design is intact."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin_common.config.loader import ConfigError
from twin_ml.config import (
    AutoencoderConfig,
    IForestConfig,
    load_anomaly_config,
    load_e5_config,
    load_robustness_config,
)
from twin_ml.testing import CONFIG_DIR

E5 = load_e5_config(CONFIG_DIR / "models" / "e5.yaml")
ADR = Path(CONFIG_DIR).parent / "docs" / "adr" / "0024-e5-unseen-fault-protocol.md"


def _section() -> str:
    text = ADR.read_text(encoding="utf-8")
    return text.split("## Pre-registration", 1)[1].split("## Consequences", 1)[0]


def test_preregistration_matches_adr_0024() -> None:
    """The primary outcomes in e5.yaml equal those named in ADR 0024 (never changed later)."""
    section = _section()
    p = E5.primary
    assert p.horizon == 30
    assert "+30 s" in section
    assert p.reference == "threshold"
    assert "threshold baseline" in section
    assert p.early_warning == {
        "miss_rate": "e5",
        "rel_lead_mean_s": "e5",
        "false_alarms_per_h": "clean",
    }
    for name, cond in p.early_warning.items():
        assert f"`{name}`, on `{cond}`" in section or f"`{name}`, on the clean" in section, name
    assert p.groups == ["e5_e1models", "e5_aug", "anomaly", "combined"]
    for name in ("e5_e1models", "e5_aug"):
        assert f"`{name}`" in section
    assert p.anomaly_percentile == 99.9
    assert "99.9th percentile" in section
    assert p.conformal == ["size_mondrian_lac"]
    assert p.conformal_conditions == ["e5", "clean"]
    assert "`size_mondrian_lac`" in section
    assert p.explanation == ["agreement_rate"]
    assert "`agreement_rate`" in section
    assert p.explanation_models == ["xgb", "lstm", "gru", "transformer"]
    assert "XGBoost (TreeSHAP)" in section
    assert "Integrated Gradients" in section
    assert "Everything else is exploratory" in section
    # The interpretation note about the threshold rule's overcurrent check is part of it.
    assert "overcurrent" in section
    assert "current-led" in section


def test_e5_design_fixed_in_advance() -> None:
    """Run count, seeds, percentiles and explanation targets as recorded in ADR 0024."""
    d = E5.e5_data
    assert (d.base_seed, d.generate_runs, d.n_runs) == (42, 270, 100)
    assert d.extra_run_indices == (200, 270)
    assert d.test_split_runs
    assert d.fault_type == "lubrication_loss"
    assert d.scenario == Path("config/scenarios/e5_lubrication_loss.yaml")
    assert E5.anomaly.threshold_percentiles == [99.0, 99.5, 99.9]
    assert E5.anomaly.per_kind
    assert E5.anomaly.seeds == [0, 1, 2]
    assert E5.combined.percentiles == [99.0, 99.5, 99.9]
    assert E5.explain.ig_steps == 64
    assert E5.explain.exploratory_models == ["rf"]
    assert E5.explain.expected_sensor == {
        "bearing_wear": "vib",
        "overheating": "temp",
        "spindle_imbalance": "vib",
        "lubrication_loss": "current",
    }
    adr = ADR.read_text(encoding="utf-8")
    phrases = (
        "Runs: 100",
        "70 fresh run indices, 200",
        "--seed 42",
        "64, trapezoid",
        "not Shapley values",
    )
    for phrase in phrases:
        assert phrase in adr, phrase


def test_e5_studies_and_models_load() -> None:
    assert set(E5.studies) == {"e5_e1models", "e5_aug", "e5_anomaly", "e5_e4"}
    for name, path in E5.anomaly.models.items():
        cfg = load_anomaly_config(CONFIG_DIR.parent / path)
        assert cfg.type == name
        assert isinstance(cfg, IForestConfig | AutoencoderConfig)
    # The reused calibration / conformal settings exist and load.
    load_robustness_config(CONFIG_DIR.parent / E5.robustness_config)


def test_e5_config_rejects_inconsistent_percentile(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "models" / "e5.yaml").read_text(encoding="utf-8")
    bad = text.replace("primary_percentile: 99.9", "primary_percentile: 99.5", 1)
    assert bad != text
    p = tmp_path / "e5.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="anomaly_percentile"):
        load_e5_config(p)
