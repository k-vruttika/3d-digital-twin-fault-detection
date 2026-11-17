"""Phase 10 pre-registration: real.yaml and the dataset configs match ADR 0027."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from twin_common.config.loader import ConfigError
from twin_ml.datasets.config import (
    load_facts,
    load_real_config,
    load_real_dataset_config,
)
from twin_ml.testing import CONFIG_DIR

REAL_PATH = CONFIG_DIR / "models" / "real.yaml"
REAL = load_real_config(REAL_PATH)
ADR = Path(CONFIG_DIR).parent / "docs" / "adr" / "0027-real-data-protocol.md"


def _adr() -> str:
    return ADR.read_text(encoding="utf-8")


def _section() -> str:
    return _adr().split("## Pre-registration", 1)[1].split("## Consequences", 1)[0]


def test_preregistration_matches_adr_0027() -> None:
    section = _section()
    az, mp = REAL.primary.azure_pdm, REAL.primary.metropt3
    assert (az.horizon, az.lookback) == (24, 48)
    assert "horizon +24 h, lookback 48 h" in section
    assert az.outcomes == ["detection_rate", "lead_mean_h", "false_alarms_per_day"]
    assert az.groups == [
        "best_supervised",
        "autoencoder",
        "combined",
        "threshold",
        "zscore",
        "persistence",
    ]
    assert az.reference == "threshold"
    assert "paired against the threshold baseline" in section
    assert az.combined_vs == "best_supervised"
    assert "also paired against the best supervised model alone" in section
    assert mp.lookback == 360
    assert "lookback 6 h" in section
    assert mp.outcomes == [
        "warned",
        "lead_min",
        "warned_count",
        "false_alarms_per_day",
        "chance_expected",
    ]
    assert mp.groups == ["autoencoder", "iforest", "threshold", "zscore"]
    for name in {*az.outcomes, *az.groups, *mp.outcomes, *mp.groups}:
        assert f"`{name}`" in section, name
    assert az.anomaly_percentile == mp.anomaly_percentile == 99.9
    assert "99.9th percentile" in section
    assert mp.evaluable_min_coverage == 0.5
    assert "at least 50 % of the lookback has data" in section
    assert "no supervised model and no combined policy for MetroPT-3" in section
    assert "Everything else is exploratory" in section
    assert "before any model was trained on real data" in _adr()


def test_design_fixed_in_advance() -> None:
    adr = _adr()
    mp, az = REAL.datasets["metropt3"], REAL.datasets["azure_pdm"]
    assert mp.supervised == []
    assert az.supervised == ["rf", "xgb", "lstm", "gru", "transformer"]
    assert mp.lookbacks == [120, 360, 1440]
    assert (mp.sustain, az.sustain, az.horizon) == (3, 3, 24)
    assert az.lookbacks == [48]
    assert (mp.zscore_trailing, az.zscore_trailing) == (1440, 168)
    assert REAL.seeds == [0, 1, 2]
    assert REAL.anomaly.threshold_percentiles == [99.0, 99.5, 99.9]
    assert REAL.baselines.threshold.low_pct == 0.1
    assert REAL.baselines.threshold.high_pct == 99.9
    assert REAL.baselines.zscore.z == 3.0
    assert REAL.selection.metric == "pr_auc_mean"
    assert REAL.selection.split == "val"
    assert "**No setting is tuned on test results.**" in adr
    assert REAL.combined.primary == ["best_supervised", "autoencoder"]
    assert REAL.bootstrap.n_resamples == 1000
    # exploratory variants fixed before any result (plan review additions)
    ho = REAL.exploratory.holdout
    assert (ho.fraction, ho.seed) == (0.2, 2027)
    hcfg = load_real_dataset_config(CONFIG_DIR.parent / ho.dataset_config)
    assert hcfg.machine_holdout is not None
    assert (hcfg.machine_holdout.fraction, hcfg.machine_holdout.seed) == (ho.fraction, ho.seed)
    assert "seed 2027" in adr
    assert REAL.exploratory.fa_by_month == ["metropt3"]
    assert REAL.exploratory.error_xgb.model == "xgb"
    assert "same machines (temporal split)" in adr


def test_dataset_configs_fixed_in_advance() -> None:
    m = load_real_dataset_config(CONFIG_DIR / "models" / "real_metropt3.yaml")
    a = load_real_dataset_config(CONFIG_DIR / "models" / "real_azure.yaml")
    assert (m.step_s, m.window, m.max_gap_rows) == (60, 60, 10)
    assert (m.labels.warning_band, m.labels.fault_context, m.labels.post_failure) == (
        360,
        1440,
        1440,
    )
    assert m.split.test == ("2020-04-11 00:00:00", "2020-09-02 00:00:00")
    assert (a.step_s, a.window, a.horizons, a.ttf_cap) == (3600, 24, [12, 24], 168)
    assert (a.labels.warning_band, a.labels.fault_context) == (24, 48)
    assert a.split.train == ("2015-01-01 00:00:00", "2015-09-01 00:00:00")
    assert a.normal_keep_frac == 0.1
    assert not a.error_channels
    # the error-count and holdout variants differ from real_azure.yaml only where intended
    for name, field in (
        ("real_azure_err", "error_channels"),
        ("real_azure_holdout", "machine_holdout"),
    ):
        v = load_real_dataset_config(CONFIG_DIR / "models" / f"{name}.yaml")
        keep = set(type(v).model_fields) - {field}
        assert v.model_dump(include=keep) == a.model_dump(include=keep), name


def test_metropt_failure_list_and_source() -> None:
    facts = load_facts(CONFIG_DIR / "datasets" / "metropt3.yaml")
    assert [(f.start, f.end) for f in facts.failures] == [
        ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
        ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
        ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
        ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
    ]
    assert {f.type for f in facts.failures} == {"air_leak"}
    assert "Data Description_Metro.pdf" in facts.failure_source
    assert facts.pdf_check is not None
    assert len(facts.pdf_check.inconsistencies) == 3
    azure = load_facts(CONFIG_DIR / "datasets" / "azure_pdm.yaml")
    assert "synthetic" in azure.origin


def test_real_config_rejects_inconsistent_primary(tmp_path: Path) -> None:
    data = yaml.safe_load(REAL_PATH.read_text(encoding="utf-8"))
    data["primary"]["metropt3"]["lookback"] = 120
    bad = tmp_path / "real.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="primary"):
        load_real_config(bad)
