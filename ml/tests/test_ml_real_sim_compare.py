"""Phase 10 report: simulator vs real-data comparison table (exploratory), from synthetic inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from twin_ml.experiments import report_real_results as rr


def _ci(p: float) -> dict[str, float]:
    return {"point": p, "lo": p, "hi": p}


def _write_sim(d: Path) -> None:
    e1 = {
        "models": [
            {
                "model": "xgb",
                "metrics": {
                    "h30.ew.miss_rate": {"mean": 0.1},
                    "h30.ew.false_alarms_per_h": {"mean": 0.5},
                },
            },
            {
                "model": "threshold",
                "metrics": {
                    "h30.ew.miss_rate": {"mean": 0.2},
                    "h30.ew.false_alarms_per_h": {"mean": 0.0},
                },
            },
        ]
    }
    e5 = {
        "primary_config": {"horizon": 30, "anomaly_percentile": 99.9},
        "primary": {
            "early_warning": [
                {
                    "kind": "reference",
                    "label": "Threshold rule",
                    "values": {"miss_rate": _ci(0.04)},
                },
                {"kind": "supervised", "label": "XGBoost", "values": {"miss_rate": _ci(0.25)}},
                {
                    "kind": "detector",
                    "label": "Autoencoder (p99.9)",
                    "values": {"miss_rate": _ci(0.0)},
                },
                {
                    "kind": "detector",
                    "label": "Isolation Forest (p99.9)",
                    "values": {"miss_rate": _ci(0.99)},
                },
            ]
        },
        "exploratory": {
            "tradeoff": [
                {
                    "label": "Autoencoder",
                    "percentile": 99.9,
                    "clean_miss_rate": _ci(0.08),
                    "fa_clean": _ci(0.5),
                },
                {
                    "label": "Autoencoder",
                    "percentile": 99.0,
                    "clean_miss_rate": _ci(0.0),
                    "fa_clean": _ci(9.0),
                },
                {
                    "label": "Isolation Forest",
                    "percentile": 99.9,
                    "clean_miss_rate": _ci(0.97),
                    "fa_clean": _ci(0.25),
                },
            ]
        },
    }
    (d / "E1_clean.json").write_text(json.dumps(e1), encoding="utf-8")
    (d / "e5_explain.json").write_text(json.dumps(e5), encoding="utf-8")


def _results() -> dict[str, Any]:
    def mean(det: float, fa: float) -> dict[str, Any]:
        return {"mean": {"detection_rate": _ci(det), "false_alarms_per_day": _ci(fa)}}

    pts = {
        "warned_count": 1.0,
        "evaluable": 4.0,
        "chance_expected": 0.5,
        "false_alarms_per_day": 2.0,
    }
    return {
        "azure": {
            "best_supervised": "xgb",
            "primary": {
                "best_supervised": mean(0.6, 0.1),
                "autoencoder": mean(0.3, 0.2),
                "threshold": mean(0.1, 0.0),
            },
            "exploratory": {
                "methods": {
                    "iforest|p99.9": {"detection_rate": _ci(0.2), "false_alarms_per_day": _ci(0.3)}
                }
            },
        },
        "metropt3": {"primary": {"autoencoder": {"seeds": {"0": {"points": pts}}}}},
    }


def test_sim_comparison_values(tmp_path: Path) -> None:
    _write_sim(tmp_path)
    c = rr.compute_sim_comparison(_results(), tmp_path)
    assert c is not None
    rows = {r["method"]: r for r in c["rows"]}
    assert list(rows) == ["xgb", "autoencoder", "iforest", "threshold"]
    assert rows["xgb"]["group"] == "best_supervised"
    assert rows["xgb"]["sim_clean_detection"] == pytest.approx(0.9)
    assert rows["xgb"]["sim_clean_fa_day"] == pytest.approx(12.0)
    assert rows["xgb"]["sim_e5_detection"] == pytest.approx(0.75)
    assert rows["xgb"]["azure_detection"] == pytest.approx(0.6)
    assert rows["autoencoder"]["sim_clean_detection"] == pytest.approx(0.92)  # p99.9 row only
    assert rows["autoencoder"]["sim_clean_fa_day"] == pytest.approx(12.0)
    assert rows["autoencoder"]["metropt3_warned"] == 1.0
    assert rows["iforest"]["azure_fa_day"] == pytest.approx(0.3)  # from the exploratory table
    assert rows["threshold"]["sim_e5_detection"] == pytest.approx(0.96)
    assert "metropt3_warned" not in rows["xgb"]
    text = "\n".join(rr.render_sim_comparison(c))
    assert "## Simulator vs real data (exploratory)" in text
    assert "`xgb` (best supervised)" in text
    assert "1.00 / 4 (chance 0.50)" in text


def test_sim_comparison_missing_files(tmp_path: Path) -> None:
    assert rr.compute_sim_comparison(_results(), tmp_path) is None


def test_sim_comparison_without_metropt(tmp_path: Path) -> None:
    _write_sim(tmp_path)
    res = _results()
    res["metropt3"] = None
    c = rr.compute_sim_comparison(res, tmp_path)
    assert c is not None
    assert all("metropt3_warned" not in r for r in c["rows"])
    assert math.isfinite(c["rows"][0]["azure_detection"])
