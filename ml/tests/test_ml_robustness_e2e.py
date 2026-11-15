"""Phase 8 end to end on quick-scenario data: E1 runs -> e1models study runs (worker
processes, tmp MLflow store) -> resume -> report numbers equal the MLflow metrics.

Also checks that a sensor-fault regeneration with the same seed keeps every state label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twin_ml.config import (
    DatasetConfig,
    EvalConfig,
    RobustnessConfig,
    load_robustness_config,
)
from twin_ml.data.dataset import Dataset
from twin_ml.testing import CONFIG_DIR, generate_sim, require_xgboost, tiny_config
from twin_ml.train import tracking
from twin_ml.train.isolated import run_isolated

SENSOR_FAULTS: dict[str, Any] = {
    "random": {
        "rate_per_machine_per_hour": 4.0,
        "types": ["stuck", "drift", "dropout"],
        "earliest_onset_s": 60,
    }
}
MODELS = ("persistence", "threshold", "rf", "xgb", "gru")


@pytest.fixture(scope="module")
def sf_dir(ml_sim_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from twin_ml.testing import ML_SIM_DURATION_S, ML_SIM_RUNS

    return generate_sim(
        tmp_path_factory.mktemp("ml_sim_sf"),
        ML_SIM_RUNS,
        ML_SIM_DURATION_S,
        sensor_faults=SENSOR_FAULTS,
    )


def test_sensor_fault_runs_keep_labels(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, sf_dir: Path
) -> None:
    from twin_ml.data.test_sets import load_series, sensor_fault_split

    clean = load_series(
        ml_dataset_cfg.source_dir, ml_dataset.meta["splits"]["test"], ml_dataset_cfg.line_config
    )
    split, rows = sensor_fault_split(ml_dataset, sf_dir, clean, ml_dataset_cfg.line_config)
    stored = ml_dataset.split("test")
    for k in ("ends", "health", "state", "fault_type", "ttf", "ep_fault"):
        assert np.array_equal(stored.arrays[k], split.arrays[k]), k
    assert rows.any()
    assert rows.shape[0] == split.n_rows
    assert not np.array_equal(stored.arrays["values"], split.arrays["values"])


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, default=str), encoding="utf-8")  # JSON is valid YAML
    return path


def _study_config(tmp: Path, ds_cfg_path: Path, sf_dir: Path, eval_cfg: EvalConfig) -> Path:
    base = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")
    data = base.model_dump(mode="json")
    data["experiment"] = "P8_test"
    data["source_experiment"] = eval_cfg.experiment
    data["progress_dir"] = str(tmp / "progress")
    data["sensor_fault_sets"] = {
        "sf_all": {
            "source_dir": str(sf_dir),
            "scenario": "config/scenarios/sf_all.yaml",
            "types": ["stuck", "drift", "dropout"],
        }
    }
    data["primary"]["conditions"] = ["noise_20", "mcar_20", "sf_all", "e4_high"]
    data["studies"] = {
        "e1models": {
            "models_from": "e1",
            "dataset_config": str(ds_cfg_path),
            "experiment": eval_cfg.experiment,
            "conditions": ["clean", "noise_20", "mcar_20", "stuck_20", "sf_all"],
        }
    }
    data["reliability_conditions"] = ["clean", "noise_20"]
    RobustnessConfig.model_validate(data)
    return _write(tmp / "robustness.yaml", data)


def test_e1models_study_resume_and_report(
    ml_dataset: Dataset,
    ml_dataset_cfg: DatasetConfig,
    ml_eval_cfg: EvalConfig,
    sf_dir: Path,
    tmp_path: Path,
) -> None:
    require_xgboost()
    from twin_ml.experiments import robustness
    from twin_ml.experiments.report_robustness import build_report

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for m in MODELS:
        tiny_config(m, models_dir)
        res = run_isolated(m, tiny_config(m, models_dir), 0, ml_dataset.path, ml_eval_cfg)
        assert res.run_id
    ds_path = _write(tmp_path / "dataset.yaml", ml_dataset_cfg.model_dump(mode="json"))
    eval_path = _write(tmp_path / "eval.yaml", ml_eval_cfg.model_dump(mode="json"))
    rcfg_path = _study_config(tmp_path, ds_path, sf_dir, ml_eval_cfg)
    argv = [
        "--study",
        "e1models",
        "--models",
        ",".join(MODELS),
        "--seeds",
        "0",
        "--robustness-config",
        str(rcfg_path),
        "--eval-config",
        str(eval_path),
        "--models-dir",
        str(models_dir),
        "--skip-existing",
    ]
    assert robustness.main(argv) == 0
    rcfg = load_robustness_config(rcfg_path)
    runs = tracking.finished_runs(
        ml_eval_cfg.tracking_dir, "P8_test", None, rcfg.phase_tag, "tags.study = 'e1models'"
    )
    assert sorted(r.data.tags["model"] for r in runs) == sorted(MODELS)
    for r in runs:
        tags, metrics = r.data.tags, r.data.metrics
        assert tags["source_run_id"]
        assert tags["study_config_sha"]
        for key in ("clean.h30.macro_f1", "noise_20.h30.ece_cal", "sf_all.h30.cov_aps"):
            assert key in metrics or tags["model"] == "persistence", key
        assert "sf_all.h30.sf_negative_h" in metrics
    import mlflow

    detail = {
        r.data.tags["model"]: mlflow.artifacts.load_dict(f"{r.info.artifact_uri}/metrics.json")
        for r in runs
    }
    for m in ("rf", "xgb", "gru"):  # reloaded / retrained models reproduce E1 exactly
        chk = detail[m]["prediction_check"]
        assert all(v["labels_identical"] for v in chk.values()), m
    assert detail["gru"]["calibration"]["heads"]["30"]["method"] == "temperature"
    assert detail["xgb"]["calibration"]["heads"]["30"]["method"] in ("isotonic", "platt")
    # persistence ignores the sensors: perturbations change nothing
    pm = {r.data.tags["model"]: r.data.metrics for r in runs}["persistence"]
    assert pm["clean.h30.macro_f1"] == pm["mcar_20.h30.macro_f1"]

    # resume: nothing left to run
    assert robustness.main(argv) == 0
    again = tracking.finished_runs(ml_eval_cfg.tracking_dir, "P8_test", None, rcfg.phase_tag)
    assert len(again) == len(runs)

    out = tmp_path / "results"
    paths = build_report(rcfg, ml_eval_cfg, out_dir=out, models_dir=models_dir)
    names = {p.name for p in paths}
    assert {"robustness.md", "robustness.json", "robustness_reliability.png"} <= names
    report = json.loads((out / "robustness.json").read_text(encoding="utf-8"))
    prim = report["primary"]["noise_20"]
    assert prim["available"] is True
    assert report["primary"]["e4_high"]["available"] is False
    xgb_run = next(r for r in runs if r.data.tags["model"] == "xgb")
    for k in ("miss_rate", "rel_lead_mean_s", "false_alarms_per_h"):
        reported = prim["values"]["xgb"][k]["point"]
        logged = xgb_run.data.metrics.get(f"noise_20.h30.{k}")
        if logged is None:
            assert reported is None
        else:
            assert reported == pytest.approx(logged)
    assert set(prim["vs_reference"]) == set(MODELS) - {"threshold"}
    md = (out / "robustness.md").read_text(encoding="utf-8")
    crlf = b"\r\n"
    assert crlf not in (out / "robustness.md").read_bytes()  # LF on every OS
    assert crlf not in (out / "robustness.json").read_bytes()
    assert "## PRIMARY: pre-registered outcomes" in md
    assert "EXPLORATORY" in md
