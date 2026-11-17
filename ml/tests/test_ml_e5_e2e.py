"""Phase 9 end to end on quick-scenario data: E1 runs -> E5 supervised and anomaly study runs
(worker processes, tmp MLflow store) -> combined policy -> resume logs nothing new."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from twin_ml.config import DatasetConfig, E5Config, EvalConfig, load_e5_config
from twin_ml.data.dataset import Dataset
from twin_ml.testing import (
    CONFIG_DIR,
    ML_SIM_DURATION_S,
    ML_SIM_RUNS,
    generate_sim,
    require_xgboost,
    tiny_config,
)
from twin_ml.train import tracking
from twin_ml.train.isolated import run_isolated

SUPERVISED = ("threshold", "xgb", "gru")
DETECTORS = ("iforest", "autoencoder")
DRIFT: dict[str, Any] = {
    "random": {"rate_per_machine_per_hour": 4.0, "types": ["drift"], "earliest_onset_s": 60}
}


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, default=str), encoding="utf-8")  # JSON is valid YAML
    return path


@pytest.fixture(scope="module")
def drift_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_sim(
        tmp_path_factory.mktemp("ml_sim_drift"), ML_SIM_RUNS, ML_SIM_DURATION_S, sensor_faults=DRIFT
    )


def _configs(
    tmp: Path,
    ds: Dataset,
    ds_cfg: DatasetConfig,
    eval_cfg: EvalConfig,
    e5_dir: Path,
    drift_dir: Path,
) -> tuple[Path, Path, Path]:
    from twin_ml.config import load_robustness_config

    ds_path = _write(tmp / "dataset.yaml", ds_cfg.model_dump(mode="json"))
    eval_path = _write(tmp / "eval.yaml", eval_cfg.model_dump(mode="json"))
    rob = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml").model_dump(mode="json")
    rob["sensor_fault_sets"] = {
        "sf_drift": {
            "source_dir": str(drift_dir),
            "scenario": "config/scenarios/sf_drift.yaml",
            "types": ["drift"],
        }
    }
    rob_path = _write(tmp / "robustness.yaml", rob)
    _write(
        tmp / "iforest.yaml",
        {
            "type": "iforest",
            "n_estimators": 20,
            "max_samples": 64,
            "max_features": 1.0,
            "n_jobs": 1,
        },
    )
    _write(
        tmp / "autoencoder.yaml",
        {
            "type": "autoencoder",
            "hidden": 16,
            "latent": 4,
            "epochs": 2,
            "batch_size": 128,
            "lr": 0.003,
            "weight_decay": 0.0,
            "eval_batch_size": 4096,
            "num_threads": 2,
        },
    )
    e5 = load_e5_config(CONFIG_DIR / "models" / "e5.yaml").model_dump(mode="json")
    test_ids = ds.meta["splits"]["test"]
    e5 |= {
        "experiment": "P9_test",
        "smoke_experiment": "P9_test_smoke",
        "robustness_config": str(rob_path),
        "progress_dir": str(tmp / "progress"),
    }
    e5["e5_data"] |= {
        "source_dir": str(e5_dir),
        "generate_runs": ML_SIM_RUNS,
        "extra_run_indices": [ML_SIM_RUNS, ML_SIM_RUNS],
        "n_runs": len(test_ids),
    }
    e5["anomaly"]["models"] = {m: str(tmp / f"{m}.yaml") for m in DETECTORS}
    conditions = ["clean", "e5", "sf_drift"]
    e5["studies"] = {
        "e5_e1models": {
            "kind": "supervised",
            "dataset_config": str(ds_path),
            "source_experiment": eval_cfg.experiment,
            "source_phase": "E1",
            "models": list(SUPERVISED),
            "conditions": conditions,
        },
        "e5_anomaly": {
            "kind": "anomaly",
            "dataset_config": str(ds_path),
            "models": list(DETECTORS),
            "conditions": conditions,
        },
    }
    E5Config.model_validate(e5)
    return _write(tmp / "e5.yaml", e5), eval_path, ds_path


def test_e5_studies_combine_and_resume(
    ml_dataset: Dataset,
    ml_dataset_cfg: DatasetConfig,
    ml_eval_cfg: EvalConfig,
    e5_sim_dir: Path,
    drift_dir: Path,
    tmp_path: Path,
) -> None:
    require_xgboost()
    import mlflow

    from twin_ml.experiments import e5 as runner

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for m in SUPERVISED:
        res = run_isolated(m, tiny_config(m, models_dir), 0, ml_dataset.path, ml_eval_cfg)
        assert res.run_id
    e5_path, eval_path, _ = _configs(
        tmp_path, ml_dataset, ml_dataset_cfg, ml_eval_cfg, e5_sim_dir, drift_dir
    )
    common = [
        "--seeds",
        "0",
        "--e5-config",
        str(e5_path),
        "--eval-config",
        str(eval_path),
        "--models-dir",
        str(models_dir),
        "--skip-existing",
    ]
    assert runner.main(["--study", "e5_e1models", *common]) == 0
    assert runner.main(["--study", "e5_anomaly", *common]) == 0
    assert runner.main(["--combine", *common]) == 0

    tdir = ml_eval_cfg.tracking_dir

    def runs(study: str) -> list[Any]:
        return tracking.finished_runs(tdir, "P9_test", None, "P9", f"tags.study = '{study}'")

    sup = {r.data.tags["model"]: r for r in runs("e5_e1models")}
    det = {r.data.tags["model"]: r for r in runs("e5_anomaly")}
    comb = runs("e5_combined")
    assert set(sup) == set(SUPERVISED)
    assert set(det) == set(DETECTORS)
    assert len(comb) == len(SUPERVISED) * len(DETECTORS)  # seed 0 only
    for m in ("xgb", "gru"):
        d = mlflow.artifacts.load_dict(f"{sup[m].info.artifact_uri}/metrics.json")
        chk = d["conditions"]["clean"]["prediction_check"]
        assert all(v["labels_identical"] for v in chk.values()), m
        assert "lubrication_loss" in d["fault_types"]
        met = sup[m].data.metrics
        for key in (
            "e5.h30.miss_rate",
            "e5.h30.wrong_rate",
            "clean.h30.false_alarms_per_h",
            "e5.h30.size_mondrian_lac",
            "sf_drift.h30.sf_segments",
        ):
            assert key in met, (m, key)
    for m in DETECTORS:
        met = det[m].data.metrics
        for p in ("99", "99.5", "99.9"):
            assert f"e5.p{p}.miss_rate" in met, (m, p)
            assert f"sf_drift.p{p}.sf_flag_rate" in met or f"sf_drift.p{p}.sf_segments" in met
        d = mlflow.artifacts.load_dict(f"{det[m].info.artifact_uri}/metrics.json")
        assert set(d["detector"]["thresholds"]["values"]) == {"99.0", "99.5", "99.9"}
    ae = mlflow.artifacts.load_dict(f"{det['autoencoder'].info.artifact_uri}/metrics.json")
    assert "sensor_error_shares" in ae["conditions"]["e5"]
    # OR can only add detections: the combined miss rate is at most each side's
    for r in comb:
        t = r.data.tags
        m_c = r.data.metrics.get("e5.p99.9.miss_rate")
        m_s = sup[t["supervised"]].data.metrics.get("e5.h30.miss_rate")
        m_d = det[t["detector"]].data.metrics.get("e5.p99.9.miss_rate")
        if m_c is not None and m_s is not None and m_d is not None:
            assert m_c <= min(m_s, m_d) + 1e-12

    # explanation sanity checks (Part B) on the reused E1 models
    assert runner.main(["--explain", "--models", "xgb,gru", *common]) == 0
    ex = {r.data.tags["model"]: r for r in runs("explain")}
    assert set(ex) == {"xgb", "gru"}
    for m, r in ex.items():
        d = mlflow.artifacts.load_dict(f"{r.info.artifact_uri}/metrics.json")
        assert d["method"] == ("treeshap" if m == "xgb" else "integrated_gradients")
        assert set(d["conditions"]) == {"clean", "e5"}
        for c in ("clean", "e5"):
            info = d["conditions"][c]
            assert info["episodes_explained"] <= info["episodes_total"]
            if info["windows"]:
                assert f"{c}.h30.del_top" in r.data.metrics
        if m == "gru" and d["conditions"]["e5"]["windows"]:
            assert r.data.metrics["e5.h30.ig_err_rel"] < 1e-2

    # exploratory Part C follow-ups: AE error shares (re-fit must reproduce the stored
    # alarms, else the worker fails) and the IG step-count sensitivity
    fu_path = _write(
        tmp_path / "e5_followup.yaml",
        {
            "ae_shares": {
                "detector": "autoencoder",
                "source_study": "e5_anomaly",
                "precursor_conditions": ["clean", "e5"],
                "drift_condition": "sf_drift",
            },
            "ig_sensitivity": {"models": ["gru"], "ig_steps": 16},
        },
    )
    fu = ["--followup-config", str(fu_path), *common]
    assert runner.main(["--ae-shares", *fu]) == 0
    assert runner.main(["--ig-sensitivity", *fu]) == 0
    (ae_run,) = runs("ae_shares")
    ad = mlflow.artifacts.load_dict(f"{ae_run.info.artifact_uri}/metrics.json")
    assert ad["source_run"] == det["autoencoder"].info.run_id
    assert set(ad["conditions"]) == {"clean", "e5", "sf_drift"}
    assert all(c["alarms_reproduced"] for c in ad["conditions"].values())
    assert ad["conditions"]["sf_drift"]["unit"] == "segment"
    (ig_run,) = runs("explain_ig_steps")
    gd = mlflow.artifacts.load_dict(f"{ig_run.info.artifact_uri}/metrics.json")
    assert gd["ig_steps"] == 16
    base = mlflow.artifacts.load_dict(f"{ex['gru'].info.artifact_uri}/metrics.json")
    for c in ("clean", "e5"):  # same alarms -> same explained windows
        assert gd["conditions"][c]["windows"] == base["conditions"][c]["windows"]

    # every worker loaded at most one of torch / xgboost (ADR 0020)
    never = {"xgb": "torch", "gru": "xgboost", "autoencoder": "xgboost", "iforest": "torch"}
    for group in (sup, det, ex, {"autoencoder": ae_run}, {"gru": ig_run}):
        for m, r in group.items():
            if m not in never:
                continue
            d = mlflow.artifacts.load_dict(f"{r.info.artifact_uri}/metrics.json")
            assert never[m] not in d["process"]["libraries"], (m, d["process"])

    # resume: nothing new is logged
    before = len(tracking.finished_runs(tdir, "P9_test", None, "P9"))
    assert runner.main(["--study", "e5_e1models", *common]) == 0
    assert runner.main(["--study", "e5_anomaly", *common]) == 0
    assert runner.main(["--combine", *common]) == 0
    assert runner.main(["--explain", "--models", "xgb,gru", *common]) == 0
    assert runner.main(["--ae-shares", *fu]) == 0
    assert runner.main(["--ig-sensitivity", *fu]) == 0
    assert len(tracking.finished_runs(tdir, "P9_test", None, "P9")) == before
    # report: reads MLflow only; its numbers equal the logged metrics
    from twin_ml.experiments.report_e5 import build_report

    out = tmp_path / "results"
    e5cfg = load_e5_config(e5_path)
    from twin_ml.config import load_e5_followup_config

    paths = build_report(
        e5cfg,
        ml_eval_cfg,
        out_dir=out,
        models_dir=models_dir,
        followup=load_e5_followup_config(fu_path),
    )
    names = {p.name for p in paths}
    assert {"e5_explain.md", "e5_explain.json", "e5_explain_early_warning.png"} <= names
    crlf = bytes([13, 10])
    for path in paths:
        if path.suffix != ".png":
            assert crlf not in path.read_bytes(), path.name  # LF on every OS
    rep = json.loads((out / "e5_explain.json").read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in rep["primary"]["early_warning"]}
    logged = sup["xgb"].data.metrics
    for m, cond in (("miss_rate", "e5"), ("false_alarms_per_h", "clean")):
        got = rows["XGBoost"]["values"][m]["point"]
        assert got == pytest.approx(logged[f"{cond}.h30.{m}"], rel=1e-5), m
    assert any(" OR " in r["label"] for r in rep["primary"]["early_warning"])
    assert {r["model"] for r in rep["primary"]["agreement"]} == {"xgb", "gru"}
    md = (out / "e5_explain.md").read_text(encoding="utf-8")
    assert "## PRIMARY: pre-registered outcomes" in md
    assert "## EXPLORATORY" in md
    assert "overcurrent" in md
    assert "Autoencoder per-sensor reconstruction-error shares" in md
    assert "Integrated Gradients at 16 steps vs 64" in md
    ae_rep = {c["condition"]: c for c in rep["exploratory"]["ae_shares"]}
    assert set(ae_rep) == {"clean", "e5", "sf_drift"}
    e5_rate = ae_run.data.metrics.get("e5.ae.agreement_rate")
    if e5_rate is not None:
        assert ae_rep["e5"]["overall_unit"]["point"] == pytest.approx(e5_rate, rel=1e-5)
    assert {r["condition"] for r in rep["exploratory"]["ig_sensitivity"]} == {"clean", "e5"}

    assert runner.main(["--study", "nope", *common]) == 2
    assert runner.main([*common]) == 2  # neither --study nor --combine
