"""Tiny end-to-end training per model type on quick-scenario data, logged to a tmp MLflow
store, and the E1 report rendered from those runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from twin_ml.config import DatasetConfig, EvalConfig, ModelConfig, NnConfig, load_model_config
from twin_ml.data.dataset import Dataset
from twin_ml.experiments.report_e1 import build_report
from twin_ml.testing import CONFIG_DIR, require_xgboost
from twin_ml.train import tracking
from twin_ml.train.trainer import run_one

TINY: dict[str, dict[str, Any]] = {
    "persistence": {"type": "persistence"},
    "threshold": {"type": "threshold", "sim_config": str(CONFIG_DIR / "simulator.yaml")},
    "rf": {
        "type": "rf",
        "class_weight": {"power": 0.5, "max": 50.0},
        "classifier": {"n_estimators": 8, "max_depth": 6, "min_samples_leaf": 2},
        "regressor": {"n_estimators": 8, "max_depth": 6, "min_samples_leaf": 2},
        "n_jobs": 2,
    },
    "xgb": {
        "type": "xgb",
        "class_weight": {"power": 0.5, "max": 50.0},
        "classifier": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        },
        "regressor": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        },
        "early_stopping_rounds": 5,
        "n_jobs": 2,
    },
}

NN_TRAINING: dict[str, Any] = {
    "early_stopping": {
        "min_epochs": 1,
        "max_epochs": 3,
        "patience": 1,
        "ema_alpha": 0.5,
        "min_delta": 0.0,
    },
    "lr_schedule": {"type": "plateau", "factor": 0.5, "patience": 0, "min_lr": 1e-5},
    "batch_size": 128,
    "lr": 0.003,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "focal_gamma": 0.0,
    "huber_delta": 0.1,
    "loss_weights": {"cls": 1.0, "ttf": 0.5, "fault_type": 0.3},
    "eval_batch_size": 1024,
    "num_threads": 2,
}
for _kind in ("lstm", "gru", "transformer"):
    TINY[_kind] = {
        "type": _kind,
        "class_weight": {"power": 0.5, "max": 50.0},
        "arch": {"hidden": 16, "layers": 2, "dropout": 0.1, "heads": 2, "ff_mult": 2},
        "training": NN_TRAINING,
    }

EXPECTED_KEYS = (
    "h10.operating.macro_f1",
    "h30.operating.macro_f1",
    "h30.operating.fault_recall",
    "h30.all.accuracy",
    "h30.ew.episodes",
    "h30.ew.false_alarms_per_h",
    "h30.ci.macro_f1.lo",
    "time.train_s",
    "latency.p50_ms",
    "latency.p95_ms",
)


def tiny_config(model: str, tmp_path: Path) -> ModelConfig:
    path = tmp_path / f"{model}.yaml"
    path.write_text(json.dumps(TINY[model]), encoding="utf-8")  # JSON is valid YAML
    return load_model_config(path)


@pytest.mark.parametrize("model", list(TINY))
def test_train_and_log_each_model(
    model: str, ml_dataset: Dataset, ml_eval_cfg: EvalConfig, tmp_path: Path
) -> None:
    if model == "xgb":
        require_xgboost()
    res = run_one(model, tiny_config(model, tmp_path), 0, ml_dataset, ml_eval_cfg)
    for key in EXPECTED_KEYS:
        assert key in res.metrics, key
    runs = tracking.finished_runs(ml_eval_cfg.tracking_dir, ml_eval_cfg.experiment)
    assert [r.info.run_id for r in runs] == [res.run_id]
    run = runs[0]
    assert run.data.tags["model"] == model
    assert run.data.tags["dataset_key"] == ml_dataset.meta["dataset_key"]
    assert run.data.tags["content_sha256"] == ml_dataset.meta["source"]["content_sha256"]
    artifacts = {Path(a.path).name for a in _artifacts(ml_eval_cfg, run.info.run_id)}
    assert {"metrics.json", "splits.json", "test_predictions.npz"} <= artifacts
    if model not in ("persistence", "threshold"):
        assert "ttf.operating.mae_s" in res.metrics
        assert "fault_type.operating.macro_f1" in res.metrics


def _artifacts(cfg: EvalConfig, run_id: str) -> list[Any]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking.tracking_uri(cfg.tracking_dir))
    return list(client.list_artifacts(run_id))


def test_report_numbers_come_from_mlflow(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, ml_eval_cfg: EvalConfig, tmp_path: Path
) -> None:
    require_xgboost()  # the report test trains every model type, XGBoost included
    results = {m: run_one(m, tiny_config(m, tmp_path), 0, ml_dataset, ml_eval_cfg) for m in TINY}
    md, js, png, curves = build_report(ml_dataset_cfg, ml_eval_cfg)
    text = md.read_text(encoding="utf-8")
    summary = json.loads(js.read_text(encoding="utf-8"))
    assert png.stat().st_size > 0
    assert [m["model"] for m in summary["models"]] == list(TINY)
    assert ml_dataset.meta["source"]["manifest_sha256"] in text
    assert ml_dataset.meta["source"]["content_sha256"] in text
    assert "| Window, horizons, ttf cap | 30 s; +10 s, +30 s; 600 s |" in text
    assert curves.name == "E1_test_learning_curves.png"
    assert curves.stat().st_size > 0
    for heading in (
        "## How to read this report",
        "## Primary result: early warning on gradual faults",
        "## Paired comparisons",
        "**Model - threshold baseline, horizon +30 s**",
        "**Best sequence model - best tree model**",
        "## Secondary: per-window classification",
        "## Sequence-model training",
    ):
        assert heading in text, heading
    assert text.index("## Primary result") < text.index("## Secondary")
    assert "Per-window WARNING precision penalises" in text
    assert "sensor-only models cannot tell IDLE, FAULT and MAINTENANCE apart" in text
    assert "## Changes since the previous configuration" not in text
    paired = summary["paired"]
    assert set(paired["vs_reference"]["30"]) == set(TINY) - {"threshold"}
    best = paired["tree_vs_sequence"]["30"]
    assert best["tree"] in ("rf", "xgb")
    assert best["sequence"] in ("lstm", "gru", "transformer")
    for d in paired["vs_reference"]["30"]["persistence"].values():
        assert d["seeds"] == 1
        if d["point"] is not None and d["lo"] is not None:
            assert d["lo"] <= d["point"] <= d["hi"]
    section = text.split("### Operating windows, horizon +30 s")[1].split("\n### ")[0]
    for model, res in results.items():
        value = res.metrics["h30.operating.macro_f1"]
        row = next(line for line in section.splitlines() if line.startswith(f"| {model} |"))
        assert re.match(rf"\| {model} \| 1 \| {value:.3f} ", row), row
        got = next(m for m in summary["models"] if m["model"] == model)
        assert got["metrics"]["h30.operating.macro_f1"]["mean"] == pytest.approx(value)
        assert got["run_ids"] == [res.run_id]


def test_report_without_runs_fails_clearly(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, ml_eval_cfg: EvalConfig
) -> None:
    with pytest.raises(LookupError, match="train-all"):
        build_report(ml_dataset_cfg, ml_eval_cfg)


def test_sequence_model_training_is_deterministic(ml_dataset: Dataset, tmp_path: Path) -> None:
    from twin_ml.models.nn import SequenceModel

    cfg = tiny_config("gru", tmp_path)
    assert isinstance(cfg, NnConfig)
    train, val = ml_dataset.split("train"), ml_dataset.split("val")

    def run(seed: int) -> Any:
        model = SequenceModel(cfg, len(ml_dataset.fault_types), ml_dataset.ttf_cap_s)
        model.fit(train, val, seed)
        return model.predict(val).probs[30]

    a, b, c = run(0), run(0), run(1)
    assert (a == b).all()
    assert not (a == c).all()


def test_report_compares_a_changed_config_with_its_previous_runs(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, ml_eval_cfg: EvalConfig, tmp_path: Path
) -> None:
    run_one("threshold", tiny_config("threshold", tmp_path), 0, ml_dataset, ml_eval_cfg)
    old = run_one("rf", tiny_config("rf", tmp_path), 0, ml_dataset, ml_eval_cfg)
    changed = json.loads(json.dumps(TINY["rf"]))
    changed["classifier"]["n_estimators"] = 5
    path = tmp_path / "rf_changed.yaml"
    path.write_text(json.dumps(changed), encoding="utf-8")
    new = run_one("rf", load_model_config(path), 0, ml_dataset, ml_eval_cfg)
    md, js, *_ = build_report(ml_dataset_cfg, ml_eval_cfg)
    text = md.read_text(encoding="utf-8")
    assert "## Changes since the previous configuration" in text
    assert "| rf (previous) |" in text
    assert "| rf (current) |" in text
    summary = json.loads(js.read_text(encoding="utf-8"))
    rf = next(m for m in summary["models"] if m["model"] == "rf")
    assert rf["run_ids"] == [new.run_id]
    assert summary["previous"][0]["run_ids"] == [old.run_id]
