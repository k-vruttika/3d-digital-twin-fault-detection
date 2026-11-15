"""Tiny end-to-end training per model type on quick-scenario data, logged to a tmp MLflow
store, and the E1 report rendered from those runs.

Every (model, seed) trains in its own worker process, as in ``train-all``: torch and XGBoost
must never share a process (two OpenMP runtimes crash on macOS; ADR 0020). The workers run
with faulthandler on, so a crash prints the crashing frame to the test output.
"""

from __future__ import annotations

import faulthandler
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from twin_ml.config import DatasetConfig, EvalConfig, ModelConfig, NnConfig, load_model_config
from twin_ml.data.dataset import Dataset
from twin_ml.experiments.report_e1 import build_report
from twin_ml.testing import REPO_ROOT, TINY, require_xgboost, tiny_config
from twin_ml.train import tracking
from twin_ml.train.isolated import (
    TrainingProcessError,
    build_request,
    describe_exit,
    run_isolated,
)
from twin_ml.train.trainer import RunResult

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


def run_one(
    model: str, cfg: ModelConfig, seed: int, ds: Dataset, eval_cfg: EvalConfig
) -> RunResult:
    """One (model, seed) in its own worker process, exactly as ``train-all`` runs it."""
    return run_isolated(model, cfg, seed, ds.path, eval_cfg)


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


# What a worker may load: a tree model never torch, a sequence model never XGBoost (ADR 0020).
WORKER_SCRIPT = """
import json, sys
from twin_ml.train.isolated import main
assert main(sys.argv[1:3]) == 0
watch = ("torch", "xgboost", "sklearn")
print(json.dumps(sorted(w for w in watch
                        if any(k == w or k.startswith(w + ".") for k in sys.modules))))
"""


@pytest.mark.parametrize(
    ("model", "loads", "never"),
    [
        ("rf", {"sklearn"}, {"torch", "xgboost"}),
        ("xgb", {"xgboost"}, {"torch"}),
        ("gru", {"torch"}, {"xgboost"}),
    ],
)
def test_training_worker_loads_only_its_own_library(
    model: str,
    loads: set[str],
    never: set[str],
    ml_dataset: Dataset,
    ml_eval_cfg: EvalConfig,
    tmp_path: Path,
) -> None:
    if model == "xgb":
        require_xgboost()
    request = build_request(model, tiny_config(model, tmp_path), 0, ml_dataset.path, ml_eval_cfg)
    req, res = tmp_path / "request.json", tmp_path / "result.json"
    req.write_text(json.dumps(request), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", WORKER_SCRIPT, str(req), str(res)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert loads <= loaded, loaded
    assert not never & loaded, loaded
    assert json.loads(res.read_text(encoding="utf-8"))["model"] == model


def test_failed_worker_raises_with_its_exit_status(ml_eval_cfg: EvalConfig, tmp_path: Path) -> None:
    cfg = tiny_config("persistence", tmp_path)
    with pytest.raises(TrainingProcessError, match=r"persistence seed 0 .*exit code 1"):
        run_isolated("persistence", cfg, 0, tmp_path / "no_dataset", ml_eval_cfg)
    assert not tracking.finished_runs(ml_eval_cfg.tracking_dir, ml_eval_cfg.experiment)


def test_describe_exit_names_the_signal() -> None:
    assert describe_exit(1) == "exit code 1"
    assert describe_exit(-11) == "signal SIGSEGV"


def test_faulthandler_is_on_in_the_test_process() -> None:
    # pytest's faulthandler plugin; disabling it (-p no:faulthandler) would hide crash frames.
    assert faulthandler.is_enabled()
