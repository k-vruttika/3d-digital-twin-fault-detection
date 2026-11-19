"""Phase 10 end to end on the synthetic stand-in datasets: build -> study runs in worker
processes (tmp MLflow store) -> combined policies -> explanations -> report numbers equal the
MLflow metrics -> resume logs nothing new -> no worker loads both torch and XGBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from twin_ml.datasets.build import build_real_dataset
from twin_ml.testing import TINY, require_xgboost
from twin_ml.testing_real import real_tree
from twin_ml.train import tracking

SEEDS = [0, 1]
RUNS = {
    "metropt3": "iforest,autoencoder,threshold,zscore",
    "azure": "persistence,threshold,zscore,xgb,gru,autoencoder",
    "azure_err": "xgb,error_rule",
    "azure_holdout": "xgb,autoencoder",
}


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    require_xgboost()
    base = tmp_path_factory.mktemp("p10_e2e")
    paths = real_tree(base)
    models = base / "config" / "models"
    for m in ("rf", "xgb", "lstm", "gru", "transformer"):
        (models / f"{m}.yaml").write_text(json.dumps(TINY[m]), encoding="utf-8")
    (models / "iforest.yaml").write_text(
        json.dumps(
            {
                "type": "iforest",
                "n_estimators": 20,
                "max_samples": 64,
                "max_features": 1.0,
                "n_jobs": 1,
            }
        ),
        encoding="utf-8",
    )
    (models / "autoencoder.yaml").write_text(
        json.dumps(
            {
                "type": "autoencoder",
                "hidden": 16,
                "latent": 4,
                "epochs": 2,
                "batch_size": 128,
                "lr": 0.003,
                "weight_decay": 0.0,
                "eval_batch_size": 2048,
                "num_threads": 1,
            }
        ),
        encoding="utf-8",
    )
    real = yaml.safe_load((base / paths["real"]).read_text(encoding="utf-8"))
    real["seeds"] = SEEDS
    real["bootstrap"]["n_resamples"] = 50
    (base / paths["real"]).write_text(yaml.safe_dump(real, sort_keys=False), encoding="utf-8")
    for name in ("real_metropt3", "real_azure", "real_azure_err", "real_azure_holdout"):
        build_real_dataset(paths[name], base)
    return base


def _runs(root: Path, study: str | None = None) -> list[Any]:
    flt = "" if study is None else f"tags.study = '{study}'"
    return tracking.finished_runs(root / "mlruns", "P10_real", None, "P10", flt)


def _detail(run: Any, root: Path) -> dict[str, Any]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking.tracking_uri(root / "mlruns"))
    path = client.download_artifacts(run.info.run_id, "metrics.json", str(root / "dl"))
    return json.loads(Path(path).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_studies_combine_explain_report_and_resume(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin_ml.experiments import real

    monkeypatch.chdir(root)
    (root / "dl").mkdir(exist_ok=True)
    seeds = ",".join(str(s) for s in SEEDS)
    for study, models in RUNS.items():
        assert real.main(["--study", study, "--models", models, "--seeds", seeds]) == 0
    n_rule = {"metropt3": 2, "azure": 3, "azure_err": 1, "azure_holdout": 0}
    for study, models in RUNS.items():
        got = _runs(root, study)
        n_models = len(models.split(","))
        assert len(got) == n_rule[study] + (n_models - n_rule[study]) * len(SEEDS), study

    # torch and XGBoost never share a worker process (ADR 0020)
    for run in _runs(root):
        libs = set(_detail(run, root)["process"]["libraries"])
        assert not {"torch", "xgboost"} <= libs
        if run.data.tags["model"] == "xgb":
            assert "torch" not in libs
        if run.data.tags["model"] in ("gru", "autoencoder"):
            assert "xgboost" not in libs

    assert real.main(["--combine"]) == 0
    comb = _runs(root, "azure_combined")
    assert len(comb) == 2 * len(SEEDS)  # (xgb, gru) x autoencoder (iforest is not run here)
    assert {r.data.tags["primary"] for r in comb} == {"true", "false"}
    assert len(_runs(root, "azure_holdout_combined")) == len(SEEDS)

    assert real.main(["--explain"]) == 0
    ex = _runs(root, "explain")
    best = comb[0].data.tags["best_supervised"]
    expected = len(SEEDS) * (2 if best != "xgb" else 1) + len(SEEDS)  # + MetroPT-3 AE
    assert len(ex) == expected

    # resume: nothing new is logged
    before = len(_runs(root))
    for study, models in RUNS.items():
        assert (
            real.main(["--study", study, "--models", models, "--seeds", seeds, "--skip-existing"])
            == 0
        )
    assert real.main(["--combine", "--skip-existing"]) == 0
    assert real.main(["--explain", "--skip-existing"]) == 0
    assert len(_runs(root)) == before

    # the report's numbers are the MLflow metrics
    from twin_ml.datasets.audit import run_audit
    from twin_ml.datasets.config import load_real_config
    from twin_ml.experiments.report_real import build_report

    for name in ("metropt3", "azure_pdm"):
        run_audit(name, Path("config/models/real.yaml"), root)
    paths = build_report(load_real_config(root / "config/models/real.yaml"), root)
    md = next(p for p in paths if p.suffix == ".md").read_text(encoding="utf-8")
    js = json.loads(next(p for p in paths if p.suffix == ".json").read_text(encoding="utf-8"))
    assert "## MetroPT-3 results" in md
    assert "## Azure PdM results" in md
    assert "## Explanations (exploratory)" in md
    res = js["results"]
    (thr,) = [r for r in _runs(root, "azure") if r.data.tags["model"] == "threshold"]
    for metric in ("detection_rate", "false_alarms_per_day"):
        want = thr.data.metrics.get(f"clean.rule.L48.{metric}")
        got = res["azure"]["primary"]["threshold"]["mean"][metric]["point"]
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-3, abs=1e-6)
    (z,) = [r for r in _runs(root, "metropt3") if r.data.tags["model"] == "zscore"]
    pt = res["metropt3"]["primary"]["zscore"]["seeds"]["0"]["points"]
    assert pt["warned_count"] == z.data.metrics["clean.rule.L360.warned_count"]
    assert pt["evaluable"] == z.data.metrics["clean.rule.L360.evaluable"]
    assert res["azure"]["best_supervised"] == best
    for p in paths:
        assert p.stat().st_size < 5 * 1024 * 1024
