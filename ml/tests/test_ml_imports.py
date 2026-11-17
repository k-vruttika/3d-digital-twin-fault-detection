"""Import isolation: inference-side modules load neither twin_sim nor heavy ML libraries."""

from __future__ import annotations

import json
import subprocess
import sys

from twin_ml.testing import REPO_ROOT

# What Phase 11 needs to run inference inside the backend, without the simulator.
INFERENCE_MODULES = (
    "twin_ml.config",
    "twin_ml.features",
    "twin_ml.features.inputs",
    "twin_ml.features.normalize",
    "twin_ml.features.summary",
    "twin_ml.data.dataset",
    "twin_ml.data.labels",
    "twin_ml.data.windows",
    "twin_ml.models.base",
    "twin_ml.models.trees",
    "twin_ml.evaluate.evaluator",
    "twin_ml.evaluate.metrics",
    "twin_ml.evaluate.early_warning",
    "twin_ml.evaluate.bootstrap",
    "twin_ml.evaluate.latency",
    "twin_ml.baselines.persistence",
)


def _loaded(modules: tuple[str, ...], watch: tuple[str, ...]) -> list[str]:
    script = (
        "import importlib, json, sys\n"
        f"for m in {list(modules)!r}: importlib.import_module(m)\n"
        f"print(json.dumps(sorted(w for w in {list(watch)!r} "
        "if any(k == w or k.startswith(w + '.') for k in sys.modules))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


def test_inference_modules_do_not_import_the_simulator_or_heavy_libraries() -> None:
    watch = ("twin_sim", "torch", "sklearn", "xgboost", "mlflow")
    assert _loaded(INFERENCE_MODULES, watch) == []


def test_simulator_only_where_allowed() -> None:
    assert _loaded(("twin_ml.data.build",), ("twin_sim",)) == ["twin_sim"]
    assert _loaded(("twin_ml.baselines.threshold",), ("twin_sim",)) == ["twin_sim"]


def test_torch_only_in_the_sequence_models() -> None:
    assert _loaded(("twin_ml.models.nn",), ("torch", "twin_sim")) == ["torch"]
    # Phase 9: the autoencoder is the only other torch module
    watch = ("torch", "sklearn", "xgboost", "twin_sim")
    assert _loaded(("twin_ml.models.autoencoder",), watch) == ["torch"]


def test_training_parents_load_no_ml_library() -> None:
    # train / train-all only start workers; each worker loads one model's library (ADR 0020).
    parents = ("twin_ml.train.__main__", "twin_ml.experiments.e1", "twin_ml.train.isolated")
    assert _loaded(parents, ("torch", "sklearn", "xgboost", "mlflow")) == []


PHASE8_PURE = (
    "twin_ml.data.perturb",
    "twin_ml.calibrate",
    "twin_ml.calibrate.ece",
    "twin_ml.calibrate.temperature",
    "twin_ml.calibrate.trees",
    "twin_ml.conformal.classification",
    "twin_ml.conformal.regression",
    "twin_ml.evaluate.run_sums",
    "twin_ml.experiments.robust_eval",
)


def test_phase8_modules_load_no_simulator_or_heavy_library() -> None:
    # calibration imports scikit-learn lazily, only inside the tree calibrators
    watch = ("twin_sim", "torch", "sklearn", "xgboost", "mlflow")
    assert _loaded(PHASE8_PURE, watch) == []


def test_phase8_parents_load_no_ml_library() -> None:
    parents = ("twin_ml.experiments.robustness", "twin_ml.experiments.robust_worker")
    assert _loaded(parents, ("torch", "sklearn", "xgboost", "mlflow")) == []


PHASE9_PURE = (
    "twin_ml.models.anomaly",
    "twin_ml.experiments.e5_eval",
)


def test_phase9_modules_load_no_simulator_or_heavy_library() -> None:
    # the Isolation Forest imports scikit-learn lazily, only when it is fitted
    watch = ("twin_sim", "torch", "sklearn", "xgboost", "mlflow")
    assert _loaded(PHASE9_PURE, watch) == []


def test_phase9_parents_load_no_ml_library() -> None:
    parents = ("twin_ml.experiments.e5", "twin_ml.experiments.e5_worker")
    assert _loaded(parents, ("torch", "sklearn", "xgboost", "mlflow")) == []


PHASE10_DATA = (
    "twin_ml.features.schema",
    "twin_ml.datasets.config",
    "twin_ml.datasets.common",
    "twin_ml.datasets.metropt3",
    "twin_ml.datasets.azure_pdm",
    "twin_ml.datasets.build",
    "twin_ml.datasets.audit",
    "twin_ml.datasets.__main__",
)


def test_phase10_data_modules_load_no_simulator_or_ml_library() -> None:
    # loaders, build and audit: pyarrow + numpy only (mlflow only inside the audit run)
    watch = ("twin_sim", "torch", "sklearn", "xgboost", "mlflow")
    assert _loaded(PHASE10_DATA, watch) == []
