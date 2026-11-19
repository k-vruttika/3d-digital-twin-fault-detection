"""XGBoost deploy bundle and live parity, run in its own pytest process.

Not collected directly (no ``test_`` prefix): ``test_ml_serve.py::
test_xgboost_bundle_and_parity_in_own_process`` runs this file through
``twin_ml.testing.run_pytest_isolated``, so XGBoost never shares a process with torch
(ADR 0020). Asserts at the end that torch was never loaded here.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from twin_ml.data.dataset import Dataset
from twin_ml.testing import (
    REPO_ROOT,
    deploy_setup,
    eval_config,
    isolated_dataset,
    run_live,
    train_e1_runs,
)


@pytest.fixture(scope="module")
def xgb_setup(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from twin_ml.config import DatasetConfig, load_deploy_config
    from twin_ml.serve.bundle import build_and_register

    ds: Dataset = isolated_dataset()
    tmp = tmp_path_factory.mktemp("xgb_serve")
    eval_cfg = eval_config(tmp / "mlruns", tmp / "results")
    preds = train_e1_runs(ds, eval_cfg, tmp / "cfg", ("xgb",))
    ds_cfg = DatasetConfig.model_validate(ds.meta["config"])
    path = deploy_setup(tmp / "cfg", ds_cfg, eval_cfg, ("xgb",))
    res = build_and_register("xgb", load_deploy_config(path), REPO_ROOT, tmp / "mlruns", False)
    return tmp / "mlruns", ds, preds["xgb"], res


def test_xgb_live_equals_offline(xgb_setup: Any) -> None:
    from twin_ml.serve.buffers import PREDICT
    from twin_ml.serve.worker import load_worker

    tracking, ds, predictor, res = xgb_setup
    assert res.summary["checks"]["test_predictions"]["h30"]["labels_identical"]
    worker = load_worker(
        ["supervised"], ("test-xgb", "latest"), None, tracking, tracking.parent / "cache", 8
    )
    test = ds.split("test")
    offline = predictor.predict(test)
    ends = test.arrays["ends"]
    series_at_end = test.arrays["series"][ends]
    compared = 0
    for sid in np.unique(series_at_end):
        rows = np.flatnonzero(test.arrays["series"] == sid)
        live = run_live(worker, test, int(sid), batch=5)
        for w in np.flatnonzero(series_at_end == sid):
            e = live[int(ends[w] - rows[0])]
            if e["status"] != PREDICT:
                continue
            for h in ds.horizons:
                np.testing.assert_allclose(
                    e["out"]["probs"][str(h)], offline.probs[h][w], atol=1e-6
                )
            assert "anomalyScore" not in e["out"]  # no monitor role in this process
            compared += 1
    assert compared > 100
    info = worker.hello()["process"]
    assert "xgboost" in info["libraries"]
    assert "torch" not in info["libraries"]
    assert "torch" not in sys.modules
