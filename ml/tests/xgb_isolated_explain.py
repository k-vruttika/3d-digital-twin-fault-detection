"""XGBoost explanation tests, run in their own pytest process (ADR 0020, ADR 0026).

Not collected on its own (no ``test_`` prefix): ``test_ml_explain.py::
test_xgboost_explanations_in_own_process`` runs this file through
``twin_ml.testing.run_pytest_isolated``. The main pytest process loads torch, and torch and
XGBoost must never share a process (two OpenMP runtimes segfault on macOS). This process loads
XGBoost only; the root conftest's guard fails it if torch shows up too.
"""

from __future__ import annotations

import numpy as np
import pytest

from twin_common.config import load_line_config
from twin_ml.data.dataset import Dataset
from twin_ml.features.inputs import SENSORS
from twin_ml.testing import CONFIG_DIR, isolated_dataset, require_xgboost, tiny_config

CNC_PARTS = next(
    m for m in load_line_config(CONFIG_DIR / "line.yaml").machines if m.id == "CNC-01"
).sensor_parts


@pytest.fixture(scope="session")
def ml_dataset() -> Dataset:
    """The parent's quick dataset (overrides the plugin fixture, which would rebuild it)."""
    return isolated_dataset()


@pytest.fixture(scope="module")
def xgb_model(ml_dataset: Dataset, tmp_path_factory: pytest.TempPathFactory) -> object:
    require_xgboost()
    from twin_ml.models.trees import TreeModel

    cfg = tiny_config("xgb", tmp_path_factory.mktemp("xgb_cfg"))
    model = TreeModel(cfg, len(ml_dataset.fault_types), ml_dataset.ttf_cap_s)  # type: ignore[arg-type]
    np.random.seed(0)
    model.fit(ml_dataset.split("train"), ml_dataset.split("val"), 0)
    return model


def test_treeshap_sums_to_margin(ml_dataset: Dataset, xgb_model: object) -> None:
    from twin_ml.explain.trees import xgb_contributions, xgb_margin

    split = ml_dataset.split("val")
    x = split.summary(np.arange(300))
    head = xgb_model.heads["h30"]  # type: ignore[attr-defined]
    for c in (1, 2):
        cls = np.full(300, c)
        contrib, bias = xgb_contributions(head, x, cls)
        assert contrib.shape == (300, 50)
        assert np.allclose(contrib.sum(axis=1) + bias, xgb_margin(head, x, cls), atol=1e-4)


def test_explain_api_for_xgboost(ml_dataset: Dataset, xgb_model: object) -> None:
    from twin_ml.explain import explain

    values, mask, _ = ml_dataset.split("val").gather(np.array([100]))
    out = explain(xgb_model, values[0], mask[0], machine_kind="CNC", sensor_parts=CNC_PARTS)
    assert len(out) == 5
    assert {d["sensor"] for d in out} == set(SENSORS)
    assert sum(abs(d["weight"]) for d in out) == pytest.approx(1.0, abs=1e-9) or all(
        d["weight"] == 0 for d in out
    )
    assert all(d["part"] == CNC_PARTS[d["sensor"]] for d in out)


def test_this_process_never_loads_torch() -> None:
    import sys

    assert "torch" not in sys.modules
