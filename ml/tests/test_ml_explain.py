"""Phase 9 explainability (ADR 0026): sensor maps, TreeSHAP / Saabas additivity, Integrated
Gradients completeness, the deletion test, agreement sums and the Phase 11 ``explain()``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twin_common.config import load_line_config
from twin_common.enums import MachineState
from twin_ml.config import NnConfig
from twin_ml.data.dataset import Dataset
from twin_ml.experiments import explain_eval
from twin_ml.explain.aggregate import (
    normalise,
    sequence_channel_sensors,
    summary_feature_sensors,
    to_parts,
    to_sensors,
)
from twin_ml.features.inputs import SENSORS, SEQUENCE_CHANNELS
from twin_ml.features.summary import summary_names
from twin_ml.testing import CONFIG_DIR, make_split, require_xgboost, tiny_config

LINE = load_line_config(CONFIG_DIR / "line.yaml")
CNC_PARTS = next(m for m in LINE.machines if m.id == "CNC-01").sensor_parts


def test_feature_maps_cover_every_input_once() -> None:
    fs = summary_feature_sensors()
    assert len(fs) == len(summary_names())
    assert fs.count(None) == 3  # kind one-hots dropped
    for s in SENSORS:
        n = fs.count(s)
        # 7 stats per value channel (+7 for vib_norm on vib) and one missing fraction
        assert n == (15 if s == "vib" else 8), s
    cs = sequence_channel_sensors()
    assert len(cs) == len(SEQUENCE_CHANNELS)
    assert cs.count(None) == 3
    assert cs[SEQUENCE_CHANNELS.index("vib_norm")] == "vib"
    assert cs[SEQUENCE_CHANNELS.index("current_missing")] == "current"
    attr = np.ones((2, len(fs)))
    assert to_sensors(attr, fs).sum() == pytest.approx(2 * (len(fs) - 3))


def test_to_parts_schema_and_normalisation() -> None:
    out = to_parts([0.1, -0.2, 0.6, 0.0, 0.1], CNC_PARTS)
    assert out[0]["sensor"] == "vib"
    assert out[0]["part"] == "spindle_bearing"
    assert sum(abs(d["weight"]) for d in out) == pytest.approx(1.0)
    assert {d["part"] for d in out} <= set(CNC_PARTS.values())
    assert set(out[0]) == {"sensor", "part", "weight"}
    assert normalise(np.zeros((1, 5))).tolist() == [[0.0] * 5]


@pytest.fixture(scope="module")
def tree_models(ml_dataset: Dataset, tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    require_xgboost()
    from twin_ml.models.trees import TreeModel

    d = tmp_path_factory.mktemp("tree_cfg")
    out: dict[str, object] = {}
    for m in ("xgb", "rf"):
        model = TreeModel(tiny_config(m, d), len(ml_dataset.fault_types), ml_dataset.ttf_cap_s)  # type: ignore[arg-type]
        np.random.seed(0)
        model.fit(ml_dataset.split("train"), ml_dataset.split("val"), 0)
        out[m] = model
    return out


def test_treeshap_sums_to_margin(ml_dataset: Dataset, tree_models: dict[str, object]) -> None:
    from twin_ml.explain.trees import xgb_contributions, xgb_margin

    split = ml_dataset.split("val")
    x = split.summary(np.arange(300))
    head = tree_models["xgb"].heads["h30"]  # type: ignore[attr-defined]
    for c in (1, 2):
        cls = np.full(300, c)
        contrib, bias = xgb_contributions(head, x, cls)
        assert contrib.shape == (300, 50)
        assert np.allclose(contrib.sum(axis=1) + bias, xgb_margin(head, x, cls), atol=1e-4)


def test_saabas_sums_to_probability(ml_dataset: Dataset, tree_models: dict[str, object]) -> None:
    from twin_ml.explain.trees import saabas_contributions

    split = ml_dataset.split("val")
    x = split.summary(np.arange(200))
    head = tree_models["rf"].heads["h30"]  # type: ignore[attr-defined]
    proba = head.proba(x)
    cls = np.where(np.arange(200) % 2 == 0, 1, 2)
    contrib, bias = saabas_contributions(head, x, cls)
    assert np.allclose(contrib.sum(axis=1) + bias, proba[np.arange(200), cls], atol=1e-6)


def test_integrated_gradients_completeness_and_linear_case() -> None:
    import torch

    from twin_ml.explain.ig import baseline_of, integrated_gradients

    torch.manual_seed(0)
    x = torch.randn(8, 30, 14)
    x[..., 11:] = 0.0
    x[:, :, 12] = 1.0  # kind CNC
    w = torch.randn(30 * 14, 3)

    def linear(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], -1) @ w

    target = torch.tensor([1, 2] * 4)
    attr, err, delta = integrated_gradients(linear, x, target, 64)
    b = baseline_of(x)
    expect = ((x - b).reshape(8, -1) * w[:, target].T).reshape(8, 30, 14).numpy()
    assert np.allclose(attr, expect, atol=1e-5)
    assert np.abs(attr[..., 11:]).max() == 0.0  # the kind channels get no attribution
    assert err.max() < 1e-4

    net = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(30 * 14, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)
    )
    attr, err, delta = integrated_gradients(net, x, target, 64)
    assert (err / np.maximum(np.abs(delta), 1e-6)).max() < 1e-3


def test_deletion_inputs_and_random_other() -> None:
    values = np.ones((3, 30, 6), dtype=np.float32)
    out = explain_eval.deletion_inputs(values, np.array([1, 2, 4]))
    assert out[0, :, 1].max() == 0.0 and out[0, :, [0, 2, 3, 4, 5]].min() == 1.0  # noqa: PT018
    assert out[1, :, 2].max() == 0.0 and out[1, :, 5].max() == 0.0  # noqa: PT018 vib + vib_norm
    assert values.min() == 1.0  # input unchanged
    top = np.array([0, 1, 2, 3, 4] * 20)
    ids = np.arange(100)
    r1 = explain_eval.random_other(top, ids, 17)
    assert (r1 != top).all()
    assert np.array_equal(r1, explain_eval.random_other(top, ids, 17))
    assert len(set(r1.tolist())) == 5


def test_deletion_sums_on_a_one_sensor_model() -> None:
    """A model that only reads temp: deleting temp moves p, a random other sensor does not."""
    rng = np.random.default_rng(0)
    values = rng.normal(2.0, 0.1, (40, 30, 6)).astype(np.float32)

    def p_alarm(v: np.ndarray) -> np.ndarray:
        return np.asarray(1 / (1 + np.exp(-(v[:, :, 1].mean(axis=1) - 1.0))), dtype=np.float64)

    top = np.full(40, SENSORS.index("temp"))
    rnd = explain_eval.random_other(top, np.arange(40), 3)
    p0 = p_alarm(values)
    s = explain_eval.deletion_sums(
        np.zeros(40, dtype=np.int64),
        p0,
        p_alarm(explain_eval.deletion_inputs(values, top)),
        p_alarm(explain_eval.deletion_inputs(values, rnd)),
        1,
    )
    from twin_ml.evaluate import run_sums

    m = run_sums.robust_metrics({"deletion": s.sum(axis=0)})
    assert m["del_top"] > 0.3
    assert m["del_rand"] == 0.0
    assert m["del_top_gt_frac"] == 1.0


def test_explained_windows_thinning_cap_and_agreement() -> None:
    n = 200
    split = make_split([MachineState.RUNNING] * n, window=3, horizons=(2,))
    a = split.arrays
    a["ep_series"] = np.array([0], dtype=np.int32)
    a["ep_fault_type"] = np.array([4], dtype=np.int8)
    a["ep_start"] = np.array([20], dtype=np.int64)  # t = 21 s
    a["ep_warn"] = np.array([-1], dtype=np.int64)
    a["ep_fault"] = np.array([150], dtype=np.int64)  # t = 151 s
    sus = np.ones(split.n_windows, dtype=bool)
    eps = explain_eval.explained_windows(split, sus, stride_s=10, max_windows=5)
    assert len(eps) == 1
    t = split.at_end("t_s")[eps[0].windows]
    assert eps[0].windows.size == 5
    assert t.min() >= 21
    assert t.max() < 151
    assert np.all(np.diff(t) >= 10)
    full = explain_eval.explained_windows(split, sus, stride_s=10, max_windows=100)
    assert full[0].windows.size == 13  # 21, 31, ..., 141
    none = explain_eval.explained_windows(split, np.zeros_like(sus), 10, 5)
    assert none == []
    w = {0: np.array([[0.1, 0.0, 0.2, 0.0, 0.7], [0.0, 0.0, 0.5, 0.0, 0.5]])}
    sums, rec = explain_eval.agreement_sums(eps, w, {4: SENSORS.index("current")}, 1, 6)
    assert sums[0, 4].tolist() == [1.0, 1.0]
    assert rec[0]["top"] == "current"
    sums2, _ = explain_eval.agreement_sums(eps, w, {4: SENSORS.index("vib")}, 1, 6)
    assert sums2[0, 4].tolist() == [1.0, 0.0]


def test_explain_api_for_trees_and_sequence_models(
    ml_dataset: Dataset, tree_models: dict[str, object], tmp_path: Path
) -> None:
    from twin_ml.explain import attribute, explain
    from twin_ml.models.nn import SequenceModel

    split = ml_dataset.split("val")
    values, mask, _ = split.gather(np.array([100]))
    cfg = tiny_config("gru", tmp_path)
    assert isinstance(cfg, NnConfig)
    gru = SequenceModel(cfg, len(ml_dataset.fault_types), ml_dataset.ttf_cap_s)
    gru.fit(ml_dataset.split("train"), split, 0)
    for model in (tree_models["xgb"], tree_models["rf"], gru):
        out = explain(model, values[0], mask[0], machine_kind="CNC", sensor_parts=CNC_PARTS)
        assert len(out) == 5
        assert {d["sensor"] for d in out} == set(SENSORS)
        assert sum(abs(d["weight"]) for d in out) == pytest.approx(1.0, abs=1e-9) or all(
            d["weight"] == 0 for d in out
        )
        assert all(d["part"] == CNC_PARTS[d["sensor"]] for d in out)
    v, m, k = split.gather(np.arange(4))
    att = attribute(gru, v, m, k, 30, ig_steps=64)
    assert att.completeness_error is not None
    assert att.delta is not None
    assert (att.completeness_error <= 1e-2 * np.maximum(np.abs(att.delta), 1.0)).all()
    assert set(att.cls.tolist()) <= {1, 2}


def test_explain_import_is_light() -> None:
    import json
    import subprocess
    import sys

    from twin_ml.testing import REPO_ROOT

    watch = ["twin_sim", "torch", "sklearn", "xgboost", "mlflow"]
    script = (
        "import json, sys, twin_ml.explain, twin_ml.explain.aggregate, "
        "twin_ml.explain.trees, twin_ml.experiments.explain_eval\n"
        f"print(json.dumps([w for w in {watch!r} if any(k == w or k.startswith(w + '.') "
        "for k in sys.modules)]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
