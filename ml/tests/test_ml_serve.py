"""Phase 11 live inference (twin_ml.serve): exported calibration, buffers and gate, bundles,
live vs offline parity, the worker process (ADR 0029 / 0030).

The XGBoost parts run in their own pytest process (``xgb_isolated_serve.py``): this process
loads torch, and the two must never share a process (ADR 0020).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twin_common.config import load_line_config
from twin_common.enums import MachineKind
from twin_ml.calibrate import HeadCalibration, fit_head
from twin_ml.config import CalibrationConfig
from twin_ml.data.dataset import Dataset, SplitData
from twin_ml.features.normalize import Normalizer
from twin_ml.serve.buffers import (
    PREDICT,
    STOPPED,
    WAITING,
    WARMING_UP,
    GateConfig,
    MachineBuffer,
    gate_over_series,
)
from twin_ml.serve.calib import LiveCalibration, LiveHeadCalibration
from twin_ml.testing import (
    CONFIG_DIR,
    REPO_ROOT,
    configure_message,
    run_live,
    run_pytest_isolated,
    series_rows,
)

OPERATING = ("RUNNING", "WARNING")
WAIT = ("STARVED", "BLOCKED")
ATOL = 1e-5  # live vs offline probabilities (batch composition changes float sums slightly)


# ---------------------------------------------------------------------------- calibration
def _probs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3)) * 2.0
    y = np.argmax(logits + rng.normal(size=(n, 3)), axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return p, y


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_exported_tree_calibration_equals_sklearn(method: str) -> None:
    p, y = _probs(800, 1)
    cfg = CalibrationConfig.model_validate(
        {"tree_methods": [method], "cv_folds": 2, "temperature_bounds": (0.05, 20.0)}
    )
    head = fit_head("tree", p, None, y, np.arange(800) % 8, cfg)
    assert head.method == method
    live = LiveHeadCalibration.from_dict(
        json.loads(json.dumps(LiveHeadCalibration.from_head(head).to_dict()))
    )
    q, _ = _probs(300, 2)
    np.testing.assert_allclose(live.apply(q), head.apply(q), rtol=0, atol=1e-12)


def test_exported_temperature_and_sets() -> None:
    p, y = _probs(500, 3)
    logits = np.log(p)
    cfg = CalibrationConfig(tree_methods=["platt"], cv_folds=2, temperature_bounds=(0.05, 20.0))
    head: HeadCalibration = fit_head("nn", p, logits, y, np.zeros(500), cfg)
    live = LiveHeadCalibration.from_head(head)
    np.testing.assert_allclose(live.apply(p, logits), head.apply(p, logits), atol=1e-12)
    cal = LiveCalibration({30: live}, {30: np.array([0.1, math.inf, 0.9])}, 0.1, None, 600.0)
    back = LiveCalibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert math.isinf(back.mondrian_q[30][1])
    assert back.ttf_interval(np.array([1.0])) is None
    sets = back.sets(30, np.array([[0.95, 0.03, 0.02]]))
    assert sets.tolist() == [[True, True, False]]


# ---------------------------------------------------------------------------- buffers
def _norm(seed: int = 0) -> Normalizer:
    rng = np.random.default_rng(seed)
    series = [
        (rng.normal(50, 5, size=(200, 5)).astype(np.float32), k)
        for k in (MachineKind.CNC, MachineKind.CONVEYOR, MachineKind.INSPECTION)
    ]
    return Normalizer.fit(series, 0.1)


def test_buffer_transform_equals_series_wide_transform() -> None:
    norm = _norm()
    rng = np.random.default_rng(5)
    raw = rng.normal(50, 5, size=(400, 5)).astype(np.float32)
    raw[:7, 2] = np.nan  # leading gap: training mean
    raw[rng.random(raw.shape) < 0.2] = np.nan  # scattered gaps, some longer than the buffer
    raw[100:180, 0] = np.nan
    full_v, full_m = norm.transform(raw, MachineKind.CNC)
    cfg = GateConfig.of(30, OPERATING, WAIT, 5)
    buf = MachineBuffer("CNC-01", MachineKind.CNC, cfg, capacity=40)
    for i in range(raw.shape[0]):
        new = buf.append(i, float(i), "RUNNING", [None if np.isnan(v) else v for v in raw[i]])
        assert new is not None
        assert new.row == i
        v, m = buf.transform(norm)
        np.testing.assert_array_equal(v, full_v[buf.base : i + 1])
        np.testing.assert_array_equal(m, full_m[buf.base : i + 1])
        if buf.has_window(i):
            wv, wm = buf.windows(norm, [i])
            np.testing.assert_array_equal(wv[0], full_v[i - 29 : i + 1])
            np.testing.assert_array_equal(wm[0], full_m[i - 29 : i + 1])


def test_gate_statuses_and_warm_up() -> None:
    cfg = GateConfig.of(3, OPERATING, WAIT, 2)
    states = ["IDLE", "RUNNING", "RUNNING", "RUNNING", "WARNING", "STARVED", "RUNNING", "FAULT"]
    buf = MachineBuffer("M", MachineKind.CNC, cfg, capacity=10)
    live = [buf.append(i, float(i), s, [1.0] * 5) for i, s in enumerate(states)]
    expected = [STOPPED, WARMING_UP, WARMING_UP, PREDICT, PREDICT, WAITING, WARMING_UP, STOPPED]
    assert [n.status for n in live if n is not None] == expected
    assert gate_over_series(states, cfg)[0] == expected


def test_buffer_gaps_duplicates_and_restart() -> None:
    cfg = GateConfig.of(3, OPERATING, WAIT, 2)
    buf = MachineBuffer("M", MachineKind.CNC, cfg, capacity=20)
    for i in range(4):
        buf.append(i, float(i), "RUNNING", [1.0] * 5)
    assert buf.append(3, 3.0, "RUNNING", [1.0] * 5) is None  # duplicate
    assert buf.ignored == 1
    n = buf.append(6, 6.0, "RUNNING", [2.0] * 5)  # 2 missing readings: filled and masked
    assert n is not None
    assert n.status == PREDICT
    assert n.row == 6
    _, mask = buf.transform(_norm())
    assert mask[-3:-1].all()
    assert not mask[-1].any()
    n = buf.append(20, 20.0, "RUNNING", [2.0] * 5)  # long gap: warm up again
    assert n is not None
    assert n.status == WARMING_UP
    assert buf.resets == 1
    n = buf.append(0, 30.0, "RUNNING", [2.0] * 5)  # sequence restart (newer timestamp)
    assert n is not None
    assert buf.resets == 2
    assert bool(np.isnan(buf.carry).all())


# ---------------------------------------------------------------------------- parity
@pytest.fixture(scope="module")
def gru_worker(serve_setup: Any) -> Any:
    from twin_ml.serve.worker import load_worker

    tracking, _, _, _ = serve_setup
    return load_worker(
        ["supervised", "monitor"],
        ("test-gru", "latest"),
        ("test-autoencoder", "latest"),
        tracking,
        tracking.parent / "cache_inproc",
        8,
    )


def test_live_equals_offline_for_the_same_windows(serve_setup: Any, gru_worker: Any) -> None:
    _, _, ds, preds = serve_setup
    test: SplitData = ds.split("test")
    offline = preds["gru"].predict(test)
    ae = gru_worker.anomaly.det.score(test)
    ends = test.arrays["ends"]
    series_at_end = test.arrays["series"][ends]
    compared = 0
    gated_out = 0
    for sid in np.unique(series_at_end):
        rows = np.flatnonzero(test.arrays["series"] == sid)
        live = run_live(gru_worker, test, int(sid), batch=7)
        for w in np.flatnonzero(series_at_end == sid):
            pos = int(ends[w] - rows[0])
            e = live[pos]
            if e["status"] != PREDICT:
                assert "out" not in e
                gated_out += 1
                continue
            out = e["out"]
            for h in ds.horizons:
                np.testing.assert_allclose(out["probs"][str(h)], offline.probs[h][w], atol=ATOL)
            assert out["ttf"] == pytest.approx(float(offline.ttf[w]), abs=1e-2)
            assert out["anomalyScore"] == pytest.approx(float(ae[w]), rel=1e-5, abs=1e-7)
            compared += 1
    assert compared > 100
    assert gated_out > 0


def test_live_attributions_equal_explain(serve_setup: Any, gru_worker: Any) -> None:
    from twin_ml.explain import explain

    _, _, ds, _ = serve_setup
    line = load_line_config(CONFIG_DIR / "line.yaml")
    sup = gru_worker.supervised
    checked = 0
    for test, sid in [(ds.split(n), s) for n in ("test", "train") for s in range(12)]:
        if sid not in set(test.arrays["series"].tolist()):
            continue
        live = run_live(gru_worker, test, int(sid), batch=1)  # every window is "newest"
        machine = str(test.arrays["series_machine"][sid])
        m = next(x for x in line.machines if x.id == machine)
        rows = np.flatnonzero(test.arrays["series"] == sid)
        values, mask = sup.normalizer.transform(test.arrays["raw"][rows], m.kind)
        for pos, e in live.items():
            att = e.get("explanation")
            if not att:
                continue
            ref = explain(
                sup.model,
                values[pos - 29 : pos + 1],
                mask[pos - 29 : pos + 1],
                machine_kind=m.kind,
                sensor_parts=m.sensor_parts,
                horizon=30,
                ig_steps=8,
            )
            assert [a["sensor"] for a in att] == [a["sensor"] for a in ref]
            np.testing.assert_allclose(
                [a["weight"] for a in att], [a["weight"] for a in ref], atol=1e-5
            )
            checked += 1
            if checked >= 5:
                return
    if checked == 0:
        pytest.skip("the tiny quick model raised no alarm on the test series")


def test_async_explanations_equal_sync(serve_setup: Any, gru_worker: Any) -> None:
    """The background explainer returns the same attributions, tagged with their window."""
    _, _, ds, _ = serve_setup
    line = load_line_config(CONFIG_DIR / "line.yaml")
    split = ds.split("train")
    for sid in range(len(split.arrays["series_machine"])):
        sync = run_live(gru_worker, split, sid, batch=10)
        want = {seq: e["explanation"] for seq, e in sync.items() if "explanation" in e}
        if want:
            break
    else:
        pytest.skip("the tiny quick model raised no alarm")
    machine = str(split.arrays["series_machine"][sid])
    m = next(x for x in line.machines if x.id == machine)
    conf = configure_message(machine, m.kind.value, dict(m.sensor_parts)) | {"explainAsync": True}
    gru_worker.configure(conf)
    got: dict[int, Any] = {}
    rows = series_rows(split, sid)
    for i in range(0, len(rows), 10):
        res = gru_worker.step({machine: rows[i : i + 10]})
        assert gru_worker.explainer is not None
        gru_worker.explainer.wait_idle(60)  # deterministic: one request at a time
        got |= {x["seq"]: x["attributions"] for x in res["explanations"]}
    got |= {x["seq"]: x["attributions"] for x in gru_worker.explainer.collect()}
    assert got
    for seq, att in got.items():
        assert [a["sensor"] for a in att] == [a["sensor"] for a in want[seq]]
        np.testing.assert_allclose(
            [a["weight"] for a in att], [a["weight"] for a in want[seq]], atol=1e-6
        )
    gru_worker.configure(conf | {"explainAsync": False})


def test_worker_process_matches_in_process(serve_setup: Any, gru_worker: Any) -> None:
    from twin_common.jsonl_process import JsonLinesProcess

    tracking, _, ds, _ = serve_setup
    test: SplitData = ds.split("test")
    sid = int(test.arrays["series"][test.arrays["ends"][-1]])
    machine = str(test.arrays["series_machine"][sid])
    line = load_line_config(CONFIG_DIR / "line.yaml")
    m = next(x for x in line.machines if x.id == machine)
    cmd = [
        sys.executable, "-m", "twin_ml.serve", "worker", "--roles", "supervised,monitor",
        "--supervised", "test-gru", "--anomaly", "test-autoencoder",
        "--tracking-dir", str(tracking), "--cache-dir", str(tracking.parent / "cache_proc"),
        "--ig-steps", "8",
    ]  # fmt: skip
    proc = JsonLinesProcess(cmd, name="test-worker", cwd=REPO_ROOT)
    proc.start()
    try:
        hello = proc.request({"op": "hello"}, 120)
        assert "xgboost" not in hello["process"]["libraries"]
        assert "torch" in hello["process"]["libraries"]
        assert hello["bundles"]["supervised"]["registered"] == "test-gru"
        conf = configure_message(machine, m.kind.value, dict(m.sensor_parts))
        proc.request(conf, 30)
        gru_worker.configure(conf)
        rows = series_rows(test, sid)
        for i in range(0, min(len(rows), 120), 10):
            remote = proc.request({"op": "step", "rows": {machine: rows[i : i + 10]}}, 30)
            local = gru_worker.step({machine: rows[i : i + 10]})
            assert remote["rows"] == json.loads(json.dumps(local["rows"]))
    finally:
        proc.close()
    assert proc.returncode == 0


def test_bundle_meta_and_checks(serve_setup: Any) -> None:
    from twin_ml.serve.bundle import resolve

    tracking, _, ds, _ = serve_setup
    rb = resolve("test-gru", "latest", tracking, tracking.parent / "cache_meta")
    meta = rb.meta
    assert meta["kind"] == "supervised"
    assert meta["model"]["family"] == "nn"
    assert meta["checks"]["test_predictions"]["h30"]["labels_identical"] is True
    assert meta["checks"]["calibration"]["status"].startswith("no reference run")
    assert meta["sustain_s"] == 3
    assert meta["dataset"]["key"] == ds.meta["dataset_key"]
    assert rb.label.startswith("test-gru v1 (gru seed 0)")
    ae = resolve("test-autoencoder", "1", tracking, tracking.parent / "cache_meta_ae")
    assert set(ae.meta["anomaly"]["thresholds"]) == {"99.0", "99.5", "99.9"}


def test_strict_build_needs_the_reference_run(serve_setup: Any, tmp_path: Path) -> None:
    from twin_ml.config import load_deploy_config
    from twin_ml.serve.bundle import BundleCheckError, build_anomaly

    tracking, deploy_path, _, _ = serve_setup
    with pytest.raises(BundleCheckError, match="no none autoencoder seed 0 run"):
        build_anomaly(load_deploy_config(deploy_path), REPO_ROOT, tracking, tmp_path, True)


def test_check_calibration_detects_a_difference() -> None:
    from twin_ml.serve.bundle import BundleCheckError, check_calibration, check_thresholds

    live: dict[str, Any] = {
        "heads": {"30": {"method": "temperature", "temperature": 0.8}},
        "mondrian_q": {"30": [0.1, 0.5, None]},
        "ttf_q": 280.0,
    }
    logged: dict[str, Any] = {
        "heads": {"30": {"method": "temperature", "temperature": 0.8}},
        "conformal": {"30": {"q": {"mondrian_lac": [0.1, 0.5, float("inf")]}}},
        "ttf": {"q_s": 280.0},
    }
    assert check_calibration(live, logged)["ttf_q"] == 280.0
    logged["heads"]["30"]["temperature"] = 0.81
    with pytest.raises(BundleCheckError, match="T="):
        check_calibration(live, logged)
    with pytest.raises(BundleCheckError):
        check_thresholds({"99.9": [1.0, 2.0]}, {"values": {"99.9": [1.0, 2.1]}})


def test_xgboost_bundle_and_parity_in_own_process(ml_dataset: Dataset) -> None:
    from twin_ml.testing import require_xgboost

    require_xgboost()
    run_pytest_isolated(Path(__file__).parent / "xgb_isolated_serve.py", dataset=ml_dataset)


def test_serve_modules_load_no_ml_library_at_import() -> None:
    code = (
        "import sys\n"
        "import twin_ml.serve, twin_ml.serve.buffers, twin_ml.serve.calib, twin_ml.serve.bundle\n"
        "import twin_ml.serve.live, twin_ml.serve.worker, twin_ml.serve.latency\n"
        "import twin_ml.serve.__main__\n"
        "bad = [m for m in ('torch','xgboost','sklearn','mlflow','twin_sim') if m in sys.modules]\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "[]"


def test_latency_report_on_the_quick_bundle(serve_setup: Any, tmp_path: Path) -> None:
    import argparse

    from twin_common.config.loader import read_yaml
    from twin_ml.serve.latency import run

    tracking, deploy_path, ds, _ = serve_setup
    inf = read_yaml(CONFIG_DIR / "inference.yaml")
    inf |= {
        "model": "gru",
        "models": {"gru": {"registered": "test-gru", "family": "nn"}},
        "anomaly": {"registered": "test-autoencoder", "version": "latest"},
        "tracking_dir": str(tracking),
    }
    inf["worker"]["cache_dir"] = str(tmp_path / "cache")
    cfg_path = tmp_path / "inference.yaml"
    cfg_path.write_text(json.dumps(inf), encoding="utf-8")
    out = tmp_path / "live_inference"
    args = argparse.Namespace(
        inference_config=cfg_path,
        line_config=CONFIG_DIR / "line.yaml",
        eval_config=deploy_path.parent / "eval.yaml",
        models="gru",
        ticks=120,
        data_dir=ds.path.parent,
        out=out,
        no_gate_effect=False,
    )
    assert run(args) == 0
    result = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    [lat] = result["latency"]
    assert lat["ticks"] == 120
    assert lat["processes"]["nn"]["libraries"] == ["torch"]
    assert lat["per_machine_ms"]["n"] == lat["ticks_with_windows"] > 0
    assert lat["round_trip_ms"]["all"]["p50"] <= lat["round_trip_ms"]["all"]["max"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert f"{lat['per_machine_ms']['p95']:.2f}" in md  # the table shows the measured numbers
    [ge] = result["gate_effect"]
    assert ge["gate_open_windows"] <= ge["windows"]
    assert set(ge["horizons"]["30"]) >= {"evaluated", "gated", "episodes_changed"}
