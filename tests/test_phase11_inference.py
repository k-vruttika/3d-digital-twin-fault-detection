"""Phase 11 live inference end to end with real worker processes (ADR 0030).

Uses the quick deploy bundles of ``twin_ml.testing.serve_setup`` (a tiny GRU and autoencoder
registered in a temporary MLflow store).

* In process: a file-source runtime with inference on publishes Prediction events; every one
  equals what an independent in-process worker computes for the same readings (live parity
  through the backend, the bus and the JSON transport).
* As a separate server process (``serve --mode minimal --inference``): the API process
  imports no ML library while the workers run, ``/predict`` reports the deployed bundles,
  predictions are stored, ``/predict/history`` and ``/replay`` return them.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from twin_backend.bus.base import Topic
from twin_backend.testing import SimExport, drain_readings, file_runtime, free_port
from twin_common.config.inference import InferenceConfig, load_inference_config
from twin_common.config.loader import read_yaml
from twin_common.contracts import Prediction, Reading

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_S = 240.0


def quick_inference_config(tracking: Path, cache: Path) -> dict[str, Any]:
    data = read_yaml(REPO_ROOT / "config" / "inference.yaml")
    data |= {
        "enabled": True,
        "model": "gru",
        "models": {"gru": {"registered": "test-gru", "family": "nn"}},
        "anomaly": {"registered": "test-autoencoder", "version": "latest"},
        "tracking_dir": str(tracking),
    }
    data["worker"]["cache_dir"] = str(cache)
    return data


def rows_of(readings: list[Reading]) -> dict[str, list[list[Any]]]:
    out: dict[str, list[list[Any]]] = {}
    for r in sorted(readings, key=lambda x: (x.machine_id, x.seq)):
        out.setdefault(r.machine_id, []).append(
            [r.seq, r.ts.timestamp(), r.state.value, r.speed, r.temp, r.vib, r.load, r.current]
        )
    return out


async def test_backend_predictions_equal_a_worker_on_the_same_readings(
    serve_setup: Any, sim_export: SimExport, tmp_path: Path
) -> None:
    from twin_ml.serve.worker import load_worker

    tracking = serve_setup[0]
    cfg = InferenceConfig.model_validate(quick_inference_config(tracking, tmp_path / "cache"))
    rt = file_runtime(sim_export, tmp_path, inference_cfg=cfg)
    assert rt.inference is not None
    readings_sub = rt.bus.subscribe(Topic.READINGS)
    preds_sub = rt.bus.subscribe(Topic.PREDICTIONS)
    await rt.inference.start()
    assert await rt.inference.wait_ready(STARTUP_TIMEOUT_S), rt.inference.status
    await rt.start()
    try:
        await rt.manager.wait_idle()
        preds: list[Prediction] = []
        quiet = 0
        while quiet < 10:  # until no new prediction for 10 batch intervals
            await asyncio.sleep(cfg.batch_interval_s)
            new = drain_readings(preds_sub)
            preds += new
            quiet = 0 if new else quiet + 1
        info = rt.inference.info
    finally:
        await rt.stop()
    readings = [r for r in drain_readings(readings_sub) if isinstance(r, Reading)]
    assert preds, "no predictions were published"
    assert info.processes["nn"]["libraries"] == ["torch"]

    worker = load_worker(
        ["supervised", "monitor"],
        ("test-gru", "latest"),
        ("test-autoencoder", "latest"),
        tracking,
        tmp_path / "cache_ref",
        cfg.ig_steps,
    )
    worker.configure(rt.inference.pool.configure_message())
    ref = worker.step(rows_of(readings))["rows"]
    expected = {(m, e["ts"]): e["out"] for m, entries in ref.items() for e in entries if "out" in e}
    checked = 0
    for p in preds:
        o = expected[(p.machine_id, p.ts.timestamp())]
        h = str(int(p.horizon_s))
        # float32 networks: sums depend slightly on the batch composition (ATOL as offline)
        assert [p.probs[c] for c in p.probs] == pytest.approx(o["cal"][h], abs=1e-5)
        assert [c.value for c in p.conformal_set] == o["set"][h]
        # float32 MLP: sums depend slightly on the batch size (service batches vs one batch)
        assert p.anomaly_score == pytest.approx(o["anomalyScore"], rel=1e-5, abs=1e-8)
        assert p.ttf_s == pytest.approx(o["ttf"], rel=1e-5)
        assert p.calibrated
        assert p.model_version == info.model_version
        checked += 1
    assert checked == len(preds)  # every prediction belongs to a gated window of the reference


# ---------------------------------------------------------------------------- server process
@pytest.fixture(scope="module")
def inference_server(
    serve_setup: Any, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Path]]:
    tracking = serve_setup[0]
    tmp = tmp_path_factory.mktemp("p11_server")
    data = tmp / "data"
    subprocess.run(
        [
            *(sys.executable, "-m", "twin_sim", "generate", "--quick", "--runs", "1"),
            *("--duration", "300", "--workers", "1", "--export", "csv", "--out", str(data)),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    config = tmp / "config"
    shutil.copytree(REPO_ROOT / "config", config)
    (config / "inference.yaml").write_text(
        json.dumps(quick_inference_config(tracking, tmp / "cache")), encoding="utf-8"
    )
    load_inference_config(config / "inference.yaml")  # valid
    port = free_port()
    env = dict(os.environ)
    env |= {
        "LINETWIN_API_KEYS_OPERATOR": "",
        "LINETWIN_API_KEYS_VIEWER": "",
        "LINETWIN_CONFIG_DIR": str(config),
    }
    cmd = [
        *(sys.executable, "-m", "twin_backend", "serve", "--mode", "minimal", "--inference"),
        *("--file", str((data / "run_0000.csv").resolve()), "--replay-speed", "4"),
        *("--no-loop", "--sqlite-path", str(tmp / "p11.sqlite"), "--port", str(port)),
    ]
    log_path = tmp / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited early:\n{log_path.read_text(encoding='utf-8')}")
            try:
                body = httpx.get(f"{base}/predict", timeout=2.0).json()
                if any(m["predictions"] for m in body["machines"].values()):
                    break
                if body["inference"]["state"] == "failed":
                    pytest.fail(f"inference failed: {body['inference']['reason']}")
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"no predictions in time:\n{log_path.read_text(encoding='utf-8')}")
            time.sleep(0.5)
        yield base, log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def test_api_process_stays_ml_free_with_inference_on(inference_server: Any) -> None:
    base, _ = inference_server
    health = httpx.get(f"{base}/health", params={"modules": "true"}).json()
    assert health["loadedOptionalModules"] == []  # the ML runs in the worker processes


def test_predict_reports_bundles_gate_and_predictions(inference_server: Any) -> None:
    base, _ = inference_server
    body = httpx.get(f"{base}/predict").json()
    inf = body["inference"]
    assert inf["enabled"] is True
    assert inf["state"] == "ready"
    assert inf["supervised"]["registered"] == "test-gru"
    assert inf["anomaly"]["registered"] == "test-autoencoder"
    assert inf["supervised"]["card"]["trainingData"] == "quick test data"
    assert inf["windowS"] == 30
    assert inf["primaryHorizonS"] == 30
    predicting = {m: v for m, v in body["machines"].items() if v["predictions"]}
    assert predicting
    _, v = next(iter(predicting.items()))
    assert set(v["predictions"]) == {"10", "30"}
    assert v["predictions"]["30"]["calibrated"] is True
    assert v["gate"]["window"] == 30


def test_predictions_are_stored_and_replayed(inference_server: Any) -> None:
    base, _ = inference_server
    deadline = time.monotonic() + 60.0
    hist: list[dict[str, Any]] = []
    while not hist:  # the storage writer flushes in batches
        rng = httpx.get(f"{base}/replay/range").json()
        if rng["start"] is not None:
            hist = httpx.get(
                f"{base}/predict/history",
                params={"from": rng["start"], "to": "2100-01-01T00:00:00Z"},
            ).json()["predictions"]
        if not hist:
            assert time.monotonic() < deadline, "no stored predictions within 60 s"
            time.sleep(0.5)
    last = hist[-1]
    t = last["ts"]
    replay = httpx.get(f"{base}/replay", params={"t": t}).json()
    stored = replay["predictions"][last["machineId"]][f"{last['horizonS']:g}"]
    assert stored["ts"] == last["ts"]
    assert stored["probs"] == last["probs"]


def test_predict_without_inference_says_off(sim_export: SimExport, tmp_path: Path) -> None:
    from twin_backend.api.routes.predict import inference_info

    rt = file_runtime(sim_export, tmp_path)
    assert rt.inference is None

    class Ctx:
        runtime = rt

    info = inference_info(Ctx())
    assert info.enabled is False
    assert info.state == "off"
