"""Phase 11 Part C: drift through the real worker pool (ADR 0031).

The backend's ``WorkerPool`` starts a real worker process (quick test bundles), configures it
with the live drift settings, and merges the answers. Clean rows give no drift; the same rows
with a load far outside the training range give a drift report that raises and is carried on
the merged entries the inference service reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from twin_backend.inference.workers import WorkerPool
from twin_common.config import load_line_config
from twin_common.config.inference import InferenceConfig
from twin_common.config.loader import read_yaml
from twin_ml.testing import series_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config"
LOAD_COL = 6  # a series row is [seq, ts, state, speed, temp, vib, load, current]


def quick_config(tracking: Path, cache: Path) -> InferenceConfig:
    data = read_yaml(CONFIG / "inference.yaml")
    data |= {
        "enabled": True,
        "model": "gru",
        "models": {"gru": {"registered": "test-gru", "family": "nn"}},
        "anomaly": {"registered": "test-autoencoder", "version": "latest"},
        "tracking_dir": str(tracking),
    }
    data["worker"]["cache_dir"] = str(cache)
    data["drift"] |= {"threshold_margin": 1.5, "sustain_s": 10, "clear_s": 20}
    return InferenceConfig.model_validate(data)


def longest_series(split: Any) -> int:
    ids, counts = np.unique(split.arrays["series"], return_counts=True)
    return int(ids[int(np.argmax(counts))])


async def run_rows(pool: WorkerPool, machine: str, rows: list[list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(rows), 50):
        res = await pool.step({machine: rows[i : i + 50]})
        out += res["rows"][machine]
    return out


async def test_drift_reports_flow_through_the_worker_pool(serve_setup: Any, tmp_path: Path) -> None:
    tracking, _, ds, _ = serve_setup
    cfg = quick_config(tracking, tmp_path / "cache")
    split = ds.split("train")
    sid = longest_series(split)
    machine = str(split.arrays["series_machine"][sid])
    line = load_line_config(CONFIG / "line.yaml")
    rows = series_rows(split, sid)
    pool = WorkerPool(cfg, line, REPO_ROOT)
    try:
        info = await pool.start()
        assert info.bundles["anomaly"]["drift"]["available"] is True
        assert pool.configure_message()["drift"]["sustainS"] == 10
        clean = await run_rows(pool, machine, rows)
        reports = [e["drift"] for e in clean if "drift" in e]
        assert reports, "operating rows carry drift reports through the pool's merge"
        assert not any(r["drifted"] for r in reports)

        # The quick test bundles are tiny: their calibrated alert level is not meaningful, so the
        # shifted run uses a low margin (the decision logic is tested in ml/tests/test_ml_drift.py).
        pool.cfg = pool.cfg.model_copy(
            update={"drift": pool.cfg.drift.model_copy(update={"threshold_margin": 0.01})}
        )
        await pool.reconfigure()  # empty buffers and fresh monitors
        shifted_rows = [
            [*r[:LOAD_COL], (r[LOAD_COL] or 0.0) + 200.0, *r[LOAD_COL + 1 :]] for r in rows
        ]
        shifted = await run_rows(pool, machine, shifted_rows)
        drifted = [e["drift"] for e in shifted if "drift" in e and e["drift"]["drifted"]]
        assert drifted, "a load far above the training range is drift"
        first = drifted[0]
        assert first["psi"]["load"] >= first["alert"]["load"]
        assert first["since"] is not None
    finally:
        await pool.close()


async def test_drift_can_be_switched_off_in_the_settings(serve_setup: Any, tmp_path: Path) -> None:
    tracking, _, ds, _ = serve_setup
    cfg = quick_config(tracking, tmp_path / "cache_off")
    cfg = cfg.model_copy(update={"drift": cfg.drift.model_copy(update={"enabled": False})})
    split = ds.split("train")
    sid = longest_series(split)
    machine = str(split.arrays["series_machine"][sid])
    pool = WorkerPool(cfg, load_line_config(CONFIG / "line.yaml"), REPO_ROOT)
    try:
        await pool.start()
        assert pool.configure_message()["drift"] is None
        entries = await run_rows(pool, machine, series_rows(split, sid)[:120])
        assert not any("drift" in e for e in entries)
    finally:
        await pool.close()
