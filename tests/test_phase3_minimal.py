"""Minimal mode with persistence (CLAUDE.md section 1): file -> ingest -> bus -> SQLite store.

Runs ``build_runtime`` in a fresh interpreter with the shipped configs (storage backend
``sqlite``), so the check "no broker, database driver, HTTP or ML module was imported" is
meaningful. Afterwards the store holds every reading and ``reconstruct`` works. CI runs it via
``uv run poe minimal-data-check`` on the ``generate --quick`` output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "aiomqtt",
    "paho",
    "asyncua",
    "websockets",
    "httpx",
    "fastapi",
    "uvicorn",
    "asyncpg",
    "sqlalchemy",
    "torch",
    "sklearn",
    "xgboost",
    "mlflow",
    "twin_ml",
    "onnxruntime",
)

SCRIPT = r"""
import asyncio, json, sys
from pathlib import Path
from twin_common import aio
from twin_common.config import load_sources_config, load_storage_config
from twin_common.config.settings import TwinSettings
from twin_common.config.sources import FieldMapping
from twin_backend.runtime import build_runtime
from twin_backend.storage.base import Stream

data_file, fmt, db, forbidden = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")

async def main():
    sources = load_sources_config("config/sources.yaml")
    file_cfg = sources.file.model_copy(update={"path": data_file, "format": fmt, "loop": False,
                                               "replay_speed": 1e6})
    if fmt == "jsonl":
        file_cfg = file_cfg.model_copy(update={"mapping": FieldMapping()})
    ingest = sources.ingest.model_copy(update={"tick_s": 0.01, "lateness_window_s": 0.05})
    sources = sources.model_copy(update={"file": file_cfg, "ingest": ingest})
    storage = load_storage_config("config/storage.yaml")
    assert storage.backend == "sqlite"
    storage = storage.model_copy(update={"sqlite": storage.sqlite.model_copy(update={"path": db})})
    rt = build_runtime(TwinSettings(), root=Path("."), sources=sources, storage_cfg=storage)
    await rt.start()
    await asyncio.wait_for(rt.manager.wait_idle(), 120)
    await rt.stop()

    store = rt.storage.store
    await store.connect()
    count = await store.count(Stream.READINGS)
    span = await store.min_max_ts([Stream.READINGS])
    state = await rt.storage.reconstruct(span[1])
    await store.close()
    print(json.dumps({
        "stored": count,
        "machines": sorted(m for m, v in state.machines.items() if v.last_reading is not None),
        "quarantine_rows": rt.manager.quarantine.total,
        "dropped": rt.storage.metrics.dropped_total(),
        "loaded": [x for x in forbidden if x in sys.modules],
    }))

aio.run(main())
"""


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    preset = os.environ.get("LINETWIN_MINIMAL_DATA_DIR")
    if preset:
        return Path(preset)
    out = tmp_path_factory.mktemp("sim_quick_p3")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twin_sim",
            "generate",
            "--quick",
            "--runs",
            "1",
            "--duration",
            "300",
            "--workers",
            "1",
            "--export",
            "csv,jsonl",
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return out


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_minimal_mode_persists_to_sqlite_without_services(
    data_dir: Path, fmt: str, tmp_path: Path
) -> None:
    data_file = (data_dir / f"run_0000.{fmt}").resolve()
    expected = sum(
        1
        for line in (data_dir / "run_0000.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    db = tmp_path / "minimal.sqlite"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(data_file), fmt, str(db), ",".join(FORBIDDEN)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["loaded"] == []  # no broker, database driver, HTTP or ML library
    assert summary["stored"] == expected
    assert summary["machines"] == ["CNC-01", "CNV-01", "CNV-02", "INSP-01"]
    assert summary["dropped"] == 0
    assert db.is_file()
