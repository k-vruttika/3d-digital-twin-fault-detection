"""Minimal mode (CLAUDE.md section 1): file -> FileReplayAdapter -> ingest -> event bus.

Runs in a fresh interpreter so the check "no broker, database, HTTP or ML module was imported"
is meaningful. CI runs it through ``uv run poe minimal-data-check`` on the real
``generate --quick`` output (``LINETWIN_MINIMAL_DATA_DIR``); without it, a small export is
generated here.
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
from twin_common.config import load_line_config, load_sources_config
from twin_common.enums import AdapterKind
from twin_backend.adapters.manager import SourceManager
from twin_backend.bus import InMemoryEventBus, Topic

data_file, fmt, forbidden = sys.argv[1], sys.argv[2], sys.argv[3].split(",")

async def main():
    sources = load_sources_config("config/sources.yaml")
    assert sources.active is AdapterKind.FILE
    file_cfg = sources.file.model_copy(update={"path": data_file, "format": fmt, "loop": False,
                                               "replay_speed": 1e6})
    if fmt == "jsonl":
        from twin_common.config.sources import FieldMapping
        file_cfg = file_cfg.model_copy(update={"mapping": FieldMapping()})
    ingest = sources.ingest.model_copy(update={"tick_s": 0.01, "lateness_window_s": 0.05})
    sources = sources.model_copy(update={"file": file_cfg, "ingest": ingest})
    bus = InMemoryEventBus(queue_size=10_000_000)
    sub = bus.subscribe(Topic.READINGS)
    mgr = SourceManager(sources, load_line_config("config/line.yaml"), bus, root=Path("."))
    await mgr.start()
    await asyncio.wait_for(mgr.wait_idle(), 120)
    await mgr.stop()
    readings = []
    while (m := sub.get_nowait()) is not None:
        readings.append(m)
    m = mgr.pipeline.metrics
    print(json.dumps({
        "readings": len(readings),
        "machines": sorted({r.machine_id for r in readings}),
        "quarantined": mgr.quarantine.total,
        "duplicates": m.value("duplicates", source="file"),
        "loaded": [x for x in forbidden if x in sys.modules],
    }))

aio.run(main())
"""


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    preset = os.environ.get("LINETWIN_MINIMAL_DATA_DIR")
    if preset:
        return Path(preset)
    out = tmp_path_factory.mktemp("sim_quick_p2")
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
def test_file_replay_through_ingest_without_services(data_dir: Path, fmt: str) -> None:
    data_file = data_dir / f"run_0000.{fmt}"
    expected = sum(
        1
        for line in (data_dir / "run_0000.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(data_file), fmt, ",".join(FORBIDDEN)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["loaded"] == []  # no broker, database, HTTP or ML library
    assert summary["readings"] == expected
    assert summary["machines"] == ["CNC-01", "CNV-01", "CNV-02", "INSP-01"]
    assert summary["duplicates"] == 0
    if fmt == "jsonl":
        assert summary["quarantined"] == 0
