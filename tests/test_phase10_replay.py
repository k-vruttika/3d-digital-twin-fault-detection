"""Phase 10 replay demo: MetroPT-format CSV -> converter -> FileReplayAdapter -> SourceManager
-> event bus, in minimal mode (CLAUDE.md section 1).

The converter runs offline (``twin_ml``); the replay runs in a fresh interpreter that must load
no broker, database, HTTP or ML module, exactly like ``tests/test_phase2_minimal.py``. CI uses a
synthetic file in the exact MetroPT-3 format; the same check on the real file runs locally and
is skipped when ``data/external/metropt3/`` is absent.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from twin_ml.datasets.replay import convert, load_replay_config
from twin_ml.testing_real import real_tree

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
from collections import Counter
from pathlib import Path
from twin_common import aio
from twin_common.config import load_line_config, load_sources_config
from twin_backend.adapters.manager import SourceManager
from twin_backend.bus import InMemoryEventBus, Topic

data_file, forbidden = sys.argv[1], sys.argv[2].split(",")

async def main():
    sources = load_sources_config("config/sources.yaml")
    file_cfg = sources.file.model_copy(update={"path": data_file, "format": "csv", "loop": False,
                                               "replay_speed": 1e6})
    ingest = sources.ingest.model_copy(update={"tick_s": 0.01, "lateness_window_s": 0.05})
    sources = sources.model_copy(update={"file": file_cfg, "ingest": ingest})
    bus = InMemoryEventBus(queue_size=10_000_000)
    sub = bus.subscribe(Topic.READINGS)
    mgr = SourceManager(sources, load_line_config("config/line_single.yaml"), bus, root=Path("."))
    await mgr.start()
    await asyncio.wait_for(mgr.wait_idle(), 300)
    await mgr.stop()
    readings = []
    while (m := sub.get_nowait()) is not None:
        readings.append(m)
    print(json.dumps({
        "readings": len(readings),
        "machines": sorted({r.machine_id for r in readings}),
        "quarantined": mgr.quarantine.total,
        "quality": dict(Counter(str(r.quality.value) for r in readings)),
        "states": dict(Counter(str(r.state.value) for r in readings)),
        "missing_speed": sum(r.speed is None for r in readings),
        "with_temp": sum(r.temp is not None for r in readings),
        "loaded": [x for x in forbidden if x in sys.modules],
    }))

aio.run(main())
"""


def _replay(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(path), ",".join(FORBIDDEN)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out: dict[str, object] = json.loads(result.stdout.strip().splitlines()[-1])
    return out


def _check(summary: dict[str, object], csv_path: Path) -> None:
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert summary["loaded"] == []  # minimal mode: no broker, database, HTTP or ML library
    assert summary["readings"] == len(rows)
    assert summary["machines"] == ["CNV-01"]
    assert summary["quarantined"] == 0
    assert summary["quality"] == {"SUSPECT": len(rows)}  # speed, vib, load are always missing
    assert summary["missing_speed"] == len(rows)
    states = summary["states"]
    assert isinstance(states, dict)
    assert set(states) <= {"RUNNING", "IDLE"}


def test_converter_output_replays_in_minimal_mode(tmp_path: Path) -> None:
    paths = real_tree(tmp_path)
    assert paths  # synthetic MetroPT-3 file + facts under tmp_path
    cfg_path = tmp_path / "replay.yaml"
    base = yaml.safe_load((REPO_ROOT / "config/datasets/metropt3_replay.yaml").read_text("utf-8"))
    base |= {
        "facts": "config/datasets/metropt3.yaml",
        "failure": 1,
        "before_h": 2.0,
        "after_h": 1.0,
        "out": "out/replay.csv",
    }
    cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    summary = convert(load_replay_config(cfg_path), tmp_path)
    out = tmp_path / "out" / "replay.csv"
    assert summary["rows"] > 0
    assert summary["rows_in_failure"] > 0
    assert summary["running_rows"] + summary["idle_rows"] == summary["rows"]
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["seq"]) for r in rows] == list(range(1, len(rows) + 1))
    assert all(r["ts"].endswith("Z") and r["speed"] == "" for r in rows)
    assert rows[0]["run_time"] == "0"
    _check(_replay(out), out)


def test_converter_rejects_unknown_failure(tmp_path: Path) -> None:
    real_tree(tmp_path)
    cfg = load_replay_config(REPO_ROOT / "config/datasets/metropt3_replay.yaml")
    with pytest.raises(ValueError, match="no failure #9"):
        convert(cfg.model_copy(update={"failure": 9}), tmp_path)


def test_real_file_replays_in_minimal_mode(tmp_path: Path) -> None:
    cfg = load_replay_config(REPO_ROOT / "config/datasets/metropt3_replay.yaml")
    facts = yaml.safe_load((REPO_ROOT / cfg.facts).read_text("utf-8"))
    if not (REPO_ROOT / facts["dir"] / facts["files"]["telemetry"]).is_file():
        pytest.skip("MetroPT-3 file not in data/external/metropt3/ (local check only)")
    out = tmp_path / "replay.csv"
    summary = convert(cfg, REPO_ROOT, out=out)
    assert summary["rows_in_failure"] > 0
    _check(_replay(out), out)
