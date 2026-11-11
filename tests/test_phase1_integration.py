"""Phase 1 integration: the minimal-mode data path (CLAUDE.md section 1).

The simulator CLI writes plain JSONL/CSV files that replay into canonical ``Reading`` contracts
with no database, broker or ML involved. CI also runs ``generate --quick --export csv,jsonl``
and points ``LINETWIN_MINIMAL_DATA_DIR`` at the result, so this test checks the real CLI output.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest

from twin_common.contracts import Reading
from twin_common.enums import MachineState

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = ("asyncpg", "aiomqtt", "asyncua", "torch", "sqlalchemy")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    preset = os.environ.get("LINETWIN_MINIMAL_DATA_DIR")
    if preset:
        return Path(preset)
    out = tmp_path_factory.mktemp("sim_quick")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twin_sim",
            "generate",
            "--quick",
            "--runs",
            "2",
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


def test_jsonl_replays_into_reading_contracts(data_dir: Path) -> None:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    for entry in manifest["run_entries"]:
        lines = (data_dir / entry["files"]["jsonl"]).read_text(encoding="utf-8").splitlines()
        readings = [Reading.model_validate_json(line) for line in lines]
        assert readings, "empty JSONL export"
        total += len(readings)
        # Time-ordered per machine with increasing seq: directly replayable.
        by_machine: dict[str, list[Reading]] = {}
        for r in readings:
            by_machine.setdefault(r.machine_id, []).append(r)
        assert set(by_machine) == {"CNV-01", "CNC-01", "INSP-01", "CNV-02"}
        for rs in by_machine.values():
            assert all(a.seq < b.seq and a.ts < b.ts for a, b in itertools.pairwise(rs))
        assert {r.state for r in readings} <= set(MachineState)
    assert total > 0


def test_csv_and_parquet_agree(data_dir: Path) -> None:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["run_entries"][0]
    pq_table = pq.read_table(data_dir / entry["files"]["parquet"])
    csv_table = pacsv.read_csv(data_dir / entry["files"]["csv"])
    assert csv_table.num_rows == pq_table.num_rows == entry["rows"]
    assert csv_table.column("state").to_pylist() == pq_table.column("state").to_pylist()


def test_simulator_needs_no_db_broker_or_ml() -> None:
    code = (
        "import sys, twin_sim.batch, twin_sim.runner, twin_sim.line;"
        f"bad=[m for m in {FORBIDDEN_MODULES!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == ""
