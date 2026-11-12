"""Export to CSV / JSONL: wire format, nulls, quality, order, other streams, and the CLI."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from twin_backend.storage.base import EventStore, Stored, Stream
from twin_backend.storage.export import CSV_COLUMNS, export_range
from twin_backend.storage.sqlite import SqliteEventStore
from twin_common.contracts import Reading
from twin_common.enums import Quality
from twin_common.testing import sample_alert, sample_reading, sample_state_change

T0 = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
ING = datetime(2026, 11, 12, tzinfo=UTC)
ORDER = ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]


def rows() -> list[Stored[Any]]:
    out: list[Stored[Any]] = []
    for s in range(6):
        for m in ("INSP-01", "CNC-01", "CNV-01"):  # not in line order on purpose
            kw: dict[str, Any] = {}
            if s == 2:
                kw = {"temp": None, "quality": Quality.SUSPECT}
            if s == 4 and m == "CNC-01":
                kw = {"quality": Quality.LATE, "vib": 0.1 + 0.2}
            out.append(
                Stored(sample_reading(machine_id=m, seq=s, ts=T0 + timedelta(seconds=s), **kw), ING)
            )
    return out


async def test_jsonl_is_canonical_reading_json(event_store: EventStore, tmp_path: Path) -> None:
    data = rows()
    await event_store.insert(Stream.READINGS, data)
    out = tmp_path / "x.jsonl"
    result = await export_range(event_store, None, None, out, "jsonl", machine_order=ORDER)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert result.rows[Stream.READINGS] == len(lines) == len(data)
    parsed = [Reading.model_validate_json(line) for line in lines]
    assert sorted(parsed, key=repr) == sorted((r.item for r in data), key=repr)
    # (ts, line machine order, seq)
    assert [r.machine_id for r in parsed[:3]] == ["CNV-01", "CNC-01", "INSP-01"]
    first_missing = json.loads(next(line for line in lines if '"temp":null' in line))
    assert first_missing["temp"] is None
    assert first_missing["quality"] == "SUSPECT"


async def test_csv_has_empty_cells_for_missing_and_exact_floats(
    event_store: EventStore, tmp_path: Path
) -> None:
    await event_store.insert(Stream.READINGS, rows())
    out = tmp_path / "x.csv"
    await export_range(
        event_store,
        T0 + timedelta(seconds=2),
        T0 + timedelta(seconds=5),
        out,
        "csv",
        machine_order=ORDER,
    )
    with out.open(encoding="utf-8", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert tuple(table[0]) == CSV_COLUMNS
    assert len(table) == 9  # seconds 2, 3, 4 (end exclusive) x 3 machines
    assert {r["temp"] for r in table if r["ts"].startswith("2026-01-01T06:00:02")} == {""}
    late = next(r for r in table if r["quality"] == "LATE")
    assert float(late["vib"]) == 0.1 + 0.2  # repr keeps every bit


async def test_all_streams_written_next_to_readings(
    event_store: EventStore, tmp_path: Path
) -> None:
    await event_store.insert(Stream.READINGS, rows())
    await event_store.insert(Stream.STATE_CHANGES, [Stored(sample_state_change(ts=T0), ING)])
    await event_store.insert(Stream.ALERTS, [Stored(sample_alert(ts=T0), ING)])
    out = tmp_path / "run.jsonl"
    result = await export_range(event_store, None, None, out, "jsonl", all_streams=True)
    assert result.paths[Stream.STATE_CHANGES] == tmp_path / "run.state_changes.jsonl"
    assert result.rows[Stream.ALERTS] == 1
    assert result.rows[Stream.COMMANDS] == 0
    assert (tmp_path / "run.commands.jsonl").read_text(encoding="utf-8") == ""


def run_cli(config: Path, cwd: Path, *args: str) -> Any:
    """Run the storage CLI in a subprocess (it configures logging process-wide)."""
    result = subprocess.run(
        [sys.executable, "-m", "twin_backend.storage", "--config-dir", str(config), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


async def test_cli_migrate_info_export(tmp_path: Path) -> None:
    repo_config = Path(__file__).resolve().parents[2] / "config"
    config = tmp_path / "config"
    config.mkdir()
    (config / "line.yaml").write_text(
        (repo_config / "line.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    storage_yaml = (repo_config / "storage.yaml").read_text(encoding="utf-8")
    db = (tmp_path / "cli.sqlite").as_posix()
    (config / "storage.yaml").write_text(
        storage_yaml.replace("path: data/store/linetwin.sqlite", f"path: {db}"), encoding="utf-8"
    )
    store = SqliteEventStore(Path(db))
    await store.connect()
    await store.migrate()
    await store.insert(Stream.READINGS, rows())
    await store.close()

    assert run_cli(config, tmp_path, "migrate")["applied"] == []
    assert run_cli(config, tmp_path, "info")["rows"]["readings"] == 18
    out = tmp_path / "exp" / "r.csv"
    exported = run_cli(
        config,
        tmp_path,
        "export",
        "--format",
        "csv",
        "--out",
        str(out),
        "--from",
        "2026-01-01T06:00:01Z",
        "--to",
        "2026-01-01T06:00:03Z",
    )
    assert exported["rows"]["readings"] == 6
    assert out.read_text(encoding="utf-8").splitlines()[1].startswith("CNV-01,1,")
