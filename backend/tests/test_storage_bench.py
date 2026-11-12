"""The storage benchmark script runs end to end (SQLite only, tiny settings) and its pieces work."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from twin_backend.storage.bench import RateResult, ReadingFactory

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reading_factory_is_seeded() -> None:
    a = [ReadingFactory(3).next() for _ in range(1)]
    first, second = ReadingFactory(3), ReadingFactory(3)
    assert [first.next() for _ in range(50)] == [second.next() for _ in range(50)]
    assert a[0] == ReadingFactory(3).next()
    batch = ReadingFactory(4).batch(400, a[0].ts)
    keys = {(r.item.machine_id, r.item.seq) for r in batch}
    assert len(keys) == 400  # no duplicates
    assert any(r.item.temp is None or r.item.vib is None for r in batch)


def test_kept_up_criterion() -> None:
    ok = RateResult(1000, 0.5, 10.0, 10_000, 9_000, 0, 20, 1.0, 2.0)
    assert ok.backlog == 1_000
    assert ok.sustainable  # 1000 <= 1000/s x 1.5 s
    growing = RateResult(1000, 0.5, 10.0, 10_000, 7_000, 0, 20, 1.0, 2.0)
    assert not growing.sustainable
    dropping = RateResult(1000, 0.5, 10.0, 10_000, 9_900, 1, 20, 1.0, 2.0)
    assert not dropping.sustainable


def test_benchmark_script_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "bench.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twin_backend.storage.bench",
            "--backends",
            "sqlite",
            "--duration",
            "0.5",
            "--rates",
            "200",
            "--rows",
            "500",
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        timeout=120,
    )
    text = out.read_text(encoding="utf-8")
    assert "# Storage write benchmark" in text
    assert "| sqlite | 200 |" in text
    assert "insert, batch 100" in text
