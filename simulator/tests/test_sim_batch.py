from __future__ import annotations

import json
from pathlib import Path

import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest

from twin_common.config.line import LineConfig
from twin_common.contracts import Reading
from twin_sim.__main__ import main
from twin_sim.batch import MANIFEST_NAME, content_sha256, file_sha256, generate
from twin_sim.config import ScenarioConfig, SimConfig
from twin_sim.recorder import SCHEMA


def test_generate_writes_files_and_manifest(
    line_cfg: LineConfig, sim_cfg: SimConfig, tmp_path: Path
) -> None:
    sc = ScenarioConfig(name="b", duration_s=120)
    manifest = generate(
        line=line_cfg,
        sim=sim_cfg,
        scenario=sc,
        runs=2,
        base_seed=3,
        out_dir=tmp_path,
        exports=("csv", "jsonl"),
        workers=1,
    )
    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["runs"] == 2
    assert on_disk["base_seed"] == 3
    assert on_disk["held_out_fault_types"] == ["lubrication_loss"]
    assert manifest["run_entries"][1]["seed"] == {"entropy": 3, "spawn_key": [1]}
    for entry in on_disk["run_entries"]:
        table = pq.read_table(tmp_path / entry["files"]["parquet"])
        assert table.schema.equals(SCHEMA)
        assert table.num_rows == entry["rows"] == 480
        assert pacsv.read_csv(tmp_path / entry["files"]["csv"]).num_rows == 480
        lines = (tmp_path / entry["files"]["jsonl"]).read_text(encoding="utf-8").splitlines()
        readings = [Reading.model_validate_json(x) for x in lines]
        assert len(readings) == 480
        assert set(entry["thresholds"]) == {"CNV-01", "CNC-01", "INSP-01", "CNV-02"}


def test_manifest_hashes_files_and_content(
    line_cfg: LineConfig, sim_cfg: SimConfig, tmp_path: Path
) -> None:
    sc = ScenarioConfig(name="h", duration_s=60)

    def gen(out: Path, workers: int) -> dict[str, object]:
        return generate(
            line=line_cfg,
            sim=sim_cfg,
            scenario=sc,
            runs=2,
            base_seed=5,
            out_dir=out,
            exports=("csv",),
            workers=workers,
            cli_args={"out": str(out)},
        )

    first = gen(tmp_path / "a", 1)
    second = gen(tmp_path / "b", 2)
    for entry in first["run_entries"]:  # type: ignore[attr-defined]
        for kind, name in entry["files"].items():
            assert entry["sha256"][kind] == file_sha256(tmp_path / "a" / name)
    # Identical data -> identical content hash, despite different paths, workers and times.
    assert first["content_sha256"] == second["content_sha256"] == content_sha256(first)
    assert first["cli_args"] != second["cli_args"]
    on_disk = json.loads((tmp_path / "a" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["content_sha256"] == first["content_sha256"]
    # Different data -> different content hash.
    other = generate(
        line=line_cfg, sim=sim_cfg, scenario=sc, runs=2, base_seed=6, out_dir=tmp_path / "c"
    )
    assert other["content_sha256"] != first["content_sha256"]


def test_generate_rejects_bad_args(
    line_cfg: LineConfig, sim_cfg: SimConfig, tmp_path: Path
) -> None:
    sc = ScenarioConfig(name="b", duration_s=10)
    with pytest.raises(ValueError, match="export"):
        generate(
            line=line_cfg,
            sim=sim_cfg,
            scenario=sc,
            runs=1,
            base_seed=1,
            out_dir=tmp_path,
            exports=("xlsx",),
        )
    with pytest.raises(ValueError, match="runs"):
        generate(line=line_cfg, sim=sim_cfg, scenario=sc, runs=0, base_seed=1, out_dir=tmp_path)


def test_cli_generate_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.chdir(config_dir.parent)
    rc = main(
        [
            "generate",
            "--quick",
            "--runs",
            "2",
            "--duration",
            "60",
            "--workers",
            "1",
            "--out",
            str(tmp_path),
            "--export",
            "jsonl",
        ]
    )
    assert rc == 0
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["scenario"]["name"] == "quick"
    assert manifest["duration_s"] == 60
    assert len(list(tmp_path.glob("run_*.parquet"))) == 2


def test_cli_reports_config_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["generate", "--scenario", str(tmp_path / "missing.yaml"), "--out", str(tmp_path)])
    assert rc == 2
    assert "config file not found" in capsys.readouterr().err
