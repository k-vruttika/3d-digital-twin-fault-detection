"""Single-station mode: config/line_single.yaml (CNV-01 only, no buffers)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from twin_common.config.line import LineConfig
from twin_common.config.loader import ConfigError
from twin_common.contracts import Command, Reading
from twin_common.enums import AckStatus, CommandType, MachineState
from twin_sim.__main__ import main
from twin_sim.batch import MANIFEST_NAME, generate
from twin_sim.config import (
    FaultSchedule,
    FaultType,
    RecoveryParams,
    ScenarioConfig,
    ScheduledFault,
    SimConfig,
    SourceParams,
    check_scenario_compat,
    load_scenario,
)
from twin_sim.line import SOURCE_ID, LineSimulator, SimSnapshot
from twin_sim.scenario import make_run_plan


def _sim(line: LineConfig, sim_cfg: SimConfig, sc: ScenarioConfig, seed: int = 1) -> LineSimulator:
    return LineSimulator(line, sim_cfg, make_run_plan(sc, sim_cfg, line, seed, 0))


def test_single_line_config(single_line_cfg: LineConfig) -> None:
    assert [m.id for m in single_line_cfg.machines] == ["CNV-01"]
    assert single_line_cfg.buffers == []


def test_healthy_single_station_runs_without_flow_states(
    single_line_cfg: LineConfig, sim_cfg: SimConfig
) -> None:
    sim = _sim(single_line_cfg, sim_cfg, ScenarioConfig(name="s", duration_s=1200))
    frames = sim.run()
    assert all(len(f.samples) == 1 for f in frames)
    assert {f.samples[0].state for f in frames} == {MachineState.RUNNING}
    assert {(f.samples[0].buffer_in, f.samples[0].buffer_out) for f in frames} == {(-1, -1)}
    # Throughput is set by the source: one part per 6.5 s on average.
    assert sim.produced == pytest.approx(1200 / 6.5, abs=3)


def test_single_station_faults_and_root_cause(
    single_line_cfg: LineConfig, sim_cfg: SimConfig
) -> None:
    sc = ScenarioConfig(
        name="s",
        duration_s=3000,
        load_ranges=["high"],
        recovery=RecoveryParams(auto=False),
        faults=FaultSchedule(
            scheduled=[ScheduledFault(type=FaultType.BEARING_WEAR, machine="CNV-01", onset_s=100)]
        ),
    )
    sim = _sim(single_line_cfg, sim_cfg, sc)
    states = [f.samples[0] for f in sim.run()]
    first_warn = next(k for k, s in enumerate(states) if s.state is MachineState.WARNING)
    first_fault = next(k for k, s in enumerate(states) if s.state is MachineState.FAULT)
    assert first_warn < first_fault
    assert first_fault + 1 - 100 >= sim_cfg.validation.min_lead_s
    assert states[first_fault].root_cause == "CNV-01"
    assert states[first_fault].fault_type == "bearing_wear"


def test_source_starvation_and_commands(single_line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    sc = ScenarioConfig(
        name="s", duration_s=600, source=SourceParams(arrival_interval_s=100.0, jitter_frac=0.0)
    )
    sim = _sim(single_line_cfg, sim_cfg, sc)
    sim.fast_forward(60)
    assert sim.machine("CNV-01").state is MachineState.STARVED
    assert sim.machine("CNV-01").root_cause == SOURCE_ID
    ack = sim.apply_command(
        Command(
            command_id="c",
            idempotency_key="k",
            machine_id="CNV-01",
            type=CommandType.STOP,
            issued_by="t",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    assert ack.status is AckStatus.EXECUTED
    assert sim.machine("CNV-01").state is MachineState.IDLE


def test_single_station_snapshot_roundtrip(single_line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    sc = ScenarioConfig(
        name="s",
        duration_s=900,
        faults=FaultSchedule(
            scheduled=[ScheduledFault(type=FaultType.BELT_JAM, machine="CNV-01", onset_s=250)]
        ),
    )
    sim = _sim(single_line_cfg, sim_cfg, sc)
    sim.fast_forward(240)
    snap = SimSnapshot.model_validate_json(sim.snapshot().model_dump_json())
    expected = [f.samples for f in sim.fast_forward(200, record=True)]
    restored = LineSimulator.from_snapshot(single_line_cfg, sim_cfg, snap)
    assert [f.samples for f in restored.fast_forward(200, record=True)] == expected


def test_full_line_scenarios_rejected_on_single_line(
    single_line_cfg: LineConfig, sim_cfg: SimConfig, config_dir: Path
) -> None:
    quick = load_scenario(config_dir / "scenarios" / "quick.yaml")
    with pytest.raises(
        ConfigError, match="unknown machine 'CNC-01' \\(not on line 'line1_single'\\)"
    ):
        check_scenario_compat(quick, sim_cfg, single_line_cfg)
    check_scenario_compat(
        load_scenario(config_dir / "scenarios" / "quick_single.yaml"), sim_cfg, single_line_cfg
    )


def test_generate_single_line(
    single_line_cfg: LineConfig, sim_cfg: SimConfig, config_dir: Path, tmp_path: Path
) -> None:
    sc = load_scenario(config_dir / "scenarios" / "quick_single.yaml").model_copy(
        update={"duration_s": 300.0}
    )
    manifest = generate(
        line=single_line_cfg,
        sim=sim_cfg,
        scenario=sc,
        runs=2,
        base_seed=4,
        out_dir=tmp_path,
        exports=("jsonl",),
        workers=1,
    )
    for entry in manifest["run_entries"]:
        table = pq.read_table(tmp_path / entry["files"]["parquet"])
        assert table.num_rows == 300
        assert set(table.column("machine_id").to_pylist()) == {"CNV-01"}
        assert table.column("buffer_in").null_count == 300
        assert table.column("buffer_out").null_count == 300
        lines = (tmp_path / entry["files"]["jsonl"]).read_text(encoding="utf-8").splitlines()
        assert all(Reading.model_validate_json(x).machine_id == "CNV-01" for x in lines)


def test_cli_line_option(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(config_dir.parent)
    single = str(config_dir / "line_single.yaml")
    scenario = str(config_dir / "scenarios" / "quick_single.yaml")
    args = ["--quick", "--runs", "1", "--duration", "60", "--workers", "1", "--out", str(tmp_path)]
    assert main(["generate", "--line", single, "--scenario", scenario, *args]) == 0
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["run_entries"][0]["rows"] == 60
    capsys.readouterr()

    # Without a single-line scenario, --quick picks quick.yaml, which references CNC-01.
    assert main(["generate", "--line", single, *args]) == 2
    assert "not on line 'line1_single'" in capsys.readouterr().err

    rc = main(
        ["run", "--line", single, "--scenario", scenario, "--speed", "1000", "--duration", "3"]
    )
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    readings = [Reading.model_validate_json(x) for x in out]
    assert [r.machine_id for r in readings] == ["CNV-01"] * 3
