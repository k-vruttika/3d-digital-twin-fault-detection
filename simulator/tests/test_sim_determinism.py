from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from twin_common.config.line import LineConfig
from twin_sim.batch import RunTask, generate, simulate_run
from twin_sim.config import FaultSchedule, FaultType, RandomFaults, ScenarioConfig, SimConfig

SCENARIO = ScenarioConfig(
    name="det",
    duration_s=900,
    faults=FaultSchedule(
        random=RandomFaults(
            rate_per_machine_per_hour=4.0,
            types=[FaultType.BEARING_WEAR, FaultType.BELT_JAM, FaultType.OVERHEATING],
            earliest_onset_s=30,
        )
    ),
)


def _table(line: LineConfig, sim: SimConfig, seed: int, idx: int = 0) -> object:
    _, rec, _ = simulate_run(RunTask(line, sim, SCENARIO, seed, idx, Path("."), ()))
    return rec.to_table()


def test_same_seed_identical(line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    assert _table(line_cfg, sim_cfg, 42).equals(_table(line_cfg, sim_cfg, 42))  # type: ignore[attr-defined]


def test_different_seed_or_run_differs(line_cfg: LineConfig, sim_cfg: SimConfig) -> None:
    a = _table(line_cfg, sim_cfg, 42)
    assert not a.equals(_table(line_cfg, sim_cfg, 43))  # type: ignore[attr-defined]
    assert not a.equals(_table(line_cfg, sim_cfg, 42, idx=1))  # type: ignore[attr-defined]


def test_output_independent_of_worker_count(
    line_cfg: LineConfig, sim_cfg: SimConfig, tmp_path: Path
) -> None:
    kwargs = {"line": line_cfg, "sim": sim_cfg, "scenario": SCENARIO, "runs": 3, "base_seed": 5}
    generate(out_dir=tmp_path / "w1", workers=1, **kwargs)  # type: ignore[arg-type]
    generate(out_dir=tmp_path / "w2", workers=2, **kwargs)  # type: ignore[arg-type]
    for i in range(3):
        name = f"run_{i:04d}.parquet"
        assert pq.read_table(tmp_path / "w1" / name).equals(pq.read_table(tmp_path / "w2" / name))
