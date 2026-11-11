from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from twin_common.config.line import LineConfig
from twin_sim.batch import RunTask, simulate_run
from twin_sim.config import FaultSchedule, FaultType, RandomFaults, ScenarioConfig, SimConfig
from twin_sim.labels import time_to_fault
from twin_sim.recorder import SCHEMA


def test_time_to_fault_unit() -> None:
    t = np.arange(1.0, 11.0)
    fault = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    ttf, cens = time_to_fault(t, fault, cap_s=2.0)
    assert ttf.tolist() == [2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 0.0]
    assert cens.tolist() == [False] * 5 + [True] * 5


def test_time_to_fault_no_fault_and_leading_fault() -> None:
    t = np.arange(0.0, 4.0)
    ttf, cens = time_to_fault(t, np.zeros(4, dtype=bool), cap_s=100.0)
    assert ttf.tolist() == [3.0, 2.0, 1.0, 0.0]
    assert cens.all()
    ttf, cens = time_to_fault(t, np.array([1, 0, 1, 0], dtype=bool), cap_s=100.0)
    assert ttf.tolist() == [0.0, 1.0, 0.0, 0.0]  # last row: lower bound t_end - t = 0
    assert cens.tolist() == [False, False, False, True]


@pytest.fixture(scope="module")
def table(line_cfg: LineConfig, sim_cfg: SimConfig) -> pa.Table:
    sc = ScenarioConfig(
        name="labels",
        duration_s=3000,
        faults=FaultSchedule(
            random=RandomFaults(
                rate_per_machine_per_hour=3.0,
                types=[FaultType.BEARING_WEAR, FaultType.OVERHEATING, FaultType.BELT_JAM],
                earliest_onset_s=30,
            )
        ),
    )
    _, rec, _ = simulate_run(RunTask(line_cfg, sim_cfg, sc, 77, 0, Path("."), ()))
    return rec.to_table()


def test_schema_and_columns(table: pa.Table) -> None:
    assert table.schema.equals(SCHEMA)
    assert table.num_rows == 4 * 3000
    assert set(table.column("load_range").to_pylist()) <= {"low", "medium", "high"}


def test_generated_labels_consistent(table: pa.Table, sim_cfg: SimConfig) -> None:
    cap = sim_cfg.labels.ttf_cap_s
    n_faults = 0
    for mid in ("CNV-01", "CNC-01", "INSP-01", "CNV-02"):
        rows = table.filter(pa.compute.equal(table.column("machine_id"), mid))
        state = np.array(rows.column("state").to_pylist())
        ttf = rows.column("time_to_fault_s").to_numpy()
        cens = rows.column("ttf_censored").to_numpy(zero_copy_only=False)
        t = rows.column("t_s").to_numpy()
        is_fault = state == "FAULT"
        onsets = np.flatnonzero(is_fault & ~np.concatenate(([False], is_fault[:-1])))
        n_faults += len(onsets)
        assert np.all(ttf[is_fault] == 0.0)
        for k in range(len(t)):
            ahead = onsets[onsets >= k]
            if is_fault[k]:
                continue
            if len(ahead) == 0:
                assert cens[k], f"{mid} row {k}: censored row expected"
                assert ttf[k] == min(t[-1] - t[k], cap)
            else:
                assert not cens[k]
                assert ttf[k] == min(t[ahead[0]] - t[k], cap)
        # Counts down by 1 s per row in the last `cap` seconds before every onset.
        for o in onsets:
            lo = max(0, o - int(cap) + 1)
            window = ttf[lo:o]
            if len(window) > 1 and not is_fault[lo:o].any():
                assert np.allclose(np.diff(window), -1.0)
                assert window[-1] == 1.0
        # fault_type is set whenever the machine is in FAULT.
        ft = np.array(rows.column("fault_type").to_pylist())
        assert np.all(ft[is_fault] != "none")
    assert n_faults >= 2
