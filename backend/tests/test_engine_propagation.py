"""Topology, root-cause propagation and the transition table, checked against the simulator.

The simulator records the true root cause of every STARVED/BLOCKED machine
(``root_cause_machine``). The engine derives root causes only from the line config and the
machine states it sees, so on simulated runs both must agree sample by sample, for the full
line (operator stop, faults) and the single station. The shipped transition table must accept
every transition in those runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from prometheus_client import CollectorRegistry

from twin_backend.engine.propagation import LineStatus, line_view, root_cause
from twin_backend.engine.state_machine import StateMachines
from twin_backend.engine.topology import SINK_ID, SOURCE_ID, Topology
from twin_backend.testing import REPO_ROOT
from twin_common.config import load_engine_config, load_line_config
from twin_common.config.line import LineConfig
from twin_common.enums import MachineState

R, S, B = MachineState.RUNNING, MachineState.STARVED, MachineState.BLOCKED
CONFIG = REPO_ROOT / "config"
LINE = load_line_config(CONFIG / "line.yaml")
SINGLE = load_line_config(CONFIG / "line_single.yaml")
ENGINE = load_engine_config(CONFIG / "engine.yaml")


def _states(**kw: MachineState | None) -> dict[str, MachineState | None]:
    return {k.replace("_", "-"): v for k, v in kw.items()}


def test_topology_full_line_follows_buffers() -> None:
    topo = Topology.from_line(LINE)
    assert topo.machine_ids == ("CNV-01", "CNC-01", "INSP-01", "CNV-02")
    assert topo.upstream == {
        "CNV-01": None,
        "CNC-01": "CNV-01",
        "INSP-01": "CNC-01",
        "CNV-02": "INSP-01",
    }
    assert topo.downstream["CNV-02"] is None
    assert topo.downstream["CNC-01"] == "INSP-01"


def test_topology_single_station() -> None:
    topo = Topology.from_line(SINGLE)
    assert topo.upstream == {"CNV-01": None}
    assert topo.downstream == {"CNV-01": None}


def test_topology_without_buffers_uses_machine_order() -> None:
    line = LineConfig.model_validate(LINE.model_dump(by_alias=True) | {"buffers": []})
    topo = Topology.from_line(line)
    assert topo.downstream["CNV-01"] == "CNC-01"
    assert topo.upstream["CNV-02"] == "INSP-01"


def test_cnc_stop_blocks_upstream_and_starves_downstream() -> None:
    topo = Topology.from_line(LINE)
    states = _states(CNV_01=B, CNC_01=MachineState.IDLE, INSP_01=S, CNV_02=S)
    view = line_view(states, topo)
    assert {m: rc.cause for m, rc in view.root_causes.items()} == {
        "CNV-01": "CNC-01",
        "CNC-01": "CNC-01",
        "INSP-01": "CNC-01",
        "CNV-02": "CNC-01",
    }
    assert view.impacted == {"CNC-01": ["CNV-01", "INSP-01", "CNV-02"]}
    assert view.status is LineStatus.STOPPED


def test_running_neighbour_is_the_bottleneck_and_ends_are_source_and_sink() -> None:
    topo = Topology.from_line(LINE)
    states = _states(CNV_01=S, CNC_01=R, INSP_01=B, CNV_02=R)
    assert root_cause("CNV-01", states, topo).cause == SOURCE_ID
    assert root_cause("INSP-01", states, topo).cause == "CNV-02"
    states["CNV-02"] = B
    assert root_cause("CNV-02", states, topo).cause == SINK_ID
    assert root_cause("CNC-01", states, topo).cause is None
    view = line_view(states, topo)
    assert view.impacted == {}  # SOURCE / SINK are not machines
    assert view.status is LineStatus.DEGRADED


def test_link_down_neighbour_is_reported_as_unknown() -> None:
    topo = Topology.from_line(LINE)
    states = _states(CNV_01=B, CNC_01=MachineState.OFFLINE, INSP_01=S, CNV_02=S)
    rc = root_cause("CNV-01", states, topo)
    assert rc.cause == "CNC-01"
    assert rc.unknown
    assert root_cause("CNC-01", states, topo).unknown
    states["CNC-01"] = None
    assert root_cause("CNV-01", states, topo).unknown
    assert line_view(states, topo).status is LineStatus.UNKNOWN


def test_invalid_transition_is_counted_and_logged_not_raised() -> None:
    registry = CollectorRegistry()
    sms = StateMachines(["CNC-01"], ENGINE, registry)
    assert sms.observe("CNC-01", MachineState.IDLE) is not None  # first state: always valid
    tr = sms.observe("CNC-01", MachineState.FAULT)  # a stopped machine cannot trip
    assert tr is not None
    assert not tr.valid
    assert sms.machines["CNC-01"].state is MachineState.FAULT  # applied anyway
    assert sms.observe("CNC-01", MachineState.FAULT) is None  # not a transition
    assert sms.invalid_count() == 1
    value = registry.get_sample_value(
        "linetwin_engine_invalid_transitions_total",
        {"machine": "CNC-01", "from_state": "IDLE", "to_state": "FAULT"},
    )
    assert value == 1.0
    assert sms.observe("unknown", MachineState.RUNNING) is None


# ---------------------------------------------------------------- against the simulator
def _generate(tmp: Path, line_name: str, scenario: str, duration: float, seed: int) -> Path:
    from twin_sim.batch import generate
    from twin_sim.config import load_scenario, load_sim_config

    sc = load_scenario(CONFIG / "scenarios" / scenario).model_copy(update={"duration_s": duration})
    generate(
        line=load_line_config(CONFIG / line_name),
        sim=load_sim_config(CONFIG / "simulator.yaml"),
        scenario=sc,
        runs=2,
        base_seed=seed,
        out_dir=tmp,
        exports=(),
        workers=1,
    )
    return tmp


RUNS = [
    ("line.yaml", "cnc_stop.yaml", 1500.0, 7),
    ("line.yaml", "quick.yaml", 1800.0, 42),
    ("line_single.yaml", "quick_single.yaml", 1800.0, 42),
]


@pytest.fixture(scope="module", params=RUNS, ids=[r[1] for r in RUNS])
def sim_runs(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    line_name, scenario, duration, seed = request.param
    out = _generate(tmp_path_factory.mktemp("prop"), line_name, scenario, duration, seed)
    rows: list[dict[str, Any]] = []
    for f in sorted(out.glob("run_*.parquet")):
        rows.extend(
            pq.read_table(
                f, columns=["run_id", "t_s", "machine_id", "state", "root_cause_machine"]
            ).to_pylist()
        )
    return load_line_config(CONFIG / line_name), rows


def test_root_cause_equals_simulator_label(sim_runs: Any) -> None:
    line, rows = sim_runs
    topo = Topology.from_line(line)
    by_t: dict[tuple[Any, float], list[dict[str, Any]]] = {}
    for r in rows:
        by_t.setdefault((r["run_id"], r["t_s"]), []).append(r)
    waiting = 0
    for group in by_t.values():
        states: dict[str, MachineState | None] = {
            r["machine_id"]: MachineState(r["state"]) for r in group
        }
        for r in group:
            got = root_cause(r["machine_id"], states, topo).cause or ""
            assert got == (r["root_cause_machine"] or ""), r
            waiting += states[r["machine_id"]] in (S, B)
    if len(topo.machine_ids) > 1:
        assert waiting > 0  # the full-line runs do contain STARVED/BLOCKED samples


def test_shipped_transition_table_accepts_simulated_runs(sim_runs: Any) -> None:
    line, rows = sim_runs
    for run_id in {r["run_id"] for r in rows}:
        sms = StateMachines([m.id for m in line.machines], ENGINE, CollectorRegistry())
        for r in (x for x in rows if x["run_id"] == run_id):
            sms.observe(r["machine_id"], MachineState(r["state"]))
        assert sms.invalid_count() == 0
