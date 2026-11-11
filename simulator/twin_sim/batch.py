"""Batch dataset generation: one Parquet file per run, optional CSV/JSONL, manifest.json."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from twin_common.config.line import LineConfig
from twin_common.logging import get_logger
from twin_sim.config import ScenarioConfig, SimConfig, check_scenario_compat
from twin_sim.line import LineSimulator, thresholds_table
from twin_sim.recorder import Recorder
from twin_sim.scenario import make_run_plan

EXPORT_FORMATS = ("csv", "jsonl")
MANIFEST_NAME = "manifest.json"

log = get_logger(__name__)


@dataclass(frozen=True)
class RunTask:
    line: LineConfig
    sim: SimConfig
    scenario: ScenarioConfig
    base_seed: int
    run_index: int
    out_dir: Path
    exports: tuple[str, ...]


def simulate_run(task: RunTask) -> tuple[LineSimulator, Recorder, list[str]]:
    """Run one plan to completion. Returns the simulator, its recorder and JSONL lines."""
    plan = make_run_plan(task.scenario, task.sim, task.line, task.base_seed, task.run_index)
    sim = LineSimulator(task.line, task.sim, plan)
    rec = Recorder(sim)
    jsonl: list[str] = []
    want_jsonl = "jsonl" in task.exports
    for _ in range(round(plan.duration_s)):
        frame = sim.step(1.0)
        rec.add(frame)
        if want_jsonl:
            jsonl.extend(r.model_dump_json() for r in sim.readings(frame))
    return sim, rec, jsonl


def run_one(task: RunTask) -> dict[str, Any]:
    """Worker entry point: simulate, write files, return the manifest entry."""
    sim, rec, jsonl = simulate_run(task)
    plan = sim.plan
    table = rec.to_table()
    task.out_dir.mkdir(parents=True, exist_ok=True)
    files = {"parquet": f"{plan.run_id}.parquet"}
    pq.write_table(table, task.out_dir / files["parquet"], compression="zstd")
    if "csv" in task.exports:
        files["csv"] = f"{plan.run_id}.csv"
        pacsv.write_csv(table, task.out_dir / files["csv"])
    if "jsonl" in task.exports:
        files["jsonl"] = f"{plan.run_id}.jsonl"
        (task.out_dir / files["jsonl"]).write_text(
            "\n".join(jsonl) + ("\n" if jsonl else ""), encoding="utf-8"
        )
    return {
        "run_id": plan.run_id,
        "run_index": plan.run_index,
        "seed": {"entropy": plan.base_seed, "spawn_key": [plan.run_index]},
        "files": files,
        "rows": table.num_rows,
        "load_range": plan.load_range,
        "ambient_c": plan.ambient_c,
        "load_mu": {m.machine_id: m.load_mu for m in plan.machines},
        "thresholds": thresholds_table(sim),
        "eligible_fault_types": [f.value for f in plan.eligible_fault_types],
        "planned_faults": [e.model_dump(mode="json") for e in plan.fault_events],
        "realised_faults": [asdict(e) for e in sim.fault_log],
        "sensor_faults": [e.model_dump(mode="json") for e in plan.sensor_fault_events],
        "parts_produced": sim.produced,
    }


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def generate(
    *,
    line: LineConfig,
    sim: SimConfig,
    scenario: ScenarioConfig,
    runs: int,
    base_seed: int,
    out_dir: Path,
    exports: tuple[str, ...] = (),
    workers: int | None = None,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate ``runs`` runs into ``out_dir`` and write the manifest. Returns the manifest."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    unknown = sorted(set(exports) - set(EXPORT_FORMATS))
    if unknown:
        raise ValueError(f"unknown export format(s) {unknown}; allowed: {list(EXPORT_FORMATS)}")
    check_scenario_compat(scenario, sim, line)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers, runs))
    tasks = [
        RunTask(line, sim, scenario, base_seed, i, out_dir, tuple(exports)) for i in range(runs)
    ]
    log.info("generate.start", runs=runs, workers=n_workers, out=str(out_dir))
    started = time.perf_counter()
    if n_workers == 1:
        entries = [run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            entries = list(pool.map(run_one, tasks))
    elapsed = time.perf_counter() - started

    scenario_json = scenario.model_dump(mode="json")
    sim_json = sim.model_dump(mode="json")
    line_json = line.model_dump(mode="json", by_alias=True)
    manifest: dict[str, Any] = {
        "generator": "twin_sim",
        "version": version("twin-sim"),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "generation_time_s": round(elapsed, 3),
        "workers": n_workers,
        "cli_args": cli_args or {},
        "base_seed": base_seed,
        "runs": runs,
        "duration_s": scenario.duration_s,
        "exports": list(exports),
        "scenario": scenario_json,
        "scenario_sha256": _sha256_json(scenario_json),
        "sim_config": sim_json,
        "sim_config_sha256": _sha256_json(sim_json),
        "line_config_sha256": _sha256_json(line_json),
        "held_out_fault_types": [f.value for f in sim.faults.held_out_types()],
        "run_entries": entries,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("generate.done", runs=runs, seconds=round(elapsed, 2))
    return manifest
