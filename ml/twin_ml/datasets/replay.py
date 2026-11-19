"""MetroPT-3 replay demo: raw slice around one failure -> canonical CSV (Phase 10 Part C).

The output has the simulator's CSV columns (``run_id, machine_id, seq, ts, state, speed, temp,
vib, load, current, run_time, quality``), so the file source replays it with the mapping in
``config/sources.yaml`` and minimal mode needs nothing new (no ML import at replay time; this
converter runs offline). Mapping (``config/datasets/metropt3_replay.yaml``):

* the raw ~10 s samples from ``before_h`` before the reported failure start to ``after_h``
  after its end, unchanged (no resampling);
* ``temp`` = Oil_temperature, ``current`` = Motor_current; ``speed``, ``vib`` and ``load`` are
  empty, so every reading is a partial reading (quality SUSPECT after validation, ADR 0007);
* ``state`` = RUNNING if the motor current is at or above ``running_current_a``, else IDLE. The
  failure report is **not** injected as a state: the twin shows what the data shows;
* ``ts`` = the naive MetroPT-3 timestamp written as UTC (the source states no time zone);
  ``seq`` counts from 1; ``run_time`` = seconds since the slice start.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from twin_common.config.line import ConfigModel
from twin_common.config.loader import load_model
from twin_common.enums import MachineState
from twin_ml.datasets.config import load_facts, parse_time
from twin_ml.datasets.metropt3 import read_raw

COLUMNS = (
    "run_id",
    "machine_id",
    "seq",
    "ts",
    "state",
    "speed",
    "temp",
    "vib",
    "load",
    "current",
    "run_time",
    "quality",
)


class ReplayConfig(ConfigModel):
    facts: Path
    failure: int
    before_h: float = Field(ge=0)
    after_h: float = Field(ge=0)
    machine_id: str
    temp_sensor: str
    current_sensor: str
    running_current_a: float = Field(gt=0)
    out: Path


def load_replay_config(path: str | Path) -> ReplayConfig:
    return load_model(path, ReplayConfig)


def _iso(seconds: int) -> str:
    return datetime.fromtimestamp(int(seconds), tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def convert(
    cfg: ReplayConfig, root: Path = Path(), failure: int | None = None, out: Path | None = None
) -> dict[str, Any]:
    """Write the replay CSV; returns a summary (rows, span, states, sample interval)."""
    facts = load_facts(root / cfg.facts)
    k = cfg.failure if failure is None else failure
    match = [f for f in facts.failures if f.id == k]
    if not match:
        raise ValueError(f"no failure #{k} in {cfg.facts}; known: {[f.id for f in facts.failures]}")
    f = match[0]
    epoch = datetime(1970, 1, 1)
    start = int((parse_time(f.start) - epoch).total_seconds())
    end = int((parse_time(f.end) - epoch).total_seconds())
    lo, hi = start - round(cfg.before_h * 3600), end + round(cfg.after_h * 3600)
    raw = read_raw(facts, root, sha=False)
    sel = (raw.t >= lo) & (raw.t <= hi)
    t = raw.t[sel]
    if t.size == 0:
        raise ValueError(f"no samples between {_iso(lo)} and {_iso(hi)}")
    names = list(facts.sensor_names)
    temp = raw.values[sel, names.index(cfg.temp_sensor)]
    cur = raw.values[sel, names.index(cfg.current_sensor)]
    running = np.nan_to_num(cur, nan=0.0) >= cfg.running_current_a
    path = root / (cfg.out if out is None else out)
    if out is None and failure is not None and failure != cfg.failure:
        path = path.with_name(path.name.replace(f"failure{cfg.failure}", f"failure{k}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"metropt3_failure{k}"

    def num(x: float) -> str:
        return "" if not np.isfinite(x) else repr(float(x))

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for i in range(t.size):
            w.writerow(
                [
                    run_id,
                    cfg.machine_id,
                    i + 1,
                    _iso(int(t[i])),
                    (MachineState.RUNNING if running[i] else MachineState.IDLE).value,
                    "",
                    num(temp[i]),
                    "",
                    "",
                    num(cur[i]),
                    int(t[i] - t[0]),
                    "OK",
                ]
            )
    d = np.diff(t)
    return {
        "failure": k,
        "failure_start": f.start,
        "failure_end": f.end,
        "out": path.as_posix(),
        "rows": int(t.size),
        "first": _iso(int(t[0])),
        "last": _iso(int(t[-1])),
        "span_h": float((t[-1] - t[0]) / 3600),
        "running_rows": int(running.sum()),
        "idle_rows": int((~running).sum()),
        "median_interval_s": float(np.median(d)) if d.size else None,
        "max_gap_s": int(d.max()) if d.size else None,
        "rows_in_failure": int(((t >= start) & (t <= end)).sum()),
    }


def log_summary(summary: dict[str, Any], real_path: Path, root: Path = Path()) -> str:
    """Log a conversion summary to MLflow (``P10_real``, ``study=replay``) for the report."""
    import json

    import mlflow

    from twin_ml.config import load_eval_config
    from twin_ml.datasets.config import load_real_config
    from twin_ml.train import tracking

    real = load_real_config(root / real_path)
    eval_cfg = load_eval_config(root / real.eval_config)
    tracking.configure(root / eval_cfg.tracking_dir, real.experiment)
    with mlflow.start_run(run_name=f"replay-metropt3-failure{summary['failure']}") as run:
        mlflow.set_tags(
            {
                "phase": real.phase_tag,
                "study": "replay",
                "dataset": "metropt3",
                "failure": str(summary["failure"]),
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in summary.items()
                if isinstance(v, int | float) and not isinstance(v, bool)
            }
        )
        mlflow.log_dict(json.loads(json.dumps(summary)), "replay.json")
        return str(run.info.run_id)
