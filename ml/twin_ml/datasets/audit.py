"""Data audit of the public datasets, logged to MLflow (Phase 10, ADR 0027).

``python -m twin_ml.datasets audit`` reads each dataset, measures rows, span, sampling, gaps,
ranges, missing values and failure events (``metropt3.audit`` / ``azure_pdm.audit``), adds the
split view of the configured mapping (series, rows and events per split, data coverage of every
failure's lookbacks), and logs one MLflow run per dataset (tag ``study=audit``) with the result
as ``audit.json``. The report reads these runs; nothing is typed by hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from twin_ml.config import SPLITS
from twin_ml.datasets import azure_pdm, metropt3
from twin_ml.datasets.build import load, split_series
from twin_ml.datasets.common import EPOCH
from twin_ml.datasets.config import (
    RealConfig,
    load_facts,
    load_real_config,
    load_real_dataset_config,
)

log = structlog.get_logger(__name__)

AUDIT_STUDY = "audit"


def split_view(name: str, real: RealConfig, root: Path) -> dict[str, Any]:
    """Series, rows and events per split of the configured mapping, and per event the data
    coverage of each lookback (share of grid rows with data) and its evaluability."""
    study = real.datasets[name]  # type: ignore[index]
    cfg = load_real_dataset_config(root / study.dataset_config)
    data = load(cfg, root, sha=False)
    series = split_series(data, cfg)
    step = cfg.step_s
    observed: dict[str, tuple[np.ndarray, np.ndarray]] = {
        g.machine_id: (g.t_s, g.observed) for g in data.grids
    }
    out: dict[str, Any] = {"step_s": step, "splits": {}}
    for split in SPLITS:
        a, b = (
            float((np.datetime64(t, "s") - EPOCH).astype(np.int64) - data.t0)
            for t in cfg.split.of(split)
        )
        machines = sorted({s.machine_id for s in series[split]})
        evs = [e for e in data.events if e.machine_id in machines and a <= e.start_s < b]
        per_event: list[dict[str, Any]] = []
        for e in evs:
            t_s, obs = observed[e.machine_id]
            cov = {}
            for lb in study.lookbacks:
                lo = e.start_s - lb * step
                sel = (t_s >= lo) & (t_s < e.start_s)
                share = float(obs[sel].sum() / lb)
                inside = lo >= a  # the whole lookback lies inside this split
                cov[str(lb)] = {
                    "coverage": share,
                    "inside_split": bool(inside),
                    "evaluable": bool(inside and share >= study.evaluable_min_coverage),
                }
            sel_f = (t_s >= e.start_s) & (t_s <= e.end_s)
            n_f = round((e.end_s - e.start_s) / step) + 1
            per_event.append(
                {
                    "id": e.id,
                    "machine_id": e.machine_id,
                    "fault_type": e.fault_type,
                    "interval_coverage": float(obs[sel_f].sum() / n_f),
                    "lookbacks": cov,
                }
            )
        prim = str(study.primary_lookback)
        out["splits"][split] = {
            "machines": len(machines),
            "series": len(series[split]),
            "rows": int(sum(s.t_s.shape[0] for s in series[split])),
            "events": len(evs),
            "evaluable_primary": sum(p["lookbacks"][prim]["evaluable"] for p in per_event),
            "by_fault_type": {
                f: sum(e.fault_type == f for e in evs) for f in sorted({e.fault_type for e in evs})
            },
            "event_detail": per_event if name == "metropt3" else None,
        }
    return out


def run_audit(
    name: str, real_path: Path, root: Path = Path(), *, smoke: bool = False
) -> dict[str, Any]:
    """Audit one dataset and log it to MLflow; returns the audit dict."""
    import mlflow

    from twin_ml.config import load_eval_config
    from twin_ml.train import tracking  # sets the MLflow env flags before mlflow is imported

    started = time.perf_counter()
    real = load_real_config(root / real_path)
    study = real.datasets[name]  # type: ignore[index]
    cfg = load_real_dataset_config(root / study.dataset_config)
    facts = load_facts(root / cfg.facts)
    if name == "metropt3":
        raw_m = metropt3.read_raw(facts, root)
        result = metropt3.audit(
            facts,
            raw_m,
            step_s=cfg.step_s,
            max_gap_rows=cfg.max_gap_rows,
            lookbacks_rows=study.lookbacks,
        )
        digest = raw_m.sha256
    else:
        raw_a = azure_pdm.read_raw(facts, root)
        result = azure_pdm.audit(facts, raw_a, step_s=cfg.step_s)
        digest = raw_a.sha256
    result["mapping"] = split_view(name, real, root)
    result["facts"] = {
        "title": facts.title,
        "provider": facts.provider,
        "licence": facts.licence,
        "origin": facts.origin,
        "time_zone": facts.time_zone,
        "fingerprint": facts.fingerprint(),
    }
    result["dataset_config_fingerprint"] = cfg.fingerprint(facts)
    result["audit_time_s"] = round(time.perf_counter() - started, 2)

    eval_cfg = load_eval_config(root / real.eval_config)
    tracking.configure(
        root / eval_cfg.tracking_dir, real.smoke_experiment if smoke else real.experiment
    )
    with mlflow.start_run(run_name=f"audit-{name}"):
        mlflow.set_tags(
            {
                "phase": real.phase_tag,
                "study": AUDIT_STUDY,
                "dataset": name,
                "source_sha256": digest,
                "facts_fingerprint": facts.fingerprint(),
                "dataset_config_fingerprint": cfg.fingerprint(facts),
            }
        )
        metrics: dict[str, float] = {}
        if name == "metropt3":
            metrics = {
                "rows": result["rows"],
                "span_days": result["span_days"],
                "gaps": result["gaps"]["count"],
                "gap_total_h": result["gaps"]["total_h"],
                "failures": result["failure_count"],
            }
        else:
            metrics = {
                "rows": result["telemetry"]["rows"],
                "machines": result["telemetry"]["machines"],
                "failure_events": result["failures"]["events"],
                "failure_rows": result["failures"]["rows"],
            }
        for split, v in result["mapping"]["splits"].items():
            metrics[f"{split}.events"] = v["events"]
            metrics[f"{split}.evaluable_primary"] = v["evaluable_primary"]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_dict(json.loads(json.dumps(result, default=float)), "audit.json")
    log.info("audit.done", dataset=name, seconds=result["audit_time_s"])
    return result
