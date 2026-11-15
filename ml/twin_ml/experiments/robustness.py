"""Phase 8 runner: ``python -m twin_ml.experiments.robustness --study <name>``
(poe task ``robustness``).

Runs every (model, seed) of a study (``config/models/robustness.yaml``), each in its own
worker process (``robust_worker``), and logs one MLflow run per pair in the robustness
experiment. ``--skip-existing`` resumes: a pair is skipped only if a finished study run with
the same study config hash exists (model config, robustness and evaluation settings,
dataset key and, for the e1models study, the reused E1 run).

Progress goes to ``<progress_dir>/<study>.jsonl``: a plan line with the estimated duration
(from the Phase 7 measured train times in the config), then one line per model, condition
and finished pair, with the elapsed time and a re-estimated remaining time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twin_common.config import ConfigError
from twin_common.logging import configure_logging, get_logger

log = get_logger(__name__)


def estimate_s(
    model: str,
    study_trains: bool,
    n_conditions: int,
    eta_train: dict[str, float],
    eta_cond: dict[str, float],
    train_scale: float = 1.0,
) -> float:
    """Rough seconds for one (model, seed): training (or the RF retrain) plus evaluation."""
    train = eta_train.get(model, 0.0) * train_scale if (study_trains or model == "rf") else 0.0
    return train + n_conditions * eta_cond.get(model, 0.0)


def _write(path: Path, **event: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.robustness")
    parser.add_argument("--study", required=True, help="e1models | e4 | aug (robustness.yaml)")
    parser.add_argument("--models", default="", help="comma-separated; default: eval.yaml models")
    parser.add_argument("--seeds", default="", help="comma-separated; default: eval.yaml seeds")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and ETA only")
    parser.add_argument(
        "--robustness-config", type=Path, default=Path("config/models/robustness.yaml")
    )
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument("--models-dir", type=Path, default=Path("config/models"))
    args = parser.parse_args(argv)
    configure_logging()

    from twin_ml.config import (
        load_dataset_config,
        load_eval_config,
        load_model_config,
        load_robustness_config,
    )
    from twin_ml.data.dataset import Dataset, DatasetNotBuiltError, resolve_dataset_dir
    from twin_ml.experiments.e1 import pending
    from twin_ml.experiments.robust_worker import (
        StudyProcessError,
        expand_conditions,
        run_study_isolated,
        source_run_id,
        study_config_sha,
    )
    from twin_ml.train import tracking
    from twin_ml.train.__main__ import parse_seeds
    from twin_ml.train.trainer import BASELINES

    try:
        rcfg = load_robustness_config(args.robustness_config)
        eval_cfg = load_eval_config(args.eval_config)
        if args.study not in rcfg.studies:
            raise ConfigError(f"unknown study {args.study!r}; known: {sorted(rcfg.studies)}")
        study = rcfg.studies[args.study]
        ds = Dataset.open(resolve_dataset_dir(load_dataset_config(study.dataset_config)))
        models = [m for m in args.models.split(",") if m] or eval_cfg.models
        cfgs = {m: load_model_config(args.models_dir / f"{m}.yaml") for m in models}
        seeds = parse_seeds(args.seeds, eval_cfg.seeds)
        key = ds.meta["dataset_key"]
        shas: dict[tuple[str, int], str] = {}
        for m in models:
            for s in seeds[:1] if m in BASELINES else seeds:
                src = source_run_id(study, rcfg, eval_cfg.tracking_dir, key, m, s, cfgs[m])
                shas[(m, s)] = study_config_sha(args.study, cfgs[m], rcfg, eval_cfg, key, src)
    except (ConfigError, DatasetNotBuiltError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    done: set[tuple[str, int, str]] = set()
    if args.skip_existing:
        flt = f"tags.study = '{args.study}'"
        for run in tracking.finished_runs(
            eval_cfg.tracking_dir, rcfg.experiment, key, rcfg.phase_tag, flt
        ):
            m, s = run.data.tags["model"], int(run.data.tags["seed"])
            if run.data.tags.get("study_config_sha") == shas.get((m, s)):
                done.add((m, s, "x"))
    todo = pending(models, seeds, dict.fromkeys(models, "x"), done)
    n_cond = len(expand_conditions(study, rcfg))
    e1_train_windows = None
    if study.models_from == "train":
        e1 = Dataset.open(
            resolve_dataset_dir(load_dataset_config(rcfg.studies["e1models"].dataset_config))
        )
        e1_train_windows = e1.meta["counts"]["train"]["windows"]
    scale = ds.meta["counts"]["train"]["windows"] / e1_train_windows if e1_train_windows else 1.0
    est = {
        (m, s): estimate_s(
            m, study.models_from == "train", n_cond, rcfg.eta_train_s, rcfg.eta_condition_s, scale
        )
        for m, s in todo
    }
    total = sum(est.values())
    progress = rcfg.progress_dir / f"{args.study}.jsonl"
    plan = [f"{m}:{s}" for m, s in todo]
    print(
        f"study {args.study}: {len(todo)} runs, {n_cond} conditions each, "
        f"estimated {total / 60:.0f} min; progress: {progress.as_posix()}",
        flush=True,
    )
    if args.dry_run:
        for (m, s), e in est.items():
            print(f"  {m} seed {s}: ~{e / 60:.1f} min")
        return 0
    _write(
        progress,
        event="plan",
        study=args.study,
        runs=plan,
        conditions=n_cond,
        estimated_s=round(total),
        train_scale=round(scale, 3),
    )
    started = time.perf_counter()
    spent_est = 0.0
    for m, s in todo:
        _write(
            progress,
            event="start",
            study=args.study,
            model=m,
            seed=s,
            estimated_s=round(est[(m, s)]),
        )
        t0 = time.perf_counter()
        try:
            res = run_study_isolated(args.study, m, cfgs[m], s, rcfg, eval_cfg, progress=progress)
        except StudyProcessError as exc:
            _write(progress, event="failed", study=args.study, model=m, seed=s, error=str(exc))
            print(
                f"error: {exc}; finished runs are kept, resume with --skip-existing",
                file=sys.stderr,
            )
            return 1
        spent_est += est[(m, s)]
        elapsed = time.perf_counter() - started
        ratio = elapsed / spent_est if spent_est > 0 else 1.0
        remaining = (total - spent_est) * ratio
        _write(
            progress,
            event="finished",
            study=args.study,
            model=m,
            seed=s,
            run_id=res["run_id"],
            seconds=round(time.perf_counter() - t0),
            elapsed_s=round(elapsed),
            remaining_est_s=round(remaining),
        )
        print(
            f"{m} seed {s}: run {res['run_id']} in {time.perf_counter() - t0:.0f} s; "
            f"~{remaining / 60:.0f} min left",
            flush=True,
        )
    _write(
        progress,
        event="study_done",
        study=args.study,
        elapsed_s=round(time.perf_counter() - started),
    )
    print(
        f"study {args.study} done in {time.perf_counter() - started:.0f} s; "
        "next: uv run poe report-robustness"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
