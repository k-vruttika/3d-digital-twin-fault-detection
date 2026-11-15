"""E1 runner: ``python -m twin_ml.experiments.e1`` (poe task ``train-all``).

Trains and evaluates every model of ``eval.yaml`` (baselines once, models once per seed) on the
current dataset and logs each (model, seed) as its own MLflow run, tagged with the model config
hash. ``--skip-existing`` resumes: it skips a (model, seed) only if a finished run with the
same config hash exists, so a changed config is always re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from twin_common.config import ConfigError
from twin_common.logging import configure_logging, get_logger

log = get_logger(__name__)


def pending(
    models: list[str],
    seeds: list[int],
    shas: dict[str, str],
    done: set[tuple[str, int, str]],
    baselines: tuple[str, ...] = ("persistence", "threshold"),
) -> list[tuple[str, int]]:
    """(model, seed) pairs still to run, in order; baselines run with the first seed only.

    A pair is done only if it finished with the model's *current* config hash.
    """
    out = []
    for m in models:
        for seed in seeds[:1] if m in baselines else seeds:
            if (m, seed, shas[m]) in done:
                log.info("e1.skip", model=m, seed=seed)
            else:
                out.append((m, seed))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.e1")
    parser.add_argument("--models", default="", help="comma-separated; default: eval.yaml models")
    parser.add_argument("--seeds", default="", help="comma-separated; default: eval.yaml seeds")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip (model, seed) already finished with the same model config hash",
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("config/models/dataset.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument("--models-dir", type=Path, default=Path("config/models"))
    args = parser.parse_args(argv)
    configure_logging()

    from twin_ml.config import load_dataset_config, load_eval_config, load_model_config
    from twin_ml.data.dataset import Dataset, DatasetNotBuiltError, resolve_dataset_dir
    from twin_ml.train import tracking
    from twin_ml.train.__main__ import parse_seeds
    from twin_ml.train.trainer import config_sha, run_one

    try:
        ds_cfg = load_dataset_config(args.dataset_config)
        eval_cfg = load_eval_config(args.eval_config)
        ds = Dataset.open(resolve_dataset_dir(ds_cfg))
        models = [m for m in args.models.split(",") if m] or eval_cfg.models
        cfgs = {m: load_model_config(args.models_dir / f"{m}.yaml") for m in models}
    except (ConfigError, DatasetNotBuiltError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    seeds = parse_seeds(args.seeds, eval_cfg.seeds)
    done: set[tuple[str, int, str]] = set()
    if args.skip_existing:
        for run in tracking.finished_runs(
            eval_cfg.tracking_dir, eval_cfg.experiment, ds.meta["dataset_key"]
        ):
            done.add(
                (
                    run.data.tags["model"],
                    int(run.data.tags["seed"]),
                    run.data.tags.get("model_config_sha", ""),
                )
            )
    started = time.perf_counter()
    todo = pending(models, seeds, {m: config_sha(c) for m, c in cfgs.items()}, done)
    for m, seed in todo:
        res = run_one(m, cfgs[m], seed, ds, eval_cfg)
        print(f"{m} seed {seed}: run {res.run_id}, trained in {res.train_s:.1f} s", flush=True)
    print(f"E1 done in {time.perf_counter() - started:.0f} s; next: uv run poe report-e1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
