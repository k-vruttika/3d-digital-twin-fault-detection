"""CLI: ``python -m twin_ml.train --model rf [--seeds 0,1,2]`` (poe task ``train``).

Each seed trains in its own worker process (``twin_ml.train.isolated``, ADR 0020).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twin_common.config import ConfigError
from twin_common.logging import configure_logging


def parse_seeds(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.train")
    parser.add_argument("--model", required=True, help="config/models/<model>.yaml")
    parser.add_argument(
        "--seeds", nargs="?", const="", default="", help="comma-separated; default: eval.yaml"
    )
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--dataset-config", type=Path, default=Path("config/models/dataset.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment (default: eval.yaml), e.g. diagnostics",
    )
    args = parser.parse_args(argv)
    configure_logging()

    from twin_ml.config import load_dataset_config, load_eval_config, load_model_config
    from twin_ml.data.dataset import Dataset, DatasetNotBuiltError, resolve_dataset_dir
    from twin_ml.train.isolated import TrainingProcessError, run_isolated
    from twin_ml.train.trainer import BASELINES

    try:
        ds_cfg = load_dataset_config(args.dataset_config)
        eval_cfg = load_eval_config(args.eval_config)
        cfg = load_model_config(args.model_config or Path(f"config/models/{args.model}.yaml"))
        ds = Dataset.open(resolve_dataset_dir(ds_cfg))
    except (ConfigError, DatasetNotBuiltError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.experiment:
        eval_cfg = eval_cfg.model_copy(update={"experiment": args.experiment})
    seeds = parse_seeds(args.seeds, eval_cfg.seeds)
    if args.model in BASELINES:
        seeds = seeds[:1]  # deterministic: one run
    for seed in seeds:
        try:
            res = run_isolated(args.model, cfg, seed, ds.path, eval_cfg)
        except TrainingProcessError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{res.model} seed {res.seed}: run {res.run_id}, trained in {res.train_s:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
