"""CLI of the public-dataset tools (Phase 10).

python -m twin_ml.datasets audit [--dataset metropt3|azure_pdm]   # data audit -> MLflow
python -m twin_ml.datasets build [--config config/models/real_<name>.yaml ...]
python -m twin_ml.datasets replay-csv [--failure N] [--out PATH] [--log]   # replay demo CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REAL = Path("config/models/real.yaml")
DEFAULT_REPLAY = Path("config/datasets/metropt3_replay.yaml")
DEFAULT_BUILDS = (
    Path("config/models/real_metropt3.yaml"),
    Path("config/models/real_azure.yaml"),
    Path("config/models/real_azure_err.yaml"),
    Path("config/models/real_azure_holdout.yaml"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="data audit of each dataset, logged to MLflow")
    a.add_argument("--dataset", choices=["metropt3", "azure_pdm"], action="append")
    a.add_argument("--real-config", type=Path, default=DEFAULT_REAL)
    a.add_argument("--smoke", action="store_true", help="log to the smoke experiment")
    b = sub.add_parser("build", help="build the windowed real-data datasets")
    b.add_argument("--config", type=Path, action="append")
    r = sub.add_parser("replay-csv", help="MetroPT-3 slice around a failure -> replay CSV")
    r.add_argument("--config", type=Path, default=DEFAULT_REPLAY)
    r.add_argument("--failure", type=int, default=None, help="failure id (default: config)")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--log", action="store_true", help="log the summary to MLflow (P10_real)")
    r.add_argument("--real-config", type=Path, default=DEFAULT_REAL)
    args = parser.parse_args(argv)

    from twin_ml.datasets.common import MissingDatasetError

    try:
        if args.cmd == "audit":
            from twin_ml.datasets.audit import run_audit

            for name in args.dataset or ["metropt3", "azure_pdm"]:
                run_audit(name, args.real_config, smoke=args.smoke)
        elif args.cmd == "replay-csv":
            import json

            from twin_ml.datasets.replay import convert, load_replay_config, log_summary

            summary = convert(load_replay_config(args.config), failure=args.failure, out=args.out)
            if args.log:
                summary["mlflow_run_id"] = log_summary(summary, args.real_config)
            print(json.dumps(summary, indent=1))
        else:
            from twin_ml.datasets.build import build_real_dataset

            for cfg in args.config or DEFAULT_BUILDS:
                path = build_real_dataset(cfg)
                print(f"{cfg.as_posix()} -> {path.as_posix()}")
    except (MissingDatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
