"""CLI: ``python -m twin_ml.data build [--config config/models/dataset.yaml]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twin_common.config import ConfigError
from twin_common.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.data")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the windowed dataset from the simulator runs")
    build.add_argument("--config", type=Path, default=Path("config/models/dataset.yaml"))
    build.add_argument(
        "--no-verify", action="store_true", help="skip checking file hashes against the manifest"
    )
    args = parser.parse_args(argv)
    configure_logging()

    from twin_ml.config import load_dataset_config
    from twin_ml.data.build import build_dataset

    try:
        cfg = load_dataset_config(args.config)
        out = build_dataset(cfg, check_hashes=not args.no_verify)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
