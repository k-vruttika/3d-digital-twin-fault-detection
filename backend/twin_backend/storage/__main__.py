"""Storage CLI: ``python -m twin_backend.storage <command>``.

- ``migrate``: apply pending migrations to the configured store.
- ``export --from ISO --to ISO --format csv|jsonl --out PATH [--all]``: write a time range
  (``from <= ts < to``; either bound may be omitted) in a format the file adapter replays.
- ``info``: row counts, time span and snapshot count of the configured store.

The store comes from ``config/storage.yaml`` (``--config-dir`` to change), connection
settings from ``.env`` via TwinSettings.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from twin_backend.storage.base import Stream
from twin_backend.storage.export import export_range
from twin_backend.storage.service import build_store
from twin_common import aio
from twin_common.config import load_line_config, load_storage_config
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings
from twin_common.logging import configure_logging


def _ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        raise argparse.ArgumentTypeError(f"timestamp needs a timezone (e.g. Z): {value}")
    return ts


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m twin_backend.storage", description=__doc__)
    p.add_argument("--config-dir", type=Path, default=None, help="default: config/")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply pending migrations")
    sub.add_parser("info", help="row counts and time span")
    ex = sub.add_parser("export", help="export a time range (replayable by the file adapter)")
    ex.add_argument("--from", dest="t0", type=_ts, default=None, help="inclusive, ISO-8601")
    ex.add_argument("--to", dest="t1", type=_ts, default=None, help="exclusive, ISO-8601")
    ex.add_argument("--format", choices=("csv", "jsonl"), required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--all", action="store_true", help="also every other stream as JSONL")
    return p


async def _main(args: argparse.Namespace) -> int:
    settings = TwinSettings()
    config_dir = args.config_dir or settings.config_dir
    cfg = load_storage_config(config_dir / "storage.yaml")
    store = build_store(cfg, root=Path.cwd(), settings=settings)
    await store.connect()
    try:
        applied = await store.migrate()
        if args.command == "migrate":
            print(json.dumps({"backend": store.backend, "applied": applied}))
        elif args.command == "info":
            span = await store.min_max_ts(list(Stream))
            counts = {s.value: await store.count(s) for s in Stream}
            print(
                json.dumps(
                    {
                        "backend": store.backend,
                        "rows": counts,
                        "from": span[0].isoformat() if span else None,
                        "to": span[1].isoformat() if span else None,
                        "snapshots": len(await store.snapshot_times()),
                    }
                )
            )
        else:
            line = load_line_config(config_dir / "line.yaml")
            result = await export_range(
                store,
                args.t0,
                args.t1,
                args.out,
                args.format,
                machine_order=[m.id for m in line.machines],
                all_streams=args.all,
            )
            print(
                json.dumps(
                    {
                        "rows": {s.value: n for s, n in result.rows.items()},
                        "files": {s.value: str(p) for s, p in result.paths.items()},
                    }
                )
            )
    finally:
        await store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(stream=sys.stderr)
    try:
        return aio.run(_main(args))
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
