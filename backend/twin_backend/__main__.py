"""Backend command line: ``python -m twin_backend serve``.

Modes:

- ``minimal``: file replay (``sources.yaml`` ``file``) + SQLite. No Docker, broker, database
  server or ML is needed or imported (CLAUDE.md section 1).
- ``dev``: the active adapter from ``sources.yaml`` + TimescaleDB (start it with
  ``uv run poe up``). ``uv run poe dev`` also starts the matching simulator publisher.
- ``config``: exactly what the config files say.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from twin_common import aio
from twin_common.config import (
    TwinSettings,
    load_sources_config,
    load_storage_config,
)
from twin_common.config.loader import ConfigError
from twin_common.enums import AdapterKind
from twin_common.logging import SecretFilter, configure_logging, get_logger

if TYPE_CHECKING:
    from twin_backend.runtime import Runtime

Mode = Literal["minimal", "dev", "config"]
log = get_logger("twin_backend")


def build_for_mode(
    mode: Mode,
    *,
    settings: TwinSettings,
    root: Path,
    source: AdapterKind | None = None,
    file_path: str | None = None,
    replay_speed: float | None = None,
    sqlite_path: str | None = None,
    loop: bool | None = None,
) -> Runtime:
    """A Runtime with the mode's overrides applied (imports stay lazy for minimal mode)."""
    from twin_backend.runtime import build_runtime

    config_dir = (
        settings.config_dir if settings.config_dir.is_absolute() else root / settings.config_dir
    )
    sources = load_sources_config(config_dir / "sources.yaml")
    storage_cfg = load_storage_config(config_dir / "storage.yaml")
    if mode == "minimal":
        sources = sources.model_copy(update={"active": AdapterKind.FILE})
        storage_cfg = storage_cfg.model_copy(update={"backend": "sqlite"})
    elif mode == "dev":
        storage_cfg = storage_cfg.model_copy(update={"backend": "timescale"})
    if source is not None:
        if getattr(sources, source.value, None) is None:
            raise ConfigError(f"sources.yaml has no '{source.value}' section")
        sources = sources.model_copy(update={"active": source})
    file_update: dict[str, object] = {}
    if file_path is not None:
        file_update["path"] = file_path
    if replay_speed is not None:
        file_update["replay_speed"] = replay_speed
    if loop is not None:
        file_update["loop"] = loop
    if file_update:
        if sources.file is None:
            raise ConfigError("sources.yaml has no 'file' section")
        sources = sources.model_copy(update={"file": sources.file.model_copy(update=file_update)})
    if sqlite_path is not None:
        storage_cfg = storage_cfg.model_copy(
            update={"sqlite": storage_cfg.sqlite.model_copy(update={"path": sqlite_path})}
        )
    return build_runtime(settings, root=root, sources=sources, storage_cfg=storage_cfg)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m twin_backend", description="LineTwin backend")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="run the API (REST + WebSocket) with the twin engine")
    add_serve_args(s)
    return p


def add_serve_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--mode", choices=["minimal", "dev", "config"], default="config")
    s.add_argument("--line", help="line file in the config dir (default: LINETWIN_LINE_FILE)")
    s.add_argument("--source", choices=[k.value for k in AdapterKind], help="override active")
    s.add_argument("--file", dest="file_path", help="replay file for the file source")
    s.add_argument("--replay-speed", type=float, help="file source replay speed")
    s.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="file source: replay forever (--loop) or stop at the end (--no-loop); "
        "default: sources.yaml",
    )
    s.add_argument("--sqlite-path", help="SQLite event store file")
    s.add_argument("--host", help="bind address (default: api.yaml host)")
    s.add_argument("--port", type=int, help="port (default: api.yaml port)")


def run_serve(args: argparse.Namespace) -> int:
    from twin_backend.api.auth import StartupRefusedError
    from twin_backend.api.server import serve

    settings = TwinSettings()
    if args.line:
        settings = settings.model_copy(update={"line_file": Path(args.line).name})
    configure_logging(settings.log_level, stream=sys.stderr)
    logging.basicConfig(level=settings.log_level.upper(), stream=sys.stderr, format="%(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter())
    try:
        runtime = build_for_mode(
            args.mode,
            settings=settings,
            root=Path.cwd(),
            source=AdapterKind(args.source) if args.source else None,
            file_path=args.file_path,
            replay_speed=args.replay_speed,
            sqlite_path=args.sqlite_path,
            loop=getattr(args, "loop", None),
        )
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2
    host = args.host or runtime.api_cfg.host
    port = args.port or runtime.api_cfg.port
    log.info(
        "backend.starting",
        mode=args.mode,
        source=runtime.sources.active.value,
        store=runtime.storage_cfg.backend,
        line=runtime.line.line_id,
        url=f"http://{host}:{port}",
    )
    try:
        started = aio.run(serve(runtime, host=host, port=port))
    except StartupRefusedError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0 if started else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return run_serve(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
