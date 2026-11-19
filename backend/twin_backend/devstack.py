"""Development launcher (``uv run poe dev``): simulator publisher + backend in one command.

Reads ``sources.yaml``. For a live source (``mock``, ``mqtt``, ``opcua``) it starts the
matching simulator publisher (``python -m twin_sim mock-server | mqtt | opcua``) as a child
process, waits until it is ready, and then runs the backend in this process with
``serve --mode dev`` (the active adapter + TimescaleDB). With ``file`` only the backend runs.

The services it needs (TimescaleDB always, Mosquitto for MQTT) must already run
(``uv run poe up``); they are probed first, with a clear hint if they are not reachable.

Cross-platform by construction: the child is started with ``subprocess.Popen`` and the current
interpreter (no shell, no asyncio subprocess APIs, which the Windows selector loop does not
support, ADR 0008) and is stopped in ``finally``: terminate, wait, then kill.

Examples::

    uv run poe dev                                  # active source from sources.yaml
    uv run poe dev --source opcua --speed 5         # override source, simulator speed
    uv run poe dev --no-sim                         # backend only (publisher runs elsewhere)
    uv run poe dev --line config/line_single.yaml --scenario config/scenarios/quick_single.yaml
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from twin_common.config import TwinSettings, load_sources_config
from twin_common.config.loader import ConfigError
from twin_common.config.sources import SourcesConfig
from twin_common.enums import AdapterKind

PUBLISHER_COMMAND = {
    AdapterKind.MOCK: "mock-server",
    AdapterKind.MQTT: "mqtt",
    AdapterKind.OPCUA: "opcua",
}
PROBE_TIMEOUT_S = 2.0
READY_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 10.0
START_HINT = "start the services with: uv run poe up"


def publisher_command(
    kind: AdapterKind,
    *,
    line: Path,
    scenario: Path | None,
    speed: float,
    duration: float | None,
    seed: int | None,
) -> list[str] | None:
    """The simulator publisher matching ``kind`` (None for the file source)."""
    sub = PUBLISHER_COMMAND.get(kind)
    if sub is None:
        return None
    cmd = [sys.executable, "-m", "twin_sim", sub, "--line", str(line), "--speed", f"{speed:g}"]
    if scenario is not None:
        cmd += ["--scenario", str(scenario)]
    if duration is not None:
        cmd += ["--duration", f"{duration:g}"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def required_services(
    kind: AdapterKind, sources: SourcesConfig, settings: TwinSettings
) -> list[tuple[str, str, int]]:
    """(name, host, port) of the services the dev stack needs."""
    need = [("TimescaleDB", settings.pg_host, settings.pg_port)]
    if kind is AdapterKind.MQTT and sources.mqtt is not None:
        need.append(("Mosquitto (MQTT broker)", sources.mqtt.host, sources.mqtt.port))
    return need


def port_open(host: str, port: int, timeout: float = PROBE_TIMEOUT_S) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ready_address(kind: AdapterKind, sources: SourcesConfig) -> tuple[str, int] | None:
    """Where the publisher listens once it is ready (None: it is a client, e.g. MQTT)."""
    if kind is AdapterKind.MOCK and sources.mock is not None:
        u = urlparse(sources.mock.base_url)
        return u.hostname or "127.0.0.1", u.port or 80
    if kind is AdapterKind.OPCUA and sources.opcua is not None:
        u = urlparse(sources.opcua.endpoint)
        return u.hostname or "127.0.0.1", u.port or 4840
    return None


def wait_ready(
    proc: subprocess.Popen[Any],
    address: tuple[str, int] | None,
    timeout: float = READY_TIMEOUT_S,
    probe: Callable[[str, int], bool] = port_open,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if address is None or probe(*address):
            return True
        time.sleep(0.2)
    return False


def stop_child(proc: subprocess.Popen[Any] | None, timeout: float = STOP_TIMEOUT_S) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m twin_backend.devstack",
        description="simulator publisher + backend (dev mode, TimescaleDB)",
    )
    p.add_argument("--source", choices=[k.value for k in AdapterKind], help="override active")
    p.add_argument("--no-sim", action="store_true", help="do not start a simulator publisher")
    p.add_argument("--line", help="line file (default: LINETWIN_LINE_FILE or line.yaml)")
    p.add_argument("--scenario", type=Path, help="simulator scenario YAML")
    p.add_argument("--speed", type=float, default=1.0, help="simulator speed multiplier")
    p.add_argument("--duration", type=float, help="simulated seconds (default: the scenario's)")
    p.add_argument("--seed", type=int, help="simulator seed")
    p.add_argument("--host", help="backend bind address (default: api.yaml)")
    p.add_argument("--port", type=int, help="backend port (default: api.yaml)")
    p.add_argument(
        "--cors-origin",
        dest="cors_origins",
        action="append",
        default=None,
        help="extra allowed CORS origin on top of api.yaml (repeatable)",
    )
    p.add_argument(
        "--inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="live inference (Phase 11); default: config/inference.yaml",
    )
    p.add_argument("--model", help="inference model (config/inference.yaml models key)")
    p.add_argument("--model-version", help="registered model version or 'latest'")
    return p


def main(
    argv: list[str] | None = None,
    *,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    serve: Callable[[argparse.Namespace], int] | None = None,
    probe: Callable[[str, int], bool] = port_open,
) -> int:
    args = _parser().parse_args(argv)
    settings = TwinSettings()
    if args.line:
        settings = settings.model_copy(update={"line_file": Path(args.line).name})
    try:
        sources = load_sources_config(settings.sources_config_path)
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2
    kind = AdapterKind(args.source) if args.source else sources.active
    missing = [
        f"  - {name} at {host}:{port}"
        for name, host, port in required_services(kind, sources, settings)
        if not probe(host, port)
    ]
    if missing:
        print("not reachable:\n" + "\n".join(missing) + f"\n{START_HINT}", file=sys.stderr)
        return 2
    cmd = (
        None
        if args.no_sim
        else publisher_command(
            kind,
            line=settings.line_config_path,
            scenario=args.scenario,
            speed=args.speed,
            duration=args.duration,
            seed=args.seed,
        )
    )
    proc: subprocess.Popen[Any] | None = None
    try:
        if cmd is not None:
            print(
                f"starting the {kind.value} simulator publisher: {' '.join(cmd[1:])}",
                file=sys.stderr,
            )
            proc = popen(cmd)
            if not wait_ready(proc, ready_address(kind, sources), probe=probe):
                print(f"the {kind.value} publisher did not become ready", file=sys.stderr)
                return 1
        serve_args = argparse.Namespace(
            mode="dev",
            line=args.line,
            source=kind.value,
            file_path=None,
            replay_speed=None,
            sqlite_path=None,
            host=args.host,
            port=args.port,
            cors_origins=args.cors_origins,
            inference=args.inference,
            model=args.model,
            model_version=args.model_version,
        )
        if serve is None:
            from twin_backend.__main__ import run_serve

            serve = run_serve
        return serve(serve_args)
    finally:
        stop_child(proc)


if __name__ == "__main__":
    sys.exit(main())
