"""Command line interface: ``python -m twin_sim {generate,run,plot-examples,mock-server}``.

Publisher subcommands (``mock-server``) import their network libraries lazily, so the
simulator core never needs them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from twin_common import aio
from twin_common.config import load_line_config, load_sources_config
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings
from twin_common.contracts import Reading
from twin_common.logging import configure_logging
from twin_sim.batch import EXPORT_FORMATS, generate
from twin_sim.config import check_scenario_compat, load_scenario, load_sim_config
from twin_sim.line import LineSimulator
from twin_sim.runner import RealtimeRunner
from twin_sim.scenario import make_run_plan

if TYPE_CHECKING:
    from twin_sim.publishers.session import SessionOptions as SessionOptionsT

QUICK_RUNS = 5
QUICK_SCENARIO = "quick.yaml"
DEFAULT_SCENARIO = "default.yaml"


class _LazyStderr:
    """Resolves ``sys.stderr`` on every write (it may be swapped, e.g. by test capture)."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def _parser(settings: TwinSettings) -> argparse.ArgumentParser:
    cfg = settings.config_dir
    p = argparse.ArgumentParser(prog="python -m twin_sim", description="LineTwin simulator")
    p.add_argument("--sim-config", type=Path, default=cfg / "simulator.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a labelled dataset (one Parquet per run)")
    g.add_argument(
        "--line",
        type=Path,
        default=settings.line_config_path,
        help="line layout YAML (default: config/line.yaml; config/line_single.yaml = one station)",
    )
    g.add_argument("--runs", type=int, default=None)
    g.add_argument("--duration", type=float, default=None, help="seconds per run")
    g.add_argument("--seed", type=int, default=None, help="base seed (default: scenario seed)")
    g.add_argument("--out", type=Path, default=Path("data/sim"))
    g.add_argument("--scenario", type=Path, default=None)
    g.add_argument("--export", default="", help=f"comma list of {','.join(EXPORT_FORMATS)}")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument(
        "--quick", action="store_true", help=f"{QUICK_RUNS} short runs of {QUICK_SCENARIO}"
    )

    r = sub.add_parser("run", help="real-time runner, JSON-lines Readings on stdout")
    r.add_argument("--line", type=Path, default=settings.line_config_path, help="line layout YAML")
    r.add_argument("--scenario", type=Path, default=cfg / "scenarios" / DEFAULT_SCENARIO)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--run-index", type=int, default=0)
    r.add_argument("--speed", type=float, default=1.0, help="speed multiplier (1 = real time)")
    r.add_argument("--duration", type=float, default=None, help="simulated seconds to run")

    m = sub.add_parser("mock-server", help="mock machine server: REST + WebSocket (FastAPI)")
    _session_args(m, settings)
    mock_host, mock_port = _mock_address(settings)
    m.add_argument("--host", default=mock_host, help="default: host of mock.base_url")
    m.add_argument("--port", type=int, default=mock_port, help="default: port of mock.base_url")
    m.add_argument("--command-timeout", type=float, default=5.0, help="seconds")

    e = sub.add_parser("plot-examples", help="plot one run of each fault type")
    e.add_argument("--out", type=Path, default=Path("experiments/results/sim_examples"))
    e.add_argument("--seed", type=int, default=7)
    return p


def _session_args(p: argparse.ArgumentParser, settings: TwinSettings) -> None:
    cfg = settings.config_dir
    p.add_argument("--line", type=Path, default=settings.line_config_path, help="line layout YAML")
    p.add_argument("--scenario", type=Path, default=cfg / "scenarios" / DEFAULT_SCENARIO)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-index", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="speed multiplier (1 = real time)")
    p.add_argument("--duration", type=float, default=None, help="simulated seconds to run")
    p.add_argument("--history-size", type=int, default=3600, help="readings kept per machine")


def _mock_address(settings: TwinSettings) -> tuple[str, int]:
    """Host and port of ``mock.base_url`` in sources.yaml (fallback 127.0.0.1:8100)."""
    try:
        mock = load_sources_config(settings.sources_config_path).mock
    except ConfigError:
        mock = None
    url = urlparse(mock.base_url if mock is not None else "http://127.0.0.1:8100")
    return url.hostname or "127.0.0.1", url.port or 8100


def _session_options(args: argparse.Namespace) -> SessionOptionsT:
    from twin_sim.publishers.session import SessionOptions

    return SessionOptions(
        line_path=args.line,
        scenario_path=args.scenario,
        sim_config_path=args.sim_config,
        seed=args.seed,
        run_index=args.run_index,
        speed=args.speed,
        duration_s=args.duration,
        history_size=args.history_size,
    )


def _cmd_mock_server(args: argparse.Namespace) -> int:
    from twin_sim.publishers.http import make_server
    from twin_sim.publishers.mock_server import create_app
    from twin_sim.publishers.session import SimSession

    session = SimSession.from_options(_session_options(args))
    app = create_app(session, own_session=True, command_timeout_s=args.command_timeout)
    sys.stderr.write(f"mock server on http://{args.host}:{args.port} (Ctrl+C to stop)\n")
    try:
        aio.run(make_server(app, args.host, args.port).serve())
    except KeyboardInterrupt:
        return 130
    return 0


def _cmd_generate(args: argparse.Namespace, settings: TwinSettings) -> int:
    scenario_dir = settings.config_dir / "scenarios"
    default_name = QUICK_SCENARIO if args.quick else DEFAULT_SCENARIO
    scenario = load_scenario(args.scenario or scenario_dir / default_name)
    if args.duration is not None:
        scenario = scenario.model_copy(update={"duration_s": args.duration})
    runs = args.runs if args.runs is not None else (QUICK_RUNS if args.quick else 1)
    seed = args.seed if args.seed is not None else scenario.seed
    exports = tuple(x.strip() for x in args.export.split(",") if x.strip())
    manifest = generate(
        line=load_line_config(args.line),
        sim=load_sim_config(args.sim_config),
        scenario=scenario,
        runs=runs,
        base_seed=seed,
        out_dir=args.out,
        exports=exports,
        workers=args.workers,
        cli_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    rows = sum(e["rows"] for e in manifest["run_entries"])
    sys.stdout.write(
        f"generated {runs} run(s), {rows} rows in {manifest['generation_time_s']:.1f} s "
        f"-> {args.out}\n"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    line = load_line_config(args.line)
    sim_cfg = load_sim_config(args.sim_config)
    scenario = load_scenario(args.scenario)
    check_scenario_compat(scenario, sim_cfg, line)
    seed = args.seed if args.seed is not None else scenario.seed
    plan = make_run_plan(scenario, sim_cfg, line, seed, args.run_index)
    sim = LineSimulator(line, sim_cfg, plan)

    def emit(reading: Reading) -> None:
        sys.stdout.write(reading.model_dump_json() + "\n")
        sys.stdout.flush()

    runner = RealtimeRunner(sim, emit, speed=args.speed)
    max_ticks = round(args.duration) if args.duration is not None else round(plan.duration_s)
    try:
        aio.run(runner.run(max_ticks=max_ticks))
    except KeyboardInterrupt:
        return 130
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from twin_sim.plots import plot_examples

    paths = plot_examples(
        line=load_line_config(TwinSettings().line_config_path),  # plots assume the full line
        sim=load_sim_config(args.sim_config),
        out_dir=args.out,
        seed=args.seed,
        scenario_dir=TwinSettings().config_dir / "scenarios",
    )
    for path in paths:
        sys.stdout.write(f"wrote {path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = TwinSettings()
    args = _parser(settings).parse_args(argv)
    # Logs go to stderr so `run` can stream Readings on stdout.
    configure_logging(settings.log_level, stream=_LazyStderr())  # type: ignore[arg-type]
    try:
        if args.command == "generate":
            return _cmd_generate(args, settings)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "mock-server":
            return _cmd_mock_server(args)
        return _cmd_plot(args)
    except (ConfigError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
