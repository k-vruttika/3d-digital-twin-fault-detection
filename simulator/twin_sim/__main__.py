"""Command line interface: ``python -m twin_sim {generate,run,plot-examples}``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from twin_common.config import load_line_config
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings
from twin_common.contracts import Reading
from twin_common.logging import configure_logging
from twin_sim.batch import EXPORT_FORMATS, generate
from twin_sim.config import check_scenario_compat, load_scenario, load_sim_config
from twin_sim.line import LineSimulator
from twin_sim.runner import RealtimeRunner
from twin_sim.scenario import make_run_plan

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

    e = sub.add_parser("plot-examples", help="plot one run of each fault type")
    e.add_argument("--out", type=Path, default=Path("experiments/results/sim_examples"))
    e.add_argument("--seed", type=int, default=7)
    return p


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
        asyncio.run(runner.run(max_ticks=max_ticks))
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
        return _cmd_plot(args)
    except (ConfigError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
