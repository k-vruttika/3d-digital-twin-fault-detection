"""``python -m twin_ml.serve``: build deploy bundles and run inference workers (ADR 0029, 0030).

* ``deploy [--models gru,xgb,autoencoder]``: build and register each bundle, every one in its
  own process (torch and XGBoost never share one; ADR 0020). poe task ``deploy-models``.
* ``build --name <bundle>``: one bundle in this process (used by ``deploy``).
* ``worker --roles supervised,monitor ...``: the inference worker started by the backend.
* ``latency`` / ``report``: the live-inference latency measurement and report (poe task
  ``live-latency``; see :mod:`twin_ml.serve.latency`).
* ``drift``: the drift detection report (poe task ``drift-report``, ``serve.drift_report``).
* ``export [--out zip]`` / ``import <zip>``: move the registered bundles to another machine as
  one verified zip (poe tasks ``export-bundles`` / ``import-bundles``, ``serve.portable``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DEPLOY_CONFIG = Path("config/models/deploy.yaml")
DEFAULT_BUNDLES = ("gru", "xgb", "autoencoder")
DEFAULT_INFERENCE_CONFIG = Path("config/inference.yaml")
EXPORTED_MODELS = ("gru", "xgb")  # keys of config/inference.yaml models (+ the autoencoder)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m twin_ml.serve")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("deploy", help="build and register bundles, one process each")
    d.add_argument("--models", default=",".join(DEFAULT_BUNDLES))
    d.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    d.add_argument("--tracking-dir", type=Path, default=None)
    d.add_argument("--non-strict", action="store_true", help="tests only: no reference runs")

    b = sub.add_parser("build", help="build and register one bundle in this process")
    b.add_argument("--name", required=True)
    b.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    b.add_argument("--tracking-dir", type=Path, default=None)
    b.add_argument("--non-strict", action="store_true")

    w = sub.add_parser("worker", help="inference worker (JSON lines on stdin / stdout)")
    w.add_argument("--roles", required=True, help="supervised, monitor or both (comma list)")
    w.add_argument("--supervised", help="registered supervised model name")
    w.add_argument("--supervised-version", default="latest")
    w.add_argument("--anomaly", help="registered autoencoder model name")
    w.add_argument("--anomaly-version", default="latest")
    w.add_argument("--tracking-dir", type=Path, required=True)
    w.add_argument("--cache-dir", type=Path, required=True)
    w.add_argument("--ig-steps", type=int, required=True)

    e = sub.add_parser("export", help="write the deployed bundles to one portable zip")
    e.add_argument("--out", type=Path, default=None, help="default dist/linetwin-bundles-v1.zip")
    e.add_argument("--models", default=",".join(EXPORTED_MODELS), help="keys of inference.yaml")
    e.add_argument(
        "--version",
        action="append",
        default=[],
        metavar="REGISTERED=N",
        help="export this version instead of the latest, e.g. linetwin-gru=1",
    )
    e.add_argument("--inference-config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    e.add_argument("--tracking-dir", type=Path, default=None, help="default: inference.yaml")

    i = sub.add_parser("import", help="verify a bundle zip and register its bundles")
    i.add_argument("archive", type=Path)
    i.add_argument("--inference-config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    i.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    i.add_argument("--tracking-dir", type=Path, default=None, help="default: inference.yaml")
    i.add_argument("--no-load-check", action="store_true", help="skip the worker reproduction")

    from twin_ml.serve import drift_report
    from twin_ml.serve.latency import add_args

    add_args(sub)
    drift_report.add_args(sub)
    return p


def run_deploy(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    results = []
    for name in names:
        cmd = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "twin_ml.serve",
            "build",
            "--name",
            name,
            "--deploy-config",
            str(args.deploy_config),
        ]
        if args.tracking_dir is not None:
            cmd += ["--tracking-dir", str(args.tracking_dir)]
        if args.non_strict:
            cmd.append("--non-strict")
        t0 = time.perf_counter()
        out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
        if out.returncode != 0:
            print(f"building bundle {name} failed (exit code {out.returncode})", file=sys.stderr)
            return out.returncode or 1
        res = json.loads(out.stdout.strip().splitlines()[-1])
        res["process_s"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        print(json.dumps(res, indent=2))
    return 0


def run_build(args: argparse.Namespace) -> int:
    from twin_common.logging import configure_logging
    from twin_ml.config import load_deploy_config
    from twin_ml.serve.bundle import build_and_register

    configure_logging("INFO", stream=sys.stderr)
    cfg = load_deploy_config(args.deploy_config)
    res = build_and_register(args.name, cfg, Path(), args.tracking_dir, strict=not args.non_strict)
    print(
        json.dumps(
            {
                "bundle": res.name,
                "registered": cfg.registry_prefix + res.name,
                "version": res.version,
                "run_id": res.run_id,
                "seconds": round(res.seconds, 1),
                **res.summary,
            },
            default=str,
        )
    )
    return 0


def run_worker(args: argparse.Namespace) -> int:
    from twin_common.logging import configure_logging
    from twin_ml.serve.worker import ROLES, load_worker, serve

    configure_logging("INFO", stream=sys.stderr)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    bad = [r for r in roles if r not in ROLES]
    if bad or not roles:
        print(f"unknown roles {bad}; use {ROLES}", file=sys.stderr)
        return 2
    proto = sys.stdout
    sys.stdout = sys.stderr  # nothing but protocol lines may reach the backend's pipe

    def factory() -> object:
        return load_worker(
            roles,
            (args.supervised, args.supervised_version) if args.supervised else None,
            (args.anomaly, args.anomaly_version) if args.anomaly else None,
            args.tracking_dir,
            args.cache_dir,
            args.ig_steps,
        )

    return serve(factory, sys.stdin, proto)


def run_export(args: argparse.Namespace) -> int:
    from twin_common.config.inference import load_inference_config
    from twin_common.logging import configure_logging
    from twin_ml.serve.portable import DEFAULT_ARCHIVE, GateSpec, export_bundles

    configure_logging("WARNING", stream=sys.stderr)
    icfg = load_inference_config(args.inference_config)
    tracking_dir = args.tracking_dir or icfg.tracking_dir
    pinned = dict(v.split("=", 1) for v in args.version)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in icfg.models]
    if unknown:
        print(
            f"unknown models {unknown}; inference.yaml has {sorted(icfg.models)}", file=sys.stderr
        )
        return 2
    registered = [icfg.models[n].registered for n in names] + [icfg.anomaly.registered]
    default_v = {icfg.anomaly.registered: icfg.anomaly.version}
    entries = [(r, pinned.get(r, default_v.get(r, "latest"))) for r in registered]
    g = icfg.gate
    gate = GateSpec([s.value for s in g.operating], [s.value for s in g.waiting], g.max_gap_rows)
    timeout = icfg.worker.start_timeout_s
    try:
        res = export_bundles(entries, tracking_dir, args.out or DEFAULT_ARCHIVE, gate, timeout)
    except LookupError as exc:
        print(f"{exc}; nothing to export, build the bundles first (uv run poe deploy-models)")
        return 1
    for b in res.bundles:
        size = sum(f["size"] for f in b["files"])
        print(
            f"  {b['label']}: {len(b['files'])} files, {size / 1e6:.2f} MB, "
            f"{b['reference_windows']} reference predictions"
        )
    print(f"wrote {res.path} ({res.size_bytes / 1e6:.2f} MB, {res.size_bytes} bytes)")
    return 0


def run_import(args: argparse.Namespace) -> int:
    from twin_common.config.inference import load_inference_config
    from twin_common.logging import configure_logging
    from twin_ml.config import load_deploy_config
    from twin_ml.serve.portable import ArchiveError, import_bundles

    configure_logging("WARNING", stream=sys.stderr)
    icfg = load_inference_config(args.inference_config)
    dcfg = load_deploy_config(args.deploy_config)
    tracking_dir = args.tracking_dir or icfg.tracking_dir
    try:
        results = import_bundles(
            args.archive,
            tracking_dir,
            dcfg.experiment,
            dcfg.phase_tag,
            check_load=not args.no_load_check,
            timeout_s=icfg.worker.start_timeout_s,
        )
    except (ArchiveError, FileNotFoundError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    for r in results:
        how = "already registered, reused" if r.reused else "registered"
        print(f"  {r.label}: {how}; checks: {', '.join(r.checks)}")
    print(
        f"imported {len(results)} bundles into {tracking_dir}; the live twin uses the latest "
        "versions (config/inference.yaml)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "deploy":
        return run_deploy(args)
    if args.command == "build":
        return run_build(args)
    if args.command == "worker":
        return run_worker(args)
    if args.command == "export":
        return run_export(args)
    if args.command == "import":
        return run_import(args)
    if args.command == "drift":
        from twin_ml.serve import drift_report

        return drift_report.run(args)
    from twin_ml.serve.latency import run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
