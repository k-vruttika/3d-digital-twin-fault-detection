"""Live-inference latency and the gate effect, measured and reported (Phase 11; ADR 0030).

``python -m twin_ml.serve latency`` (poe task ``live-latency``):

1. **Latency.** Starts the inference workers exactly as the backend does
   (``InferenceConfig.worker_plans``, ``JsonLinesProcess``), configures them with the line's
   machines, and replays one test-split run of the deployed model's dataset: one reading per
   machine per tick, sent the way the backend sends a batch. Per tick it records the round trip
   seen by the client (IPC included) and the workers' per-stage timings. Reported: p50 / p95 /
   max per tick (all machines) and per machine, separately for ticks with an explanation,
   against the 100 ms per machine per tick budget.
2. **Gate effect** (exploratory). The live gate predicts only windows whose rows are all
   operating; the evaluation of Phases 7-10 alarmed on every window. On the deployed model's
   stored clean test predictions (its E1 / source run), early warning is computed with the
   evaluated rule and again with gated windows forced to "no alarm".

Writes ``<out>.json`` and ``<out>.md`` (generated, never edited by hand).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

BUDGET_MS = 100.0  # per machine per tick (Phase 11 task)
EPOCH_BASE = datetime(2026, 1, 1, tzinfo=UTC).timestamp()


def add_args(sub: Any) -> None:
    p = sub.add_parser("latency", help="measure live-inference latency and the gate effect")
    p.add_argument("--inference-config", type=Path, default=Path("config/inference.yaml"))
    p.add_argument("--line-config", type=Path, default=Path("config/line.yaml"))
    p.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    p.add_argument("--models", default="gru,xgb", help="models of inference.yaml to measure")
    p.add_argument("--ticks", type=int, default=0, help="readings per machine (0 = the whole run)")
    p.add_argument("--data-dir", type=Path, default=Path("data/ml"))
    p.add_argument("--out", type=Path, default=Path("experiments/results/live_inference"))
    p.add_argument("--no-gate-effect", action="store_true")


def percentiles(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def pick_run(split: Any) -> str:
    """The test run with the most gradual-fault episodes (ties: first run id)."""
    a = split.arrays
    runs = [str(a["series_run"][s]) for s in a["ep_series"]]
    counts = {r: runs.count(r) for r in split.run_ids}
    ids: list[str] = split.run_ids
    return max(ids, key=lambda r: (counts[r], -ids.index(r)))


def run_rows(split: Any, run_id: str) -> dict[str, list[list[Any]]]:
    """Per machine of ``run_id``: ``[seq, ts, state, speed, temp, vib, load, current]`` rows."""
    from twin_ml.data.labels import STATES

    a = split.arrays
    out: dict[str, list[list[Any]]] = {}
    for sid, (run, machine) in enumerate(zip(a["series_run"], a["series_machine"], strict=True)):
        if str(run) != run_id:
            continue
        rows = np.flatnonzero(a["series"] == sid)
        out[str(machine)] = [
            [
                i,
                EPOCH_BASE + float(a["t_s"][r]),
                STATES[int(a["state"][r])].value,
                *[None if np.isnan(v) else float(v) for v in a["raw"][r]],
            ]
            for i, r in enumerate(rows)
        ]
    return out


def measure(args: argparse.Namespace, root: Path, model: str) -> dict[str, Any]:
    from twin_common.config import load_line_config
    from twin_common.config.inference import load_inference_config
    from twin_common.jsonl_process import JsonLinesProcess
    from twin_ml.data.dataset import Dataset

    cfg = load_inference_config(root / args.inference_config).with_model(model)
    line = load_line_config(root / args.line_config)
    plans = cfg.worker_plans()
    procs = [JsonLinesProcess(p.command(cfg, root), name=f"worker-{p.name}") for p in plans]
    hello: dict[str, Any] = {}
    try:
        t0 = time.perf_counter()
        for proc in procs:
            proc.start()
        for proc, plan in zip(procs, plans, strict=True):
            hello[plan.name] = proc.request({"op": "hello"}, cfg.worker.start_timeout_s)
        start_s = time.perf_counter() - t0
        sup = next(
            h["bundles"]["supervised"] for h in hello.values() if "supervised" in h["bundles"]
        )
        ds = Dataset.open(root / args.data_dir / sup["datasetKey"])
        test = ds.split("test")
        run_id = pick_run(test)
        rows = run_rows(test, run_id)
        machines = {
            m.id: {"kind": m.kind.value, "sensorParts": dict(m.sensor_parts)}
            for m in line.machines
            if m.id in rows
        }
        conf = {
            "op": "configure",
            "machines": machines,
            "gate": {
                "operating": [s.value for s in cfg.gate.operating],
                "waiting": [s.value for s in cfg.gate.waiting],
                "maxGapRows": cfg.gate.max_gap_rows,
            },
            "explainAsync": cfg.explain_async,
            "capacity": cfg.buffer_capacity_rows,
        }
        for proc in procs:
            proc.request(conf, cfg.worker.request_timeout_s)
        n_run = min(len(r) for r in rows.values())
        n_ticks = n_run if args.ticks <= 0 else min(args.ticks, n_run)
        ticks: list[dict[str, Any]] = []
        explained: list[dict[str, Any]] = []
        t_replay = time.perf_counter()
        for k in range(n_ticks):
            batch = {m: [r[k]] for m, r in rows.items()}
            t_tick = time.perf_counter()
            results = [
                proc.request({"op": "step", "rows": batch}, cfg.worker.request_timeout_s)
                for proc in procs
            ]
            rt_ms = (time.perf_counter() - t_tick) * 1000.0
            windows = max(r["windows"] for r in results)
            done = [e for r in results for e in r.get("explanations", [])]
            explained += done
            stages: dict[str, float] = {}
            for r in results:
                for s, v in r["timings_ms"].items():
                    stages[s] = stages.get(s, 0.0) + v
            ticks.append({"rt_ms": rt_ms, "windows": windows, "explained": bool(done), **stages})
        replay_s = time.perf_counter() - t_replay
        # collect explanations still running in the background (empty steps)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            time.sleep(0.25)
            for proc in procs:
                r = proc.request({"op": "step", "rows": {}}, cfg.worker.request_timeout_s)
                explained += r.get("explanations", [])
    finally:
        for proc in procs:
            proc.close()
    pred = [t for t in ticks if t["windows"] > 0]
    expl = [t for t in pred if t["explained"]]
    plain = [t for t in pred if not t["explained"]]
    stage_names = sorted({s for t in pred for s in t if s not in ("rt_ms", "windows", "explained")})
    per_machine = [t["rt_ms"] / t["windows"] for t in pred]
    return {
        "model": cfg.model,
        "plans": [{"name": p.name, "roles": list(p.roles)} for p in plans],
        "bundles": {n: h["bundles"] for n, h in hello.items()},
        "processes": {n: h["process"] for n, h in hello.items()},
        "start_s": start_s,
        "run_id": run_id,
        "machines": sorted(rows),
        "ticks": len(ticks),
        "ticks_with_windows": len(pred),
        "ticks_with_explanation": len(expl),
        "round_trip_ms": {
            "all": percentiles([t["rt_ms"] for t in pred]),
            "no_explanation": percentiles([t["rt_ms"] for t in plain]),
            "with_explanation": percentiles([t["rt_ms"] for t in expl]),
        },
        "per_machine_ms": percentiles(per_machine),
        "stages_ms": {s: percentiles([t.get(s, 0.0) for t in pred]) for s in stage_names},
        "explain_async": cfg.explain_async,
        "replay_s": replay_s,
        "ticks_per_s": len(ticks) / replay_s if replay_s > 0 else None,
        "explanations": {
            "n": len(explained),
            "errors": sum(1 for e in explained if e.get("error")),
            "compute_ms_per_window": percentiles([e["computeMs"] for e in explained]),
            "age_ms": percentiles([e["ageMs"] for e in explained]),
            "batch": percentiles([float(e["batch"]) for e in explained]),
        },
        "budget_ms": BUDGET_MS,
        "within_budget": bool(per_machine) and max(per_machine) < BUDGET_MS,
        "within_budget_p95": bool(per_machine)
        and float(np.percentile(per_machine, 95)) < BUDGET_MS,
    }


def gate_effect(bundle: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Early warning on the stored clean test predictions, evaluated rule vs live gate."""
    from twin_common.config.inference import load_inference_config
    from twin_ml.config import load_eval_config
    from twin_ml.data.dataset import Dataset
    from twin_ml.data.labels import NORMAL, STATES
    from twin_ml.evaluate.early_warning import summarize
    from twin_ml.evaluate.evaluator import horizon_run_stats, subset_masks
    from twin_ml.experiments.robust_worker import _download
    from twin_ml.serve.buffers import PREDICT, GateConfig, gate_over_series
    from twin_ml.train import tracking

    cfg = load_inference_config(root / args.inference_config)
    eval_cfg = load_eval_config(root / args.eval_config)
    ds = Dataset.open(root / args.data_dir / bundle["datasetKey"])
    test = ds.split("test")
    import mlflow

    tracking.configure(root / cfg.tracking_dir, eval_cfg.experiment)
    with tempfile.TemporaryDirectory(prefix="twin_p11_gate_") as tmp:
        run = mlflow.get_run(bundle["model"]["sourceRunId"])
        with np.load(_download(run.info.artifact_uri, "test_predictions.npz", Path(tmp))) as z:
            stored = {k: z[k] for k in z.files}
    gcfg = GateConfig.of(
        ds.window,
        [s.value for s in cfg.gate.operating],
        [s.value for s in cfg.gate.waiting],
        cfg.gate.max_gap_rows,
    )
    a = test.arrays
    open_rows = np.zeros(test.n_rows, dtype=bool)
    for sid in np.unique(a["series"]):
        rows = np.flatnonzero(a["series"] == sid)
        states = [STATES[int(c)].value for c in a["state"][rows]]
        status, _ = gate_over_series(states, gcfg)
        open_rows[rows] = np.array([s == PREDICT for s in status])
    gate_open = open_rows[a["ends"]]
    masks = subset_masks(test, eval_cfg)
    rw = test.window_run()
    out: dict[str, Any] = {
        "windows": int(test.n_windows),
        "gate_open_windows": int(gate_open.sum()),
        "operating_windows": int(masks["operating"].sum()),
        "operating_gate_closed": int((masks["operating"] & ~gate_open).sum()),
        "horizons": {},
    }
    for h in ds.horizons:
        labels = np.argmax(stored[f"probs_h{h}"], axis=1)
        res: dict[str, Any] = {}
        for name, lab in (("evaluated", labels), ("gated", np.where(gate_open, labels, NORMAL))):
            eps, stats = horizon_run_stats(
                test, lab, h, masks["operating"], ds.fault_types, eval_cfg, rw
            )
            res[name] = summarize(eps, int(stats.false_alarms.sum()), float(stats.negative_s.sum()))
            res[name]["detected_episodes"] = [
                f"{e.run}:{e.machine}:{e.t_fault:g}" for e in eps if e.detected
            ]
        changed = sorted(
            set(res["evaluated"]["detected_episodes"]) ^ set(res["gated"]["detected_episodes"])
        )
        res["episodes_changed"] = changed
        out["horizons"][str(h)] = res
    return out


def _fmt(x: Any, nd: int = 1) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Live inference: latency and gate effect",
        "",
        f"Generated by `uv run poe live-latency` at {result['generated_at']}. Every number comes "
        "from running the code; do not edit by hand.",
        "",
    ]
    for lat in result["latency"]:
        lines += [
            f"## Latency: `{lat['model']}`",
            "",
            "- Workers: "
            + ", ".join(f"`{p['name']}` ({'+'.join(p['roles'])})" for p in lat["plans"])
            + "; libraries per process: "
            + "; ".join(f"{n}: {', '.join(v['libraries'])}" for n, v in lat["processes"].items())
            + ".",
            "- Bundles: "
            + "; ".join(b["label"] for h in lat["bundles"].values() for b in h.values())
            + ".",
            f"- Replay: test run `{lat['run_id']}` of the bundle's dataset, machines "
            f"{', '.join(lat['machines'])}; {lat['ticks']} ticks (one reading per machine per "
            f"tick), {lat['ticks_with_windows']} with predicted windows, "
            f"{lat['ticks_with_explanation']} that returned an explanation.",
            f"- Worker start (bundle download + model load): {_fmt(lat['start_s'])} s.",
            "",
            "| Measure (ms) | n | p50 | p95 | max |",
            "|---|---|---|---|---|",
        ]
        rows = [
            ("Round trip per tick, all machines", lat["round_trip_ms"]["all"]),
            ("  ticks without explanation", lat["round_trip_ms"]["no_explanation"]),
            ("  ticks with explanation", lat["round_trip_ms"]["with_explanation"]),
            ("Per machine per tick (round trip / windows)", lat["per_machine_ms"]),
            *[(f"Worker stage: {s}", v) for s, v in lat["stages_ms"].items()],
        ]
        for name, v in rows:
            if v.get("n", 0):
                lines.append(
                    f"| {name} | {v['n']} | {_fmt(v['p50'], 2)} | {_fmt(v['p95'], 2)} | "
                    f"{_fmt(v['max'], 2)} |"
                )
        verdict = "within" if lat["within_budget"] else "NOT within"
        mode = (
            "beside the prediction path (background thread; not in the round trips above)"
            if lat["explain_async"]
            else "inside the prediction path (synchronous; included in the round trips)"
        )
        ex = lat["explanations"]
        lines += [
            "",
            f"Budget: {lat['budget_ms']:g} ms per machine per tick on CPU. The maximum per-machine "
            f"time is {verdict} the budget (p95 within: {lat['within_budget_p95']}).",
            "",
            f"Explanations ({ex['n']} windows, {ex['errors']} errors), computed {mode}. The "
            f"replay sent {_fmt(lat['ticks_per_s'])} ticks per second (live: 1 reading per "
            "machine per second); a newer request replaces a pending one of the same machine, "
            "so at replay speed fewer windows are explained than alarmed. Live, an explanation "
            "that arrives within a second describes the previous reading's window:",
            "",
            "| Explanation (ms) | n | p50 | p95 | max |",
            "|---|---|---|---|---|",
        ]
        for name, key in (
            ("Compute time per window", "compute_ms_per_window"),
            ("Age at delivery (request to result)", "age_ms"),
        ):
            v = ex[key]
            if v.get("n", 0):
                lines.append(
                    f"| {name} | {v['n']} | {_fmt(v['p50'], 1)} | {_fmt(v['p95'], 1)} | "
                    f"{_fmt(v['max'], 1)} |"
                )
        lines.append("")
    for ge in result.get("gate_effect", []):
        lines += [
            f"## Gate effect (exploratory): `{ge['bundle']}`",
            "",
            f"Model: `{ge['bundle']}`, stored clean test predictions of its source run. The live "
            "gate predicts only windows whose rows are all RUNNING / WARNING; the evaluated rule "
            "alarmed on every window (false alarms counted on operating windows).",
            "",
            f"- Test windows: {ge['windows']}; gate open: {ge['gate_open_windows']}; operating "
            f"(evaluated subset): {ge['operating_windows']}, of which gate closed: "
            f"{ge['operating_gate_closed']}.",
            "",
            "| Horizon | Rule | Episodes | Miss rate | Lead mean s | vs sim WARNING mean s | "
            "False alarms / h |",
            "|---|---|---|---|---|---|---|",
        ]
        for h, res in ge["horizons"].items():
            for name in ("evaluated", "gated"):
                r = res[name]
                lines.append(
                    f"| +{h} s | {name} | {r['episodes']} | {_fmt(r['miss_rate'], 3)} | "
                    f"{_fmt(r['lead_mean_s'])} | {_fmt(r['rel_lead_mean_s'])} | "
                    f"{_fmt(r['false_alarms_per_h'], 3)} |"
                )
            lines.append(
                f"| +{h} s | episodes whose detection differs | {len(res['episodes_changed'])} "
                "| | | | |"
            )
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> int:
    from twin_common.logging import configure_logging

    configure_logging("INFO")
    root = Path()
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "latency": [],
        "gate_effect": [],
    }
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        lat = measure(args, root, model)
        result["latency"].append(lat)
        if not args.no_gate_effect:
            bundle = next(h["supervised"] for h in lat["bundles"].values() if "supervised" in h)
            ge = gate_effect(bundle, args, root)
            ge["bundle"] = bundle["label"]
            result["gate_effect"].append(ge)
        print(json.dumps({"model": model, "per_machine_ms": lat["per_machine_ms"]}))
    write_report(result, args.out)
    return 0
