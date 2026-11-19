"""Phase 10 runner: ``python -m twin_ml.experiments.real --study <name>`` (poe task ``real``).

Runs every (model, seed) of a study (``real_worker.study_specs``: ``metropt3``, ``azure``,
``azure_err``, ``azure_holdout``; ``all`` runs them in that order), each in its own worker
process (ADR 0020), and logs one MLflow run per pair in ``P10_real``. ``--skip-existing``
resumes: a pair is skipped only if a finished run with the same study config hash exists.

``--combine`` logs the combined policies (supervised sustained alarm OR detector sustained
alarm, seed i with seed i) from the stored alarms of finished ``azure`` and ``azure_holdout``
runs: every supervised model x every detector, all percentiles in one run. The primary pair
(best supervised by seed-mean validation PR-AUC at +24 h, OR the autoencoder) is tagged.
Pure numpy; no model is loaded.

``--explain`` (Part C) logs the exploratory explanation runs (``real_explain``).
``--smoke`` sends every run to the smoke experiment, which is never part of a result.
Progress goes to ``<progress_dir>/p10_<study>.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twin_common.config import ConfigError
from twin_common.logging import configure_logging, get_logger

log = get_logger(__name__)

STUDY_ORDER = ("metropt3", "azure", "azure_err", "azure_holdout")
COMBINE_STUDIES = ("azure", "azure_holdout")


def _write(path: Path, **event: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def expected_shas(
    study: str, models: list[str], seeds: list[int], real: Any, rcfg: Any, eval_cfg: Any, root: Path
) -> tuple[dict[tuple[str, int], str], dict[str, Any], str]:
    """(hash per (model, seed), model configs, dataset key) of a study."""
    from twin_ml.data.dataset import Dataset
    from twin_ml.datasets.build import resolve_real_dataset
    from twin_ml.experiments.real_worker import (
        model_config,
        study_config_sha,
        study_seeds,
        study_specs,
    )

    spec = study_specs(real)[study]
    key = Dataset.open(resolve_real_dataset(spec.dataset_config, root)).meta["dataset_key"]
    cfgs = {m: model_config(real, root, m) for m in models}
    shas = {
        (m, s): study_config_sha(spec, m, cfgs[m], real, rcfg, eval_cfg, key)
        for m in models
        for s in study_seeds(m, seeds)
    }
    return shas, cfgs, key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.real")
    parser.add_argument(
        "--study", default="", help="metropt3 | azure | azure_err | azure_holdout | all"
    )
    parser.add_argument("--combine", action="store_true", help="log the combined policy runs")
    parser.add_argument("--explain", action="store_true", help="exploratory explanations (Part C)")
    parser.add_argument("--models", default="", help="comma-separated; default: the study's")
    parser.add_argument("--seeds", default="", help="comma-separated; default: real.yaml seeds")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--smoke", action="store_true", help="log to the smoke experiment")
    parser.add_argument("--real-config", type=Path, default=Path("config/models/real.yaml"))
    args = parser.parse_args(argv)
    configure_logging()
    if sum((bool(args.study), args.combine, args.explain)) != 1:
        print("error: give exactly one of --study, --combine or --explain", file=sys.stderr)
        return 2

    from twin_ml.config import load_eval_config, load_robustness_config
    from twin_ml.data.dataset import DatasetNotBuiltError
    from twin_ml.datasets.common import MissingDatasetError
    from twin_ml.datasets.config import load_real_config
    from twin_ml.experiments.real_worker import study_specs
    from twin_ml.train.__main__ import parse_seeds

    root = Path()
    try:
        real = load_real_config(root / args.real_config)
        eval_cfg = load_eval_config(root / real.eval_config)
        rcfg = load_robustness_config(root / real.robustness_config)
        experiment = real.smoke_experiment if args.smoke else real.experiment
        seeds = parse_seeds(args.seeds, real.seeds)
        specs = study_specs(real)
        if args.combine:
            return run_combine(
                real, rcfg, eval_cfg, root, experiment, seeds, args.skip_existing, args.dry_run
            )
        if args.explain:
            from twin_ml.experiments.real_explain import run_explain_study

            return run_explain_study(
                real, rcfg, eval_cfg, root, experiment, seeds, args.skip_existing, args.dry_run
            )
        studies = list(STUDY_ORDER) if args.study == "all" else [args.study]
        for st in studies:
            if st not in specs:
                raise ConfigError(f"unknown study {st!r}; known: {sorted(specs)} or all")
    except (ConfigError, DatasetNotBuiltError, MissingDatasetError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for st in studies:
        try:
            models = [m for m in args.models.split(",") if m] or list(specs[st].models)
            unknown = sorted(set(models) - set(specs[st].models))
            if unknown:
                if args.study == "all":
                    models = [m for m in models if m in specs[st].models]
                else:
                    raise ConfigError(f"models {unknown} are not in study {st}")
            shas, cfgs, _ = expected_shas(st, models, seeds, real, rcfg, eval_cfg, root)
        except (ConfigError, DatasetNotBuiltError, MissingDatasetError, LookupError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        rc = run_study(
            st, shas, cfgs, real, eval_cfg, root, experiment, args.skip_existing, args.dry_run
        )
        if rc:
            return rc
    return 0


def run_study(
    study: str,
    shas: dict[tuple[str, int], str],
    cfgs: dict[str, Any],
    real: Any,
    eval_cfg: Any,
    root: Path,
    experiment: str,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    from twin_ml.experiments.real_worker import (
        RealProcessError,
        latest_valid_runs,
        run_real_isolated,
    )

    tdir = root / eval_cfg.tracking_dir
    done: set[tuple[str, int]] = set()
    if skip_existing:
        done = set(latest_valid_runs(tdir, experiment, real.phase_tag, study, shas))
    todo = [k for k in shas if k not in done]
    progress = root / real.progress_dir / f"p10_{study}.jsonl"
    print(
        f"study {study}: {len(todo)} runs ({len(done)} done); experiment {experiment}; "
        f"progress: {progress.as_posix()}",
        flush=True,
    )
    if dry_run:
        for m, s in todo:
            print(f"  {m} seed {s}")
        return 0
    _write(
        progress,
        event="plan",
        study=study,
        runs=[f"{m}:{s}" for m, s in todo],
        experiment=experiment,
    )
    started = time.perf_counter()
    for m, s in todo:
        _write(progress, event="start", study=study, model=m, seed=s)
        t0 = time.perf_counter()
        try:
            res = run_real_isolated(
                study, m, cfgs[m], s, real, eval_cfg, root, None, progress, experiment
            )
        except RealProcessError as exc:
            _write(progress, event="failed", study=study, model=m, seed=s, error=str(exc))
            print(
                f"error: {exc}; finished runs are kept, resume with --skip-existing",
                file=sys.stderr,
            )
            return 1
        sec = time.perf_counter() - t0
        _write(
            progress,
            event="finished",
            study=study,
            model=m,
            seed=s,
            run_id=res["run_id"],
            seconds=round(sec),
            elapsed_s=round(time.perf_counter() - started),
        )
        print(f"{study} {m} seed {s}: run {res['run_id']} in {sec:.0f} s", flush=True)
    _write(
        progress, event="study_done", study=study, elapsed_s=round(time.perf_counter() - started)
    )
    print(f"study {study} done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------- selection


def best_supervised(runs: dict[tuple[str, int], Any], real: Any, seeds: list[int]) -> str | None:
    """Highest seed-mean validation PR-AUC (WARNING / FAULT mean, operating, +24 h) among the
    supervised models with a finished run for every seed. Validation only (ADR 0027)."""
    import numpy as np

    from twin_ml.experiments.real_worker import SUPERVISED

    key = f"val.h{real.selection.horizon}.{real.selection.metric}"
    best, best_v = None, -np.inf
    for m in SUPERVISED:
        vals = [runs[(m, s)].data.metrics.get(key) for s in seeds if (m, s) in runs]
        if len(vals) != len(seeds) or any(v is None for v in vals):
            continue
        v = float(np.mean(vals))
        if v > best_v:
            best, best_v = m, v
    return best


def selection_table(
    runs: dict[tuple[str, int], Any], real: Any, seeds: list[int]
) -> dict[str, Any]:
    from twin_ml.experiments.real_worker import SUPERVISED

    key = f"val.h{real.selection.horizon}.{real.selection.metric}"
    return {
        m: [runs[(m, s)].data.metrics.get(key) if (m, s) in runs else None for s in seeds]
        for m in SUPERVISED
    }


# ---------------------------------------------------------------------------- combine


def combined_sha(sup_run: str, det_run: str, real: Any) -> str:
    from twin_ml.experiments.real_worker import EVAL_VERSION

    data = {
        "version": EVAL_VERSION,
        "supervised_run": sup_run,
        "detector_run": det_run,
        "combined": real.combined.model_dump(mode="json"),
        "anomaly": real.anomaly.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def study_runs(
    study: str, real: Any, rcfg: Any, eval_cfg: Any, root: Path, experiment: str, seeds: list[int]
) -> tuple[dict[tuple[str, int], Any], str]:
    from twin_ml.experiments.real_worker import latest_valid_runs, study_specs

    spec = study_specs(real)[study]
    shas, _, key = expected_shas(study, list(spec.models), seeds, real, rcfg, eval_cfg, root)
    tdir = root / eval_cfg.tracking_dir
    return latest_valid_runs(tdir, experiment, real.phase_tag, study, shas), key


def run_combine(
    real: Any,
    rcfg: Any,
    eval_cfg: Any,
    root: Path,
    experiment: str,
    seeds: list[int],
    skip_existing: bool,
    dry_run: bool,
) -> int:
    import mlflow
    import numpy as np

    from twin_ml.data.dataset import Dataset
    from twin_ml.datasets.build import resolve_real_dataset
    from twin_ml.evaluate import run_sums
    from twin_ml.experiments import e5_eval, real_eval
    from twin_ml.experiments.e5_worker import load_alarms
    from twin_ml.experiments.real_worker import (
        COMBINED_SUFFIX,
        DETECTORS,
        SUPERVISED,
        study_specs,
    )
    from twin_ml.train import tracking

    tdir = root / eval_cfg.tracking_dir
    started = time.perf_counter()
    n_logged = 0
    for study in COMBINE_STUDIES:
        spec = study_specs(real)[study]
        st = real.datasets[spec.dataset]
        runs, key = study_runs(study, real, rcfg, eval_cfg, root, experiment, seeds)
        best = best_supervised(runs, real, seeds)
        cstudy = f"{study}{COMBINED_SUFFIX}"
        existing: set[str] = set()
        if skip_existing:
            for r in tracking.finished_runs(
                tdir, experiment, None, real.phase_tag, f"tags.study = '{cstudy}'"
            ):
                existing.add(r.data.tags.get("combined_sha", ""))
        sup = [m for m in SUPERVISED if m in spec.models]
        dets = [d for d in DETECTORS if d in spec.models]
        pairs = []
        for m in sup:
            for s in seeds:
                for d in dets:
                    sr, dr = runs.get((m, s)), runs.get((d, s))
                    if sr is None or dr is None:
                        ran_m = any(k[0] == m for k in runs)
                        ran_d = any(k[0] == d for k in runs)
                        if ran_m and ran_d:  # a partial study: say which pair is missing
                            print(f"  {study}: missing source run for {m}:{s} + {d}:{s}; skipped")
                        continue
                    sha = combined_sha(sr.info.run_id, dr.info.run_id, real)
                    if sha not in existing:
                        pairs.append((m, s, d, sr, dr, sha))
        print(
            f"combined {study}: best supervised (validation) = {best}; {len(pairs)} runs to log",
            flush=True,
        )
        if dry_run or not pairs:
            continue
        ds = Dataset.open(resolve_real_dataset(spec.dataset_config, root))
        test = ds.split("test")
        fault_types = list(ds.fault_types)
        index = real_eval.SplitIndex.of(test, fault_types)
        events = real_eval.split_events(ds.meta, "test")
        split_lo, _ = real_eval.split_bounds(ds.meta, "test")
        n, digest = test.n_windows, e5_eval.ends_digest(test)
        h = st.horizon
        pcts = list(real.anomaly.threshold_percentiles)
        tracking.configure(tdir, experiment)
        with tempfile.TemporaryDirectory(prefix="twin_p10_comb_") as tmp_s:
            tmp = Path(tmp_s)
            cache: dict[str, dict[str, np.ndarray]] = {}
            for m, s, d, sr, dr, sha in pairs:
                for r in (sr, dr):
                    if r.info.run_id not in cache:
                        cache[r.info.run_id] = load_alarms(r, tmp)
                sa, da = cache[sr.info.run_id], cache[dr.info.run_id]
                for side in (sa, da):
                    if int(side["n:clean"][0]) != n or str(side["digest:clean"][0]) != digest:
                        raise ValueError("alarm artifacts do not match the test split's windows")
                a = e5_eval.unpack_alarm(sa[f"clean|h{h}"], n)
                groups: dict[str, run_sums.Sums] = {}
                metrics: dict[str, float] = {}
                detail: dict[str, Any] = {}
                for p in pcts:
                    both = a | e5_eval.unpack_alarm(da[f"clean|p{p:g}"], n)
                    ev = real_eval.evaluate_alarm(
                        test,
                        index,
                        both,
                        events,
                        list(st.lookbacks),
                        split_lo,
                        st.evaluable_min_coverage,
                    )
                    for lk, sm in ev.sums.items():
                        g = f"clean|p{p:g}|{lk}"
                        groups[g] = sm
                        for k_, v in ev.points[lk].items():
                            if np.isfinite(v):
                                metrics[f"clean.p{p:g}.{lk}.{k_}"] = float(v)
                        detail[g] = {"points": ev.points[lk], "events": ev.events[lk]}
                primary = m == best and d == real.combined.primary[1]
                with mlflow.start_run(run_name=f"{cstudy}-{m}+{d}-seed{s}") as run:
                    mlflow.set_tags(
                        {
                            "phase": real.phase_tag,
                            "study": cstudy,
                            "source_study": study,
                            "model": f"{m}+{d}",
                            "supervised": m,
                            "detector": d,
                            "seed": str(s),
                            "dataset_key": key,
                            "supervised_run_id": sr.info.run_id,
                            "detector_run_id": dr.info.run_id,
                            "best_supervised": str(best),
                            "primary": str(primary).lower(),
                            "combined_sha": sha,
                        }
                    )
                    mlflow.log_metrics(metrics)
                    mlflow.log_dict(
                        json.loads(
                            json.dumps(
                                {
                                    "groups": detail,
                                    "selection": selection_table(runs, real, seeds),
                                    "best_supervised": best,
                                },
                                default=float,
                            )
                        ),
                        "metrics.json",
                    )
                    arrays = run_sums.save_npz_dict(groups)
                    arrays["machines"] = np.array(index.machines, dtype=str)
                    path = tmp / f"comb_{run.info.run_id}" / "runstats.npz"
                    path.parent.mkdir()
                    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
                    mlflow.log_artifact(str(path))
                    n_logged += 1
                _write(
                    root / real.progress_dir / f"p10_{cstudy}.jsonl",
                    event="combined",
                    supervised=m,
                    seed=s,
                    detector=d,
                    run_id=run.info.run_id,
                )
    print(f"combined: {n_logged} runs logged in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
