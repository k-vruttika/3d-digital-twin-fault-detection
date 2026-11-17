"""Phase 9 runner: ``python -m twin_ml.experiments.e5 --study <name>`` (poe task ``e5``).

Runs every (model, seed) of a study in ``config/models/e5.yaml``, each in its own worker
process (``e5_worker``; ADR 0020), and logs one MLflow run per pair in the E5 experiment.
``--skip-existing`` resumes: a pair is skipped only if a finished run with the same study
config hash exists (model config, E5 / robustness / evaluation settings, dataset key, the
reused source run and the E5 data's content hash).

``--combine`` logs the combined policy (supervised sustained alarm OR detector sustained
alarm) from the stored alarms of finished study runs, one run per (supervised model and
seed, detector and seed): E1 and augmentation-trained models with the E1-trained detectors,
E4 models with the E4-trained detectors. Pure numpy; no model is loaded.

``--smoke`` sends every run to the smoke experiment, which is never part of a result.
Progress goes to ``<progress_dir>/<study>.jsonl``.
"""

from __future__ import annotations

import argparse
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

COMBINE_SPECS = (  # (supervised study, detector study)
    ("e5_e1models", "e5_anomaly"),
    ("e5_aug", "e5_anomaly"),
    ("e5_e4", "e5_e4"),
)


def _write(path: Path, **event: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def study_seeds(model: str, seeds: list[int], anomaly_seeds: list[int]) -> list[int]:
    from twin_ml.experiments.e5_worker import is_anomaly
    from twin_ml.train.trainer import BASELINES

    if model in BASELINES:
        return seeds[:1]
    return [s for s in anomaly_seeds if s in seeds] if is_anomaly(model) else seeds


def estimate_s(
    model: str, n_cond: float, eta_train: dict[str, float], eta_cond: dict[str, float]
) -> float:
    from twin_ml.experiments.e5_worker import is_anomaly

    train = eta_train.get(model, 0.0) if (model == "rf" or is_anomaly(model)) else 0.0
    return train + n_cond * eta_cond.get(model, 0.0)


def expected_shas(
    e5: Any,
    study: str,
    models: list[str],
    seeds: list[int],
    cfgs: dict[str, Any],
    rcfg: Any,
    eval_cfg: Any,
    key: str,
    tdir: Path,
    root: Path,
) -> dict[tuple[str, int], str]:
    from twin_ml.experiments.e5_worker import e5_content_sha, source_id, study_config_sha

    content = e5_content_sha(e5, root)
    out: dict[tuple[str, int], str] = {}
    for m in models:
        for s in study_seeds(m, seeds, e5.anomaly.seeds):
            src = source_id(e5, study, tdir, key, m, s, cfgs[m])
            out[(m, s)] = study_config_sha(study, m, cfgs[m], e5, rcfg, eval_cfg, key, src, content)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.e5")
    parser.add_argument("--study", default="", help="e5_e1models | e5_aug | e5_anomaly | e5_e4")
    parser.add_argument("--combine", action="store_true", help="log the combined policy runs")
    parser.add_argument("--explain", action="store_true", help="explanation sanity checks (Part B)")
    parser.add_argument(
        "--ae-shares", action="store_true", help="exploratory AE per-sensor error shares (Part C)"
    )
    parser.add_argument(
        "--ig-sensitivity", action="store_true", help="exploratory IG step-count check (Part C)"
    )
    parser.add_argument(
        "--followup-config", type=Path, default=Path("config/models/e5_followup.yaml")
    )
    parser.add_argument("--models", default="", help="comma-separated; default: the study's")
    parser.add_argument("--seeds", default="", help="comma-separated; default: eval.yaml seeds")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--smoke", action="store_true", help="log to the smoke experiment")
    parser.add_argument("--e5-config", type=Path, default=Path("config/models/e5.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument("--models-dir", type=Path, default=Path("config/models"))
    args = parser.parse_args(argv)
    configure_logging()
    modes = (bool(args.study), args.combine, args.explain, args.ae_shares, args.ig_sensitivity)
    if sum(modes) != 1:
        print(
            "error: give exactly one of --study, --combine, --explain, --ae-shares or "
            "--ig-sensitivity",
            file=sys.stderr,
        )
        return 2

    from twin_ml.config import (
        load_dataset_config,
        load_e5_config,
        load_eval_config,
        load_robustness_config,
    )
    from twin_ml.data.dataset import Dataset, DatasetNotBuiltError, resolve_dataset_dir
    from twin_ml.experiments.e5_worker import model_config
    from twin_ml.train.__main__ import parse_seeds

    root = Path()
    try:
        e5 = load_e5_config(args.e5_config)
        eval_cfg = load_eval_config(args.eval_config)
        rcfg = load_robustness_config(e5.robustness_config)
        experiment = e5.smoke_experiment if args.smoke else e5.experiment
        seeds = parse_seeds(args.seeds, eval_cfg.seeds)
        if args.ae_shares or args.ig_sensitivity:
            from twin_ml.config import load_e5_followup_config

            return run_followup_study(
                "ae_shares" if args.ae_shares else "ig_sensitivity",
                e5,
                load_e5_followup_config(args.followup_config),
                rcfg,
                eval_cfg,
                args.models_dir,
                root,
                experiment,
                seeds,
                args.skip_existing,
                args.dry_run,
            )
        if args.explain:
            return run_explain_study(
                e5,
                eval_cfg,
                args.models_dir,
                root,
                experiment,
                seeds,
                args.skip_existing,
                args.dry_run,
                [m for m in args.models.split(",") if m],
            )
        if args.combine:
            return run_combine(
                e5,
                rcfg,
                eval_cfg,
                args.models_dir,
                root,
                experiment,
                seeds,
                args.skip_existing,
                args.dry_run,
            )
        if args.study not in e5.studies:
            raise ConfigError(f"unknown study {args.study!r}; known: {sorted(e5.studies)}")
        study = e5.studies[args.study]
        ds = Dataset.open(resolve_dataset_dir(load_dataset_config(study.dataset_config)))
        key = ds.meta["dataset_key"]
        models = [m for m in args.models.split(",") if m] or list(study.models)
        unknown = sorted(set(models) - set(study.models))
        if unknown:
            raise ConfigError(f"models {unknown} are not in study {args.study}")
        cfgs = {m: model_config(e5, root, args.models_dir, m) for m in models}
        shas = expected_shas(
            e5, args.study, models, seeds, cfgs, rcfg, eval_cfg, key, eval_cfg.tracking_dir, root
        )
    except (ConfigError, DatasetNotBuiltError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from twin_ml.experiments.e5_worker import E5ProcessError, latest_valid_runs, run_e5_isolated

    done: set[tuple[str, int]] = set()
    if args.skip_existing:
        done = set(
            latest_valid_runs(eval_cfg.tracking_dir, experiment, e5.phase_tag, args.study, shas)
        )
    todo = [k for k in shas if k not in done]
    weight = {"e5": 3.3}  # E5 has about 3.3 x the clean test windows
    n_cond = sum(weight.get(c, 1.0) for c in study.conditions)
    est = {k: estimate_s(k[0], n_cond, e5.eta_train_s, e5.eta_condition_s) for k in todo}
    total = sum(est.values())
    progress = e5.progress_dir / f"{args.study}.jsonl"
    print(
        f"study {args.study}: {len(todo)} runs ({len(done)} done), conditions "
        f"{study.conditions}, estimated {total / 60:.0f} min; experiment {experiment}; "
        f"progress: {progress.as_posix()}",
        flush=True,
    )
    if args.dry_run:
        for (m, s), e in est.items():
            print(f"  {m} seed {s}: ~{e / 60:.1f} min")
        return 0
    _write(
        progress,
        event="plan",
        study=args.study,
        runs=[f"{m}:{s}" for m, s in todo],
        estimated_s=round(total),
        experiment=experiment,
    )
    started = time.perf_counter()
    for m, s in todo:
        _write(
            progress,
            event="start",
            study=args.study,
            model=m,
            seed=s,
            estimated_s=round(est[(m, s)]),
        )
        t0 = time.perf_counter()
        try:
            res = run_e5_isolated(
                args.study, m, cfgs[m], s, e5, eval_cfg, progress=progress, experiment=experiment
            )
        except E5ProcessError as exc:
            _write(progress, event="failed", study=args.study, model=m, seed=s, error=str(exc))
            print(
                f"error: {exc}; finished runs are kept, resume with --skip-existing",
                file=sys.stderr,
            )
            return 1
        sec = time.perf_counter() - t0
        _write(
            progress,
            event="finished",
            study=args.study,
            model=m,
            seed=s,
            run_id=res["run_id"],
            seconds=round(sec),
            elapsed_s=round(time.perf_counter() - started),
        )
        print(f"{m} seed {s}: run {res['run_id']} in {sec:.0f} s", flush=True)
    _write(
        progress,
        event="study_done",
        study=args.study,
        elapsed_s=round(time.perf_counter() - started),
    )
    print(f"study {args.study} done in {time.perf_counter() - started:.0f} s")
    return 0


def run_explain_study(
    e5: Any,
    eval_cfg: Any,
    models_dir: Path,
    root: Path,
    experiment: str,
    seeds: list[int],
    skip_existing: bool,
    dry_run: bool,
    models: list[str],
) -> int:
    """One explanation run per (model, seed) of the E1 models (study ``explain``)."""
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.experiments.e5_worker import (
        EXPLAIN_SOURCE_STUDY,
        EXPLAIN_STUDY,
        E5ProcessError,
        e5_content_sha,
        explain_config_sha,
        latest_valid_runs,
        model_config,
        run_e5_isolated,
        source_id,
    )

    ex = e5.explain
    names = models or [*ex.primary_models, *ex.exploratory_models]
    st = e5.studies[EXPLAIN_SOURCE_STUDY]
    key = Dataset.open(
        resolve_dataset_dir(load_dataset_config(root / st.dataset_config), root)
    ).meta["dataset_key"]
    tdir = eval_cfg.tracking_dir
    cfgs = {m: model_config(e5, root, models_dir, m) for m in names}
    content = e5_content_sha(e5, root)
    shas = {
        (m, s): explain_config_sha(
            m,
            cfgs[m],
            e5,
            eval_cfg,
            key,
            source_id(e5, EXPLAIN_SOURCE_STUDY, tdir, key, m, s, cfgs[m]),
            content,
        )
        for m in names
        for s in seeds
    }
    done = (
        set(latest_valid_runs(tdir, experiment, e5.phase_tag, EXPLAIN_STUDY, shas))
        if skip_existing
        else set()
    )
    todo = [k for k in shas if k not in done]
    progress = e5.progress_dir / f"{EXPLAIN_STUDY}.jsonl"
    print(f"explain: {len(todo)} runs ({len(done)} done); experiment {experiment}", flush=True)
    if dry_run:
        for m, s in todo:
            print(f"  {m} seed {s}")
        return 0
    started = time.perf_counter()
    for m, s in todo:
        _write(progress, event="start", study=EXPLAIN_STUDY, model=m, seed=s)
        t0 = time.perf_counter()
        try:
            res = run_e5_isolated(
                EXPLAIN_STUDY, m, cfgs[m], s, e5, eval_cfg, progress=progress, experiment=experiment
            )
        except E5ProcessError as exc:
            _write(progress, event="failed", study=EXPLAIN_STUDY, model=m, seed=s, error=str(exc))
            print(f"error: {exc}; resume with --skip-existing", file=sys.stderr)
            return 1
        sec = time.perf_counter() - t0
        _write(
            progress,
            event="finished",
            study=EXPLAIN_STUDY,
            model=m,
            seed=s,
            run_id=res["run_id"],
            seconds=round(sec),
        )
        print(f"{m} seed {s}: run {res['run_id']} in {sec:.0f} s", flush=True)
    _write(
        progress,
        event="study_done",
        study=EXPLAIN_STUDY,
        elapsed_s=round(time.perf_counter() - started),
    )
    print(f"explain done in {time.perf_counter() - started:.0f} s")
    return 0


def followup_shas(
    kind: str,
    e5: Any,
    followup: Any,
    rcfg: Any,
    eval_cfg: Any,
    models_dir: Path,
    root: Path,
    seeds: list[int],
) -> tuple[str, dict[tuple[str, int], str], dict[str, Any]]:
    """(study name, expected hash per (model, seed), model configs) of a follow-up study."""
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.experiments.e5_worker import (
        AE_SHARES_STUDY,
        EXPLAIN_SOURCE_STUDY,
        IG_SENSITIVITY_STUDY,
        ae_shares_sha,
        ae_source_run,
        e5_content_sha,
        explain_config_sha,
        model_config,
        source_id,
    )

    tdir = eval_cfg.tracking_dir
    content = e5_content_sha(e5, root)
    if kind == "ae_shares":
        a = followup.ae_shares
        st = e5.studies[a.source_study]
        key = Dataset.open(
            resolve_dataset_dir(load_dataset_config(root / st.dataset_config), root)
        ).meta["dataset_key"]
        cfg = model_config(e5, root, models_dir, a.detector)
        shas = {}
        for s in [s for s in e5.anomaly.seeds if s in seeds]:
            src = ae_source_run(e5, followup, rcfg, eval_cfg, cfg, key, s, tdir, root)
            shas[(a.detector, s)] = ae_shares_sha(
                cfg, e5, followup, eval_cfg, key, src.info.run_id, content
            )
        return AE_SHARES_STUDY, shas, {a.detector: cfg}
    ig = followup.ig_sensitivity
    st = e5.studies[EXPLAIN_SOURCE_STUDY]
    key = Dataset.open(
        resolve_dataset_dir(load_dataset_config(root / st.dataset_config), root)
    ).meta["dataset_key"]
    cfgs = {m: model_config(e5, root, models_dir, m) for m in ig.models}
    shas = {
        (m, s): explain_config_sha(
            m,
            cfgs[m],
            e5,
            eval_cfg,
            key,
            source_id(e5, EXPLAIN_SOURCE_STUDY, tdir, key, m, s, cfgs[m]),
            content,
            IG_SENSITIVITY_STUDY,
            ig.ig_steps,
        )
        for m in ig.models
        for s in seeds
    }
    return IG_SENSITIVITY_STUDY, shas, cfgs


def run_followup_study(
    kind: str,
    e5: Any,
    followup: Any,
    rcfg: Any,
    eval_cfg: Any,
    models_dir: Path,
    root: Path,
    experiment: str,
    seeds: list[int],
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Exploratory Part C follow-ups (``e5_followup.yaml``), one worker process per run."""
    from twin_ml.experiments.e5_worker import E5ProcessError, latest_valid_runs, run_e5_isolated

    study, shas, cfgs = followup_shas(kind, e5, followup, rcfg, eval_cfg, models_dir, root, seeds)
    tdir = eval_cfg.tracking_dir
    done = (
        set(latest_valid_runs(tdir, experiment, e5.phase_tag, study, shas))
        if skip_existing
        else set()
    )
    todo = [k for k in shas if k not in done]
    progress = e5.progress_dir / f"{study}.jsonl"
    print(f"{study}: {len(todo)} runs ({len(done)} done); experiment {experiment}", flush=True)
    if dry_run:
        for m, s in todo:
            print(f"  {m} seed {s}")
        return 0
    started = time.perf_counter()
    for m, s in todo:
        _write(progress, event="start", study=study, model=m, seed=s)
        t0 = time.perf_counter()
        try:
            res = run_e5_isolated(
                study,
                m,
                cfgs[m],
                s,
                e5,
                eval_cfg,
                progress=progress,
                experiment=experiment,
                followup=followup,
            )
        except E5ProcessError as exc:
            _write(progress, event="failed", study=study, model=m, seed=s, error=str(exc))
            print(f"error: {exc}; resume with --skip-existing", file=sys.stderr)
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
        )
        print(f"{m} seed {s}: run {res['run_id']} in {sec:.0f} s", flush=True)
    _write(
        progress, event="study_done", study=study, elapsed_s=round(time.perf_counter() - started)
    )
    print(f"{study} done in {time.perf_counter() - started:.0f} s")
    return 0


def run_combine(
    e5: Any,
    rcfg: Any,
    eval_cfg: Any,
    models_dir: Path,
    root: Path,
    experiment: str,
    seeds: list[int],
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Log one combined-policy run per (supervised run, detector run) pair."""
    import mlflow
    import numpy as np

    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.test_sets import extended_fault_types
    from twin_ml.evaluate import run_sums
    from twin_ml.experiments.e5_worker import (
        COMBINED_STUDY,
        _Splits,
        combine_pair,
        combined_sha,
        latest_valid_runs,
        load_alarms,
        model_config,
    )
    from twin_ml.train import tracking
    from twin_ml.train.trainer import BASELINES

    tdir = eval_cfg.tracking_dir
    existing: set[str] = set()
    if skip_existing:
        for run in tracking.finished_runs(
            tdir, experiment, None, e5.phase_tag, f"tags.study = '{COMBINED_STUDY}'"
        ):
            existing.add(run.data.tags.get("combined_sha", ""))
    progress = e5.progress_dir / f"{COMBINED_STUDY}.jsonl"
    started = time.perf_counter()
    n_logged = 0
    for sup_study, det_study in COMBINE_SPECS:
        if sup_study not in e5.studies or det_study not in e5.studies:
            continue
        sst, dst = e5.studies[sup_study], e5.studies[det_study]
        ds_cfg = load_dataset_config(root / sst.dataset_config)
        ds = Dataset.open(resolve_dataset_dir(ds_cfg, root))
        key = ds.meta["dataset_key"]
        sup_models = [m for m in e5.combined.supervised if m in sst.models]
        det_models = [m for m in e5.combined.detectors if m in dst.models]
        cfgs = {m: model_config(e5, root, models_dir, m) for m in sup_models + det_models}
        dkey = Dataset.open(
            resolve_dataset_dir(load_dataset_config(root / dst.dataset_config), root)
        ).meta["dataset_key"]
        sup_runs = latest_valid_runs(
            tdir,
            experiment,
            e5.phase_tag,
            sup_study,
            expected_shas(e5, sup_study, sup_models, seeds, cfgs, rcfg, eval_cfg, key, tdir, root),
        )
        det_runs = latest_valid_runs(
            tdir,
            experiment,
            e5.phase_tag,
            det_study,
            expected_shas(e5, det_study, det_models, seeds, cfgs, rcfg, eval_cfg, dkey, tdir, root),
        )
        pairs = []
        for sm in sup_models:
            for ss in study_seeds(sm, seeds, e5.anomaly.seeds):
                for dm in det_models:
                    dseeds = study_seeds(dm, seeds, e5.anomaly.seeds) if sm in BASELINES else [ss]
                    for ds_ in dseeds:
                        sr, dr = sup_runs.get((sm, ss)), det_runs.get((dm, ds_))
                        if sr is None or dr is None:
                            print(f"  missing source run for {sm}:{ss} + {dm}:{ds_}; skipped")
                            continue
                        sha = combined_sha(sr.info.run_id, dr.info.run_id, e5, eval_cfg)
                        if sha not in existing:
                            pairs.append((sm, ss, dm, ds_, sr, dr, sha))
        conditions = [c for c in sst.conditions if c in dst.conditions]
        print(
            f"combined {sup_study} + {det_study}: {len(pairs)} runs to log, conditions "
            f"{conditions}",
            flush=True,
        )
        if dry_run or not pairs:
            continue
        fault_types = extended_fault_types(ds, [e5.e5_data.fault_type])
        splits = _Splits(e5, rcfg, ds, ds_cfg, root)
        results: dict[int, dict[str, Any]] = {i: {} for i in range(len(pairs))}
        run_ids: dict[str, list[str]] = {}
        with tempfile.TemporaryDirectory(prefix="twin_p9_comb_") as tmp_s:
            tmp = Path(tmp_s)
            alarms: dict[str, dict[str, np.ndarray]] = {}
            for p in pairs:
                for r in (p[4], p[5]):
                    if r.info.run_id not in alarms:
                        alarms[r.info.run_id] = load_alarms(r, tmp)
            for cond in conditions:
                split, sf_rows, _ = splits.get(cond)
                run_ids[cond] = split.run_ids
                for i, (_sm, _ss, _dm, _ds, sr, dr, _sha) in enumerate(pairs):
                    groups = combine_pair(
                        split,
                        cond,
                        alarms[sr.info.run_id],
                        alarms[dr.info.run_id],
                        list(e5.combined.percentiles),
                        e5.primary.horizon,
                        fault_types,
                        eval_cfg,
                        sf_rows,
                    )
                    for g, s in groups.items():
                        results[i][f"{cond}|{g}"] = s
            tracking.configure(tdir, experiment)
            for i, (sm, ss, dm, ds_, sr, dr, sha) in enumerate(pairs):
                metrics = {
                    f"{gk.replace('|', '.')}.{m}": float(v)
                    for gk, s in results[i].items()
                    for m, v in run_sums.point(s).items()
                    if np.isfinite(v)
                }
                with mlflow.start_run(run_name=f"{sup_study}-{sm}+{dm}-seed{ss}-{ds_}") as run:
                    mlflow.set_tags(
                        {
                            "phase": e5.phase_tag,
                            "study": COMBINED_STUDY,
                            "source_study": sup_study,
                            "detector_study": det_study,
                            "model": f"{sm}+{dm}",
                            "supervised": sm,
                            "detector": dm,
                            "seed": str(ss),
                            "detector_seed": str(ds_),
                            "dataset_key": key,
                            "supervised_run_id": sr.info.run_id,
                            "detector_run_id": dr.info.run_id,
                            "combined_sha": sha,
                        }
                    )
                    mlflow.log_metrics(metrics)
                    arrays = run_sums.save_npz_dict(results[i])
                    for cond, ids in run_ids.items():
                        arrays[f"runs:{cond}"] = np.array(ids, dtype=str)
                    path = tmp / f"pair_{i}" / "runstats.npz"
                    path.parent.mkdir()
                    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
                    mlflow.log_artifact(str(path))
                    n_logged += 1
                    _write(
                        progress,
                        event="combined",
                        supervised_study=sup_study,
                        supervised=sm,
                        seed=ss,
                        detector=dm,
                        detector_seed=ds_,
                        run_id=run.info.run_id,
                    )
    print(f"combined: {n_logged} runs logged in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
