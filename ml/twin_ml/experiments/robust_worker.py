"""One Phase 8 study run, i.e. one (study, model, seed), in its own process (ADR 0020, 0021).

Steps:

1. **Model.** Either load it from a finished training run with the same model config hash
   (the E1 run for the ``e1models`` study; the study's own training run when resuming a
   training study), or train it (``train.trainer.train_and_log``, which logs the training
   run). XGBoost and sequence models are loaded from their artifacts. Random Forests are
   never saved, so they are retrained from the seed; for a reused run their clean test
   predictions must reproduce the stored ones exactly.
2. **Calibration and conformal predictors,** fitted on the clean validation split.
3. **Every test condition of the study:** clean, the perturbation grid, simulator
   sensor-fault sets, or the E4 high-load and in-distribution splits.
4. **One MLflow run** in the robustness experiment. Tags: study, model, seed, dataset key,
   source run and the study config hash. Point metrics per condition; ``metrics.json`` holds
   every detail and CI; ``runstats.npz`` holds the per-run sums for the report's bootstraps.

Parent side: :func:`run_study_isolated`. Worker: ``python -m twin_ml.experiments.robust_worker
<request.json> <result.json>``.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import EvalConfig, ModelConfig, RobustnessConfig, StudyConfig

WORKER_MODULE = "twin_ml.experiments.robust_worker"
SPLIT_CONDITIONS = {"clean": "test", "e4_high": "test", "e4_in": "test_in"}
LOADABLE = ("persistence", "threshold", "xgb", "lstm", "gru", "transformer")
PRED_ATOL = 1e-5  # reloaded model vs stored test predictions


class StudyProcessError(RuntimeError):
    def __init__(self, study: str, model: str, seed: int, returncode: int) -> None:
        from twin_ml.train.isolated import describe_exit

        super().__init__(
            f"study {study} {model} seed {seed} failed in its worker process "
            f"({describe_exit(returncode)}); its output is printed above"
        )


class PredictionMismatchError(RuntimeError):
    """A reused model does not reproduce its stored clean test predictions."""


# ---------------------------------------------------------------------------- helpers


def expand_conditions(study: StudyConfig, rcfg: RobustnessConfig) -> list[str]:
    from twin_ml.data.perturb import (
        PERTURBATIONS,
        SIGNAL_NOISE,
        grid_conditions,
        signal_noise_conditions,
    )

    out: list[str] = []
    for c in study.conditions:
        if c == PERTURBATIONS:
            out += [g.name for g in grid_conditions(rcfg.perturbations)]
        elif c == SIGNAL_NOISE:
            if rcfg.signal_noise is None:
                raise ValueError("a study lists signal_noise but robustness.yaml has none")
            out += [g.name for g in signal_noise_conditions(rcfg.signal_noise.levels_pct)]
        else:
            out.append(c)
    return out


def study_config_sha(
    study: str,
    model_cfg: ModelConfig,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    dataset_key: str,
    source: str,
) -> str:
    """Hash of everything that determines a study run's results (resume key).

    Exploratory extension studies (``extension: true``) and the signal-noise settings are
    left out of the shared part, so adding them later keeps every earlier run's hash; an
    extension study's own hash adds its study config and the signal-noise settings.
    """
    ext = {k for k, v in rcfg.studies.items() if v.extension}
    shared = rcfg.model_dump(
        mode="json",
        # display and bookkeeping settings do not change any result
        exclude={
            "eta_train_s",
            "eta_condition_s",
            "progress_dir",
            "reliability_min_count",
            "signal_noise",
        },
    )
    shared["studies"] = {k: v for k, v in shared["studies"].items() if k not in ext}
    for v in shared["studies"].values():
        v.pop("extension", None)
    data: dict[str, Any] = {
        "study": study,
        "model": model_cfg.model_dump(mode="json"),
        "robustness": shared,
        "eval": eval_cfg.model_dump(mode="json", include={"subsets", "early_warning", "bootstrap"}),
        "dataset_key": dataset_key,
        "source": source,
    }
    if study in ext:
        data["extension"] = {
            "study": rcfg.studies[study].model_dump(mode="json"),
            "signal_noise": None
            if rcfg.signal_noise is None
            else rcfg.signal_noise.model_dump(mode="json"),
        }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def find_run(
    tracking_dir: Path,
    experiment: str,
    phase: str,
    dataset_key: str,
    model: str,
    seed: int,
    cfg: ModelConfig,
) -> Any | None:
    """Newest finished training run of (model, seed) trained with exactly this model config.

    A run tagged ``model_config_sha`` matches on the hash. Runs logged before the tag existed
    (some E1 runs) match when every logged ``model.*`` param equals the current config.
    """
    from twin_ml.train import tracking
    from twin_ml.train.trainer import config_sha

    sha = config_sha(cfg)
    params = tracking.flatten(cfg.model_dump(mode="json"), "model.")
    flt = f"tags.model = '{model}' and tags.seed = '{seed}'"
    runs = tracking.finished_runs(tracking_dir, experiment, dataset_key, phase, flt)
    for run in runs:
        if run.data.tags.get("model_config_sha") == sha:
            return run
    for run in runs:
        if "model_config_sha" in run.data.tags:
            continue
        logged = {k: v for k, v in run.data.params.items() if k.startswith("model.")}
        if logged == params:
            return run
    return None


def source_run_id(
    study: StudyConfig,
    rcfg: RobustnessConfig,
    tracking_dir: Path,
    dataset_key: str,
    model: str,
    seed: int,
    cfg: ModelConfig,
) -> str:
    """The E1 run reused by an ``e1`` study ("train" for training studies)."""
    if study.models_from != "e1":
        return "train"
    from twin_ml.train.trainer import BASELINES

    run = find_run(
        tracking_dir,
        rcfg.source_experiment,
        "E1",
        dataset_key,
        model,
        0 if model in BASELINES else seed,
        cfg,
    )
    if run is None:
        raise LookupError(
            f"no finished {rcfg.source_experiment} run of {model} seed {seed} with the current "
            f"config for dataset {dataset_key}; run uv run poe train-all first"
        )
    return str(run.info.run_id)


def clean_sigma(root: Path, rcfg: RobustnessConfig) -> np.ndarray:
    """Normal-operation std per kind and sensor from the *clean* E1 training split."""
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.perturb import normal_std

    ds_cfg = load_dataset_config(root / rcfg.studies["e1models"].dataset_config)
    path = resolve_dataset_dir(ds_cfg, root) / "train.npz"
    meta = Dataset.open(path.parent).meta
    with np.load(path, allow_pickle=False) as npz:
        raw, state, kind, ft = npz["raw"], npz["state"], npz["kind"], npz["fault_type"]
    no_fault = list(meta["fault_types"]).index("none")
    return normal_std(raw, state, kind, ft, no_fault, rcfg.perturbations.normal_states)


def clean_signal_scale(root: Path, rcfg: RobustnessConfig, sigma: np.ndarray) -> np.ndarray:
    """Typical rise before FAULT per kind and sensor, from the clean E1 training episodes."""
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.perturb import episode_rises, signal_scale

    ds = Dataset.open(
        resolve_dataset_dir(
            load_dataset_config(root / rcfg.studies["e1models"].dataset_config), root
        )
    )
    a = ds.split("train").arrays
    rises = episode_rises(
        a["raw"],
        a["kind"],
        a["t_s"],
        a["ep_start"],
        a["ep_fault"],
        a["ep_fault_type"],
        int(ds.meta["config"]["window_s"]),
    )
    return signal_scale(rises, sigma)


@dataclass
class _Model:
    predictor: Any
    source: str  # run id of a reused run, or "trained"
    train_s: float
    stored_preds: dict[str, np.ndarray] | None  # stored clean test predictions to check


def _download(artifact_uri: str, rel: str, dst: Path) -> Path:
    import mlflow

    return Path(
        mlflow.artifacts.download_artifacts(artifact_uri=f"{artifact_uri}/{rel}", dst_path=str(dst))
    )


def obtain_model(
    study_name: str,
    study: StudyConfig,
    model: str,
    cfg: ModelConfig,
    seed: int,
    ds: Any,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    root: Path,
    tracking_dir: Path,
    tmp: Path,
) -> _Model:
    from twin_ml.train.trainer import BASELINES, make_predictor, train_and_log

    key = ds.meta["dataset_key"]
    if study.models_from == "e1":
        run = find_run(
            tracking_dir,
            rcfg.source_experiment,
            "E1",
            key,
            model,
            0 if model in BASELINES else seed,
            cfg,
        )
        if run is None:
            raise LookupError(f"no E1 run of {model} seed {seed} with the current config")
    else:
        run = find_run(tracking_dir, study.experiment, rcfg.phase_tag, key, model, seed, cfg)
        if run is not None and model not in LOADABLE:
            run = None  # Random Forest: retrain (and re-log) rather than trust a reproduction
    if run is None:
        study_eval = eval_cfg.model_copy(update={"experiment": study.experiment})
        res, predictor = train_and_log(
            model,
            cfg,
            seed,
            ds,
            study_eval,
            root,
            tracking_dir,
            phase=rcfg.phase_tag,
            extra_tags={"study": study_name},
        )
        return _Model(predictor, res.run_id, res.train_s, None)

    uri = run.info.artifact_uri
    npz_path = _download(uri, "test_predictions.npz", tmp)
    with np.load(npz_path) as npz:
        stored = {k: npz[k] for k in npz.files}
    predictor = make_predictor(cfg, ds, root, eval_cfg)
    t0 = time.perf_counter()
    if model in BASELINES:
        pass
    elif model == "xgb":
        from twin_ml.models.trees import TreeModel

        assert isinstance(predictor, TreeModel)
        predictor.load(_download(uri, "model", tmp), ds.split("train"))
    elif model in ("lstm", "gru", "transformer"):
        from twin_ml.models.nn import SequenceModel

        assert isinstance(predictor, SequenceModel)
        path = _download(uri, "model/model.pt", tmp)
        predictor.load(path, ds.horizons, int(ds.meta["config"]["window_s"]))
    else:  # rf: retrain from the seed, exactly as the trainer did
        np.random.seed(seed)
        predictor.fit(ds.split("train"), ds.split("val"), seed)
    return _Model(predictor, str(run.info.run_id), time.perf_counter() - t0, stored)


def check_predictions(
    preds: Any, stored: dict[str, np.ndarray], horizons: tuple[int, ...]
) -> dict[str, Any]:
    """Reloaded model vs stored clean test predictions: identical labels, close probs."""
    out: dict[str, Any] = {}
    for h in horizons:
        a, b = preds.probs[h], stored[f"probs_h{h}"]
        if a.shape != b.shape:
            raise PredictionMismatchError(f"h{h}: shape {a.shape} vs stored {b.shape}")
        diff = float(np.abs(a - b).max()) if a.size else 0.0
        same = bool(np.array_equal(np.argmax(a, axis=1), np.argmax(b, axis=1)))
        out[f"h{h}"] = {"max_abs_diff": diff, "labels_identical": same}
        if not same or diff > PRED_ATOL:
            raise PredictionMismatchError(
                f"h{h}: reloaded model does not reproduce the stored predictions "
                f"(max abs diff {diff:.3g}, labels identical: {same})"
            )
    return out


def _progress(path: Path | None, **event: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------- study run


def run_study(
    study_name: str,
    model: str,
    cfg: ModelConfig,
    seed: int,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
) -> dict[str, Any]:
    import mlflow

    from twin_common.logging import get_logger
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.perturb import parse_condition, signal_noise_conditions
    from twin_ml.data.test_sets import load_series, perturbed_split, sensor_fault_split
    from twin_ml.evaluate import run_sums
    from twin_ml.experiments.robust_eval import (
        evaluate_condition,
        fit_calibration,
        flat_condition_metrics,
    )
    from twin_ml.train import tracking
    from twin_ml.train.trainer import config_sha, family

    log = get_logger(__name__)
    started = time.perf_counter()
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    study = rcfg.studies[study_name]
    ds_cfg = load_dataset_config(root / study.dataset_config)
    ds = Dataset.open(resolve_dataset_dir(ds_cfg, root))
    key = ds.meta["dataset_key"]
    fam = family(cfg)
    conditions = expand_conditions(study, rcfg)
    sigma = clean_sigma(root, rcfg)
    scale = (
        clean_signal_scale(root, rcfg, sigma)
        if any(c.startswith("snoise_") for c in conditions)
        else None
    )
    timings: dict[str, float] = {}
    details: dict[str, Any] = {}
    groups: dict[str, run_sums.Sums] = {}
    run_ids: dict[str, list[str]] = {}
    metrics: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="twin_p8_") as tmp_s:
        tmp = Path(tmp_s)
        got = obtain_model(study_name, study, model, cfg, seed, ds, rcfg, eval_cfg, root, tdir, tmp)
        _progress(
            progress,
            event="model",
            study=study_name,
            model=model,
            seed=seed,
            source=got.source,
            seconds=round(got.train_s, 1),
        )
        predictor = got.predictor
        val = ds.split("val")
        cal = fit_calibration(fam, val, predictor.predict(val), rcfg, eval_cfg, ds.ttf_cap_s)
        clean_series = None
        check: dict[str, Any] | None = None
        for name in conditions:
            t0 = time.perf_counter()
            sf_rows = None
            if name in SPLIT_CONDITIONS:
                split = ds.split(SPLIT_CONDITIONS[name])
            else:
                if clean_series is None:
                    clean_series = load_series(
                        root / ds_cfg.source_dir,
                        ds.meta["splits"]["test"],
                        root / ds_cfg.line_config,
                    )
                if name in rcfg.sensor_fault_sets:
                    split, sf_rows = sensor_fault_split(
                        ds,
                        root / rcfg.sensor_fault_sets[name].source_dir,
                        clean_series,
                        root / ds_cfg.line_config,
                    )
                elif name.startswith("snoise_"):
                    assert scale is not None
                    assert rcfg.signal_noise is not None
                    cond = next(
                        c
                        for c in signal_noise_conditions(rcfg.signal_noise.levels_pct)
                        if c.name == name
                    )
                    split = perturbed_split(
                        ds, clean_series, cond, rcfg.perturbations, scale, rcfg.perturbation_seed
                    )
                else:
                    cond = parse_condition(name, rcfg.perturbations)
                    split = perturbed_split(
                        ds, clean_series, cond, rcfg.perturbations, sigma, rcfg.perturbation_seed
                    )
            preds = predictor.predict(split)
            if got.stored_preds is not None and SPLIT_CONDITIONS.get(name) == "test":
                check = check_predictions(preds, got.stored_preds, ds.horizons)
            res = evaluate_condition(
                name, split, preds, cal, rcfg, eval_cfg, ds.fault_types, ds.ttf_cap_s, sf_rows
            )
            details[name] = res.detail
            run_ids[name] = res.run_ids
            for g, s in res.sums.items():
                groups[f"{name}|{g}"] = s
            metrics |= flat_condition_metrics(name, res)
            timings[name] = time.perf_counter() - t0
            log.info(
                "robust.condition",
                study=study_name,
                model=model,
                seed=seed,
                condition=name,
                seconds=round(timings[name], 1),
            )
            _progress(
                progress,
                event="condition",
                study=study_name,
                model=model,
                seed=seed,
                condition=name,
                seconds=round(timings[name], 1),
            )

        total_s = time.perf_counter() - started
        metrics["time.model_s"] = got.train_s
        metrics["time.total_s"] = total_s
        for h, head in cal.heads.items():
            if head.temperature is not None:
                metrics[f"cal.h{h}.temperature"] = head.temperature
        sha = config_sha(cfg)
        source = got.source if study.models_from == "e1" else "train"
        result = {
            "study": study_name,
            "model": model,
            "family": fam,
            "seed": seed,
            "dataset": {
                "key": key,
                "counts": ds.meta["counts"],
                "splits": ds.meta["splits"],
                "run_filter": ds.meta.get("run_filter"),
            },
            "source_run": got.source,
            "prediction_check": check,
            "calibration": cal.describe(),
            "normal_std": sigma.tolist(),
            "signal_scale": None if scale is None else scale.tolist(),
            "conditions": details,
            "timings_s": timings,
            "total_s": total_s,
        }
        tracking.configure(tdir, rcfg.experiment)
        with mlflow.start_run(run_name=f"{study_name}-{model}-seed{seed}") as run:
            mlflow.set_tags(
                {
                    "phase": rcfg.phase_tag,
                    "study": study_name,
                    "model": model,
                    "family": fam,
                    "seed": str(seed),
                    "dataset_key": key,
                    "source_run_id": got.source,
                    "model_config_sha": sha,
                    "study_config_sha": study_config_sha(
                        study_name, cfg, rcfg, eval_cfg, key, source
                    ),
                }
            )
            mlflow.log_params(
                {
                    "seed": str(seed),
                    "conditions": ",".join(conditions),
                    "alpha": str(rcfg.conformal.alpha),
                    "ece_bins": str(rcfg.ece_bins),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(json.loads(json.dumps(result, default=float)), "metrics.json")
            arrays = run_sums.save_npz_dict(groups)
            for name, ids in run_ids.items():
                arrays[f"runs:{name}"] = np.array(ids, dtype=str)
            path = tmp / "runstats.npz"
            np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id
    _progress(
        progress,
        event="done",
        study=study_name,
        model=model,
        seed=seed,
        run_id=run_id,
        seconds=round(total_s, 1),
    )
    return {
        "run_id": run_id,
        "study": study_name,
        "model": model,
        "seed": seed,
        "total_s": total_s,
        "source": got.source,
    }


# ---------------------------------------------------------------------------- process


def run_study_isolated(
    study: str,
    model: str,
    cfg: ModelConfig,
    seed: int,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
) -> dict[str, Any]:
    request = {
        "study": study,
        "model": model,
        "config": cfg.model_dump(mode="json"),
        "seed": seed,
        "robustness": rcfg.model_dump(mode="json"),
        "eval_config": eval_cfg.model_dump(mode="json"),
        "root": str(root),
        "tracking_dir": None if tracking_dir is None else str(tracking_dir),
        "progress": None if progress is None else str(progress),
    }
    with tempfile.TemporaryDirectory(prefix="twin_p8_req_") as tmp:
        req, res = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        cmd = [sys.executable, "-X", "faulthandler", "-m", WORKER_MODULE, str(req), str(res)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not res.is_file():
            raise StudyProcessError(study, model, seed, proc.returncode)
        out: dict[str, Any] = json.loads(res.read_text(encoding="utf-8"))
    return out


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(f"usage: python -m {WORKER_MODULE} <request.json> <result.json>", file=sys.stderr)
        return 2
    from twin_common.logging import configure_logging
    from twin_ml.config import parse_model_config

    configure_logging()
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out = run_study(
        req["study"],
        req["model"],
        parse_model_config(req["config"]),
        int(req["seed"]),
        RobustnessConfig.model_validate(req["robustness"]),
        EvalConfig.model_validate(req["eval_config"]),
        Path(req["root"]),
        None if req["tracking_dir"] is None else Path(req["tracking_dir"]),
        None if req["progress"] is None else Path(req["progress"]),
    )
    Path(args[1]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
