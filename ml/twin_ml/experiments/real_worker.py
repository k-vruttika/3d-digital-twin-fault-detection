"""One Phase 10 study run, i.e. one (study, model, seed), in its own process (ADR 0020, 0027).

Studies (:func:`study_specs`, all derived from ``config/models/real.yaml``):

* ``metropt3``: detectors (Isolation Forest, autoencoder) and the threshold / z-score rules on
  MetroPT-3 (no supervised model: no failure lies in the training period);
* ``azure``: supervised models, detectors and the persistence / threshold / z-score rules on
  Azure PdM, plus the exploratory robustness conditions for the methods listed in
  ``exploratory.robustness`` (every supervised model is run, so the validation-chosen best one
  is always covered);
* ``azure_err`` (exploratory): XGBoost with per-hour error counts as inputs, and the error-log
  rule (any code, and each code);
* ``azure_holdout`` (exploratory): every supervised model and the autoencoder on the 20 %
  machine-holdout build (the best model is chosen within the variant, on validation).

Supervised models are trained here (one process per (model, seed); torch and XGBoost never
share a process). Detectors are fitted on normal training windows, thresholds per kind from
validation normal windows. Every method is reduced to per-window sustained alarms, evaluated
on the test split's failure events with ``real_eval``.

One MLflow run per (study, model, seed) in ``P10_real``: tags, point metrics, ``metrics.json``
(per-event records, calibration and selection details), ``runstats.npz`` (per-machine sums for
the machine bootstrap), ``alarms.npz`` (clean sustained alarms, for the combined policy),
``scores.npz`` (detectors: clean test scores and thresholds), ``test_predictions.npz`` and the
saved model (XGBoost, sequence models; Random Forests are retrained from their seed).

Parent side: :func:`run_real_isolated`. Worker:
``python -m twin_ml.experiments.real_worker <request.json> <result.json>``.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import EvalConfig
from twin_ml.datasets.config import RealConfig

WORKER_MODULE = "twin_ml.experiments.real_worker"
EVAL_VERSION = "p10-b1"  # bump when the evaluation code changes results (resume key)
SUPERVISED = ("rf", "xgb", "lstm", "gru", "transformer")
DETECTORS = ("iforest", "autoencoder")
RULES = ("persistence", "threshold", "zscore", "error_rule")
CLEAN = "clean"
COMBINED_SUFFIX = "_combined"


class RealProcessError(RuntimeError):
    def __init__(self, study: str, model: str, seed: int, returncode: int) -> None:
        from twin_ml.train.isolated import describe_exit

        super().__init__(
            f"P10 study {study} {model} seed {seed} failed in its worker process "
            f"({describe_exit(returncode)}); its output is printed above"
        )


# ---------------------------------------------------------------------------- studies


@dataclass(frozen=True)
class StudySpec:
    name: str
    dataset: str  # key of real.datasets (evaluation settings)
    dataset_config: Path
    models: tuple[str, ...]
    robustness: bool = False


def study_specs(real: RealConfig) -> dict[str, StudySpec]:
    mp, az = real.datasets["metropt3"], real.datasets["azure_pdm"]
    ex = real.exploratory
    return {
        "metropt3": StudySpec(
            "metropt3", "metropt3", mp.dataset_config, (*mp.detectors, *mp.baselines)
        ),
        "azure": StudySpec(
            "azure",
            "azure_pdm",
            az.dataset_config,
            (*az.supervised, *az.detectors, *az.baselines),
            robustness=True,
        ),
        "azure_err": StudySpec(
            "azure_err",
            "azure_pdm",
            ex.error_xgb.dataset_config,
            (ex.error_xgb.model, "error_rule"),
        ),
        "azure_holdout": StudySpec(
            "azure_holdout",
            "azure_pdm",
            ex.holdout.dataset_config,
            (*az.supervised, "autoencoder"),
        ),
    }


def kind_of(model: str) -> str:
    if model in SUPERVISED:
        return "supervised"
    if model in DETECTORS:
        return "detector"
    if model in RULES:
        return "rule"
    raise ValueError(f"unknown Phase 10 method {model!r}")


def study_seeds(model: str, seeds: list[int]) -> list[int]:
    return seeds[:1] if kind_of(model) == "rule" else list(seeds)


def model_config(real: RealConfig, root: Path, model: str) -> Any:
    """Model / detector config, or the rule settings (a JSON-able dict) for a rule."""
    from twin_ml.config import load_anomaly_config, load_model_config

    k = kind_of(model)
    if k == "supervised":
        return load_model_config(root / real.models_dir / f"{model}.yaml")
    if k == "detector":
        return load_anomaly_config(root / real.models_dir / f"{model}.yaml")
    return None


def _dump(cfg: Any) -> Any:
    return None if cfg is None else cfg.model_dump(mode="json")


def robustness_models(real: RealConfig) -> set[str]:
    want = set(real.exploratory.robustness.models)
    out = {m for m in want if m in (*SUPERVISED, *DETECTORS, *RULES)}
    if "best_supervised" in want:
        out |= set(SUPERVISED)
    return out


def conditions(spec: StudySpec, model: str, real: RealConfig) -> list[str]:
    if spec.robustness and model in robustness_models(real):
        return [CLEAN, *real.exploratory.robustness.conditions]
    return [CLEAN]


def study_config_sha(
    spec: StudySpec,
    model: str,
    cfg: Any,
    real: RealConfig,
    rcfg: Any,
    eval_cfg: EvalConfig,
    dataset_key: str,
) -> str:
    """Hash of everything that determines a Phase 10 study run's results (resume key)."""
    data = {
        "version": EVAL_VERSION,
        "study": spec.name,
        "dataset": spec.dataset,
        "model": model,
        "model_config": _dump(cfg),
        "real": real.model_dump(
            mode="json",
            include={"seeds", "bootstrap", "anomaly", "baselines", "selection", "exploratory"},
        ),
        "dataset_study": real.datasets[spec.dataset].model_dump(mode="json"),  # type: ignore[index]
        "conditions": conditions(spec, model, real),
        "robustness": rcfg.model_dump(
            mode="json",
            include={
                "ece_bins",
                "eval_subset",
                "calibration",
                "conformal",
                "perturbations",
                "perturbation_seed",
            },
        ),
        "eval": eval_cfg.model_dump(mode="json", include={"subsets"}),
        "dataset_key": dataset_key,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def latest_valid_runs(
    tracking_dir: Path,
    experiment: str,
    phase: str,
    study: str,
    shas: dict[tuple[str, int], str],
) -> dict[tuple[str, int], Any]:
    """Newest finished run per (model, seed) of a study whose config hash is current."""
    from twin_ml.experiments.e5_worker import latest_valid_runs as latest

    return latest(tracking_dir, experiment, phase, study, shas)


def process_info() -> dict[str, Any]:
    from twin_ml.experiments.e5_worker import process_info as info

    return info()


def _progress(path: Path | None, **event: Any) -> None:
    from twin_ml.experiments.robust_worker import _progress as write

    write(path, **event)


# ---------------------------------------------------------------------------- methods


@dataclass
class Method:
    """A fitted method: ``alarms(split)`` -> raw per-window alarms per group."""

    alarms: Callable[[Any], dict[str, np.ndarray]]
    info: dict[str, Any]
    fit_s: float
    predictor: Any = None
    detector: Any = None
    thresholds: Any = None
    val_preds: Any = None


def _classification(split: Any, probs: Any, mask: Any, h: int) -> dict[str, Any]:
    from twin_ml.evaluate.evaluator import classification_block

    b = classification_block(split.target(h), probs, mask)
    return {
        "pr_auc": b["pr_auc"],
        "pr_auc_mean": b["pr_auc_macro"],
        "macro_f1": b["macro_f1"],
        "fault_recall": b["fault_recall"],
        "n": int(mask.sum()),
    }


def fit_method(
    model: str,
    cfg: Any,
    seed: int,
    ds: Any,
    real: RealConfig,
    spec: StudySpec,
    eval_cfg: EvalConfig,
    root: Path,
) -> Method:
    from twin_common.enums import MachineState
    from twin_ml.baselines import rules
    from twin_ml.evaluate.evaluator import subset_masks
    from twin_ml.models.anomaly import AnomalyThresholds, make_detector, normal_windows

    study = real.datasets[spec.dataset]  # type: ignore[index]
    train, val = ds.split("train"), ds.split("val")
    no_fault = list(ds.fault_types).index("none")
    horizon = study.horizon if study.horizon is not None else ds.horizons[0]
    k = kind_of(model)
    t0 = time.perf_counter()
    np.random.seed(seed)
    if k == "supervised":
        from twin_ml.train.trainer import make_predictor

        predictor = make_predictor(cfg, ds, root, eval_cfg)
        info = predictor.fit(train, val, seed)
        fit_s = time.perf_counter() - t0
        vp = predictor.predict(val)
        op = subset_masks(val, eval_cfg)[real.selection.subset]
        info = {
            "train": {k_: v for k_, v in info.items() if k_ != "history"},
            "history": info.get("history"),
            "val": {f"h{h}": _classification(val, vp.probs[h], op, h) for h in ds.horizons},
        }

        def sup_alarms(split: Any) -> dict[str, np.ndarray]:
            p = predictor.predict(split)
            return {f"h{h}": p.label(h) >= 1 for h in split.horizons}

        return Method(sup_alarms, info, fit_s, predictor=predictor, val_preds=vp)
    if k == "detector":
        a = real.anomaly
        states = [MachineState.RUNNING]
        det = make_detector(cfg, no_fault, horizon, states)
        fit_info = det.fit(train, val, seed)
        vidx = normal_windows(val, no_fault, horizon, states)
        thr = AnomalyThresholds(list(a.threshold_percentiles), a.per_kind, ds.schema.kinds).fit(
            det.score(val, vidx), val.at_end("kind", vidx)
        )
        info = {
            "fit": fit_info,
            "thresholds": thr.describe(),
            "val_normal_windows": int(vidx.shape[0]),
            "horizon_rows": horizon,
        }
        cache: dict[str, np.ndarray] = {}

        def det_alarms(split: Any) -> dict[str, np.ndarray]:
            scores = det.score(split)
            cache["scores"] = scores
            kinds = split.at_end("kind")
            return {f"p{p:g}": thr.alarm(scores, kinds, p) for p in thr.percentiles}

        m = Method(det_alarms, info, time.perf_counter() - t0, detector=det, thresholds=thr)
        m.info["_cache"] = cache
        return m
    b = real.baselines
    if model == "threshold":
        idx = normal_windows(train, no_fault, horizon, [MachineState.RUNNING])
        rule = rules.ThresholdRule(b.threshold.low_pct, b.threshold.high_pct).fit(train, idx)
        return Method(
            lambda s: {"rule": rule.alarm(s)},
            {"rule": rule.describe(), "fit_windows": int(idx.shape[0])},
            time.perf_counter() - t0,
        )
    if model == "zscore":
        trailing = study.zscore_trailing
        return Method(
            lambda s: {"rule": rules.zscore_alarm(s, trailing, b.zscore.z, b.zscore.min_std)},
            {
                "trailing_rows": trailing,
                "z": b.zscore.z,
                "min_std": b.zscore.min_std,
                "min_trailing_rows": ds.window,
            },
            0.0,
        )
    if model == "persistence":
        return Method(
            lambda s: {"rule": rules.persistence_alarm(s)}, {"observable_only": True}, 0.0
        )
    if model == "error_rule":
        codes = list(real.exploratory.error_rule.codes)
        error_codes = [c for c in codes if c != rules.ANY_CODE]
        return Method(
            lambda s: {c: rules.error_alarm(s, c, error_codes) for c in codes},
            {"codes": codes},
            0.0,
        )
    raise ValueError(f"no fit for {model}")


# ---------------------------------------------------------------------------- exploratory


def calibration_detail(
    method: Method,
    model: str,
    cfg: Any,
    ds: Any,
    test: Any,
    preds: Any,
    rcfg: Any,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    """Exploratory: calibration fitted on validation (Phase 8 code), ECE and Mondrian conformal
    on test operating windows, per horizon, with counts."""
    from twin_ml.calibrate.ece import (
        bin_sums,
        class_sums,
        ece_from_sums,
        reliability,
        top_label_sums,
    )
    from twin_ml.conformal.classification import SUM_FIELDS, set_sums
    from twin_ml.data.labels import FAULT, HEALTH
    from twin_ml.evaluate.evaluator import subset_masks
    from twin_ml.experiments.robust_eval import _logits, fit_calibration, uniforms
    from twin_ml.train.trainer import family

    val = ds.split("val")
    cal = fit_calibration(family(cfg), val, method.val_preds, rcfg, eval_cfg, ds.ttf_cap_s)
    sub = subset_masks(test, eval_cfg)[rcfg.eval_subset]
    bins = rcfg.ece_bins
    out: dict[str, Any] = {"calibration": cal.describe(), "heads": {}}
    for h in test.horizons:
        y = test.target(h).astype(np.int64)[sub]
        p_unc = preds.probs[h].astype(np.float64)
        p_cal = cal.heads[h].apply(preds.probs[h], _logits(preds, h))
        u = uniforms(rcfg.conformal.aps_seed, "test", h, test.n_windows)
        sets = cal.conformal[h].sets(p_cal, u)
        pu, pc = p_unc[sub], p_cal[sub]
        head: dict[str, Any] = {
            "n": int(sub.sum()),
            "class_counts": np.bincount(y, minlength=3).tolist(),
        }
        for tag, p in (("uncal", pu), ("cal", pc)):
            top = top_label_sums(p, y, bins)
            fault = class_sums(p, y, FAULT, bins)
            pos = bin_sums(p[:, 1:].sum(axis=1), y >= 1, bins)
            head[f"ece_top_{tag}"] = ece_from_sums(top)
            head[f"ece_fault_{tag}"] = ece_from_sums(fault)
            head[f"ece_pos_{tag}"] = ece_from_sums(pos)
            head[f"reliability_pos_{tag}"] = reliability(pos)
        conf: dict[str, Any] = {}
        for m, st in sets.items():
            ss = set_sums(st[sub], y, len(HEALTH))
            ni, ci, si = (SUM_FIELDS.index(f) for f in ("n", "covered", "size"))
            conf[m] = {
                "coverage": float(ss[:, ci].sum() / max(ss[:, ni].sum(), 1)),
                "size": float(ss[:, si].sum() / max(ss[:, ni].sum(), 1)),
                "per_class": {
                    HEALTH[c].value: {
                        "n": int(ss[c, ni]),
                        "coverage": float(ss[c, ci] / ss[c, ni]) if ss[c, ni] else None,
                        "size": float(ss[c, si] / ss[c, ni]) if ss[c, ni] else None,
                    }
                    for c in range(len(HEALTH))
                },
            }
        head["conformal"] = conf
        head["alpha"] = rcfg.conformal.alpha
        out["heads"][str(h)] = head
    return out


# ---------------------------------------------------------------------------- study run


def run_real(
    study_name: str,
    model: str,
    cfg: Any,
    seed: int,
    real: RealConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    import mlflow

    from twin_common.logging import get_logger
    from twin_ml.config import load_robustness_config
    from twin_ml.data.dataset import Dataset
    from twin_ml.data.perturb import parse_condition
    from twin_ml.datasets.build import resolve_real_dataset
    from twin_ml.evaluate import run_sums
    from twin_ml.evaluate.evaluator import subset_masks
    from twin_ml.experiments import e5_eval, real_eval
    from twin_ml.features.normalize import Normalizer
    from twin_ml.train import tracking

    log = get_logger(__name__)
    started = time.perf_counter()
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    rcfg = load_robustness_config(root / real.robustness_config)
    spec = study_specs(real)[study_name]
    study = real.datasets[spec.dataset]  # type: ignore[index]
    ds = Dataset.open(resolve_real_dataset(spec.dataset_config, root))
    key = ds.meta["dataset_key"]
    fault_types = list(ds.fault_types)
    test = ds.split("test")
    index = real_eval.SplitIndex.of(test, fault_types)
    events = real_eval.split_events(ds.meta, "test")
    split_lo, _ = real_eval.split_bounds(ds.meta, "test")
    lookbacks = list(study.lookbacks)
    conds = conditions(spec, model, real)

    method = fit_method(model, cfg, seed, ds, real, spec, eval_cfg, root)
    _progress(
        progress,
        event="model",
        study=study_name,
        model=model,
        seed=seed,
        seconds=round(method.fit_s, 1),
    )
    cache = method.info.pop("_cache", {})
    groups: dict[str, run_sums.Sums] = {}
    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}
    alarm_arrays: dict[str, np.ndarray] = {
        "n:clean": np.array([test.n_windows]),
        "digest:clean": np.array([e5_eval.ends_digest(test)]),
    }
    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}
    sigma = norm = None
    with tempfile.TemporaryDirectory(prefix="twin_p10_") as tmp_s:
        tmp = Path(tmp_s)
        for cond in conds:
            t0 = time.perf_counter()
            if cond == CLEAN:
                split = test
            else:
                if sigma is None:
                    sigma = real_eval.normal_sigma(
                        ds.split("train"), fault_types, list(rcfg.perturbations.normal_states)
                    )
                    norm = Normalizer.from_dict(ds.meta["normalizer"])
                split = real_eval.perturbed_split(
                    test,
                    norm,
                    parse_condition(cond, rcfg.perturbations),
                    rcfg.perturbations,
                    sigma,
                    rcfg.perturbation_seed,
                )
            raw_alarms = method.alarms(split)
            cdetail: dict[str, Any] = {"groups": {}}
            for g, a in raw_alarms.items():
                sus = real_eval.sustain(split, a, study.sustain)
                ev = real_eval.evaluate_alarm(
                    split, index, sus, events, lookbacks, split_lo, study.evaluable_min_coverage
                )
                for lk, s in ev.sums.items():
                    groups[f"{cond}|{g}|{lk}"] = s
                    for m, v in ev.points[lk].items():
                        if np.isfinite(v):
                            metrics[f"{cond}.{g}.{lk}.{m}"] = float(v)
                gd: dict[str, Any] = {
                    "points": ev.points,
                    "alarm_frac_raw": float(a.mean()) if a.size else float("nan"),
                }
                if cond == CLEAN:
                    gd["events"] = ev.events
                    alarm_arrays[f"clean|{g}"] = e5_eval.pack_alarm(sus)
                    if spec.dataset in real.exploratory.fa_by_month:
                        gd["fa_by_month"] = real_eval.fa_by_month(ds.meta, split, index, sus)
                cdetail["groups"][g] = gd
            if cond == CLEAN and kind_of(model) == "detector" and "scores" in cache:
                scores = cache["scores"]
                op = subset_masks(test, eval_cfg)["operating"]
                y = test.target(test.horizons[-1] if study.horizon is None else study.horizon)
                from twin_ml.evaluate.metrics import average_precision

                pos = op & (y >= 1)
                cdetail["score_quantiles_negative"] = (
                    np.quantile(scores[index.negative], [0.5, 0.9, 0.99, 0.999]).tolist()
                    if index.negative.any()
                    else None
                )
                cdetail["auprc_precursor"] = (
                    average_precision(pos[op], scores[op]) if pos.any() else None
                )
                np.savez_compressed(
                    tmp / "scores.npz",
                    scores=scores.astype(np.float32),
                    t_s=test.at_end("t_s").astype(np.float64),
                    machine=index.window_machine.astype(np.int32),
                    kind=test.at_end("kind").astype(np.int8),
                )
            if cond == CLEAN and kind_of(model) == "supervised":
                preds = method.predictor.predict(test)
                np.savez_compressed(tmp / "test_predictions.npz", **preds.to_arrays())
                op = subset_masks(test, eval_cfg)["operating"]
                cdetail["classification"] = {
                    f"h{h}": _classification(test, preds.probs[h], op, h) for h in test.horizons
                }
                extra["exploratory_calibration"] = calibration_detail(
                    method, model, cfg, ds, test, preds, rcfg, eval_cfg
                )
            details[cond] = cdetail
            timings[cond] = time.perf_counter() - t0
            log.info(
                "real.condition",
                study=study_name,
                model=model,
                seed=seed,
                condition=cond,
                seconds=round(timings[cond], 1),
            )
            _progress(
                progress,
                event="condition",
                study=study_name,
                model=model,
                seed=seed,
                condition=cond,
                seconds=round(timings[cond], 1),
            )

        if kind_of(model) == "supervised":
            for h, v in method.info["val"].items():
                for m in ("pr_auc_mean", "macro_f1"):
                    if np.isfinite(v[m]):
                        metrics[f"val.{h}.{m}"] = float(v[m])
        total_s = time.perf_counter() - started
        metrics["time.fit_s"] = method.fit_s
        metrics["time.total_s"] = total_s
        sha = study_config_sha(spec, model, cfg, real, rcfg, eval_cfg, key)
        b = real.bootstrap
        for g, s in groups.items():
            if g.startswith(f"{CLEAN}|"):
                fn = real_eval.metric_fn(real_eval.group_lookback_s(g, test.step_s))
                details[CLEAN].setdefault("bootstrap", {})[g] = run_sums.bootstrap(
                    s, b.n_resamples, b.seed, b.ci, metric_fn=fn
                )
        result = {
            "study": study_name,
            "dataset": spec.dataset,
            "model": model,
            "kind": kind_of(model),
            "seed": seed,
            "dataset_key": key,
            "counts": ds.meta["counts"],
            "machines": index.machines,
            "fault_types": fault_types,
            "lookbacks": lookbacks,
            "sustain_rows": study.sustain,
            "step_s": test.step_s,
            "method": method.info,
            **extra,
            "conditions": details,
            "timings_s": timings,
            "total_s": total_s,
            "process": process_info(),
        }
        tracking.configure(tdir, experiment or real.experiment)
        with mlflow.start_run(run_name=f"{study_name}-{model}-seed{seed}") as run:
            mlflow.set_tags(
                {
                    "phase": real.phase_tag,
                    "study": study_name,
                    "dataset": spec.dataset,
                    "model": model,
                    "family": kind_of(model),
                    "seed": str(seed),
                    "dataset_key": key,
                    "study_config_sha": sha,
                }
            )
            mlflow.log_params({"seed": str(seed), "conditions": ",".join(conds)})
            mlflow.log_metrics(metrics)
            mlflow.log_dict(json.loads(json.dumps(result, default=_jsonable)), "metrics.json")
            arrays = run_sums.save_npz_dict(groups)
            arrays["machines"] = np.array(index.machines, dtype=str)
            np.savez_compressed(tmp / "runstats.npz", **arrays)  # type: ignore[arg-type]
            np.savez_compressed(tmp / "alarms.npz", **alarm_arrays)  # type: ignore[arg-type]
            for name in ("runstats.npz", "alarms.npz", "scores.npz", "test_predictions.npz"):
                if (tmp / name).is_file():
                    mlflow.log_artifact(str(tmp / name))
            saver = getattr(method.predictor, "save", None)
            if saver is not None:
                for p in saver(tmp / "model"):
                    mlflow.log_artifact(str(p), artifact_path="model")
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
    return {"run_id": run_id, "study": study_name, "model": model, "seed": seed, "total_s": total_s}


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------------------- process


def run_real_isolated(
    study: str,
    model: str,
    cfg: Any,
    seed: int,
    real: RealConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
    experiment: str | None = None,
    task: str = "study",
) -> dict[str, Any]:
    request = {
        "task": task,
        "study": study,
        "model": model,
        "config": _dump(cfg),
        "seed": seed,
        "real": real.model_dump(mode="json"),
        "eval_config": eval_cfg.model_dump(mode="json"),
        "root": str(root),
        "tracking_dir": None if tracking_dir is None else str(tracking_dir),
        "progress": None if progress is None else str(progress),
        "experiment": experiment,
    }
    with tempfile.TemporaryDirectory(prefix="twin_p10_req_") as tmp:
        req, res = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        cmd = [sys.executable, "-X", "faulthandler", "-m", WORKER_MODULE, str(req), str(res)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not res.is_file():
            raise RealProcessError(study, model, seed, proc.returncode)
        out: dict[str, Any] = json.loads(res.read_text(encoding="utf-8"))
    return out


def parse_config(model: str, data: Any) -> Any:
    from twin_ml.config import parse_anomaly_config, parse_model_config

    if data is None:
        return None
    return (parse_anomaly_config if kind_of(model) == "detector" else parse_model_config)(data)


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(f"usage: python -m {WORKER_MODULE} <request.json> <result.json>", file=sys.stderr)
        return 2
    from twin_common.logging import configure_logging

    configure_logging()
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    model = req["model"]
    cfg = parse_config(model, req["config"])
    real = RealConfig.model_validate(req["real"])
    eval_cfg = EvalConfig.model_validate(req["eval_config"])
    kwargs = {
        "root": Path(req["root"]),
        "tracking_dir": None if req["tracking_dir"] is None else Path(req["tracking_dir"]),
        "progress": None if req["progress"] is None else Path(req["progress"]),
        "experiment": req.get("experiment"),
    }
    if req.get("task", "study") == "explain":
        from twin_ml.experiments.real_explain import run_real_explain

        out = run_real_explain(req["study"], model, cfg, int(req["seed"]), real, eval_cfg, **kwargs)
    else:
        out = run_real(req["study"], model, cfg, int(req["seed"]), real, eval_cfg, **kwargs)
    Path(args[1]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
