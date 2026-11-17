"""One Phase 9 study run, i.e. one (study, model, seed), in its own process (ADR 0020, 0024).

* **Supervised models** (``e5_e1models``, ``e5_aug``, and the supervised part of ``e5_e4``)
  are reused, never retrained (the Random Forest is retrained from its seed, as in Phase 8).
  Each must reproduce its stored clean test predictions. Calibration and conformal sets are
  fitted on clean validation data with the Phase 8 code, and every condition gets the
  Phase 8 metrics plus the exploratory confidence-of-wrong-predictions and precursor set
  sums.
* **Anomaly detectors** (``e5_anomaly``, the detector part of ``e5_e4``) are trained on the
  study dataset's normal-operation training windows; thresholds are per-kind percentiles of
  the validation normal-operation scores.
* Every run stores its per-window *sustained* alarm (primary horizon, every percentile) as
  ``alarms.npz``, which the combined policy (:func:`combine`, run by the parent) reads.

One MLflow run per (study, model, seed) in the E5 experiment: tags, point metrics per
condition, ``metrics.json`` (details, CIs), ``runstats.npz`` (per-run sums for the report's
bootstraps) and ``alarms.npz``.

Parent side: :func:`run_e5_isolated`. Worker:
``python -m twin_ml.experiments.e5_worker <request.json> <result.json>``.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import E5Config, EvalConfig

WORKER_MODULE = "twin_ml.experiments.e5_worker"
ANOMALY_MODELS = ("iforest", "autoencoder")
E5_CONDITION = "e5"
DRIFT_CONDITION = "sf_drift"
SPLIT_CONDITIONS = {"clean": "test", "e4_high": "test", "e4_in": "test_in"}
COMBINED_STUDY = "e5_combined"


class E5ProcessError(RuntimeError):
    def __init__(self, study: str, model: str, seed: int, returncode: int) -> None:
        from twin_ml.train.isolated import describe_exit

        super().__init__(
            f"E5 study {study} {model} seed {seed} failed in its worker process "
            f"({describe_exit(returncode)}); its output is printed above"
        )


# ---------------------------------------------------------------------------- helpers


def is_anomaly(model: str) -> bool:
    return model in ANOMALY_MODELS


def model_config(e5: E5Config, root: Path, models_dir: Path, model: str) -> Any:
    from twin_ml.config import load_anomaly_config, load_model_config

    if is_anomaly(model):
        return load_anomaly_config(root / e5.anomaly.models[model])
    return load_model_config(root / models_dir / f"{model}.yaml")


def config_sha_any(cfg: Any) -> str:
    data = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def e5_content_sha(e5: E5Config, root: Path) -> str | None:
    path = root / e5.e5_data.source_dir / "manifest.json"
    if not path.is_file():
        return None
    from twin_ml.data.load import read_manifest

    sha = read_manifest(root / e5.e5_data.source_dir).get("content_sha256")
    return None if sha is None else str(sha)


def study_config_sha(
    study: str,
    model: str,
    cfg: Any,
    e5: E5Config,
    rcfg: Any,
    eval_cfg: EvalConfig,
    dataset_key: str,
    source: str,
    e5_content: str | None,
) -> str:
    """Hash of everything that determines a Phase 9 study run's results (resume key)."""
    data = {
        "study": study,
        "study_config": e5.studies[study].model_dump(mode="json"),
        "model": model,
        "model_config": cfg.model_dump(mode="json"),
        "e5": e5.model_dump(
            mode="json",
            include={"e5_data", "anomaly", "confidence", "primary"},
        ),
        "robustness": rcfg.model_dump(
            mode="json",
            include={"ece_bins", "eval_subset", "calibration", "conformal", "sensor_fault_sets"},
        ),
        "eval": eval_cfg.model_dump(mode="json", include={"subsets", "early_warning", "bootstrap"}),
        "dataset_key": dataset_key,
        "source": source,
        "e5_content": e5_content if E5_CONDITION in e5.studies[study].conditions else None,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def supervised_source(
    e5: E5Config,
    study: str,
    tracking_dir: Path,
    dataset_key: str,
    model: str,
    seed: int,
    cfg: Any,
) -> Any:
    """The finished training run whose model a supervised E5 study reuses (never trains)."""
    from twin_ml.experiments.robust_worker import find_run
    from twin_ml.train.trainer import BASELINES

    st = e5.studies[study]
    assert st.source_experiment is not None
    assert st.source_phase is not None
    run = find_run(
        tracking_dir,
        st.source_experiment,
        st.source_phase,
        dataset_key,
        model,
        0 if model in BASELINES else seed,
        cfg,
    )
    if run is None:
        raise LookupError(
            f"no finished {st.source_experiment} run of {model} seed {seed} with the current "
            f"config for dataset {dataset_key}; E5 reuses models and never trains them"
        )
    return run


def source_id(
    e5: E5Config, study: str, tracking_dir: Path, key: str, model: str, seed: int, cfg: Any
) -> str:
    if is_anomaly(model):
        return "train"
    return str(supervised_source(e5, study, tracking_dir, key, model, seed, cfg).info.run_id)


def _progress(path: Path | None, **event: Any) -> None:
    from twin_ml.experiments.robust_worker import _progress as write

    write(path, **event)


@dataclass
class _Splits:
    """Builds and caches the test conditions of one study dataset."""

    e5: E5Config
    rcfg: Any
    ds: Any
    ds_cfg: Any
    root: Path
    clean_series: Any = None
    cache: dict[str, Any] = field(default_factory=dict)

    def e5_run_ids(self) -> list[str]:
        from twin_ml.config import load_dataset_config
        from twin_ml.data.dataset import Dataset, resolve_dataset_dir
        from twin_ml.data.test_sets import e5_run_ids

        e1_cfg = load_dataset_config(self.root / self.e5.studies["e5_e1models"].dataset_config)
        e1 = Dataset.open(resolve_dataset_dir(e1_cfg, self.root))
        return e5_run_ids(self.e5.e5_data, e1.meta["splits"]["test"])

    def get(self, name: str) -> tuple[Any, Any, Any]:
        """(split, sensor-fault row mask or None, faulty sensor per row or None)."""
        if name in self.cache:
            return self.cache[name]  # type: ignore[no-any-return]
        from twin_ml.data.load import read_manifest
        from twin_ml.data.test_sets import (
            check_shared_plan,
            e5_split,
            load_series,
            sensor_fault_split_detail,
        )

        line = self.root / self.ds_cfg.line_config
        out: tuple[Any, Any, Any]
        if name in SPLIT_CONDITIONS:
            out = (self.ds.split(SPLIT_CONDITIONS[name]), None, None)
        elif name == E5_CONDITION:
            d = self.e5.e5_data
            ids = self.e5_run_ids()
            check_shared_plan(
                read_manifest(self.root / self.ds_cfg.source_dir),
                read_manifest(self.root / d.source_dir),
                ids,
            )
            out = (e5_split(self.ds, d, self.root / d.source_dir, ids, line), None, None)
        elif name in self.rcfg.sensor_fault_sets:
            if self.clean_series is None:
                self.clean_series = load_series(
                    self.root / self.ds_cfg.source_dir, self.ds.meta["splits"]["test"], line
                )
            split, rows, sensors = sensor_fault_split_detail(
                self.ds,
                self.root / self.rcfg.sensor_fault_sets[name].source_dir,
                self.clean_series,
                line,
            )
            out = (split, rows, sensors)
        else:
            raise ValueError(f"unknown E5 condition {name!r}")
        self.cache = {name: out}  # keep one condition in memory (E5 has 1.4 M windows)
        return out


# ---------------------------------------------------------------------------- study run


def run_e5(
    study_name: str,
    model: str,
    cfg: Any,
    seed: int,
    e5: E5Config,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    import mlflow

    from twin_common.logging import get_logger
    from twin_ml.config import load_dataset_config, load_robustness_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.test_sets import extended_fault_types
    from twin_ml.evaluate import run_sums
    from twin_ml.experiments import e5_eval
    from twin_ml.train import tracking

    log = get_logger(__name__)
    started = time.perf_counter()
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    rcfg = load_robustness_config(root / e5.robustness_config)
    study = e5.studies[study_name]
    ds_cfg = load_dataset_config(root / study.dataset_config)
    ds = Dataset.open(resolve_dataset_dir(ds_cfg, root))
    key = ds.meta["dataset_key"]
    fault_types = extended_fault_types(ds, [e5.e5_data.fault_type])
    h_primary = e5.primary.horizon
    splits = _Splits(e5, rcfg, ds, ds_cfg, root)
    details: dict[str, Any] = {}
    groups: dict[str, run_sums.Sums] = {}
    run_ids: dict[str, list[str]] = {}
    metrics: dict[str, float] = {}
    alarm_arrays: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="twin_p9_") as tmp_s:
        tmp = Path(tmp_s)
        evaluator: Any
        if is_anomaly(model):
            fam = "anomaly"
            evaluator = _AnomalyEval(e5, ds, cfg, seed, h_primary, fault_types, eval_cfg)
            source, model_s = "train", evaluator.train_s
            extra["detector"] = evaluator.info
        else:
            from twin_ml.train.trainer import family

            fam = family(cfg)
            evaluator = _SupervisedEval(
                e5, rcfg, ds, cfg, study_name, model, seed, fault_types, eval_cfg, root, tdir, tmp
            )
            source, model_s = evaluator.source, evaluator.model_s
            extra["calibration"] = evaluator.cal.describe()
        _progress(
            progress,
            event="model",
            study=study_name,
            model=model,
            seed=seed,
            source=source,
            seconds=round(model_s, 1),
        )
        for name in study.conditions:
            t0 = time.perf_counter()
            split, sf_rows, sf_sensors = splits.get(name)
            detail, sums, alarms = evaluator.condition(name, split, sf_rows, sf_sensors)
            details[name] = detail
            run_ids[name] = split.run_ids
            for g, s in sums.items():
                groups[f"{name}|{g}"] = s
                for m, v in run_sums.point(s).items():
                    if np.isfinite(v):
                        metrics[f"{name}.{g}.{m}"] = float(v)
            for g, a in alarms.items():
                alarm_arrays[f"{name}|{g}"] = e5_eval.pack_alarm(a)
            alarm_arrays[f"n:{name}"] = np.array([split.n_windows])
            alarm_arrays[f"digest:{name}"] = np.array([e5_eval.ends_digest(split)])
            timings[name] = time.perf_counter() - t0
            log.info(
                "e5.condition",
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
        metrics["time.model_s"] = model_s
        metrics["time.total_s"] = total_s
        e5_content = e5_content_sha(e5, root)
        sha = study_config_sha(study_name, model, cfg, e5, rcfg, eval_cfg, key, source, e5_content)
        result = {
            "study": study_name,
            "model": model,
            "family": fam,
            "seed": seed,
            "dataset": {"key": key, "counts": ds.meta["counts"], "splits": ds.meta["splits"]},
            "fault_types": fault_types,
            "source_run": source,
            "e5_content_sha256": e5_content,
            **extra,
            "conditions": details,
            "timings_s": timings,
            "total_s": total_s,
        }
        tracking.configure(tdir, experiment or e5.experiment)
        with mlflow.start_run(run_name=f"{study_name}-{model}-seed{seed}") as run:
            mlflow.set_tags(
                {
                    "phase": e5.phase_tag,
                    "study": study_name,
                    "model": model,
                    "family": fam,
                    "seed": str(seed),
                    "dataset_key": key,
                    "source_run_id": source,
                    "model_config_sha": config_sha_any(cfg),
                    "study_config_sha": sha,
                }
            )
            mlflow.log_params(
                {"seed": str(seed), "conditions": ",".join(study.conditions), "kind": fam}
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(json.loads(json.dumps(result, default=float)), "metrics.json")
            arrays = run_sums.save_npz_dict(groups)
            for name, ids in run_ids.items():
                arrays[f"runs:{name}"] = np.array(ids, dtype=str)
            np.savez_compressed(tmp / "runstats.npz", **arrays)  # type: ignore[arg-type]
            np.savez_compressed(tmp / "alarms.npz", **alarm_arrays)  # type: ignore[arg-type]
            mlflow.log_artifact(str(tmp / "runstats.npz"))
            mlflow.log_artifact(str(tmp / "alarms.npz"))
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
        "source": source,
    }


class _SupervisedEval:
    """A reused supervised model: Phase 8 metrics plus the Phase 9 extras per condition."""

    def __init__(
        self,
        e5: E5Config,
        rcfg: Any,
        ds: Any,
        cfg: Any,
        study: str,
        model: str,
        seed: int,
        fault_types: list[str],
        eval_cfg: EvalConfig,
        root: Path,
        tdir: Path,
        tmp: Path,
    ) -> None:
        from twin_ml.experiments.robust_eval import fit_calibration
        from twin_ml.experiments.robust_worker import load_from_run
        from twin_ml.train.trainer import family

        run = supervised_source(e5, study, tdir, ds.meta["dataset_key"], model, seed, cfg)
        got = load_from_run(run, model, cfg, seed, ds, eval_cfg, root, tmp)
        self.e5, self.rcfg, self.ds, self.eval_cfg = e5, rcfg, ds, eval_cfg
        self.fault_types = fault_types
        self.predictor = got.predictor
        self.stored = got.stored_preds
        self.source, self.model_s = got.source, got.train_s
        val = ds.split("val")
        self.cal = fit_calibration(
            family(cfg), val, self.predictor.predict(val), rcfg, eval_cfg, ds.ttf_cap_s
        )
        self.prediction_check: dict[str, Any] | None = None

    def condition(
        self, name: str, split: Any, sf_rows: Any, sf_sensors: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
        from twin_ml.evaluate import run_sums
        from twin_ml.evaluate.evaluator import subset_masks
        from twin_ml.experiments import e5_eval
        from twin_ml.experiments.robust_eval import _logits, evaluate_condition, uniforms
        from twin_ml.experiments.robust_worker import check_predictions

        preds = self.predictor.predict(split)
        if self.stored is not None and SPLIT_CONDITIONS.get(name) == "test":
            self.prediction_check = check_predictions(preds, self.stored, self.ds.horizons)
        rcfg, ecfg = self.rcfg, self.eval_cfg
        res = evaluate_condition(
            name, split, preds, self.cal, rcfg, ecfg, self.fault_types, self.ds.ttf_cap_s, sf_rows
        )
        masks = subset_masks(split, ecfg)
        sub = masks[rcfg.eval_subset]
        alarms: dict[str, np.ndarray] = {}
        for h in split.horizons:
            s = res.sums[f"h{h}"]
            p_cal = self.cal.heads[h].apply(preds.probs[h], _logits(preds, h))
            u = uniforms(rcfg.conformal.aps_seed, name, h, split.n_windows)
            sets = self.cal.conformal[h].sets(p_cal, u)
            s |= e5_eval.supervised_extras(split, p_cal, sets, h, sub, self.e5.confidence.high)
            sus = e5_eval.sustained_alarm(split, preds.label(h) >= 1, ecfg.early_warning.sustain_s)
            if sf_rows is not None:
                neg = e5_eval.negative_windows(split, ecfg, self.fault_types, h)
                s["drift"] = e5_eval.drift_flag_sums(split, sus, neg, sf_rows)
            if h == self.e5.primary.horizon:
                alarms[f"h{h}"] = sus
            b = ecfg.bootstrap
            res.detail["robust"][str(h)]["bootstrap"] = run_sums.bootstrap(
                s, b.n_resamples, b.seed, b.ci
            )
        if preds.fault_type is not None and name == E5_CONDITION:
            # exploratory: which trained fault type the head assigns to unseen-fault precursors
            pre = masks["gradual"]
            counts = np.bincount(
                np.argmax(preds.fault_type[pre], axis=1), minlength=preds.fault_type.shape[1]
            )
            res.detail["fault_type_head_precursor"] = {
                self.ds.fault_types[i]: int(c) for i, c in enumerate(counts)
            }
        res.detail["prediction_check"] = self.prediction_check
        # the per-window episode lists stay; drop nothing else (same artifact as Phase 8)
        return res.detail, res.sums, alarms


class _AnomalyEval:
    """A detector trained on normal-operation windows, thresholds from validation."""

    def __init__(
        self,
        e5: E5Config,
        ds: Any,
        cfg: Any,
        seed: int,
        horizon: int,
        fault_types: list[str],
        eval_cfg: EvalConfig,
    ) -> None:
        from twin_ml.models.anomaly import AnomalyThresholds, make_detector, normal_windows

        a = e5.anomaly
        self.e5, self.ds, self.eval_cfg = e5, ds, eval_cfg
        self.horizon, self.fault_types = horizon, fault_types
        self.no_fault = list(ds.fault_types).index("none")
        t0 = time.perf_counter()
        np.random.seed(seed)
        self.det = make_detector(cfg, self.no_fault, horizon, a.normal_states)
        train, val = ds.split("train"), ds.split("val")
        fit_info = self.det.fit(train, val, seed)
        vidx = normal_windows(val, self.no_fault, horizon, a.normal_states)
        vscores = self.det.score(val, vidx)
        self.thr = AnomalyThresholds(list(a.threshold_percentiles), a.per_kind).fit(
            vscores, val.at_end("kind", vidx)
        )
        self.train_s = time.perf_counter() - t0
        self.info = {
            "fit": fit_info,
            "thresholds": self.thr.describe(),
            "val_normal_windows": int(vidx.shape[0]),
            "train_s": self.train_s,
        }

    def condition(
        self, name: str, split: Any, sf_rows: Any, sf_sensors: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
        from twin_ml.evaluate.evaluator import subset_masks
        from twin_ml.experiments import e5_eval

        scores = self.det.score(split)
        kinds = split.at_end("kind")
        alarms = {p: self.thr.alarm(scores, kinds, p) for p in self.thr.percentiles}
        detail, groups, sustained = e5_eval.detector_condition(
            split, scores, alarms, self.horizon, self.fault_types, self.eval_cfg, sf_rows
        )
        if hasattr(self.det, "sensor_errors"):
            detail["sensor_error_shares"] = self._sensor_shares(
                split, sustained[self.e5.anomaly.primary_percentile], sf_rows, sf_sensors
            )
        masks = subset_masks(split, self.eval_cfg)
        detail["operating_windows"] = int(masks["operating"].sum())
        return detail, groups, {f"p{p:g}": a for p, a in sustained.items()}

    def _sensor_shares(self, split: Any, sus: Any, sf_rows: Any, sf_sensors: Any) -> Any:
        """Exploratory: mean per-sensor share of the AE reconstruction error on alarmed
        precursor windows (by fault type) and on alarmed sensor-fault windows (by the faulty
        sensor)."""
        from twin_ml.evaluate.evaluator import subset_masks
        from twin_ml.features.inputs import SENSORS

        masks = subset_masks(split, self.eval_cfg)
        out: dict[str, Any] = {"sensors": list(SENSORS)}
        pre = np.flatnonzero(masks["gradual"] & sus)
        if pre.size:
            err = self.det.sensor_errors(split, pre)
            share = err / np.maximum(err.sum(axis=1, keepdims=True), 1e-12)
            ft = split.at_end("fault_type", pre)
            out["precursor"] = {
                self.fault_types[int(f)]: {
                    "n": int((ft == f).sum()),
                    "mean_share": share[ft == f].mean(axis=0).tolist(),
                    "top_counts": np.bincount(
                        np.argmax(share[ft == f], axis=1), minlength=len(SENSORS)
                    ).tolist(),
                }
                for f in np.unique(ft)
            }
        if sf_rows is not None and sf_sensors is not None:
            ends = split.arrays["ends"]
            w = np.flatnonzero(sf_rows[ends] & sus)
            if w.size:
                err = self.det.sensor_errors(split, w)
                share = err / np.maximum(err.sum(axis=1, keepdims=True), 1e-12)
                faulty = sf_sensors[ends[w]]
                out["sensor_fault"] = {
                    SENSORS[int(f)]: {
                        "n": int((faulty == f).sum()),
                        "mean_share": share[faulty == f].mean(axis=0).tolist(),
                        "top_is_faulty": float(np.mean(np.argmax(share[faulty == f], axis=1) == f)),
                    }
                    for f in np.unique(faulty)
                    if f >= 0
                }
        return out


# ---------------------------------------------------------------------------- combined


def combined_sha(sup_run: str, det_run: str, e5: E5Config, eval_cfg: EvalConfig) -> str:
    data = {
        "supervised_run": sup_run,
        "detector_run": det_run,
        "combined": e5.combined.model_dump(mode="json"),
        "horizon": e5.primary.horizon,
        "eval": eval_cfg.model_dump(mode="json", include={"subsets", "early_warning"}),
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
    from twin_ml.train import tracking

    out: dict[tuple[str, int], Any] = {}
    for run in tracking.finished_runs(
        tracking_dir, experiment, None, phase, f"tags.study = '{study}'"
    ):
        k = (run.data.tags["model"], int(run.data.tags["seed"]))
        if k not in out and run.data.tags.get("study_config_sha") == shas.get(k):
            out[k] = run
    return out


def load_alarms(run: Any, tmp: Path) -> dict[str, np.ndarray]:
    from twin_ml.experiments.robust_worker import _download

    path = _download(run.info.artifact_uri, "alarms.npz", tmp / run.info.run_id)
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def combine_pair(
    split: Any,
    condition: str,
    sup: dict[str, np.ndarray],
    det: dict[str, np.ndarray],
    percentiles: list[float],
    horizon: int,
    fault_types: list[str],
    eval_cfg: EvalConfig,
    sf_rows: Any = None,
) -> dict[str, Any]:
    """Per-percentile per-run sums of (supervised sustained alarm OR detector sustained alarm)."""
    from twin_ml.experiments import e5_eval

    n = split.n_windows
    digest = e5_eval.ends_digest(split)
    for side in (sup, det):
        if int(side[f"n:{condition}"][0]) != n or str(side[f"digest:{condition}"][0]) != digest:
            raise ValueError(f"{condition}: alarm artifacts do not match the split's windows")
    a = e5_eval.unpack_alarm(sup[f"{condition}|h{horizon}"], n)
    neg = e5_eval.negative_windows(split, eval_cfg, fault_types, horizon)
    out: dict[str, Any] = {}
    for p in percentiles:
        both = a | e5_eval.unpack_alarm(det[f"{condition}|p{p:g}"], n)
        s = e5_eval.alarm_run_sums(split, both, horizon, fault_types, eval_cfg)
        if sf_rows is not None:
            from twin_ml.evaluate.evaluator import subset_masks

            op = subset_masks(split, eval_cfg)["operating"]
            sf = e5_eval.alarm_run_sums(
                split, both, horizon, fault_types, eval_cfg, op & sf_rows[split.arrays["ends"]]
            )
            s["sf_false_alarms"] = sf["false_alarms"]
            s["sf_negative_s"] = sf["negative_s"]
            s["drift"] = e5_eval.drift_flag_sums(split, both, neg, sf_rows)
        out[f"p{p:g}"] = s
    return out


# ---------------------------------------------------------------------------- explain

EXPLAIN_STUDY = "explain"
EXPLAIN_SOURCE_STUDY = "e5_e1models"  # the E1 models are explained
EXPLAIN_CONDITIONS = ("clean", E5_CONDITION)  # seen fault types, the held-out type


def explain_config_sha(
    model: str,
    cfg: Any,
    e5: E5Config,
    eval_cfg: EvalConfig,
    dataset_key: str,
    source: str,
    e5_content: str | None,
) -> str:
    data = {
        "study": EXPLAIN_STUDY,
        "model": model,
        "model_config": cfg.model_dump(mode="json"),
        "explain": e5.explain.model_dump(mode="json"),
        "e5_data": e5.e5_data.model_dump(mode="json"),
        "eval": eval_cfg.model_dump(mode="json", include={"subsets", "early_warning", "bootstrap"}),
        "dataset_key": dataset_key,
        "source": source,
        "e5_content": e5_content,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def run_explain(
    model: str,
    cfg: Any,
    seed: int,
    e5: E5Config,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Attributions on alarmed precursor windows; agreement, deletion test, IG completeness."""
    import mlflow

    from twin_ml.config import load_dataset_config, load_robustness_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.data.test_sets import extended_fault_types
    from twin_ml.evaluate import run_sums
    from twin_ml.experiments import e5_eval, explain_eval
    from twin_ml.experiments.robust_worker import load_from_run
    from twin_ml.explain import attribute, probabilities
    from twin_ml.explain.aggregate import normalise
    from twin_ml.features.inputs import SENSORS
    from twin_ml.train import tracking
    from twin_ml.train.trainer import family

    started = time.perf_counter()
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    ex = e5.explain
    h = ex.horizon
    rcfg = load_robustness_config(root / e5.robustness_config)
    study = e5.studies[EXPLAIN_SOURCE_STUDY]
    ds_cfg = load_dataset_config(root / study.dataset_config)
    ds = Dataset.open(resolve_dataset_dir(ds_cfg, root))
    key = ds.meta["dataset_key"]
    fault_types = extended_fault_types(ds, [e5.e5_data.fault_type])
    expected = {
        fault_types.index(ft): SENSORS.index(s)
        for ft, s in ex.expected_sensor.items()
        if ft in fault_types
    }
    splits = _Splits(e5, rcfg, ds, ds_cfg, root)
    groups: dict[str, run_sums.Sums] = {}
    details: dict[str, Any] = {}
    run_ids: dict[str, list[str]] = {}
    metrics: dict[str, float] = {}
    batch = 256
    with tempfile.TemporaryDirectory(prefix="twin_p9_ex_") as tmp_s:
        tmp = Path(tmp_s)
        run = supervised_source(e5, EXPLAIN_SOURCE_STUDY, tdir, key, model, seed, cfg)
        got = load_from_run(run, model, cfg, seed, ds, eval_cfg, root, tmp)
        predictor = got.predictor
        for name in EXPLAIN_CONDITIONS:
            t0 = time.perf_counter()
            split, _, _ = splits.get(name)
            preds = predictor.predict(split)
            sus = e5_eval.sustained_alarm(
                split, preds.label(h) >= 1, eval_cfg.early_warning.sustain_s
            )
            eps = explain_eval.explained_windows(
                split, sus, ex.window_stride_s, ex.max_windows_per_episode
            )
            n_runs = len(split.run_ids)
            if eps:
                allw = np.concatenate([e.windows for e in eps])
                run_of = np.concatenate([np.full(e.windows.size, e.run) for e in eps])
            else:
                allw = np.zeros(0, dtype=np.int64)
                run_of = np.zeros(0, dtype=np.int64)
            sens = np.zeros((allw.size, len(SENSORS)))
            err_parts: list[np.ndarray] = []
            delta_parts: list[np.ndarray] = []
            p0 = np.zeros(allw.size)
            p_top = np.zeros(allw.size)
            p_rand = np.zeros(allw.size)
            for i in range(0, allw.size, batch):
                w = allw[i : i + batch]
                values, mask, kind = split.gather(w)
                att = attribute(predictor, values, mask, kind, h, ex.ig_steps)
                sens[i : i + batch] = normalise(att.sensors)
                if att.completeness_error is not None and att.delta is not None:
                    err_parts.append(att.completeness_error)
                    delta_parts.append(att.delta)
                top = np.argmax(att.sensors, axis=1)
                rnd = explain_eval.random_other(top, w, ex.deletion_seed)
                pr = probabilities(predictor, values, mask, kind, h)
                p0[i : i + batch] = pr[:, 1:].sum(axis=1)
                pt = probabilities(
                    predictor, explain_eval.deletion_inputs(values, top), mask, kind, h
                )
                p_top[i : i + batch] = pt[:, 1:].sum(axis=1)
                pn = probabilities(
                    predictor, explain_eval.deletion_inputs(values, rnd), mask, kind, h
                )
                p_rand[i : i + batch] = pn[:, 1:].sum(axis=1)
            weights: dict[int, np.ndarray] = {}
            pos = 0
            for e in eps:
                weights[e.episode] = sens[pos : pos + e.windows.size]
                pos += e.windows.size
            agree, records = explain_eval.agreement_sums(
                eps, weights, expected, n_runs, len(fault_types)
            )
            s: run_sums.Sums = {
                "agree": agree,
                "deletion": explain_eval.deletion_sums(run_of, p0, p_top, p_rand, n_runs),
            }
            if err_parts:
                s["ig_complete"] = explain_eval.completeness_sums(
                    run_of, np.concatenate(err_parts), np.concatenate(delta_parts), n_runs
                )
            groups[f"{name}|h{h}"] = s
            run_ids[name] = split.run_ids
            b = eval_cfg.bootstrap
            by_type: dict[str, Any] = {}
            for code in sorted({e.fault_type for e in eps}):
                rows = np.concatenate([np.full(e.windows.size, e.fault_type == code) for e in eps])
                by_type[fault_types[code]] = {
                    "windows": int(rows.sum()),
                    "episodes": int(sum(e.fault_type == code for e in eps)),
                    "mean_weight": sens[rows].mean(axis=0).tolist(),
                    "top_counts": np.bincount(
                        np.argmax(sens[rows], axis=1), minlength=len(SENSORS)
                    ).tolist(),
                }
            details[name] = {
                "episodes_total": int(split.arrays["ep_series"].shape[0]),
                "episodes_explained": len(eps),
                "windows": int(allw.size),
                "sensors": list(SENSORS),
                "by_fault_type": by_type,
                "episodes": records,
                "bootstrap": run_sums.bootstrap(s, b.n_resamples, b.seed, b.ci),
                "seconds": time.perf_counter() - t0,
            }
            for m, v in run_sums.point(s).items():
                if np.isfinite(v):
                    metrics[f"{name}.h{h}.{m}"] = float(v)
            _progress(
                progress,
                event="condition",
                study=EXPLAIN_STUDY,
                model=model,
                seed=seed,
                condition=name,
                seconds=round(time.perf_counter() - t0, 1),
            )
        total_s = time.perf_counter() - started
        metrics["time.total_s"] = total_s
        content = e5_content_sha(e5, root)
        sha = explain_config_sha(model, cfg, e5, eval_cfg, key, got.source, content)
        result = {
            "study": EXPLAIN_STUDY,
            "model": model,
            "family": family(cfg),
            "seed": seed,
            "method": {"xgb": "treeshap", "rf": "saabas"}.get(model, "integrated_gradients"),
            "fault_types": fault_types,
            "expected_sensor": ex.expected_sensor,
            "source_run": got.source,
            "conditions": details,
            "total_s": total_s,
        }
        tracking.configure(tdir, experiment or e5.experiment)
        with mlflow.start_run(run_name=f"{EXPLAIN_STUDY}-{model}-seed{seed}") as mrun:
            mlflow.set_tags(
                {
                    "phase": e5.phase_tag,
                    "study": EXPLAIN_STUDY,
                    "model": model,
                    "family": family(cfg),
                    "seed": str(seed),
                    "dataset_key": key,
                    "source_run_id": got.source,
                    "model_config_sha": config_sha_any(cfg),
                    "study_config_sha": sha,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(json.loads(json.dumps(result, default=float)), "metrics.json")
            arrays = run_sums.save_npz_dict(groups)
            for name, ids in run_ids.items():
                arrays[f"runs:{name}"] = np.array(ids, dtype=str)
            np.savez_compressed(tmp / "runstats.npz", **arrays)  # type: ignore[arg-type]
            mlflow.log_artifact(str(tmp / "runstats.npz"))
            run_id = mrun.info.run_id
    return {
        "run_id": run_id,
        "study": EXPLAIN_STUDY,
        "model": model,
        "seed": seed,
        "total_s": total_s,
        "source": got.source,
    }


# ---------------------------------------------------------------------------- process


def run_e5_isolated(
    study: str,
    model: str,
    cfg: Any,
    seed: int,
    e5: E5Config,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    progress: Path | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    request = {
        "study": study,
        "model": model,
        "config": cfg.model_dump(mode="json"),
        "seed": seed,
        "e5": e5.model_dump(mode="json"),
        "eval_config": eval_cfg.model_dump(mode="json"),
        "root": str(root),
        "tracking_dir": None if tracking_dir is None else str(tracking_dir),
        "progress": None if progress is None else str(progress),
        "experiment": experiment,
    }
    with tempfile.TemporaryDirectory(prefix="twin_p9_req_") as tmp:
        req, res = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        cmd = [sys.executable, "-X", "faulthandler", "-m", WORKER_MODULE, str(req), str(res)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not res.is_file():
            raise E5ProcessError(study, model, seed, proc.returncode)
        out: dict[str, Any] = json.loads(res.read_text(encoding="utf-8"))
    return out


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(f"usage: python -m {WORKER_MODULE} <request.json> <result.json>", file=sys.stderr)
        return 2
    from twin_common.logging import configure_logging
    from twin_ml.config import parse_anomaly_config, parse_model_config

    configure_logging()
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    model = req["model"]
    cfg = (parse_anomaly_config if is_anomaly(model) else parse_model_config)(req["config"])
    if req["study"] == EXPLAIN_STUDY:
        out = run_explain(
            model,
            cfg,
            int(req["seed"]),
            E5Config.model_validate(req["e5"]),
            EvalConfig.model_validate(req["eval_config"]),
            Path(req["root"]),
            None if req["tracking_dir"] is None else Path(req["tracking_dir"]),
            None if req["progress"] is None else Path(req["progress"]),
            req.get("experiment"),
        )
        Path(args[1]).write_text(json.dumps(out), encoding="utf-8")
        return 0
    out = run_e5(
        req["study"],
        model,
        cfg,
        int(req["seed"]),
        E5Config.model_validate(req["e5"]),
        EvalConfig.model_validate(req["eval_config"]),
        Path(req["root"]),
        None if req["tracking_dir"] is None else Path(req["tracking_dir"]),
        None if req["progress"] is None else Path(req["progress"]),
        req.get("experiment"),
    )
    Path(args[1]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
