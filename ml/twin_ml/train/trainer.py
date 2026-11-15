"""Config-driven training + test evaluation + MLflow logging for one (model, seed)."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twin_common.logging import get_logger
from twin_ml.config import (
    EvalConfig,
    ModelConfig,
    NnConfig,
    PersistenceConfig,
    RfConfig,
    ThresholdConfig,
    XgbConfig,
)
from twin_ml.data.dataset import Dataset
from twin_ml.evaluate.evaluator import evaluate, flat_metrics
from twin_ml.evaluate.latency import measure
from twin_ml.models.base import Predictor
from twin_ml.train import tracking

log = get_logger(__name__)

BASELINES = ("persistence", "threshold")


def family(cfg: ModelConfig) -> str:
    if isinstance(cfg, PersistenceConfig | ThresholdConfig):
        return "baseline"
    if isinstance(cfg, RfConfig | XgbConfig):
        return "tree"
    return "nn"


def make_predictor(
    cfg: ModelConfig, ds: Dataset, root: Path, eval_cfg: EvalConfig | None = None
) -> Predictor:
    if isinstance(cfg, PersistenceConfig):
        from twin_ml.baselines.persistence import PersistenceBaseline

        return PersistenceBaseline()
    if isinstance(cfg, ThresholdConfig):
        from twin_ml.baselines.threshold import ThresholdBaseline

        return ThresholdBaseline(root / cfg.sim_config, ds.meta["normalizer"])
    if isinstance(cfg, RfConfig | XgbConfig):
        from twin_ml.models.trees import TreeModel

        return TreeModel(cfg, len(ds.fault_types), ds.ttf_cap_s)
    if isinstance(cfg, NnConfig):
        from twin_ml.models.nn import SequenceModel

        if eval_cfg is None:
            return SequenceModel(cfg, len(ds.fault_types), ds.ttf_cap_s)
        return SequenceModel(
            cfg, len(ds.fault_types), ds.ttf_cap_s, operating=eval_cfg.subsets.operating
        )
    raise TypeError(f"unknown model config {type(cfg).__name__}")


def config_sha(cfg: ModelConfig) -> str:
    """Short hash of a model config; runs with different settings form separate groups."""
    data = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    model: str
    seed: int
    metrics: dict[str, float]
    train_s: float


def _set_threads(predictor: Predictor, n: int) -> Any:
    """Restrict inference threads; returns a callable that restores the previous setting."""
    restore: list[Any] = []
    setter = getattr(predictor, "set_threads", None)
    if setter is not None:
        setter(n)
    try:
        import sys

        if "torch" in sys.modules:
            import torch

            prev = torch.get_num_threads()
            torch.set_num_threads(n)
            restore.append(lambda: torch.set_num_threads(prev))
    except ImportError:  # pragma: no cover
        pass
    return lambda: [f() for f in restore]


def run_one(
    model: str,
    cfg: ModelConfig,
    seed: int,
    ds: Dataset,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
) -> RunResult:
    """Train ``model`` with ``seed``, evaluate it on the test split and log one MLflow run."""
    return train_and_log(model, cfg, seed, ds, eval_cfg, root, tracking_dir)[0]


def train_and_log(
    model: str,
    cfg: ModelConfig,
    seed: int,
    ds: Dataset,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    phase: str = tracking.PHASE_TAG,
    extra_tags: dict[str, str] | None = None,
) -> tuple[RunResult, Predictor]:
    """:func:`run_one` that also returns the fitted predictor (Phase 8 studies evaluate the
    model further in the same process; Random Forests are never saved)."""
    import mlflow

    np.random.seed(seed)  # libraries that read the legacy global state
    train, val, test = ds.split("train"), ds.split("val"), ds.split("test")
    predictor = make_predictor(cfg, ds, root, eval_cfg)
    log.info("train.start", model=model, seed=seed, train_windows=train.n_windows)
    t0 = time.perf_counter()
    info = predictor.fit(train, val, seed)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = predictor.predict(test)
    predict_s = time.perf_counter() - t0
    result = evaluate(test, preds, eval_cfg, ds.fault_types, ds.ttf_cap_s)
    restore = _set_threads(predictor, eval_cfg.latency.num_threads)
    try:
        latency = measure(
            lambda i: predictor.predict_one(test, i), test.n_windows, eval_cfg.latency, seed
        )
    finally:
        restore()
    result.update(
        {
            "model": model,
            "family": family(cfg),
            "seed": seed,
            "train_info": info,
            "train_s": train_s,
            "predict_s": predict_s,
            "latency": latency,
            "dataset": {
                "key": ds.meta["dataset_key"],
                "source": ds.meta["source"],
                "counts": ds.meta["counts"],
            },
            "machine": {"platform": platform.platform(), "processor": platform.processor()},
        }
    )
    metrics = flat_metrics(result)
    metrics["time.train_s"] = train_s
    metrics["time.predict_s"] = predict_s
    for k, v in latency.items():
        metrics[f"latency.{k}"] = v

    exp_dir = (root / eval_cfg.tracking_dir) if tracking_dir is None else tracking_dir
    tracking.configure(exp_dir, eval_cfg.experiment)
    src = ds.meta["source"]
    with mlflow.start_run(run_name=f"{model}-seed{seed}") as run:
        mlflow.set_tags(
            {
                "phase": phase,
                "model": model,
                "family": family(cfg),
                "seed": str(seed),
                "dataset_key": ds.meta["dataset_key"],
                "manifest_sha256": src["manifest_sha256"],
                "content_sha256": src["content_sha256"],
                "model_config_sha": config_sha(cfg),
                **(extra_tags or {}),
            }
        )
        params = tracking.flatten(cfg.model_dump(mode="json"), "model.")
        params |= tracking.flatten(ds.meta["config"], "dataset.")
        params |= {"seed": str(seed), "dataset_fingerprint": ds.meta["config_fingerprint"]}
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for entry in info.get("history", []):  # learning curves, one step per epoch
            step = int(entry["epoch"])
            curve = {f"curve.{k}": v for k, v in entry.items() if k != "epoch"}
            mlflow.log_metrics(curve, step=step)
        mlflow.log_dict(json.loads(json.dumps(result, default=float)), "metrics.json")
        mlflow.log_dict(ds.meta["splits"], "splits.json")
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "test_predictions.npz"
            np.savez_compressed(pred_path, **preds.to_arrays())  # type: ignore[arg-type]
            mlflow.log_artifact(str(pred_path))
            saver = getattr(predictor, "save", None)
            if saver is not None:
                for p in saver(Path(tmp) / "model"):
                    mlflow.log_artifact(str(p), artifact_path="model")
    log.info(
        "train.done",
        model=model,
        seed=seed,
        train_s=round(train_s, 1),
        macro_f1_h30=metrics.get(f"h{max(ds.horizons)}.operating.macro_f1"),
        run_id=run.info.run_id,
    )
    return RunResult(run.info.run_id, model, seed, metrics, train_s), predictor
