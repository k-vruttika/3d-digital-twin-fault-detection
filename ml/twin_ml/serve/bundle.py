"""Deploy bundles: build from the evaluated runs, register in MLflow, load (ADR 0029).

Two kinds of bundle, each built in its own process (``python -m twin_ml.serve build``), so
torch and XGBoost never meet (ADR 0020):

* **supervised** (``gru``, ``xgb``, ``gru_e4`` in ``config/models/deploy.yaml``): the trained
  model of one seed, reused from its training run (never retrained), chosen by validation
  macro F1; it must reproduce its stored test predictions. Calibration and Mondrian conformal
  maps are refitted on the clean validation split with the Phase 8 code and must equal the
  numbers the Phase 8 study logged for that run. Exported as plain numbers (``serve.calib``).
* **autoencoder**: the Phase 9 detector refitted from its fixed seed; its per-kind validation
  thresholds must equal the ones the Phase 9 study logged.

A bundle directory holds ``bundle.json`` plus the model files. It is logged as an MLflow run
(experiment ``P11_deploy``) and registered as ``<prefix><name>`` (a new version per build).
``strict=False`` (tests on quick data) records missing reference runs instead of failing.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import DeployConfig

BUNDLE_FORMAT = 1
BUNDLE_FILE = "bundle.json"
ANOMALY_NAME = "autoencoder"
REL_TOL = 1e-9  # refitted calibration / thresholds vs the logged numbers


class BundleCheckError(RuntimeError):
    """A rebuilt map does not reproduce the logged numbers of the evaluated run."""


# ---------------------------------------------------------------------------- helpers


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12)


def _q_list(q: Any) -> list[float | None]:
    return [None if (x is None or math.isinf(float(x))) else float(x) for x in np.ravel(q)]


def reference_run(tracking_dir: Path, ref: Any, model: str, seed: int, **tags: str) -> Any | None:
    """Newest finished study run of (study, model, seed) with the given extra tags."""
    from twin_ml.train import tracking

    flt = f"tags.study = '{ref.study}' and tags.model = '{model}' and tags.seed = '{seed}'"
    for k, v in tags.items():
        flt += f" and tags.{k} = '{v}'"
    runs = tracking.finished_runs(tracking_dir, ref.experiment, None, ref.phase, flt)
    return runs[0] if runs else None


def _download_json(run: Any, rel: str, tmp: Path) -> dict[str, Any]:
    from twin_ml.experiments.robust_worker import _download

    data: dict[str, Any] = json.loads(_download(run.info.artifact_uri, rel, tmp).read_text())
    return data


def validation_score(split: Any, preds: Any, eval_cfg: Any, horizon: int, subset: str) -> float:
    from twin_ml.evaluate.evaluator import classification_block, subset_masks

    mask = subset_masks(split, eval_cfg)[subset]
    return float(
        classification_block(split.target(horizon), preds.probs[horizon], mask)["macro_f1"]
    )


def check_calibration(live: dict[str, Any], logged: dict[str, Any]) -> dict[str, Any]:
    """Refitted temperatures, Mondrian thresholds and the TTF quantile vs a logged
    ``Calibration.describe()``; raises :class:`BundleCheckError` on any difference."""
    out: dict[str, Any] = {}
    for h, head in live["heads"].items():
        lt = logged["heads"][h].get("temperature")
        lm = logged["heads"][h]["method"]
        if head["method"] != lm or not _close(head.get("temperature"), lt):
            raise BundleCheckError(
                f"h{h}: calibration {head['method']} T={head.get('temperature')} differs from "
                f"the logged {lm} T={lt}"
            )
        mine = live["mondrian_q"][h]
        theirs = _q_list(logged["conformal"][h]["q"]["mondrian_lac"])
        if len(mine) != len(theirs) or not all(
            _close(a, b) for a, b in zip(mine, theirs, strict=True)
        ):
            raise BundleCheckError(f"h{h}: Mondrian thresholds {mine} differ from {theirs}")
        out[f"h{h}"] = {"method": lm, "temperature": lt, "mondrian_q": theirs}
    if not _close(live.get("ttf_q"), logged["ttf"]["q_s"]):
        raise BundleCheckError(f"ttf quantile {live.get('ttf_q')} != {logged['ttf']['q_s']}")
    out["ttf_q"] = logged["ttf"]["q_s"]
    return out


def check_thresholds(live: dict[str, list[float]], logged: dict[str, Any]) -> dict[str, Any]:
    for p, values in live.items():
        theirs = logged["values"][p]
        if not all(_close(a, b) for a, b in zip(values, theirs, strict=True)):
            raise BundleCheckError(f"p{p}: thresholds {values} differ from the logged {theirs}")
    return {"thresholds": logged["values"]}


def _registered(client: Any, name: str) -> bool:
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
    except MlflowException:
        return False
    return True


def card(cfg: DeployConfig) -> dict[str, Any]:
    return cfg.card.model_dump(mode="json")


# ---------------------------------------------------------------------------- build


@dataclass
class BuildResult:
    name: str
    run_id: str
    version: str
    seconds: float
    summary: dict[str, Any]


def build_supervised(
    name: str, cfg: DeployConfig, root: Path, tracking_dir: Path, out: Path, strict: bool
) -> dict[str, Any]:
    """Write the bundle of supervised model ``name`` to ``out``; returns ``bundle.json``."""
    from twin_common.logging import get_logger
    from twin_ml.config import (
        load_dataset_config,
        load_eval_config,
        load_model_config,
        load_robustness_config,
    )
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.experiments.robust_eval import fit_calibration
    from twin_ml.experiments.robust_worker import check_predictions, find_run, load_from_run
    from twin_ml.serve.calib import LiveCalibration
    from twin_ml.train.trainer import family

    log = get_logger(__name__)
    m = cfg.models[name]
    eval_cfg = load_eval_config(root / cfg.eval_config)
    rcfg = load_robustness_config(root / cfg.robustness_config)
    mcfg = load_model_config(root / m.config_file)
    model_type = str(mcfg.type)
    fam = family(mcfg)
    if fam not in ("nn", "tree") or model_type == "rf":
        raise ValueError(f"{name}: only saved models can be deployed (xgb, lstm, gru, transformer)")
    ds = Dataset.open(resolve_dataset_dir(load_dataset_config(root / m.dataset_config), root))
    key = ds.meta["dataset_key"]
    val, test = ds.split("val"), ds.split("test")
    sel = cfg.seed_selection
    scores: dict[int, float] = {}
    loaded: dict[int, Any] = {}
    with tempfile.TemporaryDirectory(prefix="twin_p11_") as tmp_s:
        tmp = Path(tmp_s)
        for seed in m.seeds:
            run = find_run(
                tracking_dir, m.source_experiment, m.source_phase, key, model_type, seed, mcfg
            )
            if run is None:
                raise LookupError(
                    f"{name}: no finished {m.source_experiment} run of {model_type} seed {seed} "
                    f"with the current config for dataset {key}"
                )
            got = load_from_run(run, model_type, mcfg, seed, ds, eval_cfg, root, tmp / str(seed))
            preds_val = got.predictor.predict(val)
            scores[seed] = validation_score(val, preds_val, eval_cfg, sel.horizon, sel.subset)
            loaded[seed] = (run, got, preds_val)
            log.info("deploy.seed", bundle=name, seed=seed, val_macro_f1=round(scores[seed], 4))
        seed = max(m.seeds, key=lambda s: (scores[s], -s))
        run, got, preds_val = loaded[seed]
        check = check_predictions(got.predictor.predict(test), got.stored_preds, ds.horizons)
        cal = fit_calibration(fam, val, preds_val, rcfg, eval_cfg, ds.ttf_cap_s)
        live = LiveCalibration.from_fitted(cal, ds.ttf_cap_s)
        live_d = live.to_dict()
        ref = reference_run(
            tracking_dir,
            m.calibration_reference,
            model_type,
            seed,
            source_run_id=str(run.info.run_id),
        )
        if ref is None:
            if strict:
                raise BundleCheckError(
                    f"{name}: no {m.calibration_reference.study} run of {model_type} seed {seed} "
                    f"for source run {run.info.run_id} in {m.calibration_reference.experiment}"
                )
            cal_check: dict[str, Any] = {"status": "no reference run (non-strict build)"}
        else:
            logged = _download_json(ref, "metrics.json", tmp)["calibration"]
            cal_check = {"status": "reproduced", "reference_run": ref.info.run_id}
            cal_check |= check_calibration(live_d, logged)
        model_dir = out / "model"
        got.predictor.save(model_dir)
        extra: dict[str, Any] = {}
        if fam == "tree":
            extra["tree_classes"] = got.predictor.present_classes(ds.split("train"))
    bundle = {
        "format": BUNDLE_FORMAT,
        "kind": "supervised",
        "name": name,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "type": model_type,
            "family": "xgb" if fam == "tree" else "nn",
            "seed": seed,
            "config": mcfg.model_dump(mode="json"),
            "source_run_id": str(run.info.run_id),
            "source_experiment": m.source_experiment,
            "files": sorted(p.name for p in model_dir.iterdir()),
            **extra,
        },
        "dataset": {
            "key": key,
            "content_sha256": ds.meta["source"]["content_sha256"],
            "window": ds.window,
            "horizons": list(ds.horizons),
            "ttf_cap_s": ds.ttf_cap_s,
            "fault_types": ds.fault_types,
            "health_classes": ds.meta["health_classes"],
            "normalizer": ds.meta["normalizer"],
        },
        "seed_selection": {
            "metric": f"val h{sel.horizon} {sel.subset} macro_f1",
            "scores": {str(s): v for s, v in scores.items()},
            "chosen": seed,
        },
        "calibration": live_d,
        "checks": {"test_predictions": check, "calibration": cal_check},
        "sustain_s": eval_cfg.early_warning.sustain_s,
        "operating_subset": [s.value for s in eval_cfg.subsets.operating],
        "card": card(cfg),
    }
    (out / BUNDLE_FILE).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def build_anomaly(
    cfg: DeployConfig, root: Path, tracking_dir: Path, out: Path, strict: bool
) -> dict[str, Any]:
    """Write the autoencoder bundle to ``out``; returns ``bundle.json``."""
    from twin_ml.config import load_anomaly_config, load_dataset_config, load_e5_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.models.anomaly import AnomalyThresholds, make_detector, normal_windows

    a = cfg.anomaly
    e5 = load_e5_config(root / cfg.e5_config)
    acfg = load_anomaly_config(root / a.config_file)
    ds = Dataset.open(resolve_dataset_dir(load_dataset_config(root / a.dataset_config), root))
    horizon = e5.primary.horizon
    no_fault = list(ds.fault_types).index("none")
    np.random.seed(a.seed)
    det = make_detector(acfg, no_fault, horizon, e5.anomaly.normal_states)
    train, val = ds.split("train"), ds.split("val")
    fit_info = det.fit(train, val, a.seed)
    vidx = normal_windows(val, no_fault, horizon, e5.anomaly.normal_states)
    thr = AnomalyThresholds(list(e5.anomaly.threshold_percentiles), e5.anomaly.per_kind).fit(
        det.score(val, vidx), val.at_end("kind", vidx)
    )
    described = thr.describe()
    live_thr = {str(k): [float(x) for x in v] for k, v in described["values"].items()}
    with tempfile.TemporaryDirectory(prefix="twin_p11_ae_") as tmp_s:
        ref = reference_run(tracking_dir, a.reference, acfg.type, a.seed)
        if ref is None:
            if strict:
                raise BundleCheckError(
                    f"no {a.reference.study} {acfg.type} seed {a.seed} run in "
                    f"{a.reference.experiment}"
                )
            thr_check: dict[str, Any] = {"status": "no reference run (non-strict build)"}
        else:
            logged = _download_json(ref, "metrics.json", Path(tmp_s))["detector"]["thresholds"]
            thr_check = {"status": "reproduced", "reference_run": ref.info.run_id}
            thr_check |= check_thresholds(live_thr, logged)
    det.save(out / "model" / "ae.pt")
    bundle = {
        "format": BUNDLE_FORMAT,
        "kind": "anomaly",
        "name": ANOMALY_NAME,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "type": acfg.type,
            "family": "nn",
            "seed": a.seed,
            "config": acfg.model_dump(mode="json"),
            "files": ["ae.pt"],
            "fit": {k: v for k, v in fit_info.items() if k != "history"},
        },
        "dataset": {
            "key": ds.meta["dataset_key"],
            "content_sha256": ds.meta["source"]["content_sha256"],
            "window": ds.window,
            "normalizer": ds.meta["normalizer"],
        },
        "anomaly": {
            "horizon": horizon,
            "normal_states": [s.value for s in e5.anomaly.normal_states],
            "per_kind": e5.anomaly.per_kind,
            "kinds": list(described["kinds"]),
            "thresholds": live_thr,
            "primary_percentile": e5.anomaly.primary_percentile,
            "n_cal": described["n_cal"],
        },
        "checks": {"thresholds": thr_check},
        "card": card(cfg),
    }
    (out / BUNDLE_FILE).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def build_and_register(
    name: str,
    cfg: DeployConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    strict: bool = True,
) -> BuildResult:
    """Build bundle ``name`` (a model of ``cfg.models`` or ``autoencoder``), log it to MLflow
    and register a new version. Runs in the calling process: call it from a fresh process."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from twin_ml.config import load_eval_config
    from twin_ml.train import tracking

    started = time.perf_counter()
    tdir = tracking_dir or root / load_eval_config(root / cfg.eval_config).tracking_dir
    with tempfile.TemporaryDirectory(prefix="twin_p11_bundle_") as tmp_s:
        out = Path(tmp_s) / "bundle"
        out.mkdir()
        if name == ANOMALY_NAME:
            bundle = build_anomaly(cfg, root, tdir, out, strict)
        elif name in cfg.models:
            bundle = build_supervised(name, cfg, root, tdir, out, strict)
        else:
            raise KeyError(f"unknown bundle {name!r}: {[*cfg.models, ANOMALY_NAME]}")
        tracking.configure(tdir, cfg.experiment)
        registered = cfg.registry_prefix + name
        with mlflow.start_run(run_name=f"bundle-{name}") as run:
            mlflow.set_tags(
                {
                    "phase": cfg.phase_tag,
                    "bundle": name,
                    "model": bundle["model"]["type"],
                    "seed": str(bundle["model"]["seed"]),
                    "dataset_key": bundle["dataset"]["key"],
                    "strict": str(strict),
                }
            )
            mlflow.log_artifacts(str(out), "bundle")
            run_id = run.info.run_id
            artifact_uri = run.info.artifact_uri
        client = MlflowClient(tracking_uri=tracking.tracking_uri(tdir))
        if not _registered(client, registered):
            client.create_registered_model(registered)
        mv = client.create_model_version(
            registered,
            f"{artifact_uri}/bundle",
            run_id=run_id,
            tags={"seed": str(bundle["model"]["seed"]), "strict": str(strict)},
        )
    summary = {
        "seed": bundle["model"]["seed"],
        "checks": bundle["checks"],
        "seed_selection": bundle.get("seed_selection"),
    }
    return BuildResult(name, run_id, str(mv.version), time.perf_counter() - started, summary)


# ---------------------------------------------------------------------------- load


@dataclass
class ResolvedBundle:
    name: str  # registered model name
    version: str
    run_id: str
    path: Path  # local directory with bundle.json
    meta: dict[str, Any]

    @property
    def label(self) -> str:
        """Short model version string, e.g. ``linetwin-gru v3 (gru seed 1)``."""
        m = self.meta["model"]
        return f"{self.name} v{self.version} ({m['type']} seed {m['seed']})"


def resolve(registered: str, version: str, tracking_dir: Path, dest: Path) -> ResolvedBundle:
    """Download a registered bundle version (``latest`` = highest version) to ``dest``."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from twin_ml.train import tracking

    uri = tracking.tracking_uri(tracking_dir)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if not _registered(client, registered):
        raise LookupError(
            f"no registered model {registered!r} in {tracking_dir}; build it with "
            "uv run poe deploy-models"
        )
    if version == "latest":
        versions = client.search_model_versions(f"name='{registered}'")
        if not versions:
            raise LookupError(f"{registered!r} has no versions")
        mv = max(versions, key=lambda v: int(v.version))
    else:
        mv = client.get_model_version(registered, version)
    mlflow.set_tracking_uri(uri)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    local = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{mv.run_id}/bundle", dst_path=str(dest)
        )
    )
    meta = json.loads((local / BUNDLE_FILE).read_text(encoding="utf-8"))
    if meta.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{registered} v{mv.version}: bundle format {meta.get('format')}")
    return ResolvedBundle(registered, str(mv.version), str(mv.run_id), local, meta)
