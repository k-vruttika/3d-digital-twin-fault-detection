"""Phase 10 explanations on real data (exploratory, ADR 0027 § Exploratory; Part C).

Study ``explain``, one worker process per (study, model, seed), logged to ``P10_real``:

* **Azure PdM**: the validation-chosen best supervised model and XGBoost (TreeSHAP; the tree
  reference) are restored from their study runs (XGBoost / sequence models from the saved
  weights, Random Forests retrained from the seed) and must reproduce their stored test
  predictions. Attributions (``twin_ml.explain.attribute``: TreeSHAP, Saabas or Integrated
  Gradients, toward the predicted alarm class) are taken on the alarmed windows inside the
  48 h lookback of every warned, evaluable test event, one per ``window_stride`` rows, at most
  ``max_windows_per_episode`` per event. The event's top sensor is the argmax of its mean
  normalised attribution. The **expected sensor per component** is derived from the training
  split only: the sensor whose mean normalised value over the 24 h before the component's
  training failures deviates most from the same machines' normal rows. Agreement = the top
  sensor is the expected sensor (a two-component event agrees with either). Chance level =
  1 / sensors (2 / sensors for two-component events).
* **MetroPT-3**: the autoencoder is re-fitted from its seed (its weights are not stored) and
  must reproduce the study run's stored sustained alarms at the primary percentile bit for
  bit. Per failure, the per-sensor shares of the reconstruction error on alarmed windows in
  the 6 h lookback ("before") and inside the reported failure interval ("during"), one window
  per ``metropt_window_stride`` rows, at most ``max_windows_per_episode``. They are compared
  with the PDF-derived air-leak-plausible signals and, as the reference, with the shares on
  negative (normal) test windows. Error shares are not attributions in a strict sense.

Parent: :func:`run_explain_study` (``python -m twin_ml.experiments.real --explain``). Report:
:func:`collect_explain`, :func:`compute_explain`, :func:`render_explain`, :func:`plot_explain`.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.config import EvalConfig
from twin_ml.datasets.config import RealConfig

EXPLAIN_STUDY = "explain"
NEG_REFERENCE_WINDOWS = 2000  # normal test windows (evenly spaced) for the share reference


# ---------------------------------------------------------------------------- targets


def explain_targets(
    real: RealConfig, rcfg: Any, eval_cfg: Any, root: Path, experiment: str
) -> tuple[list[tuple[str, str, int]], dict[tuple[str, str, int], Any], str | None]:
    """(study, model, seed) to explain, their source runs, and the best supervised model."""
    from twin_common.config import ConfigError
    from twin_ml.experiments.real import best_supervised, study_runs

    seeds = list(real.seeds)
    targets: list[tuple[str, str, int]] = []
    sources: dict[tuple[str, str, int], Any] = {}
    best = None
    try:
        az, _ = study_runs("azure", real, rcfg, eval_cfg, root, experiment, seeds)
    except (FileNotFoundError, ConfigError):
        az = {}
    if az:
        best = best_supervised(az, real, seeds)
        models = [m for m in dict.fromkeys([best, "xgb"]) if m is not None]
        for m in models:
            for s in seeds:
                if (m, s) in az:
                    targets.append(("azure", m, s))
                    sources[("azure", m, s)] = az[(m, s)]
    try:
        mp, _ = study_runs("metropt3", real, rcfg, eval_cfg, root, experiment, seeds)
    except (FileNotFoundError, ConfigError):
        mp = {}
    for s in seeds:
        if ("autoencoder", s) in mp:
            targets.append(("metropt3", "autoencoder", s))
            sources[("metropt3", "autoencoder", s)] = mp[("autoencoder", s)]
    return targets, sources, best


def explain_sha(study: str, model: str, seed: int, source_run: str, real: RealConfig) -> str:
    from twin_ml.experiments.real_worker import EVAL_VERSION

    data = {
        "version": EVAL_VERSION,
        "study": EXPLAIN_STUDY,
        "source_study": study,
        "model": model,
        "seed": seed,
        "source_run": source_run,
        "explain": real.exploratory.explain.model_dump(mode="json"),
        "anomaly": real.anomaly.model_dump(mode="json"),
        "neg_reference": NEG_REFERENCE_WINDOWS,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def run_explain_study(
    real: RealConfig,
    rcfg: Any,
    eval_cfg: Any,
    root: Path,
    experiment: str,
    seeds: list[int],
    skip_existing: bool,
    dry_run: bool,
) -> int:
    from twin_ml.experiments.real import _write
    from twin_ml.experiments.real_worker import (
        RealProcessError,
        model_config,
        run_real_isolated,
    )
    from twin_ml.train import tracking

    tdir = root / eval_cfg.tracking_dir
    targets, sources, best = explain_targets(real, rcfg, eval_cfg, root, experiment)
    targets = [t for t in targets if t[2] in seeds]
    done: set[str] = set()
    if skip_existing:
        for r in tracking.finished_runs(
            tdir, experiment, None, real.phase_tag, f"tags.study = '{EXPLAIN_STUDY}'"
        ):
            done.add(r.data.tags.get("study_config_sha", ""))
    todo = [t for t in targets if explain_sha(*t, sources[t].info.run_id, real) not in done]
    progress = root / real.progress_dir / f"p10_{EXPLAIN_STUDY}.jsonl"
    print(
        f"explain: best supervised (validation) = {best}; {len(todo)} runs "
        f"({len(targets) - len(todo)} done); experiment {experiment}",
        flush=True,
    )
    if dry_run:
        for st, m, s in todo:
            print(f"  {st} {m} seed {s}")
        return 0
    started = time.perf_counter()
    for st, m, s in todo:
        _write(progress, event="start", study=st, model=m, seed=s)
        t0 = time.perf_counter()
        try:
            res = run_real_isolated(
                st,
                m,
                model_config(real, root, m),
                s,
                real,
                eval_cfg,
                root,
                None,
                progress,
                experiment,
                task="explain",
            )
        except RealProcessError as exc:
            _write(progress, event="failed", study=st, model=m, seed=s, error=str(exc))
            print(f"error: {exc}; resume with --skip-existing", file=sys.stderr)
            return 1
        sec = time.perf_counter() - t0
        _write(
            progress,
            event="finished",
            study=st,
            model=m,
            seed=s,
            run_id=res["run_id"],
            seconds=round(sec),
        )
        print(f"explain {st} {m} seed {s}: run {res['run_id']} in {sec:.0f} s", flush=True)
    print(f"explain done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------- worker


def expected_sensors(train: Any, meta: dict[str, Any], fault_types: list[str]) -> dict[str, Any]:
    """Per component: the sensor whose mean normalised value over the warning band before the
    component's training failures deviates most from the same machines' normal rows."""
    from twin_ml.experiments.real_eval import RUNNING_STATE

    a = train.arrays
    band = int(meta["config"]["labels"]["warning_band"])
    step = float(train.step_s)
    sensors = list(train.schema.sensors)
    machine_of_row = np.asarray(a["series_machine"])[a["series"]]
    normal = (a["state"] == RUNNING_STATE) & (a["fault_type"] == fault_types.index("none"))
    obs = ~a["mask"].astype(bool)
    vals = np.where(obs, a["values"][:, : len(sensors)], np.nan)
    devs: dict[str, list[np.ndarray]] = {}
    normal_mean: dict[str, np.ndarray] = {}
    for e in meta["events"]["train"]:
        m = str(e["machine_id"])
        if m not in normal_mean:
            sel = (machine_of_row == m) & normal
            normal_mean[m] = (
                np.nanmean(vals[sel], axis=0) if sel.any() else np.full(len(sensors), np.nan)
            )
        rows = (
            (machine_of_row == m)
            & (a["t_s"] >= e["start_s"] - band * step)
            & (a["t_s"] < e["start_s"])
        )
        if not rows.any():
            continue
        d = np.nanmean(vals[rows], axis=0) - normal_mean[m]
        comps = list(e.get("components") or []) or [str(e["fault_type"])]
        if len(comps) > 1:
            continue  # two-component events do not define either component's signature
        devs.setdefault(comps[0], []).append(d)
    out: dict[str, Any] = {"sensors": sensors, "band_rows": band, "components": {}}
    for c, ds in sorted(devs.items()):
        mean = np.nanmean(np.stack(ds), axis=0)
        out["components"][c] = {
            "events": len(ds),
            "mean_deviation": [None if not math.isfinite(x) else float(x) for x in mean],
            "expected": sensors[int(np.nanargmax(np.abs(mean)))],
        }
    return out


def _load_supervised(
    source: Any, model: str, cfg: Any, seed: int, ds: Any, eval_cfg: Any, root: Path, tmp: Path
) -> Any:
    from twin_ml.experiments.robust_worker import _download, check_predictions
    from twin_ml.train.trainer import make_predictor

    uri = source.info.artifact_uri
    with np.load(_download(uri, "test_predictions.npz", tmp)) as z:
        stored = {k: z[k] for k in z.files}
    predictor: Any = make_predictor(cfg, ds, root, eval_cfg)
    if model == "xgb":
        predictor.load(_download(uri, "model", tmp), ds.split("train"))
    elif model in ("lstm", "gru", "transformer"):
        predictor.load(_download(uri, "model/model.pt", tmp), ds.horizons, ds.window, ds.schema)
    else:  # rf: retrained from its seed, as in the study run
        np.random.seed(seed)
        predictor.fit(ds.split("train"), ds.split("val"), seed)
    check = check_predictions(predictor.predict(ds.split("test")), stored, ds.horizons)
    return predictor, check


def run_real_explain(
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
) -> dict[str, Any]:
    import mlflow

    from twin_ml.config import load_robustness_config
    from twin_ml.experiments.real_worker import _jsonable, process_info
    from twin_ml.train import tracking

    started = time.perf_counter()
    exp = experiment or real.experiment
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    rcfg = load_robustness_config(root / real.robustness_config)
    _, sources, best = explain_targets(real, rcfg, eval_cfg, root, exp)
    source = sources.get((study, model, seed))
    if source is None:
        raise LookupError(f"no finished {study} run of {model} seed {seed} with the current config")
    with tempfile.TemporaryDirectory(prefix="twin_p10_ex_") as tmp_s:
        tmp = Path(tmp_s)
        if study == "metropt3":
            result = _explain_metropt(source, model, cfg, seed, real, eval_cfg, root, tmp)
        else:
            result = _explain_azure(source, model, cfg, seed, real, eval_cfg, root, tmp)
        total_s = time.perf_counter() - started
        result |= {
            "study": EXPLAIN_STUDY,
            "source_study": study,
            "model": model,
            "seed": seed,
            "best_supervised": best,
            "source_run": source.info.run_id,
            "total_s": total_s,
            "process": process_info(),
        }
        sha = explain_sha(study, model, seed, source.info.run_id, real)
        tracking.configure(tdir, exp)
        with mlflow.start_run(run_name=f"explain-{study}-{model}-seed{seed}") as run:
            mlflow.set_tags(
                {
                    "phase": real.phase_tag,
                    "study": EXPLAIN_STUDY,
                    "source_study": study,
                    "model": model,
                    "seed": str(seed),
                    "source_run_id": source.info.run_id,
                    "study_config_sha": sha,
                }
            )
            metrics = {
                k: float(v)
                for k, v in result.get("metrics", {}).items()
                if v is not None and math.isfinite(v)
            }
            metrics["time.total_s"] = total_s
            mlflow.log_metrics(metrics)
            mlflow.log_dict(json.loads(json.dumps(result, default=_jsonable)), "metrics.json")
            run_id = run.info.run_id
    return {"run_id": run_id, "study": study, "model": model, "seed": seed, "total_s": total_s}


def _event_windows(
    index: Any,
    sus: Any,
    t_lo: float,
    t_hi: float,
    machine: str,
    stride_s: float,
    cap: int,
    inclusive: bool = False,
) -> Any:
    from twin_ml.experiments.explain_eval import thin

    w_idx, w_t = index.win[machine]
    sel = (w_t >= t_lo) & ((w_t <= t_hi) if inclusive else (w_t < t_hi)) & sus[w_idx]
    cand = w_idx[sel]
    if not cand.size:
        return cand
    t_end = np.zeros(int(w_idx.max()) + 1 if w_idx.size else 0)
    t_end[w_idx] = w_t
    return thin(cand, t_end, int(stride_s), cap)


def _explain_azure(
    source: Any,
    model: str,
    cfg: Any,
    seed: int,
    real: RealConfig,
    eval_cfg: Any,
    root: Path,
    tmp: Path,
) -> dict[str, Any]:
    from twin_ml.data.dataset import Dataset
    from twin_ml.datasets.build import resolve_real_dataset
    from twin_ml.experiments import e5_eval, real_eval
    from twin_ml.experiments.e5_worker import load_alarms
    from twin_ml.explain import attribute
    from twin_ml.explain.aggregate import normalise

    st = real.datasets["azure_pdm"]
    ex = real.exploratory.explain
    ds = Dataset.open(resolve_real_dataset(st.dataset_config, root))
    fault_types = list(ds.fault_types)
    test = ds.split("test")
    predictor, check = _load_supervised(source, model, cfg, seed, ds, eval_cfg, root, tmp)
    alarms = load_alarms(source, tmp)
    sus = e5_eval.unpack_alarm(alarms[f"clean|h{st.horizon}"], test.n_windows)
    index = real_eval.SplitIndex.of(test, fault_types)
    events = real_eval.split_events(ds.meta, "test")
    split_lo, _ = real_eval.split_bounds(ds.meta, "test")
    lb = st.primary_lookback
    res = real_eval.event_results(index, sus, events, lb, split_lo, st.evaluable_min_coverage)
    exp = expected_sensors(ds.split("train"), ds.meta, fault_types)
    sensors = exp["sensors"]
    comps = {str(e["id"]): list(e.get("components") or []) for e in ds.meta["events"]["test"]}
    records = []
    step = float(test.step_s)
    for ev, r in zip(events, res, strict=True):
        if not (r.evaluable and r.warned):
            continue
        lo = ev.start_s - lb * step
        w = _event_windows(
            index,
            sus,
            lo,
            ev.start_s,
            ev.machine_id,
            ex.window_stride * step,
            ex.max_windows_per_episode,
        )
        if not w.size:
            continue
        values, mask, kind = test.gather(w)
        assert st.horizon is not None
        att = attribute(predictor, values, mask, kind, st.horizon, ex.ig_steps, schema=test.schema)
        weights = normalise(att.sensors)
        mean = weights.mean(axis=0)
        top = sensors[int(np.argmax(mean))]
        cs = comps.get(ev.id) or [ev.fault_type]
        want = [exp["components"][c]["expected"] for c in cs if c in exp["components"]]
        records.append(
            {
                "id": ev.id,
                "machine_id": ev.machine_id,
                "fault_type": ev.fault_type,
                "components": cs,
                "windows": int(w.size),
                "mean_weight": mean.tolist(),
                "top": top,
                "expected": want,
                "agrees": bool(want) and top in want,
                "completeness_error": None
                if att.completeness_error is None
                else float(np.mean(att.completeness_error)),
            }
        )
    by_type: dict[str, Any] = {}
    for ft in sorted({r["fault_type"] for r in records}):
        sel = [r for r in records if r["fault_type"] == ft]
        by_type[ft] = {
            "events": len(sel),
            "agree": sum(r["agrees"] for r in sel),
            "mean_weight": np.mean([r["mean_weight"] for r in sel], axis=0).tolist(),
            "top_counts": {s: sum(r["top"] == s for r in sel) for s in sensors},
        }
    n, k = len(records), sum(r["agrees"] for r in records)
    chance = (
        float(np.mean([min(len(r["expected"]), len(sensors)) / len(sensors) for r in records]))
        if records
        else float("nan")
    )
    return {
        "dataset": "azure_pdm",
        "method": {"xgb": "treeshap", "rf": "saabas"}.get(model, "integrated_gradients"),
        "prediction_check": check,
        "sensors": sensors,
        "expected": exp,
        "events_warned_evaluable": int(sum(r.evaluable and r.warned for r in res)),
        "events_explained": n,
        "records": records,
        "by_fault_type": by_type,
        "metrics": {
            "agreement_rate": k / n if n else float("nan"),
            "events_explained": n,
            "chance_rate": chance,
        },
    }


def _explain_metropt(
    source: Any,
    model: str,
    cfg: Any,
    seed: int,
    real: RealConfig,
    eval_cfg: Any,
    root: Path,
    tmp: Path,
) -> dict[str, Any]:
    from twin_ml.data.dataset import Dataset
    from twin_ml.datasets.build import resolve_real_dataset
    from twin_ml.experiments import e5_eval, real_eval
    from twin_ml.experiments.e5_worker import load_alarms
    from twin_ml.experiments.real_worker import fit_method, study_specs

    st = real.datasets["metropt3"]
    ex = real.exploratory.explain
    spec = study_specs(real)["metropt3"]
    ds = Dataset.open(resolve_real_dataset(st.dataset_config, root))
    fault_types = list(ds.fault_types)
    test = ds.split("test")
    method = fit_method(model, cfg, seed, ds, real, spec, eval_cfg, root)
    pct = real.primary.metropt3.anomaly_percentile
    alarm = method.alarms(test)[f"p{pct:g}"]
    sus = real_eval.sustain(test, alarm, st.sustain)
    stored = load_alarms(source, tmp)
    ref = e5_eval.unpack_alarm(stored[f"clean|p{pct:g}"], test.n_windows)
    if not np.array_equal(sus, ref):
        raise RuntimeError(
            f"the re-fitted autoencoder (seed {seed}) does not reproduce the stored alarms of run "
            f"{source.info.run_id} ({int((sus != ref).sum())} of {test.n_windows} windows differ)"
        )
    det = method.detector
    index = real_eval.SplitIndex.of(test, fault_types)
    sensors = list(test.schema.sensors)
    plausible = [s for s in ex.air_leak_plausible if s in sensors]
    pmask = np.array([s in plausible for s in sensors])
    step = float(test.step_s)
    lb = st.primary_lookback

    def shares(w: Any) -> Any:
        err = det.sensor_errors(test, w)
        return err / np.maximum(err.sum(axis=1, keepdims=True), 1e-12)

    neg = np.flatnonzero(index.negative)
    ref_w = (
        neg[
            np.round(np.linspace(0, neg.size - 1, min(NEG_REFERENCE_WINDOWS, neg.size))).astype(
                np.int64
            )
        ]
        if neg.size
        else neg
    )
    ref_sh = shares(ref_w) if ref_w.size else np.zeros((0, len(sensors)))
    reference = {
        "windows": int(ref_w.size),
        "mean_share": ref_sh.mean(axis=0).tolist() if ref_w.size else None,
        "plausible_mass": float(ref_sh[:, pmask].sum(axis=1).mean()) if ref_w.size else None,
        "top_plausible_frac": float(pmask[np.argmax(ref_sh, axis=1)].mean())
        if ref_w.size
        else None,
    }
    records = []
    for e in real_eval.split_events(ds.meta, "test"):
        rec: dict[str, Any] = {"id": e.id}
        for name, lo, hi, inc in (
            ("before", e.start_s - lb * step, e.start_s, False),
            ("during", e.start_s, e.end_s, True),
        ):
            w = _event_windows(
                index,
                sus,
                lo,
                hi,
                e.machine_id,
                ex.metropt_window_stride * step,
                ex.max_windows_per_episode,
                inc,
            )
            if not w.size:
                rec[name] = {"windows": 0}
                continue
            sh = shares(w)
            mean = sh.mean(axis=0)
            rec[name] = {
                "windows": int(w.size),
                "mean_share": mean.tolist(),
                "top": sensors[int(np.argmax(mean))],
                "top_plausible": bool(pmask[int(np.argmax(mean))]),
                "plausible_mass": float(mean[pmask].sum()),
            }
        records.append(rec)
    return {
        "dataset": "metropt3",
        "method": "autoencoder_error_shares",
        "alarms_reproduced": True,
        "percentile": pct,
        "sensors": sensors,
        "plausible": plausible,
        "reference_negative": reference,
        "records": records,
        "metrics": {
            "before_explained": sum(r["before"]["windows"] > 0 for r in records),
            "during_explained": sum(r["during"]["windows"] > 0 for r in records),
        },
    }


# ---------------------------------------------------------------------------- report


def collect_explain(
    real: RealConfig,
    rcfg: Any,
    eval_cfg: Any,
    tracking_dir: Path,
    root: Path,
    experiment: str,
    client: Any,
    tmp: Path,
) -> dict[str, Any]:
    from twin_ml.train import tracking

    targets, sources, best = explain_targets(real, rcfg, eval_cfg, root, experiment)
    want = {explain_sha(*t, sources[t].info.run_id, real): t for t in targets}
    out: dict[str, Any] = {"best_supervised": best, "runs": {}}
    for r in tracking.finished_runs(
        tracking_dir, experiment, None, real.phase_tag, f"tags.study = '{EXPLAIN_STUDY}'"
    ):
        t = want.get(r.data.tags.get("study_config_sha", ""))
        key = None if t is None else "|".join(str(x) for x in t)
        if key is None or key in out["runs"]:
            continue
        d = tmp / f"ex_{r.info.run_id}"
        d.mkdir(parents=True, exist_ok=True)
        path = Path(client.download_artifacts(r.info.run_id, "metrics.json", str(d)))
        out["runs"][key] = {
            "run_id": r.info.run_id,
            "detail": json.loads(path.read_text(encoding="utf-8")),
        }
    return out if out["runs"] else {}


def compute_explain(ex: dict[str, Any], real: RealConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_supervised": ex.get("best_supervised"),
        "azure": {},
        "metropt3": {},
    }
    for key, e in sorted(ex["runs"].items()):
        study, model, seed = key.split("|")
        d = e["detail"]
        if study == "azure":
            m = out["azure"].setdefault(
                model,
                {
                    "method": d["method"],
                    "seeds": {},
                    "expected": d["expected"],
                    "sensors": d["sensors"],
                },
            )
            m["seeds"][seed] = {
                "run_id": e["run_id"],
                "events_explained": d["events_explained"],
                "events_warned_evaluable": d["events_warned_evaluable"],
                "agreement_rate": d["metrics"]["agreement_rate"],
                "chance_rate": d["metrics"]["chance_rate"],
                "by_fault_type": d["by_fault_type"],
                "prediction_check": d["prediction_check"],
            }
        else:
            out["metropt3"][seed] = {
                "run_id": e["run_id"],
                "sensors": d["sensors"],
                "plausible": d["plausible"],
                "reference": d["reference_negative"],
                "records": d["records"],
            }
    return out


def render_explain(x: dict[str, Any], real: RealConfig) -> list[str]:
    from twin_ml.experiments.report_e1 import NA, _table

    def v(y: Any, d: int = 2) -> str:
        return NA if y is None or (isinstance(y, float) and not math.isfinite(y)) else f"{y:.{d}f}"

    out = ["## Explanations (exploratory)", ""]
    if x.get("azure"):
        first = next(iter(x["azure"].values()))
        exp = first["expected"]
        out += [
            "### Azure PdM: attributions by failure component",
            "",
            f"Expected sensor per component, derived from the **training split only**: the "
            f"sensor whose mean normalised value over the {exp['band_rows']} h before the "
            "component's training failures deviates most from the same machines' normal rows "
            "(two-component events excluded). No external claim is used.",
            "",
        ]
        out += _table(
            ["Component", "Training events", *[f"Δ {s}" for s in exp["sensors"]], "Expected"],
            [
                [c, str(d["events"]), *[v(y) for y in d["mean_deviation"]], f"`{d['expected']}`"]
                for c, d in exp["components"].items()
            ],
        )
        out += [
            "",
            "Agreement: the event's top sensor (mean normalised attribution over its "
            "alarmed lookback windows) is the expected sensor; chance level = 1 / sensors "
            "(two-component events: 2 / sensors).",
            "",
        ]
        rows = []
        for model, m in x["azure"].items():
            for s, sd in sorted(m["seeds"].items()):
                rows.append(
                    [
                        f"`{model}` ({m['method']})",
                        s,
                        f"{sd['events_explained']} / {sd['events_warned_evaluable']}",
                        v(sd["agreement_rate"], 3),
                        v(sd["chance_rate"], 3),
                        "; ".join(
                            f"{ft} {b['agree']}/{b['events']}"
                            for ft, b in sd["by_fault_type"].items()
                        ),
                    ]
                )
        out += _table(
            [
                "Model",
                "Seed",
                "Events explained / warned",
                "Agreement",
                "Chance",
                "By component (agree / events)",
            ],
            rows,
        )
        out += ["", "![Azure explanations](real_data_explain_azure.png)", ""]
    if x.get("metropt3"):
        sd0 = next(iter(x["metropt3"].values()))
        ref = sd0["reference"]
        out += [
            "### MetroPT-3: autoencoder error shares around each failure",
            "",
            "Per-sensor shares of the reconstruction error on alarmed windows in the 6 h before "
            "the reported start (*before*) and inside the reported interval (*during*), against "
            "the air-leak-plausible signals from the PDF semantics "
            f"({', '.join(sd0['plausible'])}; "
            f"{len(sd0['plausible'])} of {len(sd0['sensors'])} signals). Reference: the same "
            f"shares on {ref['windows']:,} evenly spaced normal test windows (seed 0: plausible "
            f"mass {v(ref['plausible_mass'], 3)}, top sensor plausible in "
            f"{v(ref['top_plausible_frac'], 3)} of windows). Error shares are not attributions "
            "in a strict sense.",
            "",
        ]
        rows = []
        for s, sd in sorted(x["metropt3"].items()):
            for r in sd["records"]:
                for name in ("before", "during"):
                    g = r[name]
                    rows.append(
                        [
                            s,
                            f"#{r['id']}",
                            name,
                            str(g["windows"]),
                            f"`{g['top']}`" if g["windows"] else NA,
                            ("yes" if g["top_plausible"] else "no") if g["windows"] else NA,
                            v(g.get("plausible_mass"), 3),
                        ]
                    )
        out += _table(
            [
                "Seed",
                "Failure",
                "Windows",
                "Alarmed windows",
                "Top sensor",
                "Top plausible",
                "Plausible mass",
            ],
            rows,
        )
        out += ["", "![MetroPT-3 error shares](real_data_explain_metropt3.png)", ""]
    return out


def plot_explain(x: dict[str, Any], out: Path, name: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from twin_ml.experiments.report_e1 import INK, INK2, SURFACE
    from twin_ml.experiments.report_real_results import _style
    from twin_ml.experiments.report_robustness import SERIES

    paths: list[Path] = []
    best = x.get("best_supervised")
    az = x.get("azure", {}).get(best) if best else None
    if az:
        sensors = az["sensors"]
        types = sorted({ft for sd in az["seeds"].values() for ft in sd["by_fault_type"]})
        fig, ax = plt.subplots(figsize=(9.5, 3.0), facecolor=SURFACE)
        _style(ax)
        w = 0.8 / len(sensors)
        for j, s in enumerate(sensors):
            vals, ns = [], []
            for ft in types:
                per = [
                    sd["by_fault_type"][ft]["mean_weight"][j]
                    for sd in az["seeds"].values()
                    if ft in sd["by_fault_type"]
                ]
                vals.append(float(np.mean(per)) if per else 0.0)
                ns.append(
                    next(
                        (
                            sd["by_fault_type"][ft]["events"]
                            for sd in az["seeds"].values()
                            if ft in sd["by_fault_type"]
                        ),
                        0,
                    )
                )
            ax.bar(
                np.arange(len(types)) + (j - (len(sensors) - 1) / 2) * w,
                vals,
                width=w,
                color=SERIES[j % len(SERIES)],
                label=s,
                lw=0,
            )
        ax.set_xticks(np.arange(len(types)))
        ax.set_xticklabels(
            [f"{ft}\n(n={n} events, seed 0)" for ft, n in zip(types, ns, strict=True)], fontsize=8
        )
        ax.set_ylabel("mean normalised attribution", color=INK2)
        ax.set_title(
            f"Azure PdM: {best} attributions by component (seed mean)",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.legend(fontsize=8, frameon=False, ncol=len(sensors))
        fig.tight_layout()
        p = out / f"{name}_explain_azure.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    mp = x.get("metropt3", {}).get("0")
    if mp:
        sensors = mp["sensors"]
        fig, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=SURFACE)
        _style(ax)
        series = []
        if mp["reference"]["mean_share"] is not None:
            series.append(("normal reference", mp["reference"]["mean_share"]))
        for r in mp["records"]:
            for nm in ("before", "during"):
                if r[nm]["windows"]:
                    series.append((f"#{r['id']} {nm} ({r[nm]['windows']} w)", r[nm]["mean_share"]))
        w = 0.8 / max(len(series), 1)
        for i, (lbl, vals) in enumerate(series):
            ax.bar(
                np.arange(len(sensors)) + (i - (len(series) - 1) / 2) * w,
                vals,
                width=w,
                color=SERIES[i % len(SERIES)],
                label=lbl,
                lw=0,
            )
        ax.set_xticks(np.arange(len(sensors)))
        ax.set_xticklabels(
            [f"{s}*" if s in mp["plausible"] else s for s in sensors],
            rotation=40,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("share of reconstruction error", color=INK2)
        ax.set_title(
            "MetroPT-3: autoencoder seed 0 error shares (* = air-leak plausible)",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.legend(fontsize=7, frameon=False, ncol=3)
        fig.tight_layout()
        p = out / f"{name}_explain_metropt3.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths
