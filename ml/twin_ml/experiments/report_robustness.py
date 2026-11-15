"""Phase 8 report: ``python -m twin_ml.experiments.report_robustness`` (poe ``report-robustness``).

Reads the finished study runs of the robustness experiment from MLflow (``metrics.json`` and
``runstats.npz`` of each run) and writes ``experiments/results/robustness.md``, ``.json`` and
plots. Every number comes from those runs; nothing is typed by hand.

Statistics (ADR 0021): a model with several seeds is compared seed by seed; a table shows the
mean over seeds of the point estimate and of the CI bounds, and in how many seeds the
individual CI excludes zero (k/n). Paired differences use the same resampled test runs on
both sides; the E4 shift effect (different run sets) resamples each side independently.
Sections are labelled PRIMARY (pre-registered in ADR 0021 / ``robustness.yaml``) or
EXPLORATORY.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twin_common.config import ConfigError
from twin_common.logging import configure_logging
from twin_ml.config import EvalConfig, RobustnessConfig
from twin_ml.data.labels import HEALTH
from twin_ml.evaluate import run_sums
from twin_ml.experiments.report_e1 import (
    ACCENT,
    AXIS,
    GRID,
    INK,
    INK2,
    LF,
    MUTED,
    NA,
    SURFACE,
    _clean,
    _git_commit,
    _table,
)
from twin_ml.train import tracking

# categorical palette (dataviz reference, light mode), fixed order by model
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
MODEL_ORDER = ("persistence", "threshold", "rf", "xgb", "lstm", "gru", "transformer")
EW_METRICS = ("miss_rate", "lead_mean_s", "rel_lead_mean_s", "false_alarms_per_h")
DEGRADATION_METRICS = ("macro_f1", "fault_recall", "miss_rate", "lead_mean_s", "false_alarms_per_h")
CURVE_METRICS = (
    ("macro_f1", "macro F1 (operating)"),
    ("fault_recall", "FAULT recall"),
    ("false_alarms_per_h", "false alarms per hour"),
    ("lead_mean_s", "mean lead time (s)"),
)
DIGITS = {
    "miss_rate": 3,
    "lead_mean_s": 0,
    "rel_lead_mean_s": 0,
    "false_alarms_per_h": 2,
    "macro_f1": 3,
    "fault_recall": 3,
    "ece_cal": 4,
    "ece_uncal": 4,
    "ece_fault_cal": 4,
    "ece_fault_uncal": 4,
    "sf_false_alarms_per_h": 2,
    "ttf_cov": 3,
    "ttf_width_s": 0,
    "ttf_certain_miss": 3,
}
EW_FIELDS = run_sums.RUNSTATS_FIELDS

Entry = dict[str, Any]
StudyRuns = dict[str, dict[int, Entry]]  # model -> seed -> entry


# ---------------------------------------------------------------------------- collect


def _load_sums(artifact_uri: str) -> tuple[dict[str, run_sums.Sums], dict[str, list[str]]]:
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{artifact_uri}/runstats.npz", dst_path=tmp
        )
        with np.load(Path(path), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    runs = {
        k[len("runs:") :]: [str(x) for x in v] for k, v in arrays.items() if k.startswith("runs:")
    }
    return run_sums.load_npz_dict(arrays), runs


def collect_study(
    tracking_dir: Path,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    study: str,
    models_dir: Path,
) -> StudyRuns:
    """Newest finished run per (model, seed) whose study config hash matches the *current*
    configs (model config, robustness and evaluation settings, dataset key, source run), so
    runs made with older settings never enter the report."""
    import mlflow

    from twin_ml.config import load_model_config
    from twin_ml.experiments.robust_worker import study_config_sha

    flt = f"tags.study = '{study}'"
    runs = tracking.finished_runs(tracking_dir, rcfg.experiment, None, rcfg.phase_tag, flt)
    cfgs: dict[str, Any] = {}
    chosen: dict[str, dict[int, Any]] = defaultdict(dict)
    source_kind = rcfg.studies[study].models_from
    for run in runs:  # newest first
        tags = run.data.tags
        model, seed = tags["model"], int(tags["seed"])
        if model not in cfgs:
            path = models_dir / f"{model}.yaml"
            cfgs[model] = load_model_config(path) if path.is_file() else None
        if cfgs[model] is None:
            continue
        source = tags.get("source_run_id", "") if source_kind == "e1" else "train"
        expected = study_config_sha(
            study, cfgs[model], rcfg, eval_cfg, tags.get("dataset_key", ""), source
        )
        if tags.get("study_config_sha") == expected:
            chosen[model].setdefault(seed, run)
    out: StudyRuns = {}
    for model in sorted(chosen, key=_model_key):
        out[model] = {}
        for seed, run in sorted(chosen[model].items()):
            sums, run_ids = _load_sums(run.info.artifact_uri)
            out[model][seed] = {
                "run_id": run.info.run_id,
                "study_config_sha": run.data.tags.get("study_config_sha"),
                "source_run_id": run.data.tags.get("source_run_id"),
                "dataset_key": run.data.tags.get("dataset_key"),
                "detail": mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/metrics.json"),
                "sums": sums,
                "run_ids": run_ids,
            }
    return out


def _model_key(m: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m)


# ---------------------------------------------------------------------------- statistics


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def stat(seeds: dict[int, Entry], cond: str, h: str, metric: str) -> dict[str, Any]:
    """Mean over seeds of the per-seed bootstrap point / lo / hi (+ std of the point)."""
    pts, los, his = [], [], []
    for e in seeds.values():
        block = e["detail"]["conditions"].get(cond, {}).get("robust", {}).get(h)
        if block is None or metric not in block["bootstrap"]:
            continue
        b = block["bootstrap"][metric]
        pts.append(b["point"])
        los.append(b["lo"])
        his.append(b["hi"])
    finite = [p for p in pts if p is not None and math.isfinite(p)]
    return {
        "point": _finite_mean(pts),
        "lo": _finite_mean(los),
        "hi": _finite_mean(his),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "seeds": len(pts),
    }


def _fields(sums: run_sums.Sums, fields: tuple[str, ...] | None) -> run_sums.Sums:
    if fields is None:
        return sums
    return {k: v for k, v in sums.items() if k in fields or any(k.startswith(f) for f in fields)}


def compare(
    a: dict[int, Entry],
    a_cond: str,
    b: dict[int, Entry],
    b_cond: str,
    h: str,
    metrics: tuple[str, ...],
    eval_cfg: EvalConfig,
    *,
    fields: tuple[str, ...] | None = None,
    mode: str = "paired",
    a_runs: list[str] | None = None,
    b_runs: list[str] | None = None,
) -> dict[str, Any]:
    """Seed-wise difference a - b, averaged over seeds (a deterministic side is reused).

    ``a_runs`` / ``b_runs``: restrict each side to these test runs (in this order), e.g. the
    test runs shared by two splits.
    """
    if len(b) == 1:
        (sb,) = b.values()
        pairs = [(sa, sb) for sa in a.values()]
    elif len(a) == 1:
        (sa,) = a.values()
        pairs = [(sa, sb) for sb in b.values()]
    else:
        pairs = [(a[s], b[s]) for s in sorted(set(a) & set(b))]
    bs = eval_cfg.bootstrap
    per_seed = []
    for ea, eb in pairs:
        ka, kb = f"{a_cond}|h{h}", f"{b_cond}|h{h}"
        if ka not in ea["sums"] or kb not in eb["sums"]:
            continue
        sa = _fields(ea["sums"][ka], fields)
        sb = _fields(eb["sums"][kb], fields)
        if a_runs is not None:
            sa = run_sums.select(sa, [ea["run_ids"][a_cond].index(r) for r in a_runs])
        if b_runs is not None:
            sb = run_sums.select(sb, [eb["run_ids"][b_cond].index(r) for r in b_runs])
        if mode == "paired":
            if a_runs is None and b_runs is None and ea["run_ids"][a_cond] != eb["run_ids"][b_cond]:
                raise ValueError(f"paired comparison over different runs: {a_cond} vs {b_cond}")
            per_seed.append(run_sums.paired(sa, sb, bs.n_resamples, bs.seed, bs.ci, metrics))
        else:
            per_seed.append(run_sums.unpaired(sa, sb, bs.n_resamples, bs.seed, bs.ci, metrics))
    out: dict[str, Any] = {}
    for m in metrics:
        vals = {k: _finite_mean([r[m][k] for r in per_seed]) for k in ("point", "lo", "hi")}
        lo, hi = vals["lo"], vals["hi"]
        out[m] = {
            **vals,
            "excludes_zero": bool(math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0)),
            "seeds_excluding_zero": int(sum(bool(r[m]["excludes_zero"]) for r in per_seed)),
            "seeds": len(per_seed),
        }
    return out


def counts(seeds: dict[int, Entry], cond: str, h: str) -> dict[str, Any]:
    """Episodes, runs and (operating) windows per true class behind a condition's figures."""
    e = next(iter(seeds.values()))
    d = e["detail"]["conditions"].get(cond)
    if d is None:
        return {}
    ew = d["horizons"][h]["early_warning"]
    sums = e["sums"].get(f"{cond}|h{h}", {})
    conf = next((v for k, v in sums.items() if k.startswith("conf_")), None)
    per_class = (
        {cls.value: int(conf.sum(axis=0)[c, 0]) for c, cls in enumerate(HEALTH)}
        if conf is not None
        else {}
    )
    return {
        "episodes": ew["episodes"],
        "episodes_per_kind": ew["episodes_per_kind"],
        "runs": d["n_runs"],
        "negative_h": ew["negative_h"],
        "windows_per_class": per_class,
        "sensor_fault_windows": d.get("sensor_fault_windows"),
    }


# ---------------------------------------------------------------------------- build


def condition_study(rcfg: RobustnessConfig, cond: str) -> str | None:
    """The first study (config order) that evaluates ``cond`` (the main models' study)."""
    from twin_ml.experiments.robust_worker import expand_conditions

    for name, st in rcfg.studies.items():
        if cond in expand_conditions(st, rcfg):
            return name
    return None


def _levels(rcfg: RobustnessConfig) -> dict[str, list[tuple[int, str]]]:
    from twin_ml.config import PERTURBATION_KINDS
    from twin_ml.data.perturb import condition_name

    return {
        k: [(0, "clean")]
        + [(lv, condition_name(k, lv)) for lv in rcfg.perturbations.grid(k).levels_pct]
        for k in PERTURBATION_KINDS
    }


def study_levels(rcfg: RobustnessConfig, st: dict[str, Any]) -> dict[str, list[tuple[int, str]]]:
    """Per perturbation type, the (level, condition) pairs the study evaluated (clean = 0)."""
    have = set(st["conditions"])
    out = {}
    for kind, lv in _levels(rcfg).items():
        kept = [(pct, c) for pct, c in lv if c in have]
        if len(kept) > 1:
            out[kind] = kept
    return out


def primary_block(
    rcfg: RobustnessConfig, eval_cfg: EvalConfig, studies: dict[str, StudyRuns]
) -> dict[str, Any]:
    p = rcfg.primary
    h = str(p.horizon)
    out: dict[str, Any] = {}
    for cond in p.conditions:
        name = condition_study(rcfg, cond)
        runs = studies.get(name or "", {})
        if not runs or p.reference not in runs:
            out[cond] = {"study": name, "available": False}
            continue
        ref = runs[p.reference]
        block: dict[str, Any] = {"study": name, "available": True, "counts": counts(ref, cond, h)}
        block["values"] = {
            m: {
                k: stat(runs[m], cond, h, k)
                for k in (*p.early_warning, *p.calibration, *p.conformal, *EXTRA_CALIBRATION)
            }
            for m in runs
        }
        block["vs_reference"] = {
            m: compare(
                runs[m], cond, ref, cond, h, tuple(p.early_warning), eval_cfg, fields=EW_FIELDS
            )
            for m in runs
            if m != p.reference
        }
        out[cond] = block
    return out


def study_block(
    name: str,
    runs: StudyRuns,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    """Exploratory results of a study with a clean condition (e1models, aug)."""
    from twin_ml.experiments.robust_worker import expand_conditions

    conds = expand_conditions(rcfg.studies[name], rcfg)
    horizons = (
        sorted(
            {
                h
                for s in runs.values()
                for e in s.values()
                for h in e["detail"]["conditions"]["clean"]["robust"]
            },
            key=int,
        )
        if runs
        else []
    )
    out: dict[str, Any] = {"conditions": conds, "horizons": horizons, "models": list(runs)}
    any_seeds = next(iter(runs.values())) if runs else {}
    out["counts"] = {h: {c: counts(any_seeds, c, h) for c in conds} for h in horizons}
    metric_names = sorted(
        {
            k
            for s in runs.values()
            for e in s.values()
            for c in conds
            for k in e["detail"]["conditions"]
            .get(c, {})
            .get("robust", {})
            .get(horizons[-1] if horizons else "", {})
            .get("bootstrap", {})
        }
    )
    out["values"] = {
        h: {m: {c: {k: stat(runs[m], c, h, k) for k in metric_names} for c in conds} for m in runs}
        for h in horizons
    }
    out["degradation"] = {
        h: {
            m: {
                c: compare(
                    runs[m], c, runs[m], "clean", h, DEGRADATION_METRICS, eval_cfg, fields=EW_FIELDS
                )
                for c in conds
                if c != "clean"
            }
            for m in runs
        }
        for h in horizons[-1:]  # the primary horizon only (exploratory)
    }
    out["calibration_methods"] = {
        m: {str(s): e["detail"]["calibration"]["heads"] for s, e in seeds.items()}
        for m, seeds in runs.items()
    }
    out["reliability"] = {
        h: {
            m: {
                c: {
                    k: _pooled_bins(seeds, c, h, k)
                    for k in ("ece_uncal", "ece_cal", "ece_fault_uncal", "ece_fault_cal")
                }
                for c in rcfg.reliability_conditions
                if c in conds
            }
            for m, seeds in runs.items()
        }
        for h in horizons
    }
    out["timings_s"] = {
        m: _finite_mean([e["detail"]["total_s"] for e in seeds.values()])
        for m, seeds in runs.items()
    }
    out["runs"] = {m: {str(s): e["run_id"] for s, e in seeds.items()} for m, seeds in runs.items()}
    out["prediction_checks"] = {
        m: {str(s): e["detail"].get("prediction_check") for s, e in seeds.items()}
        for m, seeds in runs.items()
    }
    return out


def _pooled_bins(seeds: dict[int, Entry], cond: str, h: str, key: str) -> list[list[float]]:
    total = None
    for e in seeds.values():
        s = e["sums"].get(f"{cond}|h{h}", {}).get(key)
        if s is None:
            continue
        v = s.sum(axis=0)
        total = v if total is None else total + v
    return [] if total is None else total.tolist()


SHIFT_METRICS = (
    *DEGRADATION_METRICS,
    "rel_lead_mean_s",
    "ece_cal",
    "ece_fault_cal",
    "cov_lac_FAULT",
    "cov_aps_FAULT",
    "cov_mondrian_lac_FAULT",
)


def _horizons_of(runs: StudyRuns, cond: str) -> list[str]:
    hs = {
        h
        for seeds in runs.values()
        for e in seeds.values()
        for h in e["detail"]["conditions"].get(cond, {}).get("robust", {})
    }
    return sorted(hs, key=int)


def e4_block(
    studies: dict[str, StudyRuns], rcfg: RobustnessConfig, eval_cfg: EvalConfig
) -> dict[str, Any] | None:
    """E4: E4 models on high load vs low + medium (unpaired), and E1 vs E4 on shared runs."""
    runs = studies.get("e4") or {}
    if not runs:
        return None
    conds = ("e4_high", "e4_in")
    hs = _horizons_of(runs, "e4_high")
    first = next(iter(next(iter(runs.values())).values()))
    out: dict[str, Any] = {
        "models": list(runs),
        "horizons": hs,
        "dataset": first["detail"]["dataset"],
        "counts": {h: {c: counts(next(iter(runs.values())), c, h) for c in conds} for h in hs},
        "values": {
            h: {
                m: {c: {k: stat(runs[m], c, h, k) for k in SHIFT_METRICS} for c in conds}
                for m in runs
            }
            for h in hs
        },
        "shift": {
            h: {
                m: compare(
                    runs[m],
                    "e4_high",
                    runs[m],
                    "e4_in",
                    h,
                    SHIFT_METRICS,
                    eval_cfg,
                    mode="unpaired",
                )
                for m in runs
            }
            for h in hs
        },
        "calibration_methods": {
            m: {str(s): e["detail"]["calibration"]["heads"] for s, e in seeds.items()}
            for m, seeds in runs.items()
        },
        "runs": {m: {str(s): e["run_id"] for s, e in seeds.items()} for m, seeds in runs.items()},
        "secondary": {},
    }
    base = studies.get("e1models") or {}
    if base:
        b_first = next(iter(next(iter(base.values())).values()))
        shared = sorted(set(first["run_ids"]["e4_high"]) & set(b_first["run_ids"]["clean"]))
        ref = runs.get(rcfg.primary.reference) or next(iter(runs.values()))
        e = next(iter(ref.values()))
        idx = [e["run_ids"]["e4_high"].index(r) for r in shared]
        h = str(rcfg.primary.horizon)
        episodes = float(e["sums"][f"e4_high|h{h}"]["episodes"][idx].sum()) if idx else 0.0
        out["secondary"] = {
            "shared_runs": shared,
            "episodes": episodes,
            "diff": {
                hh: {
                    m: compare(
                        runs[m],
                        "e4_high",
                        base[m],
                        "clean",
                        hh,
                        SHIFT_METRICS,
                        eval_cfg,
                        a_runs=shared,
                        b_runs=shared,
                    )
                    for m in runs
                    if m in base and shared
                }
                for hh in hs
            },
        }
    return out


def seen_status(rcfg: RobustnessConfig, cond: str) -> str:
    """seen / unseen / mixed: whether the condition's perturbation types were in augmentation."""
    seen = set(rcfg.seen_in_augmentation)
    if cond in rcfg.sensor_fault_sets:
        types = set(rcfg.sensor_fault_sets[cond].types)
        if types <= seen:
            return "seen"
        return "unseen" if not types & seen else "mixed"
    if cond == "clean":
        return "clean"
    return "seen" if cond.split("_", 1)[0] in seen else "unseen"


def ablation_block(
    studies: dict[str, StudyRuns], rcfg: RobustnessConfig, eval_cfg: EvalConfig
) -> dict[str, Any] | None:
    """Augmentation-trained minus clean-trained, paired on the same test runs, per condition."""
    aug, base = studies.get("aug") or {}, studies.get("e1models") or {}
    if not aug or not base:
        return None
    from twin_ml.experiments.robust_worker import expand_conditions
    from twin_ml.train.trainer import BASELINES

    conds = [
        c
        for c in expand_conditions(rcfg.studies["aug"], rcfg)
        if c in expand_conditions(rcfg.studies["e1models"], rcfg)
    ]
    learned = [m for m in aug if m in base and m not in BASELINES]
    hs = _horizons_of(aug, "clean")
    return {
        "models": learned,
        "horizons": hs,
        "conditions": conds,
        "seen": {c: seen_status(rcfg, c) for c in conds},
        "diff": {
            h: {
                m: {c: compare(aug[m], c, base[m], c, h, SHIFT_METRICS, eval_cfg) for c in conds}
                for m in learned
            }
            for h in hs[-1:]
        },
    }


def _e1_dataset(rcfg: RobustnessConfig, root: Path) -> Any:
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir

    ds_cfg = load_dataset_config(root / rcfg.studies["e1models"].dataset_config)
    return Dataset.open(resolve_dataset_dir(ds_cfg, root)), ds_cfg


def perturbation_strength(
    studies: dict[str, StudyRuns], rcfg: RobustnessConfig, root: Path
) -> dict[str, Any] | None:
    """Perturbation sizes relative to the fault signal, from the clean *training* split
    (the same episodes that scale the signal-relative noise grid; no test data).

    Per (machine kind, gradual fault type): the rise of every sensor from the first window
    after the degradation start to the last window before the FAULT onset (window means,
    window = the dataset's window length; ``perturb.episode_rises``), in units of the
    sensor's normal-operation std. The leading sensor has the largest median |rise| / std.
    Noise sigma at p % is p % of that std, so ``p * std / |rise|`` is the noise against the
    signal. ``signal_scale`` is the typical rise per kind and sensor (signal-noise grid).
    """
    base = studies.get("e1models") or {}
    if not base:
        return None
    from twin_ml.data.perturb import episode_rises, leading_sensors, signal_scale
    from twin_ml.features.inputs import KINDS, SENSORS

    first = next(iter(next(iter(base.values())).values()))
    sigma = np.asarray(first["detail"]["normal_std"], dtype=np.float64)
    ds, _ = _e1_dataset(rcfg, root)
    a = ds.split("train").arrays
    w = int(ds.meta["config"]["window_s"])
    rises = episode_rises(
        a["raw"], a["kind"], a["t_s"], a["ep_start"], a["ep_fault"], a["ep_fault_type"], w
    )
    lead = leading_sensors(rises, sigma)
    noise = rcfg.perturbations.noise.levels_pct
    rows = []
    for (k, ft), s_lead in sorted(lead.items()):
        items = [r for r in rises if (r.kind, r.fault_type) == (k, ft)]
        med = float(np.median([float(r.rise[s_lead]) for r in items]))
        std = float(sigma[k][s_lead])
        rows.append(
            {
                "kind": KINDS[k].value,
                "fault_type": ds.fault_types[ft],
                "episodes": len(items),
                "precursor_s": float(np.median([r.precursor_s for r in items])),
                "leading_sensor": SENSORS[s_lead],
                "rise": med,
                "normal_std": std,
                "rise_in_std": abs(med) / std if std else float("nan"),
                "noise_vs_rise": {
                    str(p): p / 100.0 * std / abs(med) if med else float("nan") for p in noise
                },
            }
        )
    scale = signal_scale(rises, sigma)
    return {
        "split": "train",
        "window_s": w,
        "rows": rows,
        "signal_scale": {
            KINDS[k].value: {SENSORS[j]: float(scale[k, j]) for j in range(len(SENSORS))}
            for k in range(len(KINDS))
        },
        "normal_std": {
            KINDS[k].value: {SENSORS[j]: float(sigma[k, j]) for j in range(len(SENSORS))}
            for k in range(len(KINDS))
        },
        "dropout_s": list(rcfg.perturbations.dropout.duration_s or ()),
        "stuck_s": list(rcfg.perturbations.stuck.duration_s or ()),
    }


def _segments(labels: list[str], t: np.ndarray) -> list[tuple[str, float, float]]:
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [lab.size]])
    return [
        (str(lab[i]), float(t[i]), float(t[j - 1]))
        for i, j in zip(starts, stops, strict=True)
        if lab[i] != "none"
    ]


def missed_episode_cases(
    studies: dict[str, StudyRuns], rcfg: RobustnessConfig, root: Path, cond: str = "sf_all"
) -> list[dict[str, Any]]:
    """Episodes of a sensor-fault set missed by every learned model (all seeds) and the
    threshold rule, with the overlapping sensor faults and the clean-data alarms."""
    base = studies.get("e1models") or {}
    if cond not in rcfg.sensor_fault_sets or not base:
        return []
    from twin_ml.data.test_sets import SENSOR_FAULT_COLUMN, load_series
    from twin_ml.features.inputs import KINDS, SENSORS

    h = str(rcfg.primary.horizon)

    def keyed(e: Entry, c: str) -> dict[tuple[str, str, float], dict[str, Any]]:
        d = e["detail"]["conditions"].get(c)
        if d is None:
            return {}
        ids = e["run_ids"][c]
        return {
            (ids[x["run"]], x["machine"], x["t_fault"]): x for x in d["horizons"][h]["episodes"]
        }

    models = [m for m in base if m != "persistence"]
    missed: set[tuple[str, str, float]] | None = None
    for m in models:
        for e in base[m].values():
            ms = {k for k, x in keyed(e, cond).items() if not x["detected"]}
            missed = ms if missed is None else missed & ms
    if not missed:
        return []
    ds, ds_cfg = _e1_dataset(rcfg, root)
    first = next(iter(next(iter(base.values())).values()))
    sigma = np.asarray(first["detail"]["normal_std"], dtype=np.float64)
    w = int(ds.meta["config"]["window_s"])
    out = []
    for run, machine, t_fault in sorted(missed):
        ep = keyed(next(iter(base[models[0]].values())), cond)[(run, machine, t_fault)]
        clean_leads = {
            m: _finite_mean(
                [
                    x["lead_s"]
                    for e in base[m].values()
                    if (x := keyed(e, "clean").get((run, machine, t_fault))) and x["detected"]
                ]
            )
            for m in models
        }
        line = root / ds_cfg.line_config
        sf = load_series(
            root / rcfg.sensor_fault_sets[cond].source_dir,
            [run],
            line,
            extra_labels=(SENSOR_FAULT_COLUMN,),
        )
        cl = load_series(root / ds_cfg.source_dir, [run], line)
        s_sf = next(s for s in sf if s.machine_id == machine)
        s_cl = next(s for s in cl if s.machine_id == machine)
        t = s_cl.t_s
        i0 = int(np.searchsorted(t, ep["t_start"]))
        i1 = int(np.searchsorted(t, t_fault))
        span = max(1, min(w, (i1 - i0) // 2))
        rise = np.nanmean(s_cl.sensors[i1 - span : i1], axis=0) - np.nanmean(
            s_cl.sensors[i0 : i0 + span], axis=0
        )
        k = [x.value for x in KINDS].index(ep["kind"])
        lead = int(np.argmax(np.abs(rise) / sigma[k]))
        finite_leads = [v for v in clean_leads.values() if math.isfinite(v)]
        earliest = t_fault - max(finite_leads) if finite_leads else ep["t_start"]
        segs = [
            (lab, a, b)
            for lab, a, b in _segments(s_sf.labels[SENSOR_FAULT_COLUMN], s_sf.t_s)
            if b >= ep["t_start"] and a <= t_fault
        ]
        observed_last = s_sf.sensors[i1 - span : i1, lead]
        out.append(
            {
                "condition": cond,
                "run": run,
                "machine": machine,
                "kind": ep["kind"],
                "fault_type": ep["fault_type"],
                "t_start": ep["t_start"],
                "t_warn": ep["t_warn"],
                "t_fault": t_fault,
                "clean_leads_s": clean_leads,
                "earliest_clean_alarm_s": earliest,
                "leading_sensor": SENSORS[lead],
                "leading_rise": float(rise[lead]),
                "leading_rise_in_std": float(abs(rise[lead]) / sigma[k][lead]),
                "sensor_faults": [{"label": lab, "from_s": a, "to_s": b} for lab, a, b in segs],
                "leading_missing_before_fault": float(np.isnan(observed_last).mean()),
            }
        )
    return out


def _live_data_quality(root: Path) -> dict[str, Any] | None:
    """Staleness thresholds of the live ingest (config/sources.yaml) for the case study."""
    from twin_common.config.loader import read_yaml

    path = root / "config" / "sources.yaml"
    if not path.is_file():
        return None
    ingest = read_yaml(path).get("ingest", {})
    if "stale_after_s" not in ingest or "offline_after_s" not in ingest:
        return None
    return {
        "stale_after_s": float(ingest["stale_after_s"]),
        "offline_after_s": float(ingest["offline_after_s"]),
    }


def _affected(label: str) -> list[str]:
    from twin_ml.features.inputs import SENSORS

    kind, _, sensor = label.partition(":")
    return list(SENSORS) if kind == "dropout" else [sensor]


def build_report(
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    out_dir: Path | None = None,
    models_dir: Path | None = None,
) -> list[Path]:
    tdir = tracking_dir or root / eval_cfg.tracking_dir
    models_dir = models_dir or root / "config" / "models"
    studies = {name: collect_study(tdir, rcfg, eval_cfg, name, models_dir) for name in rcfg.studies}
    if not any(studies.values()):
        raise LookupError(
            f"no finished {rcfg.experiment} study runs; run uv run poe robustness --study e1models"
        )
    report: dict[str, Any] = {
        "title": "Phase 8: robustness, calibration and conformal prediction",
        "experiment": rcfg.experiment,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "preregistration": rcfg.primary.model_dump(),
        "alpha": rcfg.conformal.alpha,
        "ece_bins": rcfg.ece_bins,
        "eval_subset": rcfg.eval_subset,
        "bootstrap": eval_cfg.bootstrap.model_dump(),
        "sustain_s": eval_cfg.early_warning.sustain_s,
        "perturbations": rcfg.perturbations.model_dump(mode="json"),
        "sensor_fault_sets": {
            k: v.model_dump(mode="json") for k, v in rcfg.sensor_fault_sets.items()
        },
        "seen_in_augmentation": rcfg.seen_in_augmentation,
        "studies_run": {k: sorted(v, key=_model_key) for k, v in studies.items()},
        "primary": primary_block(rcfg, eval_cfg, studies),
        "studies": {},
        "plots": {},
    }
    for name, runs in studies.items():
        if runs and "clean" in rcfg.studies[name].conditions and not rcfg.studies[name].extension:
            report["studies"][name] = study_block(name, runs, rcfg, eval_cfg)
    report["perturbation_strength"] = perturbation_strength(studies, rcfg, root)
    report["missed_cases"] = missed_episode_cases(studies, rcfg, root)
    report["live_data_quality"] = _live_data_quality(root)
    report["signal_noise"] = snoise_block(studies, rcfg, eval_cfg)
    report["e4"] = e4_block(studies, rcfg, eval_cfg)
    report["ablation"] = ablation_block(studies, rcfg, eval_cfg)
    out = out_dir or root / eval_cfg.report.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "e1models" in report["studies"]:
        paths += plot_degradation(report, rcfg, "e1models", out)
        paths += plot_reliability(report, rcfg, "e1models", out)
        paths += plot_coverage(report, rcfg, "e1models", out)
    if report["e4"]:
        paths += plot_e4(report, rcfg, out)
    if report["ablation"]:
        paths += plot_ablation(report, rcfg, out)
    if report["signal_noise"]:
        paths += plot_snoise(report, rcfg, out)
    name = rcfg.report_name
    md = out / f"{name}.md"
    # LF on every OS (the repo uses eol=lf; Windows write_text would emit CRLF)
    md.write_text(render_markdown(report, rcfg), encoding="utf-8", newline=LF)
    js = out / f"{name}.json"
    # compact, 6 significant digits: stays well below the 5 MB results limit
    js.write_text(
        json.dumps(_round(_clean(_strip(report))), separators=(",", ":")),
        encoding="utf-8",
        newline=LF,
    )
    return [md, js, *paths]


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v) for v in obj]
    return obj


def _strip(report: dict[str, Any]) -> dict[str, Any]:
    """The JSON summary without the raw reliability bins (they are in the plots)."""
    out = dict(report)
    out["studies"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "reliability"}
        for k, v in report["studies"].items()
    }
    return out


# ---------------------------------------------------------------------------- markdown


def _f(x: Any, digits: int) -> str:
    if x is None or not isinstance(x, int | float) or not math.isfinite(x):
        return NA
    return f"{x:.{digits}f}"


def _cell(st: dict[str, Any], digits: int) -> str:
    if not st or not st.get("seeds"):
        return NA
    p = _f(st["point"], digits)
    if p == NA:
        return NA
    lo, hi = _f(st["lo"], digits), _f(st["hi"], digits)
    return p if lo == NA else f"{p} [{lo}, {hi}]"


def _diff(d: dict[str, Any], digits: int) -> str:
    if not d or not d.get("seeds"):
        return NA
    p = _f(d["point"], digits)
    if p == NA:
        return NA
    mark = " *" if d["excludes_zero"] else ""
    sign = "+" if d["point"] > 0 else ""
    ci = f"[{_f(d['lo'], digits)}, {_f(d['hi'], digits)}]"
    return f"{sign}{p} {ci}{mark} ({d['seeds_excluding_zero']}/{d['seeds']})"


def _count_line(c: dict[str, Any]) -> str:
    if not c:
        return ""
    kinds = ", ".join(f"{k} {v}" for k, v in c["episodes_per_kind"].items())
    wpc = ", ".join(f"{k} {v:,}" for k, v in c.get("windows_per_class", {}).items())
    sf = c.get("sensor_fault_windows")
    sf_txt = f"; windows in a sensor fault: {sf:,}" if sf else ""
    return (
        f"{c['runs']} test runs, {c['episodes']} gradual-fault episodes ({kinds}), "
        f"{c['negative_h']:.1f} h of negative (normal operating) windows; "
        f"{eval_subset_word()} windows per true class: {wpc}{sf_txt}."
    )


def eval_subset_word() -> str:
    return "operating"


NOTES = [
    "- **Primary vs exploratory.** Sections marked **PRIMARY** report the pre-registered outcomes "
    "(ADR 0021, fixed before any study run). Everything marked **EXPLORATORY** is descriptive; "
    "with many comparisons some CIs exclude zero by chance.",
    "- **Cells:** `point [lo, hi]` = mean over seeds of the point estimate and of the 95 % run "
    "bootstrap CI bounds (1000 resamples of whole test runs). Differences: `* ` = the averaged "
    "CI excludes zero; `(k/n)` = seeds whose own CI excludes zero.",
    "- **Alarms** use the uncalibrated argmax (the E1 rule) sustained for the configured seconds; "
    "calibration changes only the probabilities used for ECE and conformal sets (ADR 0022).",
    "- **Lead relative to the threshold WARNING** (`rel_lead_mean_s`) = simulator WARNING time "
    "minus alarm time; positive = earlier than the threshold rule's onset.",
    "- **Persistence** reads the true current state by design, so sensor perturbations cannot "
    "change it; it is a reference line.",
    "- **ECE** is top-label with 15 equal-width bins on operating windows, where most windows are "
    "confidently NORMAL, so it is small in absolute terms; the FAULT-class ECE and the "
    "reliability diagrams show the rare-class behaviour.",
    "- **Conformal sets** at alpha 0.1, calibrated on clean validation operating windows. LAC and "
    "APS guarantee coverage only on average over classes (dominated by NORMAL); Mondrian LAC "
    "per class. Randomised APS may return empty sets (counted as not covered).",
    "- **Baselines and conformal sets:** persistence and threshold output one-hot "
    "probabilities and are not calibrated. Their conformal sets degenerate (a Mondrian set "
    "contains almost every class, see the set sizes), so their coverage figures say nothing "
    "about confidence; they are listed for completeness.",
    "- **Coverage needs the set size next to it:** a coverage of 1.0 with sets of size 3 is "
    "trivial, and a marginal coverage far below 0.9 under a perturbation means the calibrated "
    "probabilities moved (empty or too small sets).",
]


def _primary_section(report: dict[str, Any], rcfg: RobustnessConfig) -> list[str]:
    p = rcfg.primary
    h = str(p.horizon)
    lines = [
        "## PRIMARY: pre-registered outcomes",
        "",
        f"Pre-registered in ADR 0021 and `robustness.yaml` before any study run: at +{h} s, "
        f"miss rate, lead relative to the threshold WARNING and false alarms per hour of each "
        f"model vs the `{p.reference}` baseline (paired); ECE after calibration; FAULT-class "
        "coverage of LAC, APS and Mondrian LAC; under "
        + ", ".join(f"`{c}`" for c in p.conditions)
        + ".",
        "",
    ]
    for cond, block in report["primary"].items():
        lines += [f"### {cond}", ""]
        if not block["available"]:
            lines += [f"Not run yet (study `{block['study']}`).", ""]
            continue
        lines += [f"Study `{block['study']}`. {_count_line(block['counts'])}", ""]
        rows = []
        for m, vals in block["values"].items():
            row = [m]
            for k in p.early_warning:
                row.append(_cell(vals[k], DIGITS.get(k, 3)))
            vs = block["vs_reference"].get(m, {})
            for k in p.early_warning:
                row.append(_diff(vs.get(k, {}), DIGITS.get(k, 3)) if vs else NA)
            rows.append(row)
        header = ["Model", *p.early_warning, *[f"Δ {k} vs {p.reference}" for k in p.early_warning]]
        lines += [f"**Early warning (+{h} s)**", "", *_table(header, rows), ""]
        rows = []
        cols = (*p.calibration, *EXTRA_CALIBRATION, *p.conformal)
        for m, vals in block["values"].items():
            rows.append([m, *[_cell(vals[k], 4 if k.startswith("ece") else 3) for k in cols]])
        header = [
            "Model",
            *p.calibration,
            *[f"{k} (exploratory)" for k in EXTRA_CALIBRATION],
            *p.conformal,
        ]
        lines += [
            f"**Calibration and FAULT-class coverage (+{h} s, "
            f"target coverage {1 - rcfg.conformal.alpha:.2f})**",
            "",
            *_table(header, rows),
            "",
            _ece_note(block["counts"]),
            "",
        ]
    return lines


EXTRA_CALIBRATION = ("ece_fault_cal",)  # exploratory, shown next to the top-label ECE


def _ece_note(c: dict[str, Any]) -> str:
    """Why top-label ECE is small: the share of NORMAL among the operating windows."""
    wpc = c.get("windows_per_class") or {}
    total = sum(wpc.values())
    if not total:
        return ""
    share = wpc.get("NORMAL", 0) / total
    return (
        f"Top-label ECE on operating windows is dominated by easy NORMAL windows "
        f"({share:.1%} of the {total:,} operating windows here are NORMAL at this horizon, "
        "mostly predicted with confidence near 1). The FAULT-class ECE (`ece_fault_cal`, "
        "p(FAULT) against the event FAULT, after calibration) shows the rare class."
    )


def _study_sections(
    report: dict[str, Any], rcfg: RobustnessConfig, name: str, title: str
) -> list[str]:
    st = report["studies"].get(name)
    if not st:
        return [f"## EXPLORATORY: {title}", "", "Not run yet.", ""]
    h = st["horizons"][-1]
    lines = [f"## EXPLORATORY: {title}", ""]
    clean_c = st["counts"][h]["clean"]
    lines += [f"Clean test set: {_count_line(clean_c)}", ""]
    levels = study_levels(rcfg, st)
    # early warning per condition
    lines += [f"### Early warning and classification under each condition (+{h} s)", ""]
    for kind, lv in levels.items():
        rows = []
        for m in st["models"]:
            for _, c in lv:
                v = st["values"][h][m][c]
                rows.append(
                    [
                        m,
                        c,
                        *[
                            _cell(v.get(k, {}), DIGITS.get(k, 3))
                            for k in ("macro_f1", "fault_recall", *EW_METRICS)
                        ],
                    ]
                )
        lines += [
            f"**{kind}**",
            "",
            *_table(["Model", "Condition", "macro F1", "FAULT recall", *EW_METRICS], rows),
            "",
        ]
    sf = [c for c in st["conditions"] if c in rcfg.sensor_fault_sets]
    if sf:
        rows = []
        for m in st["models"]:
            for c in ["clean", *sf]:
                v = st["values"][h][m][c]
                rows.append(
                    [
                        m,
                        c,
                        *[
                            _cell(v.get(k, {}), DIGITS.get(k, 3))
                            for k in ("macro_f1", *EW_METRICS, "sf_false_alarms_per_h")
                        ],
                    ]
                )
        lines += [
            "### Simulator sensor faults",
            "",
            "Sets: "
            + "; ".join(f"`{k}` = {', '.join(rcfg.sensor_fault_sets[k].types)}" for k in sf)
            + ". `sf_false_alarms_per_h` counts alarms raised inside a sensor-fault segment "
            "while the machine is healthy.",
            "",
            *[f"- `{c}`: {_count_line(st['counts'][h][c])}" for c in sf],
            "",
            *_table(["Model", "Condition", "macro F1", *EW_METRICS, "sf FA/h"], rows),
            "",
        ]
    # degradation
    lines += [
        f"### Degradation: perturbed minus clean, paired per model (+{h} s)",
        "",
        "Same test runs on both sides; positive Δ miss rate / Δ false alarms = worse, "
        "negative Δ macro F1 / recall / lead = worse.",
        "",
    ]
    rows = []
    for m in st["models"]:
        for c in st["conditions"]:
            if c == "clean":
                continue
            d = st["degradation"][h][m][c]
            rows.append([m, c, *[_diff(d[k], DIGITS.get(k, 3)) for k in DEGRADATION_METRICS]])
    lines += [*_table(["Model", "Condition", *[f"Δ {k}" for k in DEGRADATION_METRICS]], rows), ""]
    # calibration
    lines += [f"### Calibration (fitted on clean validation data, +{h} s)", ""]
    rows = []
    for m in st["models"]:
        heads = st["calibration_methods"][m]
        methods = sorted(
            {
                f"{v['method']}" + (f" T={v['temperature']:.2f}" if v.get("temperature") else "")
                for s in heads.values()
                for hh, v in s.items()
                if hh == h
            }
        )
        rows.append([m, "; ".join(methods)])
    lines += [*_table(["Model", "method per seed (+30 s head)"], rows), ""]
    rows = []
    for m in st["models"]:
        for c in st["conditions"]:
            v = st["values"][h][m][c]
            rows.append(
                [
                    m,
                    c,
                    *[
                        _cell(v.get(k, {}), 4)
                        for k in ("ece_uncal", "ece_cal", "ece_fault_uncal", "ece_fault_cal")
                    ],
                ]
            )
    lines += [
        *_table(
            [
                "Model",
                "Condition",
                "ECE before",
                "ECE after",
                "FAULT ECE before",
                "FAULT ECE after",
            ],
            rows,
        ),
        "",
        _ece_note(clean_c),
        "",
    ]
    # conformal
    lines += [f"### Conformal prediction sets (alpha {rcfg.conformal.alpha}, +{h} s)", ""]
    rows = []
    for m in st["models"]:
        for c in st["conditions"]:
            v = st["values"][h][m][c]
            row = [m, c]
            for meth in rcfg.conformal.methods:
                row += [_cell(v.get(f"cov_{meth}", {}), 3), _cell(v.get(f"size_{meth}", {}), 2)]
            rows.append(row)
    header = ["Model", "Condition"] + [
        x for meth in rcfg.conformal.methods for x in (f"{meth} cov", f"{meth} size")
    ]
    lines += ["**Marginal coverage and average set size**", "", *_table(header, rows), ""]
    rows = []
    for m in st["models"]:
        for c in st["conditions"]:
            v = st["values"][h][m][c]
            row = [m, c]
            for meth in rcfg.conformal.methods:
                row += [_f(v.get(f"cov_{meth}_{cls.value}", {}).get("point"), 3) for cls in HEALTH]
            rows.append(row)
    header = ["Model", "Condition"] + [
        f"{meth} {cls.value}" for meth in rcfg.conformal.methods for cls in HEALTH
    ]
    lines += [
        "**Coverage per true class** (point estimates; CIs in the JSON)",
        "",
        *_table(header, rows),
        "",
    ]
    rows = []
    for m in st["models"]:
        for c in st["conditions"]:
            v = st["values"][h][m][c]
            rows.append(
                [
                    m,
                    c,
                    *[
                        _cell(v.get(k, {}), DIGITS.get(k, 3))
                        for k in ("ttf_cov", "ttf_width_s", "ttf_certain_miss")
                    ],
                ]
            )
    lines += [
        "**Time-to-fault intervals** (absolute-residual split conformal; coverage on known "
        "targets below the cap; on right-censored rows only the certain-miss rate, interval "
        "below the known lower bound, is measurable)",
        "",
        *_table(
            ["Model", "Condition", "coverage", "mean width (s)", "censored: certain miss"], rows
        ),
        "",
    ]
    if len(st["horizons"]) > 1:
        h0 = st["horizons"][0]
        rows = []
        for m in st["models"]:
            for c in st["conditions"]:
                v = st["values"][h0][m][c]
                rows.append(
                    [
                        m,
                        c,
                        *[
                            _f(v.get(k, {}).get("point"), DIGITS.get(k, 3))
                            for k in (
                                "macro_f1",
                                "fault_recall",
                                *EW_METRICS,
                                "ece_cal",
                                "cov_mondrian_lac_FAULT",
                            )
                        ],
                    ]
                )
        lines += [
            f"### Horizon +{h0} s (point estimates)",
            "",
            *_table(
                [
                    "Model",
                    "Condition",
                    "macro F1",
                    "FAULT recall",
                    *EW_METRICS,
                    "ECE after",
                    "Mondrian FAULT cov",
                ],
                rows,
            ),
            "",
        ]
    rows = [
        [m, _f(st["timings_s"][m], 0), ", ".join(f"{s}: `{r}`" for s, r in st["runs"][m].items())]
        for m in st["models"]
    ]
    lines += [
        "### Runs and cost",
        "",
        *_table(["Model", "time per study run (s)", "MLflow runs"], rows),
        "",
    ]
    checks = [
        f"{m} seed {s}: max |Δp| {max(v['max_abs_diff'] for v in chk.values()):.1e}"
        for m, seeds in st["prediction_checks"].items()
        for s, chk in seeds.items()
        if chk
    ]
    if checks:
        lines += [
            "Reused models reproduce their stored clean test predictions (identical labels): "
            + "; ".join(checks)
            + ".",
            "",
        ]
    return lines


def _strength_section(report: dict[str, Any], rcfg: RobustnessConfig) -> list[str]:
    ps = report.get("perturbation_strength")
    if not ps or not ps["rows"]:
        return []
    noise = [str(p) for p in rcfg.perturbations.noise.levels_pct]
    lines = [
        "## EXPLORATORY: how strong were the perturbations? (relative to the fault signal)",
        "",
        "Clean training split (the same episodes scale the signal-relative noise grid below; "
        "no test data), per machine kind and gradual fault type. *Rise* = mean of the last "
        f"{ps['window_s']} s before the FAULT onset minus the mean of the first {ps['window_s']} s "
        "after the degradation start (median over episodes), for the *leading sensor* (largest "
        "median |rise| / normal-operation std). Noise at p % has sigma = p % of that std, so "
        "`noise p % / rise` is the noise sigma as a fraction of the whole precursor rise. "
        f"Dropout segments last {ps['dropout_s'][0]} to {ps['dropout_s'][1]} s and stuck segments "
        f"{ps['stuck_s'][0]} to {ps['stuck_s'][1]} s; compare them with the median precursor "
        "length.",
        "",
    ]
    rows = []
    for r in ps["rows"]:
        rows.append(
            [
                r["kind"],
                r["fault_type"],
                str(r["episodes"]),
                _f(r["precursor_s"], 0),
                r["leading_sensor"],
                _f(r["rise"], 2),
                _f(r["normal_std"], 3),
                _f(r["rise_in_std"], 1),
                *[_f(r["noise_vs_rise"][p], 3) for p in noise],
            ]
        )
    header = [
        "Kind",
        "Fault type",
        "Episodes",
        "Median precursor (s)",
        "Leading sensor",
        "Rise (sensor units)",
        "Normal std",
        "Rise / std",
        *[f"noise {p} % / rise" for p in noise],
    ]
    lines += [*_table(header, rows), ""]
    from twin_ml.features.inputs import SENSORS

    scale_rows = [
        [kind, *[f"{_f(sc[s], 3)} ({_f(ps['normal_std'][kind][s], 3)})" for s in SENSORS]]
        for kind, sc in ps["signal_scale"].items()
    ]
    lines += [
        "**Typical rise before FAULT per kind and sensor** (sigma of the signal-relative "
        "noise grid at 100 %; normal-operation std in brackets). A sensor that leads some "
        "fault types of a kind uses its median |rise| over those episodes, any other sensor "
        "its median |rise| over all of the kind's episodes.",
        "",
        *_table(["Kind", *SENSORS], scale_rows),
        "",
    ]
    return lines


SNOISE_METRICS = (
    "miss_rate",
    "lead_mean_s",
    "rel_lead_mean_s",
    "false_alarms_per_h",
    "macro_f1",
    "ece_cal",
    "ece_fault_cal",
    "cov_mondrian_lac_FAULT",
)
SNOISE_STUDIES = (("e1models_snoise", "E1 models"), ("aug_snoise", "augmentation-trained"))


def snoise_block(
    studies: dict[str, StudyRuns], rcfg: RobustnessConfig, eval_cfg: EvalConfig
) -> dict[str, Any] | None:
    """Exploratory signal-relative noise grid: values per level and paired degradation."""
    if rcfg.signal_noise is None:
        return None
    from twin_ml.experiments.robust_worker import expand_conditions

    out: dict[str, Any] = {"levels_pct": rcfg.signal_noise.levels_pct, "studies": {}}
    h = str(rcfg.primary.horizon)
    for name, _ in SNOISE_STUDIES:
        runs = studies.get(name) or {}
        if not runs or name not in rcfg.studies:
            continue
        conds = expand_conditions(rcfg.studies[name], rcfg)
        out["studies"][name] = {
            "models": list(runs),
            "conditions": conds,
            "counts": counts(next(iter(runs.values())), "clean", h),
            "values": {
                m: {c: {k: stat(runs[m], c, h, k) for k in SNOISE_METRICS} for c in conds}
                for m in runs
            },
            "degradation": {
                m: {
                    c: compare(runs[m], c, runs[m], "clean", h, SNOISE_METRICS, eval_cfg)
                    for c in conds
                    if c != "clean"
                }
                for m in runs
            },
        }
    return out if out["studies"] else None


def _snoise_section(report: dict[str, Any]) -> list[str]:
    sn = report.get("signal_noise")
    lines = [
        "## EXPLORATORY: signal-relative noise, where does early warning break down?",
        "",
    ]
    if not sn:
        return [*lines, "Not run yet.", ""]
    h = report["preregistration"]["horizon"]
    lines += [
        "Added after the pre-registration (Part C); the primary outcomes are unchanged. "
        "Evaluation only: no model is retrained on it (Random Forests are refitted from their "
        "seed, as everywhere). Gaussian noise with sigma per machine kind and sensor = "
        f"{', '.join(str(p) for p in sn['levels_pct'])} % of that sensor's typical rise before "
        "FAULT (table above). At 100 % the noise on the leading sensor is as large as the "
        f"whole precursor rise. +{h} s; paired Δ = level minus clean, same models and runs.",
        "",
    ]
    for name, title in SNOISE_STUDIES:
        st = sn["studies"].get(name)
        if not st:
            lines += [f"**{title}:** not run yet.", ""]
            continue
        lines += [f"**{title}** (`{name}`): {_count_line(st['counts'])}", ""]
        rows = []
        for m in st["models"]:
            for c in st["conditions"]:
                v = st["values"][m][c]
                rows.append([m, c, *[_cell(v[k], DIGITS.get(k, 3)) for k in SNOISE_METRICS]])
        lines += [*_table(["Model", "Condition", *SNOISE_METRICS], rows), ""]
        rows = []
        keys = ("miss_rate", "lead_mean_s", "false_alarms_per_h", "cov_mondrian_lac_FAULT")
        for m in st["models"]:
            for c, d in st["degradation"][m].items():
                rows.append([m, c, *[_diff(d[k], DIGITS.get(k, 3)) for k in keys]])
        lines += [*_table(["Model", "Condition", *[f"Δ {k}" for k in keys]], rows), ""]
    return lines


def plot_snoise(report: dict[str, Any], rcfg: RobustnessConfig, out: Path) -> list[Path]:
    """Early warning, calibration and coverage vs signal-relative noise level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sn = report.get("signal_noise")
    if not sn:
        return []
    panels = (
        ("miss_rate", "miss rate"),
        ("lead_mean_s", "mean lead time (s)"),
        ("false_alarms_per_h", "false alarms per hour"),
        ("ece_fault_cal", "FAULT-class ECE after calibration"),
        ("cov_mondrian_lac_FAULT", "Mondrian FAULT coverage"),
        ("macro_f1", "macro F1 (operating)"),
    )
    xs = [0, *sn["levels_pct"]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), facecolor=SURFACE, layout="constrained")
    for ax, (k, label) in zip(axes.flat, panels, strict=True):
        _style(ax)
        for name, dashed in (("e1models_snoise", False), ("aug_snoise", True)):
            st = sn["studies"].get(name)
            if not st:
                continue
            colors = _colors(st["models"])
            for m in st["models"]:
                if m == "persistence":
                    continue
                vals = [st["values"][m][c][k] for c in st["conditions"]]
                y = [v["point"] for v in vals]
                if not dashed:
                    ax.fill_between(
                        xs,
                        [v["lo"] for v in vals],
                        [v["hi"] for v in vals],
                        color=colors[m],
                        alpha=0.10,
                        lw=0,
                    )
                ax.plot(
                    xs,
                    y,
                    color=colors[m],
                    lw=1.2 if dashed else 2,
                    ls=(0, (3, 2)) if dashed else "-",
                    marker=None if dashed else "o",
                    ms=4,
                    label=None if dashed else m,
                )
        if k == "cov_mondrian_lac_FAULT":
            ax.axhline(1 - rcfg.conformal.alpha, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_title(label, color=INK, fontsize=9.5, loc="left")
        ax.set_xticks(xs)
        ax.set_xlabel("noise sigma, % of the typical rise before FAULT", color=INK2, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"Signal-relative noise at +{rcfg.primary.horizon} s (exploratory): E1 models solid "
        "(band = mean 95 % run-bootstrap CI), augmentation-trained dashed; mean over seeds",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    path = out / f"{rcfg.report_name}_signal_noise.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    report["plots"]["signal_noise"] = path.name
    return [path]


def _case_section(report: dict[str, Any]) -> list[str]:
    cases = report.get("missed_cases") or []
    if not cases:
        return []
    lines = ["## EXPLORATORY: case study, the episode every model missed under sensor faults", ""]
    for c in cases:
        leads = ", ".join(f"{m} {_f(v, 0)} s" for m, v in c["clean_leads_s"].items())
        faults = "; ".join(
            f"`{f['label']}` {_f(f['from_s'], 0)} to {_f(f['to_s'], 0)} s"
            for f in c["sensor_faults"]
        )
        lead_sensor = c["leading_sensor"]
        hiding = [
            f
            for f in c["sensor_faults"]
            if lead_sensor in _affected(f["label"])
            and f["from_s"] <= c["earliest_clean_alarm_s"]
            and f["to_s"] >= c["t_fault"]
        ]
        lines += [
            f"**{c['condition']}: {c['run']}, {c['machine']} ({c['kind']}), {c['fault_type']}.** "
            f"Degradation starts at {_f(c['t_start'], 0)} s, the simulator's threshold WARNING "
            f"at {_f(c['t_warn'], 0)} s, FAULT at {_f(c['t_fault'], 0)} s.",
            "",
            f"- **Clean data:** every model and the threshold rule detect it; mean lead over "
            f"seeds: {leads}. The earliest clean-data alarm is at "
            f"{_f(c['earliest_clean_alarm_s'], 0)} s.",
            f"- **Leading signal:** `{lead_sensor}` rises by {_f(c['leading_rise'], 2)} "
            f"({_f(c['leading_rise_in_std'], 1)} normal-operation std) from degradation start "
            "to FAULT on the clean run.",
            f"- **Overlapping sensor faults in `{c['condition']}`:** {faults or 'none'}. In the "
            f"last window before the FAULT onset, {c['leading_missing_before_fault']:.0%} of the "
            f"`{lead_sensor}` values are missing.",
        ]
        if hiding:
            f = hiding[0]
            what = (
                "every sensor was missing (dropout)"
                if f["label"] == "dropout"
                else f"`{lead_sensor}` was affected (`{f['label']}`)"
            )
            lines += [
                f"- **Why it was missed:** from {_f(f['from_s'], 0)} s to {_f(f['to_s'], 0)} s "
                f"{what}, which covers the whole interval from the earliest clean-data alarm "
                "to the FAULT onset. Forward fill repeats the last values observed before the "
                "fault segment, so the rising signal is never seen. The missing masks are on, "
                "but no model was trained on missing data. Neither the models nor the "
                "threshold rule can raise a sustained alarm before the FAULT.",
            ]
            live = report.get("live_data_quality")
            if f["label"] == "dropout" and live:
                dur = f["to_s"] - f["from_s"]
                lines += [
                    f"- **In the live system this blind spot is covered by the data-quality "
                    f"layer:** a reading without any sensor value is quarantined (ADR 0007), so "
                    f"a {_f(dur, 0)} s dropout makes the machine STALE after "
                    f"{_f(live['stale_after_s'], 0)} s and OFFLINE after "
                    f"{_f(live['offline_after_s'], 0)} s without valid data "
                    "(`config/sources.yaml`), raising the CRITICAL `machine_offline` alert and "
                    "marking the data quality DEGRADED (`config/engine.yaml`). The operator is "
                    "told the machine cannot be monitored, even though no model can warn.",
                ]
        else:
            lines += [
                "- **Why it was missed:** no single sensor fault covers the leading sensor over "
                "the whole alarm interval; see the overlapping segments above.",
            ]
        lines += [""]
    return lines


def _drift_section(report: dict[str, Any], rcfg: RobustnessConfig) -> list[str]:
    st = report["studies"].get("e1models")
    cond = "sf_drift"
    if not st or cond not in st["conditions"]:
        return []
    h = st["horizons"][-1]
    vals = {
        m: st["values"][h][m][cond].get("sf_false_alarms_per_h", {}).get("point")
        for m in st["models"]
        if m != "persistence"
    }
    finite = {m: v for m, v in vals.items() if v is not None and math.isfinite(v)}
    if not finite or not all(v > 0 for v in finite.values()):
        return []
    lo, hi = min(finite.values()), max(finite.values())
    return [
        "## EXPLORATORY: sensor drift looks like degradation",
        "",
        f"Under `{cond}` (simulator drift, never seen in training) every learned model and "
        f"the threshold rule raise false alarms inside drift segments while the machine is "
        f"healthy: {_f(lo, 2)} to {_f(hi, 2)} per hour of such windows (+{h} s; per model in "
        "the sensor-fault table). A slowly drifting temperature, vibration or current reading "
        "has the same shape as a developing fault on that sensor, so this suggests that "
        "models trained only on machine faults cannot distinguish sensor drift from machine "
        "degradation. This is a motivation for Phase 9 (anomaly detection and unseen faults): "
        "separating sensor faults from machine degradation likely needs a model of normal "
        "sensor behaviour across sensors (for example, cross-sensor consistency) rather than "
        "a fault classifier alone.",
        "",
    ]


E4_TABLE = (
    "macro_f1",
    "fault_recall",
    "miss_rate",
    "rel_lead_mean_s",
    "false_alarms_per_h",
    "ece_cal",
    "ece_fault_cal",
    "cov_mondrian_lac_FAULT",
)


def _e4_section(report: dict[str, Any], rcfg: RobustnessConfig) -> list[str]:
    e4 = report.get("e4")
    lines = [
        "## EXPLORATORY: E4 condition shift (trained on low + medium load, tested on high load)",
        "",
    ]
    if not e4:
        return [*lines, "Not run yet.", ""]
    h = e4["horizons"][-1]
    rf = e4["dataset"].get("run_filter") or {}
    if rf:
        parts = []
        for k, v in rf.items():
            bands = ", ".join(f"{b} {n}" for b, n in v["by_load"].items())
            parts.append(f"{k}: {v['runs']} runs ({bands})")
        lines += [
            "Run selection (by the simulator's per-run load band, on the fixed 2026 split): "
            + "; ".join(parts)
            + ".",
            "",
        ]
    for c in ("e4_high", "e4_in"):
        lines += [f"- `{c}`: {_count_line(e4['counts'][h][c])}"]
    lines += [
        "",
        "The primary `e4_high` outcomes (vs the threshold baseline) are in the PRIMARY section. "
        f"**Shift effect** (+{h} s): the same E4 model on the high-load runs minus on the "
        "low + medium test runs. The run sets differ, so each side is resampled independently "
        "(unpaired bootstrap); the training set is the same on both sides, so its smaller size "
        "does not confound the shift effect.",
        "",
    ]
    rows = []
    for m in e4["models"]:
        for c in ("e4_in", "e4_high"):
            v = e4["values"][h][m][c]
            rows.append([m, c, *[_cell(v[k], DIGITS.get(k, 3)) for k in E4_TABLE]])
    lines += [*_table(["Model", "Test set", *E4_TABLE], rows), ""]
    rows = [
        [m, *[_diff(e4["shift"][h][m][k], DIGITS.get(k, 3)) for k in E4_TABLE]]
        for m in e4["models"]
    ]
    lines += [
        f"**Shift effect: e4_high minus e4_in, unpaired (+{h} s)**",
        "",
        *_table(["Model", *[f"Δ {k}" for k in E4_TABLE]], rows),
        "",
    ]
    sec = e4.get("secondary") or {}
    if sec.get("shared_runs"):
        rows = [
            [m, *[_diff(d[k], DIGITS.get(k, 3)) for k in E4_TABLE]]
            for m, d in sec["diff"][h].items()
        ]
        lines += [
            f"**Secondary check: E4 model minus E1 model on the {len(sec['shared_runs'])} "
            f"high-load runs of the original test split** ({sec['episodes']:.0f} gradual-fault "
            f"episodes at +{rcfg.primary.horizon} s; paired). The E1 models trained on 140 runs "
            "including high load, so this mixes the shift with the training-set size.",
            "",
            *_table(["Model", *[f"Δ {k}" for k in E4_TABLE]], rows),
            "",
        ]
    return lines


ABLATION_TABLE = (
    "macro_f1",
    "miss_rate",
    "rel_lead_mean_s",
    "false_alarms_per_h",
    "ece_cal",
    "ece_fault_cal",
    "cov_mondrian_lac_FAULT",
)


def _ablation_section(report: dict[str, Any], rcfg: RobustnessConfig) -> list[str]:
    ab = report.get("ablation")
    lines = ["## EXPLORATORY: ablation, augmentation-trained minus clean-trained models", ""]
    if not ab:
        return [*lines, "Not run yet.", ""]
    h = ab["horizons"][-1]
    lines += [
        f"Paired on the same test runs and perturbations, seed i vs seed i (+{h} s). "
        f"Perturbation types seen in augmentation: {', '.join(rcfg.seen_in_augmentation)}. "
        "`sf_drift` (simulator drift) was never seen; `sf_all` mixes seen and unseen types.",
        "",
    ]
    for status in ("clean", "seen", "mixed", "unseen"):
        cs = [c for c in ab["conditions"] if ab["seen"][c] == status]
        if not cs:
            continue
        rows = [
            [m, c, *[_diff(ab["diff"][h][m][c][k], DIGITS.get(k, 3)) for k in ABLATION_TABLE]]
            for m in ab["models"]
            for c in cs
        ]
        header = ["Model", "Condition", *[f"Δ {k}" for k in ABLATION_TABLE]]
        lines += [f"**{status} conditions**", "", *_table(header, rows), ""]
    return lines


def render_markdown(report: dict[str, Any], rcfg: RobustnessConfig) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"Generated by `uv run poe report-robustness` from the MLflow experiment "
        f"`{report['experiment']}` on {report['generated_at']} "
        f"(git `{report['git_commit'] or 'unknown'}`). "
        "Do not edit by hand; definitions are in [docs/experiments.md](../../docs/experiments.md), "
        "the protocol in [ADR 0021](../../docs/adr/0021-robustness-protocol.md) and "
        "[ADR 0022](../../docs/adr/0022-calibration-conformal.md).",
        "",
        "Studies run: "
        + "; ".join(
            f"`{k}`: {', '.join(v) if v else 'not run'}" for k, v in report["studies_run"].items()
        )
        + ".",
        "",
        "## How to read this report",
        "",
        *NOTES,
        "",
        *_primary_section(report, rcfg),
        *_study_sections(
            report,
            rcfg,
            "e1models",
            "E2 noise, E3 missing data and sensor faults (E1 models, no retraining)",
        ),
        *_strength_section(report, rcfg),
        *_snoise_section(report),
        *_case_section(report),
        *_drift_section(report, rcfg),
        *_e4_section(report, rcfg),
        *_ablation_section(report, rcfg),
    ]
    if "aug" in report["studies"]:
        lines += _study_sections(
            report, rcfg, "aug", "augmentation-trained models under every condition"
        )
    plots = report["plots"]
    if plots:
        lines += ["## Plots", ""]
        lines += [f"![{k}]({v})" for k, v in plots.items()]
        lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- plots


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _colors(models: list[str]) -> dict[str, str]:
    return {
        m: SERIES[MODEL_ORDER.index(m) % len(SERIES)] if m in MODEL_ORDER else INK2 for m in models
    }


def plot_degradation(
    report: dict[str, Any], rcfg: RobustnessConfig, study: str, out: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = report["studies"][study]
    h = st["horizons"][-1]
    colors = _colors(st["models"])
    paths = []
    for kind, lv in study_levels(rcfg, st).items():
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), facecolor=SURFACE, layout="constrained")
        for ax, (metric, label) in zip(axes.flat, CURVE_METRICS, strict=True):
            _style(ax)
            for m in st["models"]:
                xs = [pct for pct, _ in lv]
                vals = [st["values"][h][m][c].get(metric, {}) for _, c in lv]
                y = [v.get("point", float("nan")) for v in vals]
                lo = [v.get("lo", float("nan")) for v in vals]
                hi = [v.get("hi", float("nan")) for v in vals]
                ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.12, lw=0)
                ax.plot(xs, y, color=colors[m], lw=2, marker="o", ms=4, label=m)
            ax.set_title(label, color=INK, fontsize=9.5, loc="left")
            ax.set_xlabel(f"{kind} level (%)", color=INK2, fontsize=8.5)
            ax.set_xticks([pct for pct, _ in lv])
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(labels),
            frameon=False,
            fontsize=8.5,
            labelcolor=INK2,
        )
        fig.suptitle(
            f"{study}: metrics vs {kind} level at +{h} s "
            "(mean over seeds; band = mean 95 % run-bootstrap CI)",
            color=INK,
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path = out / f"{rcfg.report_name}_degradation_{kind}.png"
        fig.savefig(path, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        report["plots"][f"degradation_{kind}"] = path.name
        paths.append(path)
    return paths


def plot_reliability(
    report: dict[str, Any], rcfg: RobustnessConfig, study: str, out: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = report["studies"][study]
    h = st["horizons"][-1]
    models = [m for m in st["models"] if m != "persistence"]
    conds = [c for c in rcfg.reliability_conditions if c in st["conditions"]]
    fig, axes = plt.subplots(
        len(models),
        len(conds),
        figsize=(2.6 * len(conds) + 0.6, 2.3 * len(models) + 0.8),
        facecolor=SURFACE,
        layout="constrained",
        squeeze=False,
    )
    for i, m in enumerate(models):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            _style(ax)
            ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (3, 3)))
            for key, color, label in (("ece_uncal", MUTED, "before"), ("ece_cal", ACCENT, "after")):
                bins = np.asarray(st["reliability"][h][m][c][key])
                if bins.size == 0:
                    continue
                n = bins[:, 0]
                ok = n >= rcfg.reliability_min_count
                conf = bins[ok, 1] / n[ok]
                acc = bins[ok, 2] / n[ok]
                ax.plot(conf, acc, color=color, lw=2, marker="o", ms=3.5, label=label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(c, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{m}\naccuracy", color=INK2, fontsize=8.5)
            if i == len(models) - 1:
                ax.set_xlabel("confidence", color=INK2, fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"{study}: top-label reliability at +{h} s, operating windows, all seeds pooled "
        f"(grey: before, blue: after calibration; bins with < {rcfg.reliability_min_count} "
        "windows hidden)",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    path = out / f"{rcfg.report_name}_reliability.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    report["plots"]["reliability"] = path.name
    return [path]


def plot_coverage(
    report: dict[str, Any], rcfg: RobustnessConfig, study: str, out: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = report["studies"][study]
    h = st["horizons"][-1]
    colors = _colors(st["models"])
    methods = list(rcfg.conformal.methods)
    levels = study_levels(rcfg, st)
    if not levels:
        return []
    target = 1 - rcfg.conformal.alpha
    fig, axes = plt.subplots(
        len(methods),
        len(levels),
        figsize=(11, 2.5 * len(methods) + 0.9),
        facecolor=SURFACE,
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    for i, meth in enumerate(methods):
        for j, (kind, lv) in enumerate(levels.items()):
            ax = axes[i][j]
            _style(ax)
            ax.axhline(target, color=INK2, lw=1, ls=(0, (4, 3)))
            xs = [pct for pct, _ in lv]
            for m in st["models"]:
                if m == "persistence":
                    continue
                fault = [
                    st["values"][h][m][c].get(f"cov_{meth}_FAULT", {}).get("point", float("nan"))
                    for _, c in lv
                ]
                marg = [
                    st["values"][h][m][c].get(f"cov_{meth}", {}).get("point", float("nan"))
                    for _, c in lv
                ]
                ax.plot(xs, fault, color=colors[m], lw=2, marker="o", ms=4, label=m)
                ax.plot(xs, marg, color=colors[m], lw=1, ls=(0, (1, 2)))
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(xs)
            if i == 0:
                ax.set_title(kind, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{meth}\ncoverage", color=INK2, fontsize=8.5)
            if i == len(methods) - 1:
                ax.set_xlabel("level (%)", color=INK2, fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"{study}: conformal coverage at +{h} s vs perturbation level; solid = FAULT windows, "
        f"dotted = all operating windows; dashed = target {target:.2f} (mean over seeds)",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    path = out / f"{rcfg.report_name}_coverage.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    report["plots"]["coverage"] = path.name
    return [path]


def plot_e4(report: dict[str, Any], rcfg: RobustnessConfig, out: Path) -> list[Path]:
    """Per model: low + medium test runs vs high-load runs, point and CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e4 = report["e4"]
    h = e4["horizons"][-1]
    models = e4["models"]
    metrics = (
        ("macro_f1", "macro F1"),
        ("miss_rate", "miss rate"),
        ("rel_lead_mean_s", "lead vs threshold WARNING (s)"),
        ("false_alarms_per_h", "false alarms per hour"),
    )
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(12, 0.45 * len(models) + 1.8),
        facecolor=SURFACE,
        layout="constrained",
        sharey=True,
    )
    sides = (("e4_in", MUTED, -0.12, "low + medium load"), ("e4_high", ACCENT, 0.12, "high load"))
    for ax, (k, label) in zip(axes, metrics, strict=True):
        _style(ax)
        for y, m in enumerate(models):
            for c, color, off, name in sides:
                v = e4["values"][h][m][c][k]
                if not math.isfinite(v["point"]):
                    continue
                ax.plot([v["lo"], v["hi"]], [y + off, y + off], color=color, lw=1.5)
                ax.plot(
                    [v["point"]], [y + off], "o", color=color, ms=6, label=name if y == 0 else None
                )
        ax.set_title(label, color=INK, fontsize=9.5, loc="left")
        ax.set_yticks(range(len(models)), models, color=INK2, fontsize=9)
    axes[0].set_ylim(len(models) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"E4 condition shift at +{h} s, models trained on low + medium load "
        "(dot = mean over seeds, line = mean 95 % run-bootstrap CI)",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    path = out / f"{rcfg.report_name}_e4.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    report["plots"]["e4"] = path.name
    return [path]


def plot_ablation(report: dict[str, Any], rcfg: RobustnessConfig, out: Path) -> list[Path]:
    """macro F1 and lead vs level: augmentation-trained (solid) vs clean-trained (dashed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, aug = report["studies"].get("e1models"), report["studies"].get("aug")
    if not base or not aug:
        return []
    h = aug["horizons"][-1]
    levels = study_levels(rcfg, aug)
    if not levels:
        return []
    models = report["ablation"]["models"]
    colors = _colors(models)
    rows = (("macro_f1", "macro F1"), ("lead_mean_s", "mean lead time (s)"))
    fig, axes = plt.subplots(
        len(rows),
        len(levels),
        figsize=(11, 5.4),
        facecolor=SURFACE,
        layout="constrained",
        squeeze=False,
    )
    for i, (k, label) in enumerate(rows):
        for j, (kind, lv) in enumerate(levels.items()):
            ax = axes[i][j]
            _style(ax)
            xs = [pct for pct, _ in lv]
            for m in models:
                for st, dashed in ((aug, False), (base, True)):
                    y = [st["values"][h][m][c].get(k, {}).get("point", float("nan")) for _, c in lv]
                    ax.plot(
                        xs,
                        y,
                        color=colors[m],
                        lw=1.2 if dashed else 2,
                        ls=(0, (3, 2)) if dashed else "-",
                        marker=None if dashed else "o",
                        ms=3.5,
                        label=None if dashed else m,
                    )
            ax.set_xticks(xs)
            if i == 0:
                ax.set_title(kind, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(label, color=INK2, fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("level (%)", color=INK2, fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"Ablation at +{h} s: augmentation-trained (solid) vs clean-trained (dashed) models, "
        "mean over seeds",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    path = out / f"{rcfg.report_name}_ablation.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    report["plots"]["ablation"] = path.name
    return [path]


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.report_robustness")
    parser.add_argument(
        "--robustness-config", type=Path, default=Path("config/models/robustness.yaml")
    )
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument("--models-dir", type=Path, default=Path("config/models"))
    args = parser.parse_args(argv)
    configure_logging()
    from twin_ml.config import load_eval_config, load_robustness_config

    try:
        paths = build_report(
            load_robustness_config(args.robustness_config),
            load_eval_config(args.eval_config),
            models_dir=args.models_dir,
        )
    except (ConfigError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
