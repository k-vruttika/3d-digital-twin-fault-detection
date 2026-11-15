"""Phase 8 evaluation of one model under one test condition (no MLflow, no simulator).

* :func:`fit_calibration`: per horizon, the calibration map and the conformal predictors,
  fitted on the model's *clean validation* predictions (operating windows by default);
* :func:`evaluate_condition`: the E1 evaluation (``evaluate.evaluator.evaluate``) plus
  calibration (ECE before / after, reliability bins), conformal sets and time-to-fault
  intervals, false alarms during simulator sensor faults, and the per-run sums that every
  bootstrap of the robustness report is computed from.

Alarms (early-warning metrics) always use the *uncalibrated* argmax, the E1 decision rule
(ADR 0022); calibrated probabilities feed only ECE, reliability diagrams and conformal sets.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.calibrate import HeadCalibration, fit_head
from twin_ml.calibrate.ece import class_sums, ece_from_sums, reliability, top_label_sums
from twin_ml.config import EvalConfig, RobustnessConfig
from twin_ml.conformal.classification import ConformalClassifier, set_sums
from twin_ml.conformal.regression import fit_ttf_quantile, interval_sums
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import FAULT, HEALTH
from twin_ml.evaluate import run_sums
from twin_ml.evaluate.evaluator import Predictions, evaluate, horizon_run_stats, subset_masks

N_CLASSES = len(HEALTH)


@dataclass
class Calibration:
    family: str
    heads: dict[int, HeadCalibration] = field(default_factory=dict)
    conformal: dict[int, ConformalClassifier] = field(default_factory=dict)
    ttf_q: float | None = None
    ttf_n_cal: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "heads": {str(h): c.describe() for h, c in self.heads.items()},
            "conformal": {str(h): c.describe() for h, c in self.conformal.items()},
            "ttf": {"q_s": self.ttf_q, "n_cal": self.ttf_n_cal},
        }


def _subset(split: SplitData, eval_cfg: EvalConfig, name: str) -> npt.NDArray[np.bool_]:
    return subset_masks(split, eval_cfg)[name]


def uniforms(seed: int, tag: str, horizon: int, n: int) -> npt.NDArray[np.float64]:
    """APS uniforms for one (condition, horizon); identical for every model."""
    ss = np.random.SeedSequence(seed, spawn_key=(zlib.crc32(tag.encode()), horizon))
    return np.random.default_rng(ss).random(n)


def _logits(preds: Predictions, h: int) -> npt.NDArray[np.float32] | None:
    logits = preds.extra.get("logits")
    return None if logits is None else logits.get(h)


def fit_calibration(
    family: str,
    val: SplitData,
    preds: Predictions,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    ttf_cap_s: float,
) -> Calibration:
    sub = _subset(val, eval_cfg, rcfg.eval_subset)
    groups = val.window_run()[sub]
    out = Calibration(family)
    for h in val.horizons:
        y = val.target(h)[sub].astype(np.int64)
        lg = _logits(preds, h)
        head = fit_head(
            family,
            preds.probs[h][sub],
            None if lg is None else lg[sub],
            y,
            groups,
            rcfg.calibration,
        )
        out.heads[h] = head
        cal_probs = head.apply(preds.probs[h][sub], None if lg is None else lg[sub])
        u = uniforms(rcfg.conformal.aps_seed, "val", h, val.n_windows)[sub]
        out.conformal[h] = ConformalClassifier(
            rcfg.conformal.alpha, list(rcfg.conformal.methods), N_CLASSES
        ).fit(cal_probs, y, u)
    if preds.ttf is not None:
        q, n = fit_ttf_quantile(
            preds.ttf[sub],
            val.at_end("ttf")[sub],
            val.at_end("ttf_censored")[sub].astype(bool),
            ttf_cap_s,
            rcfg.conformal.alpha,
        )
        out.ttf_q, out.ttf_n_cal = q, n
    return out


@dataclass
class ConditionResult:
    detail: dict[str, Any]  # JSON-serialisable
    sums: dict[str, run_sums.Sums]  # "h<h>" -> per-run sums
    run_ids: list[str]


def evaluate_condition(
    name: str,
    split: SplitData,
    preds: Predictions,
    cal: Calibration,
    rcfg: RobustnessConfig,
    eval_cfg: EvalConfig,
    fault_types: list[str],
    ttf_cap_s: float,
    sensor_fault_rows: npt.NDArray[np.bool_] | None = None,
) -> ConditionResult:
    """Every Phase 8 metric of one model on one test condition."""
    detail = evaluate(split, preds, eval_cfg, fault_types, ttf_cap_s)
    masks = subset_masks(split, eval_cfg)
    operating = masks["operating"]
    sub = masks[rcfg.eval_subset]
    rw = split.window_run()
    runs = split.run_ids
    n_runs = len(runs)
    bins = rcfg.ece_bins
    sf_win = None if sensor_fault_rows is None else sensor_fault_rows[split.arrays["ends"]]
    sums: dict[str, run_sums.Sums] = {}
    robust: dict[str, Any] = {}
    for h in split.horizons:
        labels = preds.label(h)
        _, rs = horizon_run_stats(split, labels, h, operating, fault_types, eval_cfg, rw)
        s = run_sums.from_runstats(rs)
        y = split.target(h).astype(np.int64)
        p_unc = preds.probs[h].astype(np.float64)
        p_cal = cal.heads[h].apply(preds.probs[h], _logits(preds, h))
        u = uniforms(rcfg.conformal.aps_seed, name, h, split.n_windows)
        sets = cal.conformal[h].sets(p_cal, u)
        for key in ("ece_uncal", "ece_cal", "ece_fault_uncal", "ece_fault_cal"):
            s[key] = np.zeros((n_runs, bins, 3))
        for m in sets:
            s[f"conf_{m}"] = np.zeros((n_runs, N_CLASSES, 5))
        if preds.ttf is not None and cal.ttf_q is not None:
            s["ttf"] = np.zeros((n_runs, 6))
        ttf_y, ttf_c = split.at_end("ttf"), split.at_end("ttf_censored").astype(bool)
        for r in range(n_runs):
            sel = sub & (rw == r)
            yr = y[sel]
            s["ece_uncal"][r] = top_label_sums(p_unc[sel], yr, bins)
            s["ece_cal"][r] = top_label_sums(p_cal[sel], yr, bins)
            s["ece_fault_uncal"][r] = class_sums(p_unc[sel], yr, FAULT, bins)
            s["ece_fault_cal"][r] = class_sums(p_cal[sel], yr, FAULT, bins)
            for m, st in sets.items():
                s[f"conf_{m}"][r] = set_sums(st[sel], yr, N_CLASSES)
            if "ttf" in s:
                assert preds.ttf is not None
                assert cal.ttf_q is not None
                s["ttf"][r] = interval_sums(
                    preds.ttf[sel], ttf_y[sel], ttf_c[sel], cal.ttf_q, ttf_cap_s
                )
        if sf_win is not None:
            _, rs_sf = horizon_run_stats(
                split, labels, h, operating & sf_win, fault_types, eval_cfg, rw
            )
            s["sf_false_alarms"] = rs_sf.false_alarms.astype(np.float64)
            s["sf_negative_s"] = rs_sf.negative_s.astype(np.float64)
        sums[f"h{h}"] = s
        all_mask = masks["all"]
        robust[str(h)] = {
            "bootstrap": run_sums.bootstrap(
                s, eval_cfg.bootstrap.n_resamples, eval_cfg.bootstrap.seed, eval_cfg.bootstrap.ci
            ),
            "reliability": {
                k: reliability(s[k].sum(axis=0))
                for k in ("ece_uncal", "ece_cal", "ece_fault_uncal", "ece_fault_cal")
            },
            "ece_all_windows": {
                "uncal": ece_from_sums(top_label_sums(p_unc[all_mask], y[all_mask], bins)),
                "cal": ece_from_sums(top_label_sums(p_cal[all_mask], y[all_mask], bins)),
            },
        }
    detail["robust"] = robust
    detail["condition"] = name
    detail["sensor_fault_windows"] = None if sf_win is None else int(sf_win.sum())
    # per-row horizon detail (episodes) stays in detail["horizons"]; drop the per-window
    # classification of every subset except the ones the report uses, to keep artifacts small
    return ConditionResult(detail=detail, sums=sums, run_ids=runs)


def flat_condition_metrics(name: str, res: ConditionResult) -> dict[str, float]:
    """MLflow metric keys ``<condition>.h<h>.<metric>`` (point estimates; the CIs are in the
    run's ``metrics.json``, which keeps the file store small); NaN skipped."""
    out: dict[str, float] = {}

    def put(key: str, v: Any) -> None:
        if isinstance(v, int | float) and np.isfinite(v):
            out[key] = float(v)

    for h, block in res.detail["robust"].items():
        for m, ci in block["bootstrap"].items():
            put(f"{name}.h{h}.{m}", ci["point"])
    for sub, v in res.detail.get("ttf", {}).items():
        put(f"{name}.ttf.{sub}.mae_s", v["mae_s"])
    return out
