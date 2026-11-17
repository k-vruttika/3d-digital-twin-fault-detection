"""Phase 9 report: ``python -m twin_ml.experiments.report_e5`` (poe task ``report-e5``).

Reads only MLflow (the E5 experiment): the study runs (supervised models, anomaly detectors),
the combined-policy runs and the explanation runs, each with its ``metrics.json`` and
``runstats.npz``. Only runs whose config hash matches the current configs are used
(supervised / detector / explanation runs), and combined runs only if both of their source
runs are among those. Every CI is a run-level bootstrap from the per-run sums (ADR 0024):
per seed, then averaged over seeds, with the number of seeds whose own CI excludes zero.

Writes ``experiments/results/e5_explain.md``, a compact ``e5_explain.json`` and plots, with
LF line endings on every OS. Sections are labelled PRIMARY (ADR 0024 § Pre-registration,
read from ``e5.yaml`` ``primary:``) or EXPLORATORY.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twin_common.config import ConfigError
from twin_common.logging import configure_logging
from twin_ml.config import E5Config, E5FollowupConfig, EvalConfig
from twin_ml.evaluate import run_sums
from twin_ml.experiments.report_e1 import (
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
from twin_ml.experiments.report_robustness import (
    MODEL_ORDER,
    SERIES,
    _cell,
    _diff,
    _load_sums,
    _round,
)
from twin_ml.train import tracking

Entry = dict[str, Any]
Group = dict[int, Entry]  # seed (detector seed for threshold + detector) -> entry

SUP_STUDIES = ("e5_e1models", "e5_aug")
DETECTORS = ("iforest", "autoencoder")
EW = ("miss_rate", "rel_lead_mean_s", "false_alarms_per_h")
DIGITS = {
    "miss_rate": 3,
    "lead_mean_s": 0,
    "rel_lead_mean_s": 0,
    "false_alarms_per_h": 2,
    "size_mondrian_lac": 3,
    "size_pre_mondrian_lac": 3,
    "cov_mondrian_lac_FAULT": 3,
    "wrong_rate": 3,
    "wrong_conf_mean": 3,
    "wrong_high_frac": 3,
    "missed_rate": 3,
    "missed_conf_mean": 3,
    "missed_high_frac": 3,
    "sf_flag_rate": 3,
    "sf_flag_delay_s": 0,
    "sf_false_alarms_per_h": 2,
    "agreement_rate": 3,
    "del_top": 3,
    "del_rand": 3,
    "del_diff": 3,
    "del_top_gt_frac": 3,
    "ig_err_rel": 4,
    "ece_cal": 4,
}
LABEL = {
    "persistence": "Persistence",
    "threshold": "Threshold rule",
    "rf": "RF",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "iforest": "Isolation Forest",
    "autoencoder": "Autoencoder",
}
STUDY_LABEL = {"e5_e1models": "E1", "e5_aug": "aug"}


def _fields_for(metric: str) -> tuple[str, ...]:
    if metric.startswith(("size_pre_", "cov_pre_")):
        return ("conf_pre_",)
    if metric.startswith(("size_", "cov_", "singleton_", "empty_")):
        return ("conf_",)
    if metric.startswith(("wrong_", "missed_", "correct_")):
        return ("confw",)
    if metric.startswith("sf_flag") or metric == "sf_segments":
        return ("drift",)
    if metric.startswith("sf_"):
        return ("sf_false_alarms", "sf_negative_s")
    if metric.startswith("agreement"):
        return ("agree",)
    if metric.startswith("del_"):
        return ("deletion",)
    if metric.startswith("ig_"):
        return ("ig_complete",)
    if metric.startswith("ece"):
        return ("ece_cal", "ece_uncal", "ece_fault_cal", "ece_fault_uncal")
    return run_sums.RUNSTATS_FIELDS


def _select(sums: run_sums.Sums, fields: tuple[str, ...]) -> run_sums.Sums:
    out = {}
    for k, v in sums.items():
        if k in fields or any(k.startswith(f) and f.endswith("_") for f in fields):
            if fields == ("conf_",) and k.startswith("conf_pre_"):
                continue
            out[k] = v
    return out


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


class Stats:
    """Per-seed run-level bootstraps from the stored sums, cached."""

    def __init__(self, eval_cfg: EvalConfig) -> None:
        self.b = eval_cfg.bootstrap
        self.cache: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}

    def _boot(self, e: Entry, key: str, fields: tuple[str, ...]) -> dict[str, Any] | None:
        if key not in e["sums"]:
            return None
        ck = (e["run_id"], key, fields)
        if ck not in self.cache:
            s = _select(e["sums"][key], fields)
            if not s:  # e.g. IG completeness of a tree model
                return None
            self.cache[ck] = run_sums.bootstrap(s, self.b.n_resamples, self.b.seed, self.b.ci)
        return self.cache[ck]

    def stat(self, group: Group, key: str, metric: str) -> dict[str, Any]:
        pts, los, his = [], [], []
        for e in group.values():
            bt = self._boot(e, key, _fields_for(metric))
            if bt is None or metric not in bt:
                continue
            pts.append(bt[metric]["point"])
            los.append(bt[metric]["lo"])
            his.append(bt[metric]["hi"])
        finite = [p for p in pts if math.isfinite(p)]
        return {
            "point": _finite_mean(pts),
            "lo": _finite_mean(los),
            "hi": _finite_mean(his),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
            "seeds": len(pts),
        }

    def diff(
        self,
        a: Group,
        ka: str,
        b: Group,
        kb: str,
        metrics: tuple[str, ...],
        mode: str = "paired",
    ) -> dict[str, Any]:
        """Seed-wise a - b averaged over seeds; a single-entry side is reused for every seed."""
        if len(b) == 1:
            pairs = [(ea, next(iter(b.values()))) for ea in a.values()]
        elif len(a) == 1:
            pairs = [(next(iter(a.values())), eb) for eb in b.values()]
        else:
            pairs = [(a[s], b[s]) for s in sorted(set(a) & set(b))]
        per: list[dict[str, Any]] = []
        for ea, eb in pairs:
            if ka not in ea["sums"] or kb not in eb["sums"]:
                continue
            fields = tuple(sorted({f for m in metrics for f in _fields_for(m)}))
            sa, sb = _select(ea["sums"][ka], fields), _select(eb["sums"][kb], fields)
            if not sa or not sb:
                continue
            ca, cb = ka.split("|")[0], kb.split("|")[0]
            if mode == "paired":
                if ea["run_ids"].get(ca) != eb["run_ids"].get(cb):
                    raise ValueError(f"paired comparison over different runs: {ka} vs {kb}")
                fn = run_sums.paired
            else:
                fn = run_sums.unpaired
            per.append(fn(sa, sb, self.b.n_resamples, self.b.seed, self.b.ci, metrics))
        out: dict[str, Any] = {}
        for m in metrics:
            vals = {k: _finite_mean([r[m][k] for r in per]) for k in ("point", "lo", "hi")}
            lo, hi = vals["lo"], vals["hi"]
            out[m] = {
                **vals,
                "excludes_zero": bool(
                    math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0)
                ),
                "seeds_excluding_zero": int(sum(bool(r[m]["excludes_zero"]) for r in per)),
                "seeds": len(per),
            }
        return out


# ---------------------------------------------------------------------------- collect


def _entry(run: Any) -> Entry:
    import mlflow

    sums, run_ids = _load_sums(run.info.artifact_uri)
    try:
        detail = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/metrics.json")
    except Exception:  # combined runs log no metrics.json
        detail = {}
    return {
        "run_id": run.info.run_id,
        "tags": dict(run.data.tags),
        "detail": detail,
        "sums": sums,
        "run_ids": run_ids,
    }


def collect(
    e5: E5Config,
    eval_cfg: EvalConfig,
    root: Path,
    models_dir: Path,
    tracking_dir: Path,
    followup: E5FollowupConfig | None = None,
) -> dict[str, Any]:
    from twin_ml.config import load_robustness_config
    from twin_ml.experiments.e5_worker import (
        COMBINED_STUDY,
        EXPLAIN_STUDY,
        combined_sha,
        e5_content_sha,
        explain_config_sha,
        is_anomaly,
        model_config,
        study_config_sha,
    )

    rcfg = load_robustness_config(root / e5.robustness_config)
    content = e5_content_sha(e5, root)
    cfgs: dict[str, Any] = {}

    def cfg(m: str) -> Any:
        if m not in cfgs:
            try:
                cfgs[m] = model_config(e5, root, models_dir, m)
            except (ConfigError, FileNotFoundError):
                cfgs[m] = None
        return cfgs[m]

    def runs(study: str) -> list[Any]:
        return tracking.finished_runs(
            tracking_dir, e5.experiment, None, e5.phase_tag, f"tags.study = '{study}'"
        )

    studies: dict[str, dict[str, Group]] = {}
    valid_ids: set[str] = set()
    for study in e5.studies:
        chosen: dict[str, dict[int, Any]] = defaultdict(dict)
        for run in runs(study):  # newest first
            t = run.data.tags
            m, s = t["model"], int(t["seed"])
            c = cfg(m)
            if c is None or s in chosen[m]:
                continue
            src = "train" if is_anomaly(m) else t.get("source_run_id", "")
            want = study_config_sha(
                study, m, c, e5, rcfg, eval_cfg, t.get("dataset_key", ""), src, content
            )
            if t.get("study_config_sha") == want:
                chosen[m][s] = run
        studies[study] = {}
        for m in sorted(chosen, key=_model_key):
            studies[study][m] = {s: _entry(r) for s, r in sorted(chosen[m].items())}
            valid_ids |= {e["run_id"] for e in studies[study][m].values()}

    combined: dict[tuple[str, str, str], Group] = defaultdict(dict)
    for run in runs(COMBINED_STUDY):
        t = run.data.tags
        if t.get("supervised_run_id") not in valid_ids or t.get("detector_run_id") not in valid_ids:
            continue
        if t.get("combined_sha") != combined_sha(
            t["supervised_run_id"], t["detector_run_id"], e5, eval_cfg
        ):
            continue
        k = (t["source_study"], t["supervised"], t["detector"])
        seed = (
            int(t["detector_seed"])
            if t["supervised"] in ("threshold", "persistence")
            else int(t["seed"])
        )
        if seed not in combined[k]:
            combined[k][seed] = _entry(run)

    explain: dict[str, Group] = defaultdict(dict)
    for run in runs(EXPLAIN_STUDY):
        t = run.data.tags
        m, s = t["model"], int(t["seed"])
        c = cfg(m)
        if c is None or s in explain[m]:
            continue
        want = explain_config_sha(
            m, c, e5, eval_cfg, t.get("dataset_key", ""), t.get("source_run_id", ""), content
        )
        if t.get("study_config_sha") == want:
            explain[m][s] = _entry(run)
    out: dict[str, Any] = {
        "studies": studies,
        "combined": dict(combined),
        "explain": dict(explain),
        "ae_shares": {},
        "ig_sensitivity": {},
    }
    if followup is None:
        return out

    # exploratory Part C follow-ups (e5_followup.yaml), validated by their own hashes
    from twin_ml.experiments.e5_worker import (
        AE_SHARES_STUDY,
        IG_SENSITIVITY_STUDY,
        ae_shares_sha,
    )

    ae: Group = {}
    det_c = cfg(followup.ae_shares.detector)
    for run in runs(AE_SHARES_STUDY):
        t = run.data.tags
        s = int(t["seed"])
        src = t.get("source_run_id", "")
        if det_c is None or s in ae or src not in valid_ids:
            continue
        want = ae_shares_sha(det_c, e5, followup, eval_cfg, t.get("dataset_key", ""), src, content)
        if t.get("study_config_sha") == want:
            ae[s] = _entry(run)
    out["ae_shares"] = dict(sorted(ae.items()))
    ig: dict[str, Group] = defaultdict(dict)
    steps = followup.ig_sensitivity.ig_steps
    for run in runs(IG_SENSITIVITY_STUDY):
        t = run.data.tags
        m, s = t["model"], int(t["seed"])
        c = cfg(m)
        if c is None or s in ig[m]:
            continue
        want = explain_config_sha(
            m,
            c,
            e5,
            eval_cfg,
            t.get("dataset_key", ""),
            t.get("source_run_id", ""),
            content,
            IG_SENSITIVITY_STUDY,
            steps,
        )
        if t.get("study_config_sha") == want:
            ig[m][s] = _entry(run)
    out["ig_sensitivity"] = {m: dict(sorted(g.items())) for m, g in ig.items()}
    return out


def _model_key(m: str) -> tuple[int, str]:
    order = (*MODEL_ORDER, *DETECTORS)
    return (order.index(m) if m in order else len(order), m)


# ---------------------------------------------------------------------------- build


def _counts(group: Group, cond: str, key: str) -> dict[str, Any]:
    """Episodes (per kind if available), runs and negative hours behind a figure."""
    if not group:
        return {}
    e = next(iter(group.values()))
    s = e["sums"].get(key)
    if s is None:
        return {}
    out: dict[str, Any] = {
        "episodes": int(s["episodes"].sum()),
        "runs": int(s["episodes"].shape[0]),
        "negative_h": float(s["negative_s"].sum()) / 3600.0,
    }
    d = e["detail"].get("conditions", {}).get(cond, {})
    ew = d.get("horizons", {}).get(key.split("|")[1][1:], {}).get("early_warning")
    if ew and "episodes_per_kind" in ew:
        out["episodes_per_kind"] = ew["episodes_per_kind"]
    return out


def build(e5: E5Config, eval_cfg: EvalConfig, data: dict[str, Any]) -> dict[str, Any]:
    st = Stats(eval_cfg)
    p = e5.primary
    h = f"h{p.horizon}"
    pk = f"p{p.anomaly_percentile:g}"
    studies, combined, explain = data["studies"], data["combined"], data["explain"]
    e1 = studies.get("e5_e1models", {})
    thr = e1.get("threshold", {})
    cond_of = p.early_warning  # metric -> condition

    def ew_row(
        group: Group, grp_key: str, label: str, kind: str, vs_sup: tuple[Group, str] | None = None
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "label": label,
            "kind": kind,
            "values": {},
            "vs_threshold": {},
            "seeds": len(group),
        }
        for m in EW:
            c = cond_of[m]
            row["values"][m] = st.stat(group, f"{c}|{grp_key}", m)
            if thr and group is not thr:
                row["vs_threshold"][m] = st.diff(group, f"{c}|{grp_key}", thr, f"{c}|{h}", (m,))[m]
            if vs_sup is not None:
                sg, sk = vs_sup
                row.setdefault("vs_supervised", {})[m] = st.diff(
                    group, f"{c}|{grp_key}", sg, f"{c}|{sk}", (m,)
                )[m]
        return row

    primary_ew: list[dict[str, Any]] = []
    for study in SUP_STUDIES:
        for m, g in studies.get(study, {}).items():
            if m == "persistence":
                continue
            lab = LABEL.get(m, m) + ("" if study == "e5_e1models" else " (aug)")
            primary_ew.append(ew_row(g, h, lab, "supervised" if m != "threshold" else "reference"))
    det = studies.get("e5_anomaly", {})
    for m in DETECTORS:
        if m in det:
            primary_ew.append(
                ew_row(det[m], pk, f"{LABEL[m]} (p{p.anomaly_percentile:g})", "detector")
            )
    for (sst, sm, dm), g in sorted(
        combined.items(),
        key=lambda kv: (
            SUP_STUDIES.index(kv[0][0]) if kv[0][0] in SUP_STUDIES else 9,
            _model_key(kv[0][1]),
            kv[0][2],
        ),
    ):
        if sst not in SUP_STUDIES:
            continue
        sup = studies.get(sst, {}).get(sm, {})
        lab = f"{LABEL.get(sm, sm)}{'' if sst == 'e5_e1models' else ' (aug)'} OR {LABEL[dm]}"
        primary_ew.append(ew_row(g, pk, lab, "combined", (sup, h) if sup else None))

    set_rows = []
    for study in SUP_STUDIES:
        for m, g in studies.get(study, {}).items():
            if m in ("persistence", "threshold"):
                continue
            row = {"label": LABEL.get(m, m) + ("" if study == "e5_e1models" else " (aug)")}
            for metric in p.conformal:
                row[metric] = {
                    "e5": st.stat(g, f"e5|{h}", metric),
                    "clean": st.stat(g, f"clean|{h}", metric),
                    "diff": st.diff(g, f"e5|{h}", g, f"clean|{h}", (metric,), "unpaired")[metric],
                }
            set_rows.append(row)

    agree_rows = []
    for m in (*e5.explain.primary_models, *e5.explain.exploratory_models):
        g = explain.get(m)
        if not g:
            continue
        e0 = g[min(g)]  # seed 0 (the episode counts in the table are seed 0)
        fts = e0["detail"].get("fault_types", [])
        arow: dict[str, Any] = {
            "model": m,
            "label": LABEL.get(m, m),
            "primary": m in p.explanation_models,
            "method": e0["detail"].get("method"),
        }
        for c in ("clean", "e5"):
            arow[c] = {
                mm: st.stat(g, f"{c}|{h}", mm)
                for mm in (
                    "agreement_rate",
                    "agreement_episodes",
                    "del_top",
                    "del_rand",
                    "del_diff",
                    "del_top_gt_frac",
                    "ig_err_rel",
                )
            }
            info = e0["detail"]["conditions"].get(c, {})
            arow[c]["episodes_total"] = info.get("episodes_total")
            arow[c]["episodes_explained"] = info.get("episodes_explained")
            arow[c]["by_type"] = {}
            for code, ft in enumerate(fts):
                s = st.stat(g, f"{c}|{h}", f"agreement_rate_t{code}")
                if s["seeds"]:
                    arow[c]["by_type"][ft] = s
            arow[c]["mean_weight"] = {
                ft: np.mean(
                    [
                        e["detail"]["conditions"][c]["by_fault_type"][ft]["mean_weight"]
                        for e in g.values()
                        if ft in e["detail"]["conditions"].get(c, {}).get("by_fault_type", {})
                    ],
                    axis=0,
                ).tolist()
                for ft in info.get("by_fault_type", {})
            }
        # pooled over both conditions: concatenate the run sums (clean runs, then E5 runs)
        pooled: Group = {}
        for s_, e in g.items():
            a = [e["sums"][f"{c}|{h}"]["agree"] for c in ("clean", "e5") if f"{c}|{h}" in e["sums"]]
            if a:
                pooled[s_] = {
                    "run_id": e["run_id"] + ":pooled",
                    "sums": {"all|h": {"agree": np.concatenate(a, axis=0)}},
                    "run_ids": {},
                }
        arow["pooled"] = st.stat(pooled, "all|h", "agreement_rate")
        agree_rows.append(arow)

    # ---------------------------------------------------------------- exploratory
    tradeoff = []
    for m in DETECTORS:
        if m not in det:
            continue
        for pct in e5.anomaly.threshold_percentiles:
            k = f"p{pct:g}"
            tradeoff.append(
                {
                    "label": LABEL[m],
                    "percentile": pct,
                    "miss_rate": st.stat(det[m], f"e5|{k}", "miss_rate"),
                    "rel_lead_mean_s": st.stat(det[m], f"e5|{k}", "rel_lead_mean_s"),
                    "fa_clean": st.stat(det[m], f"clean|{k}", "false_alarms_per_h"),
                    "clean_miss_rate": st.stat(det[m], f"clean|{k}", "miss_rate"),
                }
            )
    comb_tradeoff = []
    for (sst, sm, dm), g in combined.items():
        if sst != "e5_e1models":
            continue
        for pct in e5.combined.percentiles:
            k = f"p{pct:g}"
            comb_tradeoff.append(
                {
                    "label": f"{LABEL.get(sm, sm)} OR {LABEL[dm]}",
                    "supervised": sm,
                    "detector": dm,
                    "percentile": pct,
                    "miss_rate": st.stat(g, f"e5|{k}", "miss_rate"),
                    "fa_clean": st.stat(g, f"clean|{k}", "false_alarms_per_h"),
                }
            )
    confidence = []
    for study in SUP_STUDIES:
        for m, g in studies.get(study, {}).items():
            if m in ("persistence", "threshold"):
                continue
            row = {"label": LABEL.get(m, m) + ("" if study == "e5_e1models" else " (aug)")}
            for metric in (
                "wrong_rate",
                "wrong_high_frac",
                "missed_rate",
                "missed_conf_mean",
                "missed_high_frac",
                "ece_cal",
                "size_pre_mondrian_lac",
            ):
                row[metric] = {
                    "e5": st.stat(g, f"e5|{h}", metric),
                    "clean": st.stat(g, f"clean|{h}", metric),
                }
            heads = [
                e["detail"]["conditions"].get("e5", {}).get("fault_type_head_precursor")
                for e in g.values()
            ]
            heads = [x for x in heads if x]
            if heads:
                tot: defaultdict[str, int] = defaultdict(int)
                for x in heads:
                    for k2, v in x.items():
                        tot[k2] += v
                n = sum(tot.values()) or 1
                row["fault_type_head"] = {k2: v / n for k2, v in sorted(tot.items())}
            row["per_seed"] = {
                str(s): {
                    mm: run_sums.point(_select(e["sums"][f"e5|{h}"], _fields_for(mm))).get(mm)
                    for mm in ("miss_rate", "rel_lead_mean_s")
                }
                for s, e in sorted(g.items())
                if f"e5|{h}" in e["sums"]
            }
            if study == "e5_e1models" and m in explain:
                row["explained_per_seed"] = {
                    str(s): e["detail"]["conditions"].get("e5", {}).get("episodes_explained")
                    for s, e in sorted(explain[m].items())
                }
            confidence.append(row)
    drift = []
    for study in SUP_STUDIES:
        for m, g in studies.get(study, {}).items():
            if m == "persistence":
                continue
            drift.append(
                _drift_row(
                    st,
                    g,
                    f"sf_drift|{h}",
                    LABEL.get(m, m) + ("" if study == "e5_e1models" else " (aug)"),
                    "supervised",
                )
            )
    for m in DETECTORS:
        if m in det:
            drift.append(_drift_row(st, det[m], f"sf_drift|{pk}", LABEL[m], "detector"))
    for (sst, sm, dm), g in combined.items():
        if sst == "e5_e1models":
            drift.append(
                _drift_row(
                    st, g, f"sf_drift|{pk}", f"{LABEL.get(sm, sm)} OR {LABEL[dm]}", "combined"
                )
            )
    e4 = []
    e4s = studies.get("e5_e4", {})
    for m, g in e4s.items():
        k = pk if m in DETECTORS else h
        e4.append(
            _e4_row(st, g, k, LABEL.get(m, m), "detector" if m in DETECTORS else "supervised")
        )
    for (sst, sm, dm), g in combined.items():
        if sst == "e5_e4":
            e4.append(_e4_row(st, g, pk, f"{LABEL.get(sm, sm)} OR {LABEL[dm]}", "combined"))
    auprc = []
    for m in DETECTORS:
        if m in det:
            for c in ("clean", "e5", "sf_drift"):
                vals = [
                    e["detail"]["conditions"].get(c, {}).get("auprc_precursor")
                    for e in det[m].values()
                ]
                auprc.append(
                    {
                        "label": LABEL[m],
                        "condition": c,
                        "mean": _finite_mean([v for v in vals if v is not None]),
                    }
                )
    ae_shares = {}
    if "autoencoder" in det:
        e0 = det["autoencoder"]
        ae_shares = {
            c: [
                e["detail"]["conditions"].get(c, {}).get("sensor_error_shares") for e in e0.values()
            ]
            for c in ("clean", "e5", "sf_drift")
        }

    any_e1: Group = next(iter(e1.values()), {}) if e1 else {}
    counts = {
        "e5": _counts(any_e1, "e5", f"e5|{h}") if any_e1 else {},
        "clean": _counts(any_e1, "clean", f"clean|{h}") if any_e1 else {},
        "sf_drift": _counts(any_e1, "sf_drift", f"sf_drift|{h}") if any_e1 else {},
    }
    if e4s:
        g0 = next(iter(e4s.values()))
        counts["e4_high"] = _counts(g0, "e4_high", f"e4_high|{h}")
        counts["e4_in"] = _counts(g0, "e4_in", f"e4_in|{h}")
    n_runs = {
        "studies": {s: {m: len(g) for m, g in v.items()} for s, v in studies.items()},
        "combined": len(combined),
        "combined_runs": sum(len(g) for g in combined.values()),
        "explain": {m: len(g) for m, g in explain.items()},
    }
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "experiment": e5.experiment,
        "primary_config": p.model_dump(mode="json"),
        "design": e5.e5_data.model_dump(mode="json"),
        "counts": counts,
        "runs": n_runs,
        "primary": {"early_warning": primary_ew, "set_size": set_rows, "agreement": agree_rows},
        "exploratory": {
            "tradeoff": tradeoff,
            "combined_tradeoff": comb_tradeoff,
            "confidence": confidence,
            "drift": drift,
            "e4": e4,
            "auprc": auprc,
            "ae_sensor_shares": ae_shares,
            "ae_shares": _ae_shares_section(st, data.get("ae_shares", {})),
            "ig_sensitivity": _ig_section(
                st, h, data.get("explain", {}), data.get("ig_sensitivity", {})
            ),
        },
        "plots": {},
    }


def _ae_shares_section(st: Stats, g: Group) -> list[dict[str, Any]]:
    """Exploratory: the autoencoder's top per-sensor error share vs the expected sensor."""
    if not g:
        return []
    e0 = g[min(g)]
    d0 = e0["detail"]
    fts: list[str] = d0.get("fault_types", [])
    expected: dict[str, str] = d0.get("expected_sensor", {})
    out = []
    for cond, info in d0.get("conditions", {}).items():
        sensors = info["sensors"]
        drift = info["unit"] == "segment"
        names = sensors if drift else fts
        types = []
        for code, name in enumerate(names):
            ep = st.stat(g, f"{cond}|ae", f"agreement_rate_t{code}")
            if not ep["seeds"]:
                continue
            shares = [
                e["detail"]["conditions"][cond]["by_type"][name]["mean_share"]
                for e in g.values()
                if name in e["detail"]["conditions"].get(cond, {}).get("by_type", {})
            ]
            types.append(
                {
                    "type": name,
                    "expected": name if drift else expected.get(name),
                    "units": info["by_type"].get(name, {}).get("units", 0),
                    "unit_rate": ep,
                    "window_rate": st.stat(g, f"{cond}|ae_w", f"agreement_rate_t{code}"),
                    "mean_share": dict(zip(sensors, np.mean(shares, axis=0).tolist(), strict=True))
                    if shares
                    else {},
                }
            )
        out.append(
            {
                "condition": cond,
                "unit": info["unit"],
                "units_total": info["units_total"],
                "units_explained": info["units_explained"],
                "overall_unit": st.stat(g, f"{cond}|ae", "agreement_rate"),
                "overall_window": st.stat(g, f"{cond}|ae_w", "agreement_rate"),
                "types": types,
                "seeds": len(g),
            }
        )
    return out


def _ig_section(
    st: Stats, h: str, explain: dict[str, Group], ig: dict[str, Group]
) -> list[dict[str, Any]]:
    """Exploratory: agreement and completeness at the sensitivity step count vs the primary 64."""
    out = []
    for m, g in ig.items():
        base = explain.get(m, {})
        if not base:
            continue
        common = {s: g[s] for s in sorted(set(g) & set(base))}
        bcommon = {s: base[s] for s in common}
        steps = next(iter(common.values()))["detail"].get("ig_steps") if common else None
        for cond in ("clean", "e5"):
            key = f"{cond}|{h}"
            changed = total = 0
            for s in common:
                a = {
                    r["episode"]: r["top"]
                    for r in common[s]["detail"]["conditions"].get(cond, {}).get("episodes", [])
                }
                b = {
                    r["episode"]: r["top"]
                    for r in bcommon[s]["detail"]["conditions"].get(cond, {}).get("episodes", [])
                }
                for ep in set(a) & set(b):
                    total += 1
                    changed += int(a[ep] != b[ep])
            out.append(
                {
                    "model": m,
                    "label": LABEL.get(m, m),
                    "condition": cond,
                    "ig_steps": steps,
                    "seeds": len(common),
                    "agree_base": st.stat(bcommon, key, "agreement_rate"),
                    "agree_sens": st.stat(common, key, "agreement_rate"),
                    "agree_diff": st.diff(common, key, bcommon, key, ("agreement_rate",))[
                        "agreement_rate"
                    ],
                    "episodes_compared": total,
                    "top_changed": changed,
                    "ig_err_base": st.stat(bcommon, key, "ig_err_rel"),
                    "ig_err_sens": st.stat(common, key, "ig_err_rel"),
                    "del_diff_sens": st.stat(common, key, "del_diff"),
                }
            )
    return out


def _drift_row(st: Stats, g: Group, key: str, label: str, kind: str) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "sf_flag_rate": st.stat(g, key, "sf_flag_rate"),
        "sf_flag_delay_s": st.stat(g, key, "sf_flag_delay_s"),
        "sf_false_alarms_per_h": st.stat(g, key, "sf_false_alarms_per_h"),
        "miss_rate": st.stat(g, key, "miss_rate"),
    }


def _e4_row(st: Stats, g: Group, k: str, label: str, kind: str) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "fa_high": st.stat(g, f"e4_high|{k}", "false_alarms_per_h"),
        "fa_in": st.stat(g, f"e4_in|{k}", "false_alarms_per_h"),
        "fa_diff": st.diff(g, f"e4_high|{k}", g, f"e4_in|{k}", ("false_alarms_per_h",), "unpaired")[
            "false_alarms_per_h"
        ],
        "miss_high": st.stat(g, f"e4_high|{k}", "miss_rate"),
        "miss_in": st.stat(g, f"e4_in|{k}", "miss_rate"),
    }


# ---------------------------------------------------------------------------- markdown


def _d(m: str) -> int:
    return DIGITS.get(m, 3)


def _count_text(c: dict[str, Any]) -> str:
    if not c:
        return NA
    kinds = c.get("episodes_per_kind")
    k = f" ({', '.join(f'{a} {b}' for a, b in kinds.items())})" if kinds else ""
    return (
        f"{c['runs']} runs, {c['episodes']} gradual-fault episodes{k}, "
        f"{c['negative_h']:.1f} h negative windows"
    )


def render(report: dict[str, Any]) -> str:
    p = report["primary_config"]
    c = report["counts"]
    pct = f"{p['anomaly_percentile']:g}"
    lines: list[str] = [
        "# E5: unseen fault, anomaly detection and explanations",
        "",
        f"Generated {report['generated_at']} from MLflow experiment `{report['experiment']}`"
        f" (commit {report['git_commit'] or NA}) by `python -m twin_ml.experiments.report_e5`."
        " Every number comes from the logged runs; nothing is typed by hand.",
        "",
        "Protocol and pre-registration: "
        "[ADR 0024](../../docs/adr/0024-e5-unseen-fault-protocol.md)."
        " Detectors: [ADR 0025](../../docs/adr/0025-anomaly-detection.md)."
        " Explanations: [ADR 0026](../../docs/adr/0026-explainability.md).",
        "",
        "## Data",
        "",
        f"- **E5** (held-out `lubrication_loss` only): {_count_text(c.get('e5', {}))}.",
        f"- **Clean test** (false alarms, set size reference): {_count_text(c.get('clean', {}))}.",
        f"- **`sf_drift`**: {_count_text(c.get('sf_drift', {}))}.",
    ]
    if "e4_high" in c:
        lines += [
            f"- **E4** `e4_high`: {_count_text(c['e4_high'])}; `e4_in`: {_count_text(c['e4_in'])}."
        ]
    lines += [
        "",
        "Cells: mean over seeds of the per-seed point estimate [mean of the per-seed 95 % run-level"
        " bootstrap CI]. Differences: `+x [lo, hi] * (k/n)`, where `*` marks a seed-averaged CI"
        " that excludes zero and k/n the seeds whose own CI excludes zero. Horizon +30 s.",
        "",
        "**Interpretation note (fixed in ADR 0024 before any result).** "
        "The threshold baseline checks"
        " the observed current against the nominal overcurrent limit, exactly the channel"
        " `lubrication_loss` trips on by design, while no supervised model was trained on a"
        " current-led fault. Any advantage of the threshold rule on E5 must be read with this in"
        " mind.",
        "",
        "## PRIMARY: pre-registered outcomes",
        "",
        f"### Early warning: E5 miss rate and lead, clean false alarms (detectors at p{pct})",
        "",
    ]
    rows = []
    for r in report["primary"]["early_warning"]:
        v, d = r["values"], r["vs_threshold"]
        rows.append(
            [
                r["label"],
                _cell(v["miss_rate"], 3),
                _diff(d.get("miss_rate", {}), 3),
                _cell(v["rel_lead_mean_s"], 0),
                _diff(d.get("rel_lead_mean_s", {}), 0),
                _cell(v["false_alarms_per_h"], 2),
                _diff(d.get("false_alarms_per_h", {}), 2),
            ]
        )
    lines += _table(
        [
            "Policy",
            "E5 miss rate",
            "Δ vs threshold",
            "E5 lead vs sim WARNING (s)",
            "Δ vs threshold",
            "Clean FA / h",
            "Δ vs threshold",
        ],
        rows,
    )
    comb = [r for r in report["primary"]["early_warning"] if "vs_supervised" in r]
    if comb:
        lines += ["", "#### Combined policy vs its supervised model alone (paired)", ""]
        lines += _table(
            ["Combined policy", "Δ E5 miss rate", "Δ E5 lead (s)", "Δ clean FA / h"],
            [
                [
                    r["label"],
                    _diff(r["vs_supervised"]["miss_rate"], 3),
                    _diff(r["vs_supervised"]["rel_lead_mean_s"], 0),
                    _diff(r["vs_supervised"]["false_alarms_per_h"], 2),
                ]
                for r in comb
            ],
        )
    lines += [
        "",
        "### Uncertainty under the unseen fault: Mondrian LAC mean set size, E5 vs clean"
        " (operating windows, unpaired)",
        "",
    ]
    lines += _table(
        ["Model", "Clean", "E5", "Δ E5 - clean"],
        [
            [
                r["label"],
                _cell(r["size_mondrian_lac"]["clean"], 3),
                _cell(r["size_mondrian_lac"]["e5"], 3),
                _diff(r["size_mondrian_lac"]["diff"], 3),
            ]
            for r in report["primary"]["set_size"]
        ],
    )
    lines += [
        "",
        "Split-conformal guarantees assume exchangeable data and do not hold on E5; set size"
        " is read as a measure of uncertainty, not as coverage.",
        "",
    ]
    lines += ["### Explanation agreement with the fault physics", ""]
    ag = [r for r in report["primary"]["agreement"] if r["primary"]]
    if ag:
        lines += _table(
            [
                "Model (method)",
                "Pooled",
                "Clean (seen types)",
                "E5 (lubrication_loss)",
                "Episodes explained clean / E5",
            ],
            [
                [
                    f"{r['label']} ({r['method']})",
                    _cell(r["pooled"], 3),
                    _cell(r["clean"]["agreement_rate"], 3),
                    _cell(r["e5"]["agreement_rate"], 3),
                    f"{r['clean'].get('episodes_explained', NA)}/"
                    f"{r['clean'].get('episodes_total', NA)}, "
                    f"{r['e5'].get('episodes_explained', NA)}/{r['e5'].get('episodes_total', NA)}",
                ]
                for r in ag
            ],
        )
        lines += [
            "",
            "Expected sensor: bearing_wear → vib, overheating → temp, spindle_imbalance → vib,"
            " lubrication_loss → current. Episodes the model never alarms on are not explained."
            " Episode counts are from seed 0.",
            "",
        ]
        types = sorted({t for r in ag for c2 in ("clean", "e5") for t in r[c2]["by_type"]})
        lines += _table(
            ["Model", *types],
            [
                [
                    r["label"],
                    *[
                        _cell(r["clean"]["by_type"].get(t) or r["e5"]["by_type"].get(t) or {}, 3)
                        for t in types
                    ],
                ]
                for r in ag
            ],
        )
    else:
        lines += ["Not available: no explanation runs with the current configs.", ""]
    lines += ["", "## EXPLORATORY", "", "### Threshold percentile trade-off (detectors)", ""]
    lines += _table(
        [
            "Detector",
            "Percentile",
            "E5 miss rate",
            "E5 lead (s)",
            "Clean FA / h",
            "Clean miss rate",
        ],
        [
            [
                r["label"],
                f"{r['percentile']:g}",
                _cell(r["miss_rate"], 3),
                _cell(r["rel_lead_mean_s"], 0),
                _cell(r["fa_clean"], 2),
                _cell(r["clean_miss_rate"], 3),
            ]
            for r in report["exploratory"]["tradeoff"]
        ],
    )
    lines += ["", "### Combined policy across percentiles (E1 models)", ""]
    lines += _table(
        ["Combined policy", "Percentile", "E5 miss rate", "Clean FA / h"],
        [
            [r["label"], f"{r['percentile']:g}", _cell(r["miss_rate"], 3), _cell(r["fa_clean"], 2)]
            for r in report["exploratory"]["combined_tradeoff"]
        ],
    )
    lines += [
        "",
        "### Confidence of calibrated predictions (operating windows)",
        "",
        "Wrong = calibrated argmax differs from the true class at +30 s; missed = a WARNING or"
        " FAULT target predicted NORMAL; high = calibrated confidence ≥ 0.9.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Wrong rate clean / E5",
            "Wrong with conf ≥ 0.9 clean / E5",
            "Missed rate clean / E5",
            "Missed: mean conf clean / E5",
            "Missed with conf ≥ 0.9 clean / E5",
            "Precursor set size clean / E5",
        ],
        [
            [r["label"]]
            + [
                f"{_cell(r[m]['clean'], 3)} / {_cell(r[m]['e5'], 3)}"
                for m in (
                    "wrong_rate",
                    "wrong_high_frac",
                    "missed_rate",
                    "missed_conf_mean",
                    "missed_high_frac",
                    "size_pre_mondrian_lac",
                )
            ]
            for r in report["exploratory"]["confidence"]
        ],
    )
    heads = [r for r in report["exploratory"]["confidence"] if r.get("fault_type_head")]
    if heads:
        names = sorted({k for r in heads for k in r["fault_type_head"]})
        lines += [
            "",
            "### Fault-type head on E5 precursor windows (share of windows per predicted"
            " type, pooled over seeds)",
            "",
        ]
        lines += _table(
            ["Model", *names],
            [[r["label"], *[_fmt(r["fault_type_head"].get(n), 3) for n in names]] for r in heads],
        )
    seeded = [r for r in report["exploratory"]["confidence"] if r.get("per_seed")]
    if seeded:
        seeds = sorted({s for r in seeded for s in r["per_seed"]}, key=int)
        lines += [
            "",
            "### Seed-to-seed variation on E5 (point estimate per seed)",
            "",
            "Explained = number of E5 episodes with at least one alarmed precursor window"
            " (explanation study, E1 models; see Data for the episode total).",
            "",
        ]
        lines += _table(
            [
                "Model",
                *[f"Miss rate s{s}" for s in seeds],
                *[f"Lead (s) s{s}" for s in seeds],
                *[f"Explained s{s}" for s in seeds],
            ],
            [
                [
                    r["label"],
                    *[_fmt(r["per_seed"].get(s, {}).get("miss_rate"), 3) for s in seeds],
                    *[_fmt(r["per_seed"].get(s, {}).get("rel_lead_mean_s"), 0) for s in seeds],
                    *[
                        NA
                        if r.get("explained_per_seed", {}).get(s) is None
                        else str(r["explained_per_seed"][s])
                        for s in seeds
                    ],
                ]
                for r in seeded
            ],
        )
    lines += [
        "",
        "### Sensor drift (`sf_drift`)",
        "",
        "Flag rate: share of drift segments with a sustained alarm on a normal-operation window"
        " inside the segment. For a detector a flag is the desired outcome; for a fault-warning"
        " policy the same alarm is a false fault warning (FA / h inside drift segments).",
        "",
    ]
    lines += _table(
        ["Policy", "Kind", "Segments flagged", "Time to flag (s)", "FA / h inside drift"],
        [
            [
                r["label"],
                r["kind"],
                _cell(r["sf_flag_rate"], 3),
                _cell(r["sf_flag_delay_s"], 0),
                _cell(r["sf_false_alarms_per_h"], 2),
            ]
            for r in report["exploratory"]["drift"]
        ],
    )
    if report["exploratory"]["e4"]:
        lines += [
            "",
            "### Load shift (E4: trained on low + medium load; `e4_high` vs `e4_in`, unpaired)",
            "",
        ]
        lines += _table(
            [
                "Policy",
                "Kind",
                "FA / h e4_in",
                "FA / h e4_high",
                "Δ high - in",
                "Miss rate in / high",
            ],
            [
                [
                    r["label"],
                    r["kind"],
                    _cell(r["fa_in"], 2),
                    _cell(r["fa_high"], 2),
                    _diff(r["fa_diff"], 2),
                    f"{_cell(r['miss_in'], 3)} / {_cell(r['miss_high'], 3)}",
                ]
                for r in report["exploratory"]["e4"]
            ],
        )
    if report["exploratory"]["auprc"]:
        lines += [
            "",
            "### Detector score AUPRC (operating windows; positive = WARNING / FAULT at"
            " +30 s; mean over seeds)",
            "",
        ]
        lines += _table(
            ["Detector", "Condition", "AUPRC"],
            [
                [r["label"], r["condition"], _fmt(r["mean"], 3)]
                for r in report["exploratory"]["auprc"]
            ],
        )
    allag = report["primary"]["agreement"]
    if allag:
        lines += [
            "",
            "### Deletion test and IG completeness",
            "",
            "Mean |Δ p(WARNING) + p(FAULT)| when the top-attributed sensor vs one random other"
            " sensor is replaced by the baseline; difference with a run-level CI.",
            "",
        ]
        lines += _table(
            [
                "Model",
                "Condition",
                "Top",
                "Random",
                "Top - random",
                "Windows top > random",
                "IG relative completeness error",
            ],
            [
                [
                    r["label"],
                    c2,
                    _cell(r[c2]["del_top"], 3),
                    _cell(r[c2]["del_rand"], 3),
                    _cell(r[c2]["del_diff"], 3),
                    _cell(r[c2]["del_top_gt_frac"], 3),
                    _cell(r[c2]["ig_err_rel"], 4),
                ]
                for r in allag
                for c2 in ("clean", "e5")
            ],
        )
        rf = [r for r in allag if not r["primary"]]
        if rf:
            lines += [
                "",
                "**Random Forest (Saabas path attributions; exploratory, not comparable with"
                " the other models: path attributions over-credit features split near the"
                " root).**",
                "",
            ]
            lines += _table(
                ["Model", "Pooled", "Clean", "E5"],
                [
                    [
                        r["label"],
                        _cell(r["pooled"], 3),
                        _cell(r["clean"]["agreement_rate"], 3),
                        _cell(r["e5"]["agreement_rate"], 3),
                    ]
                    for r in rf
                ],
            )
    lines += _render_followups(report["exploratory"])
    if report["plots"]:
        lines += ["", "## Plots", ""]
        lines += [f"![{k}]({v})" for k, v in report["plots"].items()]
    return LF.join(lines) + LF


def _render_followups(ex: dict[str, Any]) -> list[str]:
    """The exploratory Part C follow-ups (``e5_followup.yaml``, ADR 0024 amendment)."""
    lines: list[str] = []
    ae = ex.get("ae_shares") or []
    if ae:
        lines += [
            "",
            "### Autoencoder per-sensor reconstruction-error shares (exploratory follow-up)",
            "",
            "Added after the results were seen (ADR 0024 § Amendment). Autoencoder at the primary"
            " percentile, re-fitted from its seed; every re-fit reproduced its stored alarms"
            " exactly. Share = a sensor's squared reconstruction error / the window's total"
            " (vib_norm folded into vib). Windows as for the explanations (alarmed, one per 10 s,"
            " at most 30 per unit). Unit agreement: the top sensor of the unit's mean share is the"
            " expected one (precursor episodes: the fault type's leading sensor; `sf_drift`: each"
            " flagged drift segment's drifting sensor, flagged = sustained alarm on a normal"
            " window inside the segment). Window agreement: each window's own top share. The"
            " shares are not attributions: an error on one input can raise the reconstruction"
            " error of another (ADR 0025).",
            "",
        ]
        rows = []
        for c in ae:
            unit = "episodes" if c["unit"] == "episode" else "flagged segments"
            rows.append(
                [
                    f"**{c['condition']}**",
                    "all",
                    NA,
                    f"{c['units_explained']} / {c['units_total']} {unit}",
                    _cell(c["overall_unit"], 3),
                    _cell(c["overall_window"], 3),
                    "",
                ]
            )
            for t in c["types"]:
                ms = t["mean_share"]
                share_txt = ", ".join(f"{k} {v:.2f}" for k, v in ms.items())
                rows.append(
                    [
                        c["condition"],
                        t["type"],
                        t["expected"] or NA,
                        str(t["units"]),
                        _cell(t["unit_rate"], 3),
                        _cell(t["window_rate"], 3),
                        share_txt,
                    ]
                )
        lines += _table(
            [
                "Condition",
                "Fault type / drifting sensor",
                "Expected top",
                "Units (seed 0)",
                "Unit agreement",
                "Window agreement",
                "Mean share per sensor",
            ],
            rows,
        )
    ig = ex.get("ig_sensitivity") or []
    if ig:
        steps = ig[0]["ig_steps"]
        lines += [
            "",
            f"### Integrated Gradients at {steps} steps vs 64 (exploratory sensitivity check)",
            "",
            "Added after the results were seen (ADR 0024 § Amendment). Same models, windows and"
            " baseline as the explanation study; only the step count differs. The 64-step"
            " results above stay primary. Δ is paired on the same runs.",
            "",
        ]
        lines += _table(
            [
                "Model",
                "Condition",
                "Agreement 64",
                f"Agreement {steps}",
                f"Δ {steps} - 64",
                "Episodes whose top sensor changed",
                "IG rel. completeness error 64",
                f"IG rel. completeness error {steps}",
                f"Deletion top - random ({steps})",
            ],
            [
                [
                    r["label"],
                    r["condition"],
                    _cell(r["agree_base"], 3),
                    _cell(r["agree_sens"], 3),
                    _diff(r["agree_diff"], 3),
                    f"{r['top_changed']} of {r['episodes_compared']} (all seeds)",
                    _cell(r["ig_err_base"], 4),
                    _cell(r["ig_err_sens"], 4),
                    _cell(r["del_diff_sens"], 3),
                ]
                for r in ig
            ],
        )
    return lines


def _fmt(x: Any, digits: int) -> str:
    if x is None or not isinstance(x, int | float) or not math.isfinite(x):
        return NA
    return f"{x:.{digits}f}"


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


def plots(report: dict[str, Any], out: Path, name: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    def save(fig: Any, key: str) -> None:
        path = out / f"{name}_{key}.png"
        fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        report["plots"][key] = path.name
        paths.append(path)

    ew = [r for r in report["primary"]["early_warning"] if r["kind"] != "combined"]
    if ew:
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(ew) + 1.2), facecolor=SURFACE)
        labels = [r["label"] for r in ew]
        y = np.arange(len(ew))
        for ax, m, title in (
            (axes[0], "miss_rate", "E5 miss rate"),
            (axes[1], "false_alarms_per_h", "Clean false alarms / h"),
        ):
            pts = [r["values"][m]["point"] for r in ew]
            lo = [r["values"][m]["lo"] for r in ew]
            hi = [r["values"][m]["hi"] for r in ew]
            col = [
                SERIES[1]
                if r["kind"] == "detector"
                else SERIES[0]
                if r["kind"] == "supervised"
                else INK2
                for r in ew
            ]
            ax.barh(y, [0 if not math.isfinite(v) else v for v in pts], color=col)
            err = [
                [
                    max(0.0, p_ - l_) if math.isfinite(l_) else 0
                    for p_, l_ in zip(pts, lo, strict=True)
                ],
                [
                    max(0.0, h_ - p_) if math.isfinite(h_) else 0
                    for p_, h_ in zip(pts, hi, strict=True)
                ],
            ]
            ax.errorbar(np.nan_to_num(pts), y, xerr=err, fmt="none", ecolor=INK, lw=0.8)
            ax.set_yticks(y, labels if ax is axes[0] else [""] * len(y), fontsize=8, color=INK)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=9, color=INK)
            _style(ax)
        save(fig, "early_warning")
    tr = report["exploratory"]["tradeoff"]
    ctr = report["exploratory"]["combined_tradeoff"]
    if tr:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
        for i, lab in enumerate(sorted({r["label"] for r in tr})):
            rr = [r for r in tr if r["label"] == lab]
            ax.plot(
                [r["fa_clean"]["point"] for r in rr],
                [r["miss_rate"]["point"] for r in rr],
                "o-",
                color=SERIES[(i + 1) % len(SERIES)],
                label=lab,
            )
            for r in rr:
                ax.annotate(
                    f"p{r['percentile']:g}",
                    (r["fa_clean"]["point"], r["miss_rate"]["point"]),
                    fontsize=7,
                    color=MUTED,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        for j, lab in enumerate(sorted({r["label"] for r in ctr})):
            rr = [r for r in ctr if r["label"] == lab]
            ax.plot(
                [r["fa_clean"]["point"] for r in rr],
                [r["miss_rate"]["point"] for r in rr],
                ".--",
                color=SERIES[j % len(SERIES)],
                lw=0.7,
                alpha=0.6,
                label=lab,
            )
        ax.set_xlabel("clean false alarms per hour", fontsize=8, color=INK)
        ax.set_ylabel("E5 miss rate", fontsize=8, color=INK)
        ax.legend(fontsize=6, frameon=False, ncol=2)
        _style(ax)
        save(fig, "tradeoff")
    ss = report["primary"]["set_size"]
    if ss:
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
        x = np.arange(len(ss))
        for k, (cond, colr) in enumerate((("clean", SERIES[0]), ("e5", SERIES[1]))):
            ax.bar(
                x + (k - 0.5) * 0.38,
                [r["size_mondrian_lac"][cond]["point"] for r in ss],
                0.38,
                color=colr,
                label=cond,
            )
        ax.set_xticks(x, [r["label"] for r in ss], rotation=30, ha="right", fontsize=7, color=INK)
        ax.set_ylabel("Mondrian LAC mean set size", fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
        save(fig, "set_size")
    ag = report["primary"]["agreement"]
    if ag:
        types = sorted({t for r in ag for c in ("clean", "e5") for t in r[c]["by_type"]})
        mat = np.array(
            [
                [
                    (
                        r["clean"]["by_type"].get(t)
                        or r["e5"]["by_type"].get(t)
                        or {"point": np.nan}
                    )["point"]
                    for t in types
                ]
                for r in ag
            ],
            dtype=float,
        )
        fig, ax = plt.subplots(
            figsize=(1.4 * len(types) + 2, 0.45 * len(ag) + 1.2), facecolor=SURFACE
        )
        ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if math.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if mat[i, j] > 0.6 else INK,
                    )
        ax.set_xticks(range(len(types)), types, fontsize=7, rotation=20, color=INK)
        ax.set_yticks(range(len(ag)), [r["label"] for r in ag], fontsize=8, color=INK)
        ax.set_title("Agreement of the top sensor with the fault physics", fontsize=9, color=INK)
        save(fig, "agreement")
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
        x = np.arange(len(ag) * 2)
        bar_labels = [f"{r['label']} {c}" for r in ag for c in ("clean", "e5")]
        top = [r[c]["del_top"]["point"] for r in ag for c in ("clean", "e5")]
        rnd = [r[c]["del_rand"]["point"] for r in ag for c in ("clean", "e5")]
        ax.bar(x - 0.19, top, 0.38, color=SERIES[0], label="top sensor deleted")
        ax.bar(x + 0.19, rnd, 0.38, color=SERIES[3], label="random sensor deleted")
        ax.set_xticks(x, bar_labels, rotation=40, ha="right", fontsize=7, color=INK)
        ax.set_ylabel("mean |Δ p(alarm)|", fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
        save(fig, "deletion")
    return paths


# ---------------------------------------------------------------------------- main


def build_report(
    e5: E5Config,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    out_dir: Path | None = None,
    models_dir: Path = Path("config/models"),
    followup: E5FollowupConfig | None = None,
) -> list[Path]:
    tdir = tracking_dir or eval_cfg.tracking_dir
    data = collect(e5, eval_cfg, root, models_dir, tdir, followup)
    if not any(data["studies"].values()):
        raise LookupError(
            f"no E5 study runs with the current configs in {e5.experiment}; run uv run poe e5"
        )
    report = build(e5, eval_cfg, data)
    out = out_dir or eval_cfg.report.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = plots(report, out, e5.report_name)
    md = out / f"{e5.report_name}.md"
    md.write_text(render(report), encoding="utf-8", newline=LF)
    js = out / f"{e5.report_name}.json"
    js.write_text(
        json.dumps(_clean(_round(report)), indent=1, sort_keys=True) + LF,
        encoding="utf-8",
        newline=LF,
    )
    return [md, js, *paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.report_e5")
    parser.add_argument("--e5-config", type=Path, default=Path("config/models/e5.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    parser.add_argument("--models-dir", type=Path, default=Path("config/models"))
    parser.add_argument(
        "--followup-config",
        type=Path,
        default=Path("config/models/e5_followup.yaml"),
        help="exploratory follow-ups; skipped if the file does not exist",
    )
    args = parser.parse_args(argv)
    configure_logging()
    from twin_ml.config import load_e5_config, load_e5_followup_config, load_eval_config

    try:
        followup = (
            load_e5_followup_config(args.followup_config)
            if args.followup_config.is_file()
            else None
        )
        paths = build_report(
            load_e5_config(args.e5_config),
            load_eval_config(args.eval_config),
            models_dir=args.models_dir,
            followup=followup,
        )
    except (ConfigError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
