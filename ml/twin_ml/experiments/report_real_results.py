"""Phase 10 report, study results (ADR 0027): collected from the MLflow ``P10_real`` runs.

:func:`collect` gathers the newest valid run of every (study, model, seed) (same config hashes as
``python -m twin_ml.experiments.real --skip-existing``), the combined-policy runs built from
exactly those runs, and the explanation runs. :func:`compute` turns them into the report dict
(primary outcomes first, then everything exploratory); :func:`render` writes the markdown and
the ``plot_*`` functions the figures. Every number comes from a run's ``metrics.json`` or from
its per-machine sums (``runstats.npz``); nothing is typed by hand.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from twin_ml.evaluate import run_sums
from twin_ml.experiments import real_eval
from twin_ml.experiments.report_e1 import (
    ACCENT,
    ACCENT2,
    AXIS,
    GRID,
    INK,
    INK2,
    MUTED,
    NA,
    SURFACE,
    _table,
)
from twin_ml.experiments.report_robustness import SERIES

PRIMARY_METRICS_AZ = ("detection_rate", "lead_mean_h", "false_alarms_per_day")
CLEAN = "clean"


# ---------------------------------------------------------------------------- collect


def _download(client: Any, run_id: str, name: str, dst: Path) -> Path:
    return Path(client.download_artifacts(run_id, name, str(dst)))


def _entry(client: Any, run: Any, tmp: Path, *, scores: bool = False) -> dict[str, Any]:
    d = tmp / run.info.run_id
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "run_id": run.info.run_id,
        "tags": dict(run.data.tags),
        "metrics": dict(run.data.metrics),
        "detail": json.loads(
            _download(client, run.info.run_id, "metrics.json", d).read_text("utf-8")
        ),
    }
    with np.load(_download(client, run.info.run_id, "runstats.npz", d), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    out["machines"] = [str(m) for m in arrays.pop("machines")]
    out["sums"] = run_sums.load_npz_dict(arrays)
    if scores:
        names = [a.path for a in client.list_artifacts(run.info.run_id)]
        if "scores.npz" in names:
            with np.load(_download(client, run.info.run_id, "scores.npz", d)) as z:
                out["scores"] = {k: z[k] for k in z.files}
    return out


def collect(
    real: Any, rcfg: Any, eval_cfg: Any, tracking_dir: Path, root: Path, experiment: str
) -> dict[str, Any]:
    """``{"studies": {study: {(model, seed): entry}}, "combined": {...}, "explain": {...}}``."""
    from mlflow.tracking import MlflowClient

    from twin_common.config import ConfigError
    from twin_ml.experiments.real import COMBINE_STUDIES, STUDY_ORDER, combined_sha, expected_shas
    from twin_ml.experiments.real_worker import COMBINED_SUFFIX, latest_valid_runs, study_specs
    from twin_ml.train import tracking

    client = MlflowClient(tracking_uri=tracking.tracking_uri(tracking_dir))
    specs = study_specs(real)
    out: dict[str, Any] = {"studies": {}, "combined": {}, "explain": {}, "keys": {}}
    with tempfile.TemporaryDirectory(prefix="twin_p10_rep_") as tmp_s:
        tmp = Path(tmp_s)
        valid_ids: dict[str, str] = {}
        for study in STUDY_ORDER:
            try:
                shas, _, key = expected_shas(
                    study, list(specs[study].models), list(real.seeds), real, rcfg, eval_cfg, root
                )
            except (FileNotFoundError, ConfigError):  # dataset not built or configs absent
                continue
            runs = latest_valid_runs(tracking_dir, experiment, real.phase_tag, study, shas)
            out["keys"][study] = key
            out["studies"][study] = {
                k: _entry(client, r, tmp, scores=study == "metropt3") for k, r in runs.items()
            }
            for k, r in runs.items():
                valid_ids[f"{study}:{k[0]}:{k[1]}"] = r.info.run_id
        for study in COMBINE_STUDIES:
            got: dict[tuple[str, str, int], Any] = {}
            for r in tracking.finished_runs(
                tracking_dir,
                experiment,
                None,
                real.phase_tag,
                f"tags.study = '{study}{COMBINED_SUFFIX}'",
            ):
                t = r.data.tags
                key3 = (t["supervised"], t["detector"], int(t["seed"]))
                sid = valid_ids.get(f"{study}:{t['supervised']}:{t['seed']}")
                did = valid_ids.get(f"{study}:{t['detector']}:{t['seed']}")
                if key3 in got or sid is None or did is None:
                    continue
                if t.get("combined_sha") == combined_sha(sid, did, real):
                    got[key3] = _entry(client, r, tmp)
            out["combined"][study] = got
        try:
            from twin_ml.experiments.real_explain import collect_explain

            out["explain"] = collect_explain(
                real, rcfg, eval_cfg, tracking_dir, root, experiment, client, tmp
            )
        except ImportError:  # pragma: no cover - Part C module missing
            out["explain"] = {}
    return out


# ---------------------------------------------------------------------------- compute


def _boot(entry: dict[str, Any], group: str, step_s: float, real: Any) -> dict[str, Any]:
    """Point and machine-bootstrap CI of every metric of one group (stored, else computed)."""
    stored = entry["detail"].get("conditions", {}).get(CLEAN, {}).get("bootstrap", {})
    if group in stored:
        return dict(stored[group])
    b = real.bootstrap
    fn = real_eval.metric_fn(real_eval.group_lookback_s(group, step_s))
    return run_sums.bootstrap(entry["sums"][group], b.n_resamples, b.seed, b.ci, metric_fn=fn)


def _finite_mean(xs: list[float]) -> float:
    f = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(f)) if f else float("nan")


def seed_view(
    entries: dict[int, dict[str, Any]], group: str, step_s: float, real: Any
) -> dict[str, Any]:
    """Per-seed point / CI and the seed means (point, lo, hi) of every metric."""
    per = {s: _boot(e, group, step_s, real) for s, e in sorted(entries.items())}
    metrics = sorted({m for v in per.values() for m in v})
    mean = {
        m: {
            k: _finite_mean([v[m][k] for v in per.values() if m in v])
            for k in ("point", "lo", "hi")
        }
        for m in metrics
    }
    return {"seeds": sorted(per), "per_seed": per, "mean": mean}


def paired_view(
    a: dict[int, dict[str, Any]],
    b: dict[int, dict[str, Any]],
    group_a: str,
    group_b: str,
    step_s: float,
    real: Any,
    metrics: tuple[str, ...] = PRIMARY_METRICS_AZ,
    machines: list[str] | None = None,
) -> dict[str, Any]:
    """Seed-wise paired machine bootstrap a - b (a deterministic side is reused for every seed);
    seed means of point and bounds, and how many seeds' CIs exclude zero."""
    if len(b) == 1:
        pairs = [(s, a[s], next(iter(b.values()))) for s in sorted(a)]
    elif len(a) == 1:
        pairs = [(s, next(iter(a.values())), b[s]) for s in sorted(b)]
    else:
        pairs = [(s, a[s], b[s]) for s in sorted(set(a) & set(b))]
    bt = real.bootstrap
    fn = real_eval.metric_fn(real_eval.group_lookback_s(group_a, step_s))
    per = []
    for s, ea, eb in pairs:
        sa, sb = ea["sums"][group_a], eb["sums"][group_b]
        if ea["machines"] != eb["machines"]:
            raise ValueError("paired comparison needs the same machines on both sides")
        if machines is not None:
            idx = [ea["machines"].index(m) for m in machines]
            sa, sb = run_sums.select(sa, idx), run_sums.select(sb, idx)
        per.append(
            (
                s,
                run_sums.paired(
                    sa, sb, bt.n_resamples, bt.seed, bt.ci, list(metrics), metric_fn=fn
                ),
            )
        )
    out: dict[str, Any] = {"seeds": [s for s, _ in per]}
    for m in metrics:
        vals = [r[m] for _, r in per]
        lo, hi = _finite_mean([v["lo"] for v in vals]), _finite_mean([v["hi"] for v in vals])
        out[m] = {
            "point": _finite_mean([v["point"] for v in vals]),
            "lo": lo,
            "hi": hi,
            "excludes_zero": bool(math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0)),
            "seeds_excluding_zero": int(sum(v["excludes_zero"] for v in vals)),
            "seeds": len(per),
            "per_seed": {str(s): r[m] for s, r in per},
        }
    return out


def by_model(runs: dict[tuple[str, int], Any], model: str) -> dict[int, dict[str, Any]]:
    return {s: e for (m, s), e in runs.items() if m == model}


def _step(runs: dict[tuple[str, int], Any]) -> float:
    return float(next(iter(runs.values()))["detail"]["step_s"])


def compute_metropt(res: dict[str, Any], real: Any) -> dict[str, Any] | None:
    runs = res["studies"].get("metropt3")
    if not runs:
        return None
    st = real.datasets["metropt3"]
    step = _step(runs)
    pct = real.primary.metropt3.anomaly_percentile
    lb = real.primary.metropt3.lookback
    groups = {
        "autoencoder": f"p{pct:g}",
        "iforest": f"p{pct:g}",
        "threshold": "rule",
        "zscore": "rule",
    }
    primary: dict[str, Any] = {}
    for method, g in groups.items():
        ent = by_model(runs, method)
        if not ent:
            continue
        seeds: dict[str, Any] = {}
        for s, e in sorted(ent.items()):
            gd = e["detail"]["conditions"][CLEAN]["groups"][g]
            pt = gd["points"][f"L{lb}"]
            seeds[str(s)] = {
                "points": pt,
                "events": [
                    {
                        "id": ev["id"],
                        "warned": (
                            "not evaluable"
                            if not ev["evaluable"]
                            else ("yes" if ev["warned"] else "no")
                        ),
                        "lead_min": None
                        if ev["lead_s"] is None or not ev["evaluable"]
                        else ev["lead_s"] / 60.0,
                        "coverage": ev["coverage"],
                        "detected_during": ev["detected_during"],
                        "delay_min": None if ev["delay_s"] is None else ev["delay_s"] / 60.0,
                    }
                    for ev in gd["events"][f"L{lb}"]
                ],
                "fa_by_month": gd.get("fa_by_month"),
            }
        primary[method] = {"group": g, "seeds": seeds}
    explo: dict[str, Any] = {}
    for method in ("autoencoder", "iforest", "threshold", "zscore"):
        ent = by_model(runs, method)
        if not ent:
            continue
        gs = (
            [f"p{p:g}" for p in real.anomaly.threshold_percentiles]
            if method in ("autoencoder", "iforest")
            else ["rule"]
        )
        for g in gs:
            for lk in st.lookbacks:
                pts = [
                    e["detail"]["conditions"][CLEAN]["groups"][g]["points"][f"L{lk}"]
                    for e in ent.values()
                ]
                explo[f"{method}|{g}|L{lk}"] = {
                    k: {
                        "mean": _finite_mean([p[k] for p in pts]),
                        "min": min(
                            (p[k] for p in pts if math.isfinite(p[k])), default=float("nan")
                        ),
                        "max": max(
                            (p[k] for p in pts if math.isfinite(p[k])), default=float("nan")
                        ),
                    }
                    for k in (
                        "warned_count",
                        "evaluable",
                        "lead_min",
                        "false_alarms_per_day",
                        "alarm_frac",
                        "chance_expected",
                        "during_rate",
                    )
                }
    first = next(iter(runs.values()))["detail"]
    return {
        "lookback_rows": lb,
        "percentile": pct,
        "step_s": step,
        "primary": primary,
        "exploratory": explo,
        "counts": first["counts"]["test"],
        "run_ids": {f"{m}:{s}": e["run_id"] for (m, s), e in runs.items()},
        "detector_info": {
            m: {
                str(s): {
                    "thresholds": e["detail"]["method"]["thresholds"]["values"],
                    "fit": {
                        k: v for k, v in e["detail"]["method"]["fit"].items() if k != "history"
                    },
                }
                for s, e in by_model(runs, m).items()
            }
            for m in ("autoencoder", "iforest")
            if by_model(runs, m)
        },
    }


def _selection(runs: dict[tuple[str, int], Any], real: Any) -> tuple[str | None, dict[str, Any]]:
    from twin_ml.experiments.real import best_supervised
    from twin_ml.experiments.real_worker import SUPERVISED

    class _R:  # adapts report entries to the runner's selection helper
        def __init__(self, metrics: dict[str, float]) -> None:
            self.data = type("D", (), {"metrics": metrics})()

    adapted = {k: _R(e["metrics"]) for k, e in runs.items()}
    best = best_supervised(adapted, real, list(real.seeds))
    key = f"val.h{real.selection.horizon}.{real.selection.metric}"
    table: dict[str, Any] = {}
    for m in SUPERVISED:
        per = [runs[(m, s)]["metrics"].get(key) if (m, s) in runs else None for s in real.seeds]
        table[m] = {
            "per_seed": per,
            "mean": _finite_mean([float("nan") if x is None else x for x in per]),
        }
    return best, table


def compute_azure(res: dict[str, Any], real: Any) -> dict[str, Any] | None:
    runs = res["studies"].get("azure")
    if not runs:
        return None
    from twin_ml.experiments.real_worker import SUPERVISED

    p = real.primary.azure_pdm
    step = _step(runs)
    lk = f"L{p.lookback}"
    pct = f"p{p.anomaly_percentile:g}"
    hg = f"h{p.horizon}"
    best, selection = _selection(runs, real)
    comb = res["combined"].get("azure", {})
    sources: dict[str, tuple[dict[int, Any], str]] = {}
    if best:
        sources["best_supervised"] = (by_model(runs, best), f"{CLEAN}|{hg}|{lk}")
    sources["autoencoder"] = (by_model(runs, "autoencoder"), f"{CLEAN}|{pct}|{lk}")
    if best:
        sources["combined"] = (
            {s: e for (m, d, s), e in comb.items() if m == best and d == real.combined.primary[1]},
            f"{CLEAN}|{pct}|{lk}",
        )
    for r in ("threshold", "zscore", "persistence"):
        sources[r] = (by_model(runs, r), f"{CLEAN}|rule|{lk}")
    primary: dict[str, Any] = {}
    for g, (ent, grp) in sources.items():
        if ent:
            primary[g] = seed_view(ent, grp, step, real) | {"group": grp}
    paired: dict[str, Any] = {}
    ref_ent, ref_grp = sources[p.reference]
    if ref_ent:
        for g, (ent, grp) in sources.items():
            if g != p.reference and ent:
                paired[f"{g}_vs_{p.reference}"] = paired_view(
                    ent, ref_ent, grp, ref_grp, step, real
                )
    if best and sources["combined"][0]:
        paired[f"combined_vs_{p.combined_vs}"] = paired_view(
            sources["combined"][0],
            sources["best_supervised"][0],
            sources["combined"][1],
            sources["best_supervised"][1],
            step,
            real,
        )
    # ------------------------------------------------------------ exploratory
    explo: dict[str, Any] = {"methods": {}, "combined": {}, "chance": {}}
    for m in (*SUPERVISED, "iforest", "autoencoder", "threshold", "zscore", "persistence"):
        ent = by_model(runs, m)
        if not ent:
            continue
        groups = sorted(next(iter(ent.values()))["detail"]["conditions"][CLEAN]["groups"])
        for g in groups:
            explo["methods"][f"{m}|{g}"] = seed_view(ent, f"{CLEAN}|{g}|{lk}", step, real)["mean"]
    for (m, d, s), e in sorted(comb.items()):
        for g in sorted({k.split("|")[1] for k in e["sums"]}):
            explo["combined"].setdefault(f"{m}+{d}|{g}", {})[s] = e
    explo["combined"] = {
        k: seed_view(v, f"{CLEAN}|{k.split('|')[1]}|{lk}", step, real)["mean"]
        for k, v in explo["combined"].items()
    }
    out: dict[str, Any] = {
        "best_supervised": best,
        "selection": selection,
        "primary": primary,
        "paired": paired,
        "exploratory": explo,
        "step_s": step,
        "lookback_rows": p.lookback,
        "counts": next(iter(runs.values()))["detail"]["counts"]["test"],
        "run_ids": {f"{m}:{s}": e["run_id"] for (m, s), e in runs.items()},
    }
    if best:
        out["calibration"] = {
            str(s): e["detail"]
            .get("exploratory_calibration", {})
            .get("heads", {})
            .get(str(p.horizon))
            for s, e in by_model(runs, best).items()
        }
        out["classification_test"] = {
            str(s): e["detail"]["conditions"][CLEAN].get("classification", {}).get(hg)
            for s, e in by_model(runs, best).items()
        }
        out["robustness"] = robustness_view(runs, best, real, step, lk, hg, pct)
        recs = by_model(runs, best)
        out["leads_h"] = {
            str(s): [
                ev["lead_s"] / 3600.0
                for ev in e["detail"]["conditions"][CLEAN]["groups"][hg]["events"][lk]
                if ev["evaluable"] and ev["warned"]
            ]
            for s, e in recs.items()
        }
        out["by_fault_type"] = by_fault_type(recs, hg, lk)
    out["error_log"] = error_view(res, runs, real, step, lk, hg)
    out["holdout"] = holdout_view(res, runs, real, step, lk, hg, pct)
    return out


def by_fault_type(recs: dict[int, Any], group: str, lk: str) -> dict[str, Any]:
    out: dict[str, dict[str, list[float]]] = {}
    for e in recs.values():
        evs = [
            x
            for x in e["detail"]["conditions"][CLEAN]["groups"][group]["events"][lk]
            if x["evaluable"]
        ]
        for ft in sorted({x["fault_type"] for x in evs}):
            sel = [x for x in evs if x["fault_type"] == ft]
            d = out.setdefault(ft, {"n": [], "rate": []})
            d["n"].append(len(sel))
            d["rate"].append(sum(x["warned"] for x in sel) / len(sel))
    return {
        ft: {"n": int(v["n"][0]), "rate_mean": _finite_mean(v["rate"])} for ft, v in out.items()
    }


def robustness_view(
    runs: dict[tuple[str, int], Any], best: str, real: Any, step: float, lk: str, hg: str, pct: str
) -> dict[str, Any]:
    conds = list(real.exploratory.robustness.conditions)
    methods = {
        "best_supervised": (best, hg),
        "autoencoder": ("autoencoder", pct),
        "threshold": ("threshold", "rule"),
        "zscore": ("zscore", "rule"),
    }
    out: dict[str, Any] = {"conditions": [CLEAN, *conds], "methods": {}}
    bt = real.bootstrap
    fn = real_eval.metric_fn(float(lk[1:]) * step)
    for name, (m, g) in methods.items():
        ent = by_model(runs, m)
        if not ent:
            continue
        rows: dict[str, Any] = {}
        for c in [CLEAN, *conds]:
            key = f"{c}|{g}|{lk}"
            if not all(key in e["sums"] for e in ent.values()):
                continue
            pts = [run_sums.point(e["sums"][key], fn) for e in ent.values()]
            row: dict[str, Any] = {
                k: _finite_mean([pt[k] for pt in pts]) for k in PRIMARY_METRICS_AZ
            }
            if c != CLEAN:
                diffs = [
                    run_sums.paired(
                        e["sums"][key],
                        e["sums"][f"{CLEAN}|{g}|{lk}"],
                        bt.n_resamples,
                        bt.seed,
                        bt.ci,
                        ["detection_rate", "false_alarms_per_day"],
                        metric_fn=fn,
                    )
                    for e in ent.values()
                ]
                row["vs_clean"] = {
                    k: {
                        "point": _finite_mean([d[k]["point"] for d in diffs]),
                        "lo": _finite_mean([d[k]["lo"] for d in diffs]),
                        "hi": _finite_mean([d[k]["hi"] for d in diffs]),
                        "seeds_excluding_zero": int(sum(d[k]["excludes_zero"] for d in diffs)),
                        "seeds": len(diffs),
                    }
                    for k in ("detection_rate", "false_alarms_per_day")
                }
            rows[c] = row
        out["methods"][name] = {"model": m, "rows": rows}
    return out


def error_view(
    res: dict[str, Any], az: dict[tuple[str, int], Any], real: Any, step: float, lk: str, hg: str
) -> dict[str, Any] | None:
    runs = res["studies"].get("azure_err")
    if not runs:
        return None
    out: dict[str, Any] = {"rule": {}, "xgb_err": None, "paired_vs_xgb": None}
    rule = by_model(runs, "error_rule")
    if rule:
        e = next(iter(rule.values()))
        for code in real.exploratory.error_rule.codes:
            out["rule"][code] = seed_view(rule, f"{CLEAN}|{code}|{lk}", step, real)["mean"]
        ref = by_model(az, "threshold")
        if ref and e["machines"] == next(iter(ref.values()))["machines"]:
            out["rule_any_vs_threshold"] = paired_view(
                rule, ref, f"{CLEAN}|any|{lk}", f"{CLEAN}|rule|{lk}", step, real
            )
    xe, xt = by_model(runs, "xgb"), by_model(az, "xgb")
    if xe:
        out["xgb_err"] = seed_view(xe, f"{CLEAN}|{hg}|{lk}", step, real)["mean"]
        key = f"val.{hg}.{real.selection.metric}"
        out["xgb_err_val"] = _finite_mean([e["metrics"].get(key) for e in xe.values()])
        if xt:
            out["xgb_tel"] = seed_view(xt, f"{CLEAN}|{hg}|{lk}", step, real)["mean"]
            out["xgb_tel_val"] = _finite_mean([e["metrics"].get(key) for e in xt.values()])
            out["paired_vs_xgb"] = paired_view(
                xe, xt, f"{CLEAN}|{hg}|{lk}", f"{CLEAN}|{hg}|{lk}", step, real
            )
            if rule:
                out["rule_any_vs_xgb"] = paired_view(
                    rule, xt, f"{CLEAN}|any|{lk}", f"{CLEAN}|{hg}|{lk}", step, real
                )
    return out


def holdout_view(
    res: dict[str, Any],
    az: dict[tuple[str, int], Any],
    real: Any,
    step: float,
    lk: str,
    hg: str,
    pct: str,
) -> dict[str, Any] | None:
    runs = res["studies"].get("azure_holdout")
    if not runs:
        return None
    best, selection = _selection(runs, real)
    held = next(iter(runs.values()))["machines"]
    out: dict[str, Any] = {
        "best_supervised": best,
        "selection": selection,
        "machines": held,
        "rows": {},
    }
    best_az, _ = _selection(az, real)
    pairs = {
        "best_supervised": ((best, hg), (best_az, hg)),
        "autoencoder": (("autoencoder", pct), ("autoencoder", pct)),
    }
    for name, ((mh, gh), (mt, gt)) in pairs.items():
        if mh is None or mt is None:
            continue
        eh, et = by_model(runs, mh), by_model(az, mt)
        if not eh or not et:
            continue
        hv = seed_view(eh, f"{CLEAN}|{gh}|{lk}", step, real)["mean"]
        fn = real_eval.metric_fn(float(lk[1:]) * step)
        idx = [next(iter(et.values()))["machines"].index(m) for m in held]
        tpts = [
            run_sums.point(run_sums.select(e["sums"][f"{CLEAN}|{gt}|{lk}"], idx), fn)
            for e in et.values()
        ]
        tv = {k: _finite_mean([p[k] for p in tpts]) for k in tpts[0]}
        out["rows"][name] = {
            "holdout_model": mh,
            "temporal_model": mt,
            "holdout": hv,
            "temporal_same_machines": tv,
        }
    return out


def compute(res: dict[str, Any], real: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metropt3": compute_metropt(res, real),
        "azure": compute_azure(res, real),
    }
    if res.get("explain"):
        from twin_ml.experiments.real_explain import compute_explain

        out["explain"] = compute_explain(res["explain"], real)
    return out


# ---------------------------------------------------------------------------- render


def _v(x: Any, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    return f"{x:.{d}f}"


def _ci(e: dict[str, float], d: int = 2) -> str:
    if not math.isfinite(e.get("point", float("nan"))):
        return NA
    return f"{_v(e['point'], d)} [{_v(e['lo'], d)}, {_v(e['hi'], d)}]"


def _diff(e: dict[str, Any], d: int = 2) -> str:
    if not math.isfinite(e.get("point", float("nan"))):
        return NA
    star = " *" if e["excludes_zero"] else ""
    seeds = e.get("seeds")
    n = len(seeds) if isinstance(seeds, list) else seeds
    return (
        f"{e['point']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]{star} "
        f"({e['seeds_excluding_zero']}/{n})"
    )


def render_metropt(m: dict[str, Any], real: Any) -> list[str]:
    lb_h = m["lookback_rows"] * m["step_s"] / 3600
    out = [
        "## MetroPT-3 results (case study, 4 failures)",
        "",
        f"**Pre-registered primary outcomes** (ADR 0027): lookback {lb_h:g} h, detectors at the "
        f"{m['percentile']:g}th percentile, test period ({m['counts']['windows']:,} windows, "
        f"{m['counts']['events']} failures). No supervised model and no combined policy (no "
        "failure lies in the training period). No confidence intervals: four events allow no "
        "statistical claim. *Warned* = a sustained alarm in the 6 h before the reported start; "
        "*chance expected* = the evaluable failures that alarms at the measured false-alarm "
        "rate would hit by chance alone (ADR 0027 § Chance check).",
        "",
    ]
    rows = []
    for method, v in m["primary"].items():
        for s, sd in v["seeds"].items():
            pt = sd["points"]
            evs = sd["events"]
            rows.append(
                [
                    method if method in ("threshold", "zscore") else f"{method} (seed {s})",
                    f"{pt['warned_count']:.0f} / {pt['evaluable']:.0f}",
                    ", ".join(
                        f"#{e['id']}: {e['warned']}"
                        + (f" ({e['lead_min']:.0f} min)" if e["lead_min"] is not None else "")
                        for e in evs
                    ),
                    _v(pt["false_alarms_per_day"], 3),
                    _v(pt["alarm_frac"] * 100, 2) + " %" if math.isfinite(pt["alarm_frac"]) else NA,
                    _v(pt["chance_expected"], 2),
                ]
            )
    out += _table(
        [
            "Method",
            "Warned / evaluable",
            "Per failure (lead)",
            "False alarms / day",
            "Alarm on (neg.)",
            "Chance expected",
        ],
        rows,
    )
    out += [
        "",
        "**Chance check.** A method's warnings carry evidence only where *warned* clearly "
        "exceeds *chance expected*. With four failures, even a perfect count is weak evidence; "
        "read every row as a case description.",
        "",
        "**Detected during the failure** (exploratory, reported separately, never counted as a "
        "warning): a sustained alarm on a window ending inside the reported failure interval.",
        "",
    ]
    rows = []
    for method, v in m["primary"].items():
        for s, sd in v["seeds"].items():
            rows.append(
                [
                    method if method in ("threshold", "zscore") else f"{method} (seed {s})",
                    ", ".join(
                        f"#{e['id']}: "
                        + (f"yes ({e['delay_min']:.0f} min)" if e["detected_during"] else "no")
                        for e in sd["events"]
                    ),
                ]
            )
    out += _table(["Method", "Per failure (delay after the reported start)"], rows)
    out += [
        "",
        "### MetroPT-3, exploratory: other lookbacks and percentiles",
        "",
        "Seed mean (min to max over seeds for the detectors).",
        "",
    ]
    rows = []
    for key, v in m["exploratory"].items():
        method, g, lk = key.split("|")
        lb = int(lk[1:]) * m["step_s"] / 3600
        rng = (
            f" ({v['warned_count']['min']:.0f}–{v['warned_count']['max']:.0f})"  # noqa: RUF001
            if method in ("autoencoder", "iforest")
            else ""
        )
        rows.append(
            [
                method,
                g,
                f"{lb:g} h",
                f"{v['warned_count']['mean']:.2f}{rng} / {v['evaluable']['mean']:.0f}",
                _v(v["lead_min"]["mean"], 0),
                _v(v["false_alarms_per_day"]["mean"], 3),
                _v(v["chance_expected"]["mean"], 2),
                _v(v["during_rate"]["mean"], 2),
            ]
        )
    out += _table(
        [
            "Method",
            "Group",
            "Lookback",
            "Warned / evaluable",
            "Lead (min)",
            "FA / day",
            "Chance expected",
            "Detected during (rate)",
        ],
        rows,
    )
    out += [
        "",
        "![MetroPT-3 timelines](real_data_metropt3_timelines.png)",
        "",
        "![MetroPT-3 false alarms by month](real_data_metropt3_fa_month.png)",
        "",
    ]
    out += render_months(m)
    return out


def render_months(m: dict[str, Any]) -> list[str]:
    out = [
        "**False alarms per day by test month** (exploratory; seed mean for the detectors; "
        "negative days in brackets):",
        "",
    ]
    months: list[str] = []
    rows = []
    for method, v in m["primary"].items():
        per = [sd["fa_by_month"] for sd in v["seeds"].values() if sd["fa_by_month"]]
        if not per:
            continue
        months = sorted(per[0])
        rows.append(
            [method]
            + [
                f"{_finite_mean([p[mo]['false_alarms_per_day'] for p in per]):.2f} "
                f"({per[0][mo]['negative_days']:.1f} d)"
                for mo in months
            ]
        )
    if rows:
        out += _table(["Method", *months], rows)
    return [*out, ""]


def render_azure(a: dict[str, Any], real: Any) -> list[str]:
    p = real.primary.azure_pdm
    best = a["best_supervised"]
    out = [
        "## Azure PdM results (public benchmark, likely synthetic)",
        "",
        f"**Pre-registered primary outcomes** (ADR 0027): horizon +{p.horizon} h, lookback "
        f"{p.lookback} h, test period ({a['counts']['windows']:,} windows, {a['counts']['events']} "
        f"failure events, {a['counts']['machines']} machines). Point estimate [95 % machine "
        "bootstrap CI, 1000 resamples]; for models with three seeds, the seed means of the "
        "point estimate and of both bounds. Same machines in training and test (temporal "
        "split): the results show forecasting for known machines, not new ones.",
        "",
        f"**Best supervised model (validation only):** `{best}`, the highest seed-mean "
        "validation PR-AUC (mean of WARNING and FAULT, operating windows, +24 h):",
        "",
    ]
    rows = [
        [m, ", ".join(_v(x, 3) for x in v["per_seed"]), _v(v["mean"], 3)]
        for m, v in a["selection"].items()
    ]
    out += _table(["Model", "Validation PR-AUC per seed", "Seed mean"], rows)
    out += ["", "**Primary outcomes:**", ""]
    rows = []
    for g, v in a["primary"].items():
        mean = v["mean"]
        name = f"best_supervised (`{best}`)" if g == "best_supervised" else g
        if g == "combined":
            name = f"combined (`{best}` OR autoencoder p{p.anomaly_percentile:g})"
        rows.append(
            [
                name,
                str(len(v["seeds"])),
                _ci(mean["detection_rate"], 3),
                _ci(mean["lead_mean_h"], 1),
                _ci(mean["false_alarms_per_day"], 4),
                f"{mean['warned_count']['point']:.1f} / {mean['evaluable']['point']:.0f}",
                _v(mean["chance_expected"]["point"], 1),
            ]
        )
    out += _table(
        [
            "Group",
            "Seeds",
            "Detection rate",
            "Lead (h)",
            "False alarms / day",
            "Warned / evaluable",
            "Chance expected",
        ],
        rows,
    )
    out += [
        "",
        "**Paired comparisons** (same bootstrap resamples of machines on both sides; "
        "difference a − b, seed means of point and bounds; `*` = the averaged CI excludes 0; "  # noqa: RUF001
        "in brackets the seeds whose own CI excludes 0):",
        "",
    ]
    rows = [
        [
            k.replace("_vs_", " − "),  # noqa: RUF001
            _diff(v["detection_rate"], 3),
            _diff(v["lead_mean_h"], 1),
            _diff(v["false_alarms_per_day"], 4),
        ]
        for k, v in a["paired"].items()
    ]
    out += _table(["Comparison", "Detection rate", "Lead (h)", "False alarms / day"], rows)
    out += ["", "![Azure early warning](real_data_azure_primary.png)", ""]
    out += render_azure_explo(a, real)
    return out


def render_azure_explo(a: dict[str, Any], real: Any) -> list[str]:
    out = [
        "### Azure PdM, exploratory",
        "",
        "**Every method** (seed means of the point "
        "estimates; supervised groups `h12` / `h24` = the head's alarm; detector groups = "
        "threshold percentile):",
        "",
    ]
    rows = []
    for k, v in {
        **a["exploratory"]["methods"],
        **{f"combined {kk}": vv for kk, vv in a["exploratory"]["combined"].items()},
    }.items():
        name, _, grp = k.rpartition("|")
        rows.append(
            [
                name,
                grp,
                _v(v["detection_rate"]["point"], 3),
                _v(v["lead_mean_h"]["point"], 1),
                _v(v["false_alarms_per_day"]["point"], 4),
                _v(v["chance_expected"]["point"], 1),
                _v(v["during_rate"]["point"], 2),
            ]
        )
    out += _table(
        [
            "Method",
            "Group",
            "Detection rate",
            "Lead (h)",
            "FA / day",
            "Chance expected",
            "Detected in failure hour",
        ],
        rows,
    )
    if a.get("by_fault_type"):
        out += [
            "",
            "**Best supervised model by failure component** (detection rate, seed mean; "
            "evaluable events):",
            "",
        ]
        out += _table(
            ["Component", "Events", "Detection rate"],
            [[k, str(v["n"]), _v(v["rate_mean"], 3)] for k, v in a["by_fault_type"].items()],
        )
    e = a.get("error_log")
    if e:
        out += [
            "",
            "**Error log** (exploratory): the error-log rule (alarm if an error occurred "
            "in the 24 h window) and XGBoost with per-hour error counts as extra inputs, "
            "against the telemetry-only XGBoost of the main study.",
            "",
        ]
        rows = [
            [
                f"error rule `{c}`",
                _v(v["detection_rate"]["point"], 3),
                _v(v["lead_mean_h"]["point"], 1),
                _v(v["false_alarms_per_day"]["point"], 4),
                _v(v["chance_expected"]["point"], 1),
                NA,
            ]
            for c, v in e["rule"].items()
        ]
        if e.get("xgb_tel"):
            rows.append(
                [
                    "xgb, telemetry only",
                    _v(e["xgb_tel"]["detection_rate"]["point"], 3),
                    _v(e["xgb_tel"]["lead_mean_h"]["point"], 1),
                    _v(e["xgb_tel"]["false_alarms_per_day"]["point"], 4),
                    _v(e["xgb_tel"]["chance_expected"]["point"], 1),
                    _v(e["xgb_tel_val"], 3),
                ]
            )
        if e.get("xgb_err"):
            rows.append(
                [
                    "xgb, telemetry + error counts",
                    _v(e["xgb_err"]["detection_rate"]["point"], 3),
                    _v(e["xgb_err"]["lead_mean_h"]["point"], 1),
                    _v(e["xgb_err"]["false_alarms_per_day"]["point"], 4),
                    _v(e["xgb_err"]["chance_expected"]["point"], 1),
                    _v(e["xgb_err_val"], 3),
                ]
            )
        out += _table(
            ["Method", "Detection rate", "Lead (h)", "FA / day", "Chance expected", "Val PR-AUC"],
            rows,
        )
        prs = [
            (k, e[k])
            for k in ("paired_vs_xgb", "rule_any_vs_xgb", "rule_any_vs_threshold")
            if e.get(k)
        ]
        if prs:
            names = {
                "paired_vs_xgb": "xgb + errors − xgb telemetry",  # noqa: RUF001
                "rule_any_vs_xgb": "error rule (any) − xgb telemetry",  # noqa: RUF001
                "rule_any_vs_threshold": "error rule (any) − threshold",  # noqa: RUF001
            }
            out += ["", "Paired:", ""]
            out += _table(
                ["Comparison", "Detection rate", "Lead (h)", "False alarms / day"],
                [
                    [
                        names[k],
                        _diff(v["detection_rate"], 3),
                        _diff(v["lead_mean_h"], 1),
                        _diff(v["false_alarms_per_day"], 4),
                    ]
                    for k, v in prs
                ],
            )
    h = a.get("holdout")
    if h:
        out += [
            "",
            f"**Machine holdout** (exploratory): {len(h['machines'])} machines held out "
            f"entirely (seed {real.exploratory.holdout.seed}); models retrained on the other "
            f"machines; best model within the variant by validation: `{h['best_supervised']}`. "
            "Side by side with the temporal-split result restricted to the same machines "
            "(seed means of the point estimates):",
            "",
        ]
        rows = []
        for name, v in h["rows"].items():
            for tag, d, mdl in (
                ("holdout (new machines)", v["holdout"], v["holdout_model"]),
                ("temporal (same machines)", v["temporal_same_machines"], v["temporal_model"]),
            ):
                pt = {k: (x["point"] if isinstance(x, dict) else x) for k, x in d.items()}
                rows.append(
                    [
                        name,
                        f"`{mdl}`",
                        tag,
                        _v(pt["detection_rate"], 3),
                        _v(pt["lead_mean_h"], 1),
                        _v(pt["false_alarms_per_day"], 4),
                        f"{pt['evaluable']:.0f}",
                    ]
                )
        out += _table(
            ["Group", "Model", "Split", "Detection rate", "Lead (h)", "FA / day", "Evaluable"], rows
        )
    r = a.get("robustness")
    if r:
        out += [
            "",
            "**Robustness** (exploratory, test data, Phase 8 perturbations applied to the "
            "raw hourly series before forward fill and normalisation): detection rate and "
            "false alarms per day (seed means), with the paired difference to clean "
            "(seeds whose CI excludes 0):",
            "",
        ]
        rows = []
        for name, v in r["methods"].items():
            for c, row in v["rows"].items():
                vc = row.get("vs_clean", {})
                rows.append(
                    [
                        f"{name} (`{v['model']}`)" if name == "best_supervised" else name,
                        c,
                        _v(row["detection_rate"], 3),
                        _v(row["false_alarms_per_day"], 4),
                        (
                            f"{vc['detection_rate']['point']:+.3f} "
                            f"({vc['detection_rate']['seeds_excluding_zero']}/{vc['detection_rate']['seeds']})"
                        )
                        if vc
                        else NA,
                        (
                            f"{vc['false_alarms_per_day']['point']:+.4f} "
                            f"({vc['false_alarms_per_day']['seeds_excluding_zero']}/{vc['false_alarms_per_day']['seeds']})"
                        )
                        if vc
                        else NA,
                    ]
                )
        out += _table(
            [
                "Method",
                "Condition",
                "Detection rate",
                "FA / day",
                "Δ detection vs clean",
                "Δ FA / day vs clean",
            ],
            rows,
        )
        out += ["", "![Azure robustness](real_data_azure_robustness.png)", ""]
    c = a.get("calibration")
    if c:
        out += [
            "",
            f"**Calibration and conformal sets** (exploratory, best supervised model "
            f"`{a['best_supervised']}`, +{real.primary.azure_pdm.horizon} h head, calibration "
            "fitted on validation, evaluated on test operating windows). Positive-class ECE "
            '= p(WARNING) + p(FAULT) against "not NORMAL" (FAULT is a single hour per '
            "event). Mondrian LAC at α = 0.1:",  # noqa: RUF001
            "",
        ]
        rows = []
        for s, hd in c.items():
            if not hd:
                continue
            mc = hd["conformal"].get("mondrian_lac", {})
            pc = mc.get("per_class", {})
            rows.append(
                [
                    s,
                    f"{hd['n']:,}",
                    f"{_v(hd['ece_top_uncal'], 4)} → {_v(hd['ece_top_cal'], 4)}",
                    f"{_v(hd['ece_fault_uncal'], 4)} → {_v(hd['ece_fault_cal'], 4)}",
                    f"{_v(hd['ece_pos_uncal'], 4)} → {_v(hd['ece_pos_cal'], 4)}",
                    " / ".join(
                        f"{k[0]} {_v(v['coverage'], 3)} (n {v['n']:,})" for k, v in pc.items()
                    ),
                    " / ".join(f"{k[0]} {_v(v['size'], 2)}" for k, v in pc.items()),
                ]
            )
        out += _table(
            [
                "Seed",
                "Windows",
                "ECE top-label",
                "ECE FAULT",
                "ECE positive",
                "Mondrian coverage N / W / F",
                "Set size N / W / F",
            ],
            rows,
        )
        out += ["", "![Azure reliability](real_data_azure_reliability.png)", ""]
    return out


# ---------------------------------------------------------------------------- simulator comparison

SIM_LABELS = {
    "rf": "RF",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "autoencoder": "Autoencoder",
    "iforest": "Isolation Forest",
}
HOURS_PER_DAY = 24.0


def _pt(e: Any) -> float:
    if isinstance(e, dict):
        v = e.get("point", e.get("mean"))
        return float("nan") if v is None else float(v)
    return float("nan")


def compute_sim_comparison(results: dict[str, Any], results_dir: Path) -> dict[str, Any] | None:
    """Simulator (E1 clean, Phase 9 E5) next to the real-data primary outcomes (exploratory).

    Reads the generated simulator summaries ``E1_clean.json`` and ``e5_explain.json`` from the
    results folder; returns None when either is missing (e.g. in tests). Simulator false alarms
    per hour of simulated operation are scaled to per day; nothing else is converted.
    """
    e1p, e5p = results_dir / "E1_clean.json", results_dir / "e5_explain.json"
    if not (e1p.exists() and e5p.exists()):
        return None
    e1 = json.loads(e1p.read_text(encoding="utf-8"))
    e5 = json.loads(e5p.read_text(encoding="utf-8"))
    h = int(e5["primary_config"]["horizon"])
    pct = float(e5["primary_config"]["anomaly_percentile"])
    e1m = {m["model"]: m["metrics"] for m in e1["models"]}
    ew = {(r["kind"], r["label"]): r["values"] for r in e5["primary"]["early_warning"]}
    trade = {r["label"]: r for r in e5["exploratory"]["tradeoff"] if float(r["percentile"]) == pct}
    az, mp = results.get("azure") or {}, results.get("metropt3") or {}
    best = az.get("best_supervised")
    methods: list[tuple[str, str]] = []
    if best:
        methods.append((best, "best_supervised"))
    methods += [("autoencoder", "autoencoder"), ("iforest", "iforest"), ("threshold", "threshold")]
    rows: list[dict[str, Any]] = []
    for name, group in methods:
        row: dict[str, Any] = {"method": name, "group": group}
        if name in ("autoencoder", "iforest"):
            t = trade.get(SIM_LABELS[name], {})
            row["sim_clean_detection"] = 1.0 - _pt(t.get("clean_miss_rate"))
            row["sim_clean_fa_day"] = _pt(t.get("fa_clean")) * HOURS_PER_DAY
            v = ew.get(("detector", f"{SIM_LABELS[name]} (p{pct:g})"), {})
        else:
            met = e1m.get(name, {})
            row["sim_clean_detection"] = 1.0 - _pt(met.get(f"h{h}.ew.miss_rate"))
            row["sim_clean_fa_day"] = _pt(met.get(f"h{h}.ew.false_alarms_per_h")) * HOURS_PER_DAY
            key = ("reference", "Threshold rule") if name == "threshold" else None
            v = ew.get(key or ("supervised", SIM_LABELS.get(name, name)), {})
        row["sim_e5_detection"] = 1.0 - _pt(v.get("miss_rate"))
        if az:
            src = az["primary"].get(group) or {}
            mean = src.get("mean") or az["exploratory"]["methods"].get(f"{name}|p{pct:g}") or {}
            row["azure_detection"] = _pt(mean.get("detection_rate"))
            row["azure_fa_day"] = _pt(mean.get("false_alarms_per_day"))
        if mp and name in mp.get("primary", {}):
            pts = [sd["points"] for sd in mp["primary"][name]["seeds"].values()]
            row["metropt3_warned"] = _finite_mean([p["warned_count"] for p in pts])
            row["metropt3_evaluable"] = _finite_mean([p["evaluable"] for p in pts])
            row["metropt3_chance"] = _finite_mean([p["chance_expected"] for p in pts])
            row["metropt3_fa_day"] = _finite_mean([p["false_alarms_per_day"] for p in pts])
        rows.append(row)
    return {"sim_horizon_s": h, "percentile": pct, "rows": rows}


def render_sim_comparison(c: dict[str, Any]) -> list[str]:
    out = [
        "## Simulator vs real data (exploratory)",
        "",
        "Simulator columns come from the generated `E1_clean.json` (clean test runs, "
        f"+{c['sim_horizon_s']} s horizon) and `e5_explain.json` (the unseen "
        "`lubrication_loss` fault; detectors at the "
        f"{c['percentile']:g}th percentile, clean false alarms from the same runs). Simulator "
        "false alarms per hour of simulated operation are multiplied by 24 to give per day. The "
        "scales differ by "
        "orders of magnitude (simulator episodes last minutes and leads are seconds; Azure leads "
        "are hours, MetroPT-3 leads minutes), the event definitions differ, and the simulator "
        "selected no single best model, so the supervised row uses the model chosen on Azure "
        "validation. Read the table as a comparison of method rankings, not of values. MetroPT-3: "
        "seed mean of warned / evaluable failures at the primary 6 h lookback.",
        "",
    ]
    rows = []
    for r in c["rows"]:
        mp = (
            f"{_v(r['metropt3_warned'], 2)} / {_v(r['metropt3_evaluable'], 0)} "
            f"(chance {_v(r['metropt3_chance'], 2)})"
            if "metropt3_warned" in r
            else NA
        )
        rows.append(
            [
                f"`{r['method']}`"
                + (" (best supervised)" if r["group"] == "best_supervised" else ""),
                _v(r["sim_clean_detection"], 3),
                _v(r["sim_clean_fa_day"], 2),
                _v(r["sim_e5_detection"], 3),
                _v(r.get("azure_detection"), 3),
                _v(r.get("azure_fa_day"), 3),
                mp,
                _v(r.get("metropt3_fa_day"), 3),
            ]
        )
    out += _table(
        [
            "Method",
            "Sim clean detection",
            "Sim clean FA / day",
            "Sim E5 detection",
            "Azure detection",
            "Azure FA / day",
            "MetroPT-3 warned",
            "MetroPT-3 FA / day",
        ],
        rows,
    )
    return [*out, ""]


def render(results: dict[str, Any], real: Any) -> list[str]:
    out: list[str] = []
    if results.get("metropt3"):
        out += render_metropt(results["metropt3"], real)
    if results.get("azure"):
        out += render_azure(results["azure"], real)
    if results.get("explain"):
        from twin_ml.experiments.real_explain import render_explain

        out += render_explain(results["explain"], real)
    if results.get("simulator"):
        out += render_sim_comparison(results["simulator"])
    return out


# ---------------------------------------------------------------------------- plots


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_metropt_timelines(
    res: dict[str, Any], m: dict[str, Any], meta_events: list[dict[str, Any]], real: Any, path: Path
) -> bool:
    """Autoencoder seed 0 score around each failure, with its p99.9 threshold, the 6 h lookback
    and the reported failure interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = res["studies"]["metropt3"]
    ent = runs.get(("autoencoder", 0))
    if ent is None or "scores" not in ent:
        return False
    sc = ent["scores"]
    thr = ent["detail"]["method"]["thresholds"]["values"][f"{m['percentile']}"][0]
    lb_s = m["lookback_rows"] * m["step_s"]
    n = len(meta_events)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.1 * n), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, meta_events, strict=True):
        a, b = ev["start_s"] - 36 * 3600, ev["end_s"] + 12 * 3600
        sel = (sc["t_s"] >= a) & (sc["t_s"] <= b)
        t = (sc["t_s"][sel] - ev["start_s"]) / 3600
        _style(ax)
        ax.axvspan(-lb_s / 3600, 0, color=ACCENT2, alpha=0.15, lw=0, label="6 h lookback")
        ax.axvspan(
            0,
            (ev["end_s"] - ev["start_s"]) / 3600,
            color=ACCENT2,
            alpha=0.4,
            lw=0,
            label="reported failure",
        )
        ax.scatter(t, sc["scores"][sel], s=1.5, color=ACCENT, lw=0, label="autoencoder score")
        ax.axhline(thr, color=INK, lw=0.8, ls="--", label=f"p{m['percentile']:g} threshold")
        ax.set_yscale("log")
        ax.set_xlim(-36, (b - ev["start_s"]) / 3600)
        ax.set_title(
            f"failure #{ev['id']} ({ev['start']} to {ev['end']}); {int(sel.sum()):,} windows",
            color=INK,
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("score", color=INK2, fontsize=8)
    axes[-1].set_xlabel("hours from the reported failure start", color=INK2, fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=4, markerscale=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_metropt_months(m: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, dict[str, float]] = {}
    days: dict[str, float] = {}
    for method, v in m["primary"].items():
        per = [sd["fa_by_month"] for sd in v["seeds"].values() if sd["fa_by_month"]]
        if per:
            series[method] = {
                mo: _finite_mean([p[mo]["false_alarms_per_day"] for p in per]) for mo in per[0]
            }
            days = {mo: per[0][mo]["negative_days"] for mo in per[0]}
    if not series:
        return False
    months = sorted(days)
    fig, ax = plt.subplots(figsize=(9.5, 3.0), facecolor=SURFACE)
    _style(ax)
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        x = np.arange(len(months)) + (i - (len(series) - 1) / 2) * w
        ax.bar(
            x,
            [vals[mo] if math.isfinite(vals[mo]) else 0 for mo in months],
            width=w,
            color=SERIES[i % len(SERIES)],
            label=name,
            lw=0,
        )
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels([f"{mo}\n{days[mo]:.1f} neg. days" for mo in months])
    ax.set_ylabel("false alarms / day", color=INK2)
    ax.set_title(
        "MetroPT-3: false alarms per day by test month (detectors: seed mean)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_azure_primary(a: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(a["primary"])
    if not groups:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), facecolor=SURFACE)
    for ax, metric, label in (
        (axes[0], "detection_rate", "detection rate (48 h lookback)"),
        (axes[1], "false_alarms_per_day", "false alarms / machine-day"),
    ):
        _style(ax)
        for i, g in enumerate(groups):
            e = a["primary"][g]["mean"][metric]
            if not math.isfinite(e["point"]):
                continue
            lo = e["point"] - e["lo"] if math.isfinite(e["lo"]) else 0
            hi = e["hi"] - e["point"] if math.isfinite(e["hi"]) else 0
            ax.errorbar(
                [i],
                [e["point"]],
                yerr=[[max(lo, 0)], [max(hi, 0)]],
                fmt="o",
                color=SERIES[i % len(SERIES)],
                capsize=3,
            )
        ax.set_xticks(range(len(groups)))
        names = [f"best ({a['best_supervised']})" if g == "best_supervised" else g for g in groups]
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_title(label, color=INK, fontsize=9, loc="left")
    ev = a["counts"]["events"]
    fig.suptitle(
        f"Azure PdM, test period: {ev} events, 95 % machine bootstrap CI (seed means)",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_azure_robustness(a: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = a.get("robustness")
    if not r or not r["methods"]:
        return False
    conds = r["conditions"]
    fig, ax = plt.subplots(figsize=(9.5, 3.0), facecolor=SURFACE)
    _style(ax)
    for i, (name, v) in enumerate(r["methods"].items()):
        ys = [v["rows"].get(c, {}).get("detection_rate", float("nan")) for c in conds]
        ax.plot(
            range(len(conds)),
            ys,
            marker="o",
            color=SERIES[i % len(SERIES)],
            label=f"{name} ({v['model']})" if name == "best_supervised" else name,
        )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("detection rate", color=INK2)
    ax.set_title(
        f"Azure PdM robustness (exploratory): detection rate per condition, "
        f"{a['counts']['events']} events",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_azure_reliability(a: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = (a.get("calibration") or {}).get("0")
    if not c:
        return False
    fig, ax = plt.subplots(figsize=(4.6, 4.0), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    for i, tag in enumerate(("uncal", "cal")):
        rel = c[f"reliability_pos_{tag}"]
        pts = [
            (x, y)
            for x, y in zip(rel["confidence"], rel["accuracy"], strict=True)
            if x is not None and y is not None
        ]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs, ys, marker="o", color=SERIES[i], label=f"{tag} (ECE {c[f'ece_pos_{tag}']:.4f})"
            )
    ax.set_xlabel("p(WARNING) + p(FAULT)", color=INK2)
    ax.set_ylabel("share not NORMAL", color=INK2)
    ax.set_title(
        f"{a['best_supervised']} seed 0, +24 h, test operating\n({c['n']:,} windows)",
        color=INK,
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True
