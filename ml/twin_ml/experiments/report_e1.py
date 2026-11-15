"""E1 report generator: ``python -m twin_ml.experiments.report_e1`` (poe task ``report-e1``).

Reads the finished E1 runs of the current dataset from MLflow (metrics + the ``metrics.json``
artifact of each run) and writes ``<report.out_dir>/<name>.md``, ``<name>.json`` and
``<name>_lead_times.png``. Every number in the output comes from MLflow; nothing is typed by
hand. Per model: mean ± std over seeds and the mean of the per-seed bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twin_common.config import ConfigError
from twin_common.logging import configure_logging
from twin_ml.config import DatasetConfig, EvalConfig
from twin_ml.data.labels import HEALTH
from twin_ml.train import tracking

CI_METRICS = (
    "macro_f1",
    "fault_recall",
    "lead_mean_s",
    "rel_lead_mean_s",
    "miss_rate",
    "false_alarms_per_h",
)
NA = "–"  # en dash: a value that does not exist (never zero)  # noqa: RUF001
SUBSETS = ("operating", "stopped", "gradual", "jam", "all")
SUBSET_TITLES = {
    "operating": "(a) operating now (RUNNING or WARNING), main result",
    "stopped": "(b) already in FAULT or MAINTENANCE",
    "gradual": "(c) operating windows before a gradual fault (onset within the ttf cap)",
    "jam": "(c) operating windows before a belt jam (onset within the ttf cap; no precursors)",
    "all": "all test windows",
}

# Report plot colours (dataviz reference palette: light chart surface, ink, one accent).
SURFACE, INK, INK2, MUTED, GRID, AXIS, ACCENT, ACCENT2 = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#2a78d6",
    "#eb6834",  # categorical slot 2; blue/orange validated as a pair (validate_palette.js)
)


# ---------------------------------------------------------------------------- collect


Runs = dict[str, dict[int, dict[str, Any]]]  # model -> seed -> run entry


def collect_runs(tracking_dir: Path, experiment: str, dataset_key: str) -> tuple[Runs, Runs]:
    """Finished runs of the dataset, grouped per model by model config (``model_config_sha``).

    Returns (current, previous): the newest run per seed of the newest config group, and of
    the group before it (if the model was re-run with a changed config). Runs logged before the
    tag existed form one "legacy" group.
    """
    import mlflow

    groups: dict[str, list[str]] = defaultdict(list)  # model -> config shas, newest first
    by_group: dict[tuple[str, str], dict[int, Any]] = defaultdict(dict)
    for run in tracking.finished_runs(tracking_dir, experiment, dataset_key):  # newest first
        model, seed = run.data.tags["model"], int(run.data.tags["seed"])
        sha = run.data.tags.get("model_config_sha", "legacy")
        if sha not in groups[model]:
            groups[model].append(sha)
        by_group[(model, sha)].setdefault(seed, run)

    def entries(model: str, sha: str) -> dict[int, dict[str, Any]]:
        return {
            seed: {
                "run_id": run.info.run_id,
                "artifact_uri": run.info.artifact_uri,
                "config_sha": sha,
                "metrics": dict(run.data.metrics),
                "detail": mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/metrics.json"),
            }
            for seed, run in sorted(by_group[(model, sha)].items())
        }

    current = {m: entries(m, shas[0]) for m, shas in groups.items()}
    previous = {m: entries(m, shas[1]) for m, shas in groups.items() if len(shas) > 1}
    return current, previous


def _stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def summarize_model(
    model: str, runs: dict[int, dict[str, Any]], horizons: list[str]
) -> dict[str, Any]:
    seeds = sorted(runs)
    keys = sorted({k for r in runs.values() for k in r["metrics"]})
    metrics = {k: _stat([runs[s]["metrics"].get(k, float("nan")) for s in seeds]) for k in keys}
    first = runs[seeds[0]]["detail"]
    ci: dict[str, Any] = {}
    early: dict[str, Any] = {}
    leads: dict[str, list[float]] = {}
    warn_leads: dict[str, list[float]] = {}
    for h in horizons:
        ci[h] = {
            m: {
                "point": _stat(
                    [runs[s]["detail"]["horizons"][h]["bootstrap"][m]["point"] for s in seeds]
                ),
                "lo": _stat(
                    [runs[s]["detail"]["horizons"][h]["bootstrap"][m]["lo"] for s in seeds]
                ),
                "hi": _stat(
                    [runs[s]["detail"]["horizons"][h]["bootstrap"][m]["hi"] for s in seeds]
                ),
            }
            for m in CI_METRICS
        }
        ew = first["horizons"][h]["early_warning"]
        early[h] = {
            "episodes": ew["episodes"],
            "episodes_per_kind": ew["episodes_per_kind"],
            "negative_h": ew["negative_h"],
        }
        eps = [e for s in seeds for e in runs[s]["detail"]["horizons"][h]["episodes"]]
        leads[h] = [float(e["lead_s"]) for e in eps if e["detected"]]
        warn_leads[h] = [
            float(e["t_fault"] - e["t_warn"])
            for e in first["horizons"][h]["episodes"]
            if e["t_warn"] is not None
        ]
    infos = [runs[s]["detail"].get("train_info", {}) for s in seeds]
    train = {
        "epochs": _stat([float(i["epochs"]) for i in infos if "epochs" in i]),
        "best_epoch": _stat([float(i["best_epoch"]) for i in infos if "best_epoch" in i]),
        "stop_reasons": sorted(
            {str(i.get("stop_reason", "patience")) for i in infos if "epochs" in i}
        ),
        "history": {s: i.get("history", []) for s, i in zip(seeds, infos, strict=True)},
    }
    return {
        "model": model,
        "family": first["family"],
        "seeds": seeds,
        "config_sha": runs[seeds[0]].get("config_sha"),
        "train": train,
        "run_ids": [runs[s]["run_id"] for s in seeds],
        "metrics": metrics,
        "ci": ci,
        "early_warning": early,
        "leads": leads,
        "threshold_warning_leads": warn_leads,
    }


# ---------------------------------------------------------------------------- render


def _f(x: float, digits: int) -> str:
    return NA if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def _ms(summary: dict[str, Any], key: str, digits: int = 3) -> str:
    st = summary["metrics"].get(key)
    if st is None or st["n"] == 0:
        return NA
    if st["n"] == 1:
        return _f(st["mean"], digits)
    return f"{_f(st['mean'], digits)} ± {_f(st['std'], digits)}"


def _ci(summary: dict[str, Any], h: str, metric: str, digits: int = 3) -> str:
    c = summary["ci"][h][metric]
    lo, hi = c["lo"]["mean"], c["hi"]["mean"]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return NA
    return f"[{_f(lo, digits)}, {_f(hi, digits)}]"


def _with_ci(summary: dict[str, Any], h: str, key: str, metric: str, digits: int = 3) -> str:
    base = _ms(summary, key, digits)
    if base == NA:
        return base
    return f"{base} {_ci(summary, h, metric, digits)}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _data_section(report: dict[str, Any]) -> list[str]:
    ds = report["dataset"]
    horizons = report["horizons"]
    lines = ["## Data", ""]
    src = ds["source"]
    lines += _table(
        ["Item", "Value"],
        [
            ["Dataset key", f"`{ds['key']}`"],
            ["Simulator manifest sha256 (file bytes)", f"`{src['manifest_sha256']}`"],
            ["Simulator content sha256", f"`{src['content_sha256']}`"],
            [
                "Source",
                f"`{src['dir']}`: scenario `{src['scenario']}`, {src['runs']} runs × "  # noqa: RUF001
                f"{_f(src['duration_s'], 0)} s, base seed {src['base_seed']}",
            ],
            [
                "Held-out fault types (not in the data)",
                ", ".join(src["held_out_fault_types"]) or NA,
            ],
            [
                "Split (runs: train / val / test)",
                " / ".join(str(ds["counts"][s]["runs"]) for s in ("train", "val", "test")),
            ],
            [
                "Windows (train / val / test)",
                " / ".join(str(ds["counts"][s]["windows"]) for s in ("train", "val", "test")),
            ],
            [
                "Strides s (train / val / test)",
                " / ".join(str(ds["config"]["strides_s"][s]) for s in ("train", "val", "test")),
            ],
            [
                "Window, horizons, ttf cap",
                f"{ds['config']['window_s']} s; "
                + ", ".join(f"+{h} s" for h in horizons)
                + f"; {_f(ds['config']['ttf_cap_s'], 0)} s",
            ],
            [
                "Training subsample of far-normal windows",
                f"keep {ds['config']['normal_keep_frac']}, kept windows weighted 1 / keep",
            ],
            ["Gradual-fault episodes in test runs", str(ds["counts"]["test"]["gradual_episodes"])],
            [
                "Bootstrap",
                f"{report['bootstrap']['n_resamples']} resamples of test runs, seed "
                f"{report['bootstrap']['seed']}, "
                f"{int(report['bootstrap']['ci'] * 100)} % percentile CI",
            ],
        ],
    )
    lines += [
        "",
        "Values are mean ± std over seeds (baselines are deterministic: one run). Brackets: "
        "95 % bootstrap CI over test runs (mean of the per-seed bounds).",
        "",
    ]
    return lines


NOTES = [
    "- **Primary result: episode-level early warning.** The research question is whether the "
    "twin warns before a fault. Per-window WARNING precision penalises a model that warns "
    "earlier than the simulator's WARNING label (those windows count as false positives), so "
    "the episode metrics below (lead time, lead relative to the threshold WARNING, miss rate, "
    "false alarms per hour) are the primary result and per-window macro F1 is secondary.",
    "- **Low FAULT recall at +10 s** means the models warn early as WARNING rather than timing "
    "the exact FAULT transition; the early-warning metrics count a WARNING or FAULT prediction "
    "as an alarm, so such a model still warns.",
    "- **Subset (b), machines already stopped:** a stopped machine reads zero on every sensor, "
    "so sensor-only models cannot tell IDLE, FAULT and MAINTENANCE apart. This follows from the "
    "input restriction (no state input); persistence wins there by construction. The live "
    "system (Phase 11) should not predict for stopped machines.",
    "- **Belt jams** have no precursors in the signals; no model can anticipate them.",
]


def _yes(x: bool) -> str:
    return "yes" if x else "no"


def _diff_cell(d: dict[str, Any], digits: int) -> str:
    if not math.isfinite(d["point"]):
        return NA
    ci = (
        f" [{_f(d['lo'], digits)}, {_f(d['hi'], digits)}]"
        if math.isfinite(d["lo"]) and math.isfinite(d["hi"])
        else ""
    )
    seeds = f", {d['seeds_excluding_zero']}/{d['seeds']} seeds" if d["seeds"] > 1 else ""
    return f"{_f(d['point'], digits)}{ci} ({_yes(d['excludes_zero'])}{seeds})"


DIFF_COLUMNS = (
    ("lead_mean_s", "Lead mean s", 0),
    ("rel_lead_mean_s", "Lead vs threshold WARNING s", 0),
    ("macro_f1", "Macro F1 (operating)", 3),
    ("false_alarms_per_h", "False alarms / h", 2),
)


def _early_warning_section(report: dict[str, Any]) -> list[str]:
    horizons, models = report["horizons"], report["models"]
    lines = ["## Primary result: early warning on gradual faults", ""]
    for h in horizons:
        first: dict[str, Any] = (
            models[0]["early_warning"][h] if models else {"episodes": 0, "episodes_per_kind": {}}
        )
        kinds = ", ".join(f"{k} {v}" for k, v in first["episodes_per_kind"].items()) or NA
        lines += [
            f"**Horizon +{h} s** - {first['episodes']} gradual-fault episodes in the test runs "
            f"({kinds}); sustained alarm = {report['sustain_s']} s of WARNING/FAULT predictions.",
            "",
        ]
        rows = [
            [
                m["model"],
                _with_ci(m, h, f"h{h}.ew.miss_rate", "miss_rate"),
                _with_ci(m, h, f"h{h}.ew.lead_mean_s", "lead_mean_s", 0),
                _ms(m, f"h{h}.ew.lead_median_s", 0),
                _with_ci(m, h, f"h{h}.ew.rel_lead_mean_s", "rel_lead_mean_s", 0),
                _ms(m, f"h{h}.ew.rel_lead_median_s", 0),
                _with_ci(m, h, f"h{h}.ew.false_alarms_per_h", "false_alarms_per_h", 2),
            ]
            for m in models
        ]
        lines += _table(
            [
                "Model",
                "Miss rate",
                "Lead mean s",
                "Lead median s",
                "vs threshold WARNING, mean s",
                "vs threshold WARNING, median s",
                "False alarms / h",
            ],
            rows,
        )
        lines += [""]
    lines += [
        "Lead = fault onset - first sustained alarm after the degradation started. "
        '"vs threshold WARNING" = simulator WARNING onset - alarm (positive = earlier than the '
        "simulator's own warning). False alarms are counted in operating windows with no "
        "active fault and a NORMAL target.",
        "",
    ]
    if report.get("plots", {}).get("leads"):
        lines += [f"![Lead times per model]({report['plots']['leads']})", ""]
    return lines


def _paired_section(report: dict[str, Any]) -> list[str]:
    paired = report.get("paired")
    if not paired:
        return []
    lines = [
        "## Paired comparisons",
        "",
        "Differences on the same resampled test runs (paired bootstrap, "
        f"{report['bootstrap']['n_resamples']} resamples, "
        f"{int(report['bootstrap']['ci'] * 100)} % CI). Each cell: mean difference over seeds "
        "[mean CI bounds] (does the CI exclude zero; in how many seeds the per-seed CI "
        "excludes zero). Lead differences compare mean leads of the detected episodes.",
        "",
    ]
    header = ["Model"] + [c[1] for c in DIFF_COLUMNS]
    for h, rows_by_model in paired["vs_reference"].items():
        lines += [f"**Model - {paired['reference']} baseline, horizon +{h} s**", ""]
        rows = [
            [m] + [_diff_cell(d[c[0]], c[2]) for c in DIFF_COLUMNS]
            for m, d in rows_by_model.items()
        ]
        lines += [*_table(header, rows), ""]
    if paired["tree_vs_sequence"]:
        lines += [
            "**Best sequence model - best tree model** (best = highest mean lead time at that "
            "horizon, or operating macro F1 if no model of a family detected an episode; the "
            "metric used is in the JSON summary; seed i paired with seed i)",
            "",
        ]
        rows = [
            [f"+{h} s: {v['sequence']} - {v['tree']}"]
            + [_diff_cell(v["diff"][c[0]], c[2]) for c in DIFF_COLUMNS]
            for h, v in paired["tree_vs_sequence"].items()
        ]
        lines += [*_table(["Horizon: models", *header[1:]], rows), ""]
    return lines


def _classification_section(report: dict[str, Any]) -> list[str]:
    horizons, models = report["horizons"], report["models"]
    lines = ["## Secondary: per-window classification", ""]
    for h in horizons:
        lines += [f"### Operating windows, horizon +{h} s", ""]
        rows = [
            [
                m["model"],
                str(len(m["seeds"])),
                _with_ci(m, h, f"h{h}.operating.macro_f1", "macro_f1"),
                _with_ci(m, h, f"h{h}.operating.fault_recall", "fault_recall"),
                _ms(m, f"h{h}.operating.accuracy"),
                _ms(m, f"h{h}.operating.pr_auc_macro"),
            ]
            for m in models
        ]
        lines += _table(
            ["Model", "Seeds", "Macro F1", "Fault recall", "Accuracy", "PR-AUC (W, F)"], rows
        )
        lines += ["", f"Per class (+{h} s, operating):", ""]
        rows = []
        for m in models:
            row = [m["model"]]
            for c in HEALTH:
                row += [
                    _ms(m, f"h{h}.operating.precision.{c.value}"),
                    _ms(m, f"h{h}.operating.recall.{c.value}"),
                    _ms(m, f"h{h}.operating.f1.{c.value}"),
                ]
            rows.append(row)
        header = ["Model"] + [f"{c.value[0]} {x}" for c in HEALTH for x in ("P", "R", "F1")]
        lines += [*_table(header, rows), ""]
    lines += ["### Other subsets", ""]
    for subset in SUBSETS[1:]:
        lines += [f"**{SUBSET_TITLES[subset]}**", ""]
        rows = []
        for m in models:
            row = [m["model"]]
            for h in horizons:
                row += [_ms(m, f"h{h}.{subset}.macro_f1"), _ms(m, f"h{h}.{subset}.fault_recall")]
            rows.append(row)
        header = ["Model"] + [f"{x} +{h} s" for h in horizons for x in ("Macro F1", "Fault recall")]
        lines += [*_table(header, rows), ""]
    return lines


def _cost_section(report: dict[str, Any]) -> list[str]:
    models = report["models"]
    rows = [
        [
            m["model"],
            _ms(m, "ttf.operating.mae_s", 1),
            _ms(m, "ttf.all.mae_s", 1),
            _ms(m, "fault_type.operating.macro_f1"),
            _ms(m, "latency.p50_ms", 2),
            _ms(m, "latency.p95_ms", 2),
            _ms(m, "time.train_s", 1),
        ]
        for m in models
    ]
    return [
        "## Time to fault, fault type, cost",
        "",
        *_table(
            [
                "Model",
                "TTF MAE s (operating)",
                "TTF MAE s (all)",
                "Fault type macro F1 (operating)",
                "Latency p50 ms",
                "Latency p95 ms",
                "Train time s",
            ],
            rows,
        ),
        "",
        "TTF MAE uses rows with a known time to fault below the cap (censored rows excluded). "
        "Latency: batch size 1 on CPU, one thread, features included.",
        "",
    ]


def _training_section(report: dict[str, Any]) -> list[str]:
    seq = [m for m in report["models"] if m["family"] == "nn"]
    if not seq:
        return []
    rows = [
        [
            m["model"],
            _stat_cell(m["train"]["epochs"], 1),
            _stat_cell(m["train"]["best_epoch"], 1),
            ", ".join(m["train"]["stop_reasons"]) or NA,
            _ms(m, "time.train_s", 0),
        ]
        for m in seq
    ]
    lines = [
        "## Sequence-model training",
        "",
        "Fixed training budget: every run trains the same number of epochs with the same "
        "learning-rate schedule (no early stopping on patience). The checkpoint kept is the epoch "
        "with the best exponential moving average (EMA) of the validation macro F1 (operating "
        "windows, mean of the horizons). Settings: `config/models/<model>.yaml`.",
        "",
        *_table(["Model", "Epochs", "Best epoch", "Stopped by", "Train time s"], rows),
        "",
    ]
    if report.get("plots", {}).get("curves"):
        lines += [f"![Learning curves]({report['plots']['curves']})", ""]
    return lines


CHANGE_KEYS = (
    ("Epochs", None, 1),
    ("Macro F1 +10 s", "h10.operating.macro_f1", 3),
    ("Macro F1 +30 s", "h30.operating.macro_f1", 3),
    ("Fault recall +30 s", "h30.operating.fault_recall", 3),
    ("Lead mean +30 s", "h30.ew.lead_mean_s", 0),
    ("vs threshold WARNING +30 s", "h30.ew.rel_lead_mean_s", 0),
    ("False alarms / h +30 s", "h30.ew.false_alarms_per_h", 2),
    ("Train time s", "time.train_s", 0),
)


def _changes_section(report: dict[str, Any]) -> list[str]:
    prev = {m["model"]: m for m in report.get("previous", [])}
    if not prev:
        return []
    lines = [
        "## Changes since the previous configuration",
        "",
        "Models whose config changed since their previous runs on this dataset (grouped by "
        "`model_config_sha`). Both rows come from MLflow.",
        "",
    ]
    rows = []
    for cur in report["models"]:
        old = prev.get(cur["model"])
        if old is None:
            continue
        for label, m in (("previous", old), ("current", cur)):
            row = [f"{cur['model']} ({label})"]
            for _, key, digits in CHANGE_KEYS:
                row.append(
                    _stat_cell(m["train"]["epochs"], digits) if key is None else _ms(m, key, digits)
                )
            rows.append(row)
    lines += [*_table(["Model", *[k[0] for k in CHANGE_KEYS]], rows), ""]
    return lines


def _stat_cell(st: dict[str, float], digits: int) -> str:
    if not st or st["n"] == 0:
        return NA
    if st["n"] == 1:
        return _f(st["mean"], digits)
    return f"{_f(st['mean'], digits)} ± {_f(st['std'], digits)}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"Generated by `uv run poe report-e1` from the MLflow experiment `{report['experiment']}` "
        f"on {report['generated_at']} (git `{report['git_commit'] or 'unknown'}`). "
        "Do not edit by hand; definitions are in [docs/experiments.md](../../docs/experiments.md).",
        "",
        *_data_section(report),
        "## How to read this report",
        "",
        *NOTES,
        "",
        *_early_warning_section(report),
        *_paired_section(report),
        *_classification_section(report),
        *_cost_section(report),
        *_training_section(report),
        *_changes_section(report),
    ]
    return "\n".join(lines)


def plot_leads(report: dict[str, Any], path: Path) -> None:
    """Strip plot of per-episode lead times per model (one panel per horizon)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons, models = report["horizons"], report["models"]
    names = [m["model"] for m in models]
    fig, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(5.2 * len(horizons), 0.45 * len(names) + 1.6),
        sharey=True,
        sharex=True,
        facecolor=SURFACE,
        layout="constrained",
    )
    axes_list = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, h in zip(axes_list, horizons, strict=True):
        ax.set_facecolor(SURFACE)
        warn = models[0]["threshold_warning_leads"][h] if models else []
        if warn:
            ax.axvline(float(np.median(warn)), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for y, m in enumerate(models):
            leads = np.asarray(m["leads"][h], dtype=float)
            if leads.size:
                ax.scatter(
                    leads,
                    y + rng.uniform(-0.18, 0.18, leads.size),
                    s=14,
                    color=ACCENT,
                    alpha=0.55,
                    linewidths=0,
                    zorder=2,
                )
                med = float(np.median(leads))
                ax.plot(
                    [med, med],
                    [y - 0.32, y + 0.32],
                    color=INK,
                    lw=2.2,
                    zorder=3,
                    solid_capstyle="round",
                )
            else:
                ax.text(
                    0.01,
                    y,
                    "no episode detected",
                    color=MUTED,
                    fontsize=8,
                    va="center",
                    transform=ax.get_yaxis_transform(),
                )
        ax.set_title(f"Horizon +{h} s", color=INK, fontsize=10, loc="left")
        ax.set_yticks(range(len(names)), names, color=INK2, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel("lead time before FAULT (s)", color=INK2, fontsize=9)
    axes_list[0].set_xlim(left=0)
    axes_list[0].set_ylim(len(names) - 0.4, -0.6)  # shared y: first model on top
    fig.suptitle(
        "Lead time per detected gradual-fault episode (dots; all seeds), median (bar); "
        "dashed: median simulator WARNING lead",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def plot_curves(report: dict[str, Any], path: Path) -> bool:
    """Learning curves of the sequence models: one row per model, (losses, val macro F1).

    Thin lines are seeds; dots mark the kept (best) epoch. Returns False if nothing to plot.
    """
    seq = [
        m for m in report["models"] if m["family"] == "nn" and any(m["train"]["history"].values())
    ]
    if not seq:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    horizons = report["horizons"]
    fig, axes = plt.subplots(
        len(seq),
        2,
        figsize=(10.4, 2.5 * len(seq) + 0.9),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
        layout="constrained",
    )
    for row, m in zip(axes, seq, strict=True):
        ax_loss, ax_f1 = row
        for hist in m["train"]["history"].values():
            if not hist:
                continue
            ep = [e["epoch"] for e in hist]
            ax_loss.plot(ep, [e["train_loss"] for e in hist], color=ACCENT, lw=1.2, alpha=0.8)
            ax_loss.plot(ep, [e["val_loss"] for e in hist], color=ACCENT2, lw=1.2, alpha=0.8)
            for h, color in zip(horizons, (ACCENT, ACCENT2), strict=False):
                ax_f1.plot(
                    ep, [e[f"val_macro_f1_h{h}"] for e in hist], color=color, lw=1.2, alpha=0.8
                )
            ax_f1.plot(
                ep, [e["val_macro_f1_ema"] for e in hist], color=INK2, lw=1.0, ls=(0, (4, 3))
            )
            best = max(hist, key=lambda e: e["val_macro_f1"])
            ax_f1.scatter([best["epoch"]], [best["val_macro_f1"]], s=22, color=INK, zorder=3)
        ax_loss.set_ylabel(m["model"], color=INK, fontsize=10)
        for ax in row:
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
    axes[0][0].set_title("Loss (weighted, all heads)", color=INK, fontsize=10, loc="left")
    axes[0][1].set_title("Validation macro F1 (operating)", color=INK, fontsize=10, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
    axes[0][0].legend(
        handles=[Line2D([], [], color=ACCENT, lw=1.5), Line2D([], [], color=ACCENT2, lw=1.5)],
        labels=["train", "validation"],
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
    )
    axes[0][1].legend(
        handles=[
            *[Line2D([], [], color=c, lw=1.5) for c in (ACCENT, ACCENT2)][: len(horizons)],
            Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3))),
            Line2D([], [], color=INK, marker="o", lw=0, markersize=5),
        ],
        labels=[*[f"+{h} s" for h in horizons][:2], "EMA (checkpoint signal)", "kept epoch"],
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
    )
    fig.suptitle(
        "Sequence-model learning curves (one thin line per seed)",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def build_report(
    ds_cfg: DatasetConfig,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Write the E1 report files; returns their paths."""
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.experiments.paired import paired_comparisons

    ds = Dataset.open(resolve_dataset_dir(ds_cfg, root))
    key = ds.meta["dataset_key"]
    runs, previous = collect_runs(
        tracking_dir or root / eval_cfg.tracking_dir, eval_cfg.experiment, key
    )
    if not runs:
        raise LookupError(
            f"no finished {eval_cfg.experiment} runs for dataset {key}; run uv run poe train-all"
        )
    horizons = [str(h) for h in ds.horizons]
    order = [m for m in eval_cfg.models if m in runs] + sorted(set(runs) - set(eval_cfg.models))
    models = [summarize_model(m, runs[m], horizons) for m in order]
    name = eval_cfg.report.name
    report: dict[str, Any] = {
        "title": "E1: early fault warning on clean data",
        "experiment": eval_cfg.experiment,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "dataset": {
            "key": key,
            "source": ds.meta["source"],
            "counts": ds.meta["counts"],
            "config": ds.meta["config"],
        },
        "horizons": horizons,
        "sustain_s": eval_cfg.early_warning.sustain_s,
        "bootstrap": eval_cfg.bootstrap.model_dump(),
        "models": models,
        "previous": [summarize_model(m, previous[m], horizons) for m in order if m in previous],
        "paired": paired_comparisons({m: runs[m] for m in order}, models, ds, eval_cfg),
        "plots": {},
    }
    out = out_dir or root / eval_cfg.report.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    lead_path = out / f"{name}_lead_times.png"
    plot_leads(report, lead_path)
    report["plots"]["leads"] = lead_path.name
    paths.append(lead_path)
    curves_path = out / f"{name}_learning_curves.png"
    if plot_curves(report, curves_path):
        report["plots"]["curves"] = curves_path.name
        paths.append(curves_path)
    md_path = out / f"{name}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(_clean(report), indent=1), encoding="utf-8")
    return [md_path, json_path, *paths]


def _clean(obj: Any) -> Any:
    """NaN/inf -> None so the JSON summary is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.report_e1")
    parser.add_argument("--dataset-config", type=Path, default=Path("config/models/dataset.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/models/eval.yaml"))
    args = parser.parse_args(argv)
    configure_logging()
    from twin_ml.config import load_dataset_config, load_eval_config
    from twin_ml.data.dataset import DatasetNotBuiltError

    try:
        paths = build_report(
            load_dataset_config(args.dataset_config), load_eval_config(args.eval_config)
        )
    except (ConfigError, DatasetNotBuiltError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
