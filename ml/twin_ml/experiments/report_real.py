"""Phase 10 report: validation on real public data (ADR 0027).

Writes ``experiments/results/real_data.md`` + ``real_data.json`` + plots, from MLflow runs only
(experiment ``P10_real``). Part A: the data audit of MetroPT-3 and Azure PdM and the mapping to
the common window format. Parts B and C add the study results.

Only runs that match the current files and configs are used: an audit run counts if its tags
carry the current source sha256, facts fingerprint and dataset config fingerprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from twin_common.config import ConfigError
from twin_common.logging import configure_logging
from twin_ml.datasets.common import MissingDatasetError, files_sha256, require_files
from twin_ml.datasets.config import (
    RealConfig,
    load_facts,
    load_real_config,
    load_real_dataset_config,
)
from twin_ml.experiments.report_e1 import (
    ACCENT,
    ACCENT2,
    AXIS,
    GRID,
    INK,
    INK2,
    LF,
    MUTED,
    NA,
    SURFACE,
    _clean,
    _table,
)
from twin_ml.experiments.report_robustness import _round

REPORT_NAME = "real_data"
DATASETS = ("metropt3", "azure_pdm")


# ---------------------------------------------------------------------------- collect


def expected_tags(name: str, real: RealConfig, root: Path) -> dict[str, str]:
    """Tags an audit run of ``name`` must carry to count for the current data and configs."""
    cfg = load_real_dataset_config(root / real.datasets[name].dataset_config)  # type: ignore[index]
    facts = load_facts(root / cfg.facts)
    paths = require_files(facts, root)
    return {
        "source_sha256": files_sha256([paths[k] for k in sorted(paths)]),
        "facts_fingerprint": facts.fingerprint(),
        "dataset_config_fingerprint": cfg.fingerprint(facts),
    }


def collect_audits(
    real: RealConfig, tracking_dir: Path, root: Path, experiment: str | None = None
) -> dict[str, dict[str, Any]]:
    """Newest valid audit run per dataset: ``{name: {"run_id", "audit"}}``."""
    from mlflow.tracking import MlflowClient

    from twin_ml.train import tracking

    exp_name = experiment or real.experiment
    out: dict[str, dict[str, Any]] = {}
    client = MlflowClient(tracking_uri=tracking.tracking_uri(tracking_dir))
    for name in DATASETS:
        want = expected_tags(name, real, root)
        runs = tracking.finished_runs(
            tracking_dir,
            exp_name,
            phase=real.phase_tag,
            extra_filter=f"tags.study = 'audit' and tags.dataset = '{name}'",
        )
        for run in runs:
            if all(run.data.tags.get(k) == v for k, v in want.items()):
                local = client.download_artifacts(run.info.run_id, "audit.json")
                out[name] = {
                    "run_id": run.info.run_id,
                    "audit": json.loads(Path(local).read_text(encoding="utf-8")),
                }
                break
    return out


# ---------------------------------------------------------------------------- render


def _f(x: Any, digits: int = 2) -> str:
    if x is None:
        return NA
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def _pct(x: Any, digits: int = 1) -> str:
    return NA if x is None else f"{100 * x:.{digits}f} %"


def _metro_section(a: dict[str, Any], real: RealConfig) -> list[str]:
    s = a["sampling"]
    g = a["gaps"]
    grid = a["grid"]
    study = real.datasets["metropt3"]
    out = [
        "### MetroPT-3",
        "",
        f"- **Origin:** {a['facts']['origin']} data. {a['facts']['provider']}; licence "
        f"{a['facts']['licence']}. Time zone: {a['facts']['time_zone']} (timestamps and failure "
        "times are read as one naive clock).",
        f"- **File:** `{a['file']}`, {_f(a['rows'])} rows, {a['first']} to {a['last']} "
        f"({_f(a['span_days'], 1)} days), {_f(a['duplicates'])} duplicate timestamps, sorted: "
        f"{'yes' if a['sorted'] else 'no'}.",
        f"- **Sampling (measured):** median interval {_f(s['median_s'], 0)} s; "
        + ", ".join(f"{t['s']} s: {_pct(t['share'])}" for t in s["top_intervals"][:5])
        + f"; min {s['min_s']} s, max {_f(s['max_s'])} s.",
        f"- **The PDF's 1 Hz claim:** it states {_f(s['pdf_rows_1hz'])} instances logged at 1 Hz; "
        f"the CSV has {s['rows_ratio_pdf_to_file']:.2f}x fewer rows at a ~10 s interval, so the "
        "file is a ~0.1 Hz version of the logged data. The models use the CSV as it is.",
        f"- **Gaps over {g['threshold_s']} s:** {g['count']}, {_f(g['total_h'], 1)} h in total. "
        "Longest: "
        + "; ".join(f"{x['start']} ({_f(x['hours'], 1)} h)" for x in g["longest"][:5])
        + ".",
        f"- **Grid used by the models:** {grid['step_s']} s rows (bin means; digital signals "
        f"become duty fractions), {_f(grid['rows'])} rows of which {_f(grid['observed_rows'])} "
        f"hold data; a run of more than {grid['max_gap_rows']} empty rows starts a new series: "
        f"{grid['series']} series (median {_f(grid['series_rows_median'], 0)} rows).",
        "",
        "**Sensor ranges** (raw file, all rows):",
        "",
    ]
    rows = []
    for x in a["sensors"]:
        extra = f"active {_pct(x.get('share_active'))}" if x["type"] == "digital" else ""
        rows.append(
            [
                f"`{x['name']}` ({x['column']})",
                x["type"],
                x["unit"] or NA,
                _f(x["missing"]),
                _f(x["min"], 3),
                _f(x["median"], 3),
                _f(x["max"], 3),
                extra or NA,
            ]
        )
    out += _table(["Signal", "Type", "Unit", "Missing", "Min", "Median", "Max", "Digital"], rows)
    out += [
        "",
        f"**Failures** ({a['failure_count']}, all air leaks). {a['label_meaning']} "
        f"Source: {a['failure_source']}",
        "",
    ]
    lbs = [str(L) for L in study.lookbacks]
    head = ["#", "Start", "End", "Hours", "Raw samples inside", "Data in interval"]
    head += [f"Data in {int(int(L) * grid['step_s'] / 3600)} h before" for L in lbs]
    rows = []
    for f in a["failures"]:
        rows.append(
            [
                f["id"],
                f["start"],
                f["end"],
                _f(f["duration_h"], 1),
                _f(f["raw_samples_inside"]),
                _pct(f["interval_coverage"]),
                *[_pct(f["lookbacks"][L]["coverage"]) for L in lbs],
            ]
        )
    out += _table(head, rows)
    pc = a.get("pdf_check")
    if pc:
        out += [
            "",
            f"- **Check against the PDF** ({pc['method']}, {pc['date']}): {pc['result']}.",
        ]
        out += [f"  - Inconsistency: {x}" for x in pc["inconsistencies"]]
    out += ["", "![MetroPT-3 data availability](real_data_metropt3_availability.png)", ""]
    out += [
        f"*Share of 1-min rows with data per day; {a['failure_count']} failure intervals "
        "shaded; split boundaries as vertical lines.*",
        "",
    ]
    return out


def _azure_section(a: dict[str, Any]) -> list[str]:
    t = a["telemetry"]
    f = a["failures"]
    e = a["errors"]
    m = a["machines"]
    out = [
        "### Azure PdM",
        "",
        f"- **Origin:** {a['facts']['origin']}. {a['facts']['provider']}. It is a public "
        "benchmark and is described here as such, not as real plant data.",
        f"- **Telemetry:** {_f(t['rows'])} rows, {t['machines']} machines, {t['first']} to "
        f"{t['last']} ({_f(t['span_days'], 0)} days). Intervals: "
        + ", ".join(f"{x['s']} s x {_f(x['count'])}" for x in t["intervals"])
        + f"; {t['gaps']} gaps; {t['duplicates']} duplicates; {t['missing_values']} missing "
        "values.",
        f"- **Machines:** {m['count']}, by model "
        + ", ".join(f"{k}: {v}" for k, v in m["by_model"].items())
        + f" (the model is the machine kind); age {m['age_min']} to {m['age_max']} years.",
        f"- **Failures:** {f['rows']} failure rows form **{f['events']} events** (machine, hour); "
        + ", ".join(f"{k}: {v}" for k, v in f["by_component"].items())
        + " rows; events with 1 / 2 failed components: "
        + " / ".join(str(v) for v in f["components_per_event"].values())
        + f"; {f['machines_with_failure']} machines with a failure; first {f['first']}, last "
        f"{f['last']}. {a['label_meaning']}",
        "- **Signs of synthetic origin (measured):** every failure is stamped at hour "
        + " or ".join(f"{h}:00 ({n})" for h, n in f["by_hour_of_day"].items())
        + f"; {f['with_maintenance_row']} of {f['rows']} failure rows have a maintenance row at "
        "the same machine, hour and component; the telemetry has no gaps and no missing values.",
        f"- **Errors:** {_f(e['rows'])} rows, "
        + ", ".join(f"{k}: {v}" for k, v in e["by_code"].items())
        + f". {_pct(e['events_with_error_in_prev_24h'])} of failure events have at least one "
        f"error on the same machine in the 24 h before ({_pct(e['events_with_error_in_prev_48h'])}"
        " in 48 h). Errors are not model inputs in the primary models (ADR 0027).",
        f"- **Maintenance records** start {a['maintenance']['first']}, before the telemetry.",
        "",
        "**Sensor ranges:**",
        "",
    ]
    rows = [
        [
            f"`{x['name']}`",
            _f(x["missing"]),
            _f(x["min"]),
            _f(x["p1"]),
            _f(x["median"]),
            _f(x["p99"]),
            _f(x["max"]),
        ]
        for x in a["sensors"]
    ]
    out += _table(["Sensor", "Missing", "Min", "p1", "Median", "p99", "Max"], rows)
    out += [""]
    return out


def _mapping_section(audits: dict[str, dict[str, Any]], real: RealConfig, root: Path) -> list[str]:
    out = [
        "## Mapping to the common window format",
        "",
        "Every length is in rows of the dataset's own grid. Decisions and limitations: "
        "`docs/experiments.md` § Phase 10; protocol: ADR 0027.",
        "",
    ]
    rows = []
    for name in DATASETS:
        study = real.datasets[name]  # type: ignore[index]
        cfg = load_real_dataset_config(root / study.dataset_config)
        h = cfg.step_s / 3600
        unit = "min" if cfg.step_s < 3600 else "h"
        scale = cfg.step_s / 60 if unit == "min" else h

        def fmt(n: int, scale: float = scale, unit: str = unit) -> str:
            v = n * scale
            if unit == "min" and v % 60 == 0:
                return f"{v / 60:g} h"
            return f"{v:g} {unit}"

        rows.append(
            [
                name,
                f"{cfg.step_s} s",
                fmt(cfg.window),
                ", ".join(fmt(x) for x in cfg.horizons)
                if study.supervised
                else "none (no supervised model)",
                fmt(cfg.labels.warning_band),
                ", ".join(fmt(x) for x in study.lookbacks)
                + f" (primary {fmt(study.primary_lookback)})",
                fmt(study.sustain),
                " / ".join(
                    f"{a} to {b}" for a, b in (cfg.split.train, cfg.split.val, cfg.split.test)
                ),
            ]
        )
    out += _table(
        [
            "Dataset",
            "Row",
            "Window",
            "Horizons",
            "WARNING band",
            "Lookbacks",
            "Sustain",
            "Train / val / test",
        ],
        rows,
    )
    out += [
        "",
        "**Events per split** (failure start inside the split; evaluable = the primary "
        "lookback lies inside the split and at least "
        f"{_pct(real.datasets['metropt3'].evaluable_min_coverage, 0)} of it has data):",
        "",
    ]
    rows = []
    for name, entry in audits.items():
        for split, v in entry["audit"]["mapping"]["splits"].items():
            rows.append(
                [
                    name,
                    split,
                    _f(v["machines"]),
                    _f(v["series"]),
                    _f(v["rows"]),
                    _f(v["events"]),
                    _f(v["evaluable_primary"]),
                    ", ".join(f"{k}: {n}" for k, n in v["by_fault_type"].items()) or NA,
                ]
            )
    out += _table(
        ["Dataset", "Split", "Machines", "Series", "Rows", "Events", "Evaluable", "By type"], rows
    )
    out += [""]
    return out


def render(report: dict[str, Any], real: RealConfig, root: Path) -> str:
    audits = report["audits"]
    lines = [
        "# Phase 10: validation on real public data",
        "",
        f"Generated by `uv run poe report-real` on {report['generated_at']} from the MLflow "
        f"experiment `{report['experiment']}` (runs: "
        + ", ".join(f"{k} audit `{v['run_id'][:8]}`" for k, v in audits.items())
        + "). Nothing here is typed by hand. Protocol and pre-registered outcomes: ADR 0027.",
        "",
    ]
    results = report.get("results") or {}
    if not any(results.values()):
        lines += [
            "> **Status:** data audit and mapping only. No study run on real data was found.",
            "",
        ]
    else:
        lines += [
            "> **Primary vs exploratory.** Only the outcomes pre-registered in ADR 0027 are "
            "primary (marked as such below). Everything else is exploratory and never a "
            "conclusion on its own.",
            "",
        ]
    lines += ["## Data audit", ""]
    if "metropt3" in audits:
        lines += _metro_section(audits["metropt3"]["audit"], real)
    if "azure_pdm" in audits:
        lines += _azure_section(audits["azure_pdm"]["audit"])
    lines += _mapping_section(audits, real, root)
    if any(results.values()):
        from twin_ml.experiments.report_real_results import render as render_results

        lines += render_results(results, real)
    return LF.join(lines) + LF


# ---------------------------------------------------------------------------- plots


def plot_availability(a: dict[str, Any], real: RealConfig, root: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    def x(t: datetime) -> float:
        return float(mdates.date2num(t))  # type: ignore[no-untyped-call]

    cov = a["daily_coverage"]
    day0 = datetime.fromisoformat(cov["first_day"])
    days = [day0 + timedelta(days=i) for i in range(len(cov["share"]))]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar([x(d) for d in days], [100 * v for v in cov["share"]], width=1.0, color=ACCENT, lw=0)
    ax.xaxis_date()
    for f in a["failures"]:
        s = datetime.fromisoformat(f["start"])
        e = datetime.fromisoformat(f["end"])
        ax.axvspan(x(s), x(max(e, s + timedelta(hours=6))), color=ACCENT2, alpha=0.35, lw=0)
        ax.text(x(s), 104, f"#{f['id']}", color=INK2, fontsize=8, ha="left")
    cfg = load_real_dataset_config(root / real.datasets["metropt3"].dataset_config)
    for name in ("val", "test"):
        b = datetime.fromisoformat(cfg.split.of(name)[0].isoformat())
        ax.axvline(x(b), color=INK, linewidth=1.0)
        ax.text(x(b), 112, f" {name} →", color=INK, fontsize=8, ha="left")
    ax.set_ylim(0, 118)
    ax.set_ylabel("rows with data (%)", color=INK2)
    ax.set_title(
        f"MetroPT-3: daily data coverage, {a['failure_count']} reported failures (shaded)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _result_plots(
    collected: dict[str, Any], results: dict[str, Any], real: RealConfig, root: Path, out: Path
) -> list[Path]:
    from twin_ml.experiments import report_real_results as rr

    paths: list[Path] = []
    m, a = results.get("metropt3"), results.get("azure")
    if m:
        from twin_ml.data.dataset import Dataset
        from twin_ml.datasets.build import resolve_real_dataset

        meta = Dataset.open(
            resolve_real_dataset(real.datasets["metropt3"].dataset_config, root)
        ).meta
        p = out / f"{REPORT_NAME}_metropt3_timelines.png"
        if rr.plot_metropt_timelines(collected, m, meta["events"]["test"], real, p):
            paths.append(p)
        p = out / f"{REPORT_NAME}_metropt3_fa_month.png"
        if rr.plot_metropt_months(m, p):
            paths.append(p)
    if a:
        for name, fn in (
            ("azure_primary", rr.plot_azure_primary),
            ("azure_robustness", rr.plot_azure_robustness),
            ("azure_reliability", rr.plot_azure_reliability),
        ):
            p = out / f"{REPORT_NAME}_{name}.png"
            if fn(a, p):
                paths.append(p)
    if results.get("explain"):
        from twin_ml.experiments.real_explain import plot_explain

        paths += plot_explain(results["explain"], out, REPORT_NAME)
    return paths


# ---------------------------------------------------------------------------- main


def build_report(
    real: RealConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
    out_dir: Path | None = None,
    experiment: str | None = None,
) -> list[Path]:
    from twin_ml.config import load_eval_config

    eval_cfg = load_eval_config(root / real.eval_config)
    tdir = tracking_dir or (root / eval_cfg.tracking_dir)
    audits = collect_audits(real, tdir, root, experiment)
    if not audits:
        raise LookupError(
            f"no audit runs for the current data and configs in {experiment or real.experiment}; "
            "run uv run poe real-audit"
        )
    from twin_ml.config import load_robustness_config
    from twin_ml.experiments import report_real_results as rr

    exp_name = experiment or real.experiment
    rcfg = load_robustness_config(root / real.robustness_config)
    collected = rr.collect(real, rcfg, eval_cfg, tdir, root, exp_name)
    results = rr.compute(collected, real)
    out = out_dir or (root / eval_cfg.report.out_dir)
    if any(results.values()):
        results["simulator"] = rr.compute_sim_comparison(results, out)
    report = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "experiment": exp_name,
        "audits": audits,
        "results": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "metropt3" in audits:
        p = out / f"{REPORT_NAME}_metropt3_availability.png"
        plot_availability(audits["metropt3"]["audit"], real, root, p)
        paths.append(p)
    paths += _result_plots(collected, results, real, root, out)
    md = out / f"{REPORT_NAME}.md"
    md.write_text(render(report, real, root), encoding="utf-8", newline=LF)
    js = out / f"{REPORT_NAME}.json"
    js.write_text(
        json.dumps(_clean(_round(report)), indent=1, sort_keys=True) + LF,
        encoding="utf-8",
        newline=LF,
    )
    return [md, js, *paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin_ml.experiments.report_real")
    parser.add_argument("--real-config", type=Path, default=Path("config/models/real.yaml"))
    args = parser.parse_args(argv)
    configure_logging()
    try:
        paths = build_report(load_real_config(args.real_config))
    except (ConfigError, LookupError, MissingDatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
