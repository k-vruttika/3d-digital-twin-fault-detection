"""The drift report (Phase 11 Part C, ADR 0031): ``python -m twin_ml.serve drift``.

Everything here is computed by running code on the simulator's E1 dataset, with the same PSI,
decision rule and ADWIN as the live monitor (no numbers are typed by hand):

1. **False drift on clean runs.** The reference and the PSI alert levels are built from the
   training split and calibrated on the validation split (``drift_reference``, exactly as the
   bundle). For a grid of margins (``threshold_margin``) the decision rule is run over the
   operating rows of every clean validation and test machine run; drift episodes per
   machine-hour of operating time are counted.
2. **The live margin** is chosen on the *validation* runs only: the smallest margin whose false
   episodes per hour are at most ``max_false_episodes_per_hour``. The test numbers are
   reported at every margin and never used for the choice.
3. **Detection of synthetic input shifts.** The operating-condition inputs of the clean test
   runs are shifted from the middle of each run (load plus a few percentage points, speed by
   a fraction of the mean training speed). This is a *synthetic* shift of recorded rows, not a
   physical simulation: it measures the detector, not how the simulator would behave under
   such a condition. Detection rate and delay (operating rows after the shift) are reported,
   and, through the full monitor, how well ADWIN dates the change.

Reference and thresholds come from the deploy config and the inference config; the report
reads no deployed bundle, so it can run before ``deploy-models``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.serve.drift import (
    Decider,
    DriftSettings,
    MachineDrift,
    operating_series,
    window_psi_array,
)

Series = list[tuple[npt.NDArray[np.float64], str]]
MARGIN_PRECISION = 3  # digits shown / compared for margins (a label, not a threshold)


def add_args(sub: Any) -> None:
    p = sub.add_parser("drift", help="drift detection report (false drift on clean runs, shifts)")
    p.add_argument("--deploy-config", type=Path, default=Path("config/models/deploy.yaml"))
    p.add_argument("--inference-config", type=Path, default=Path("config/inference.yaml"))
    p.add_argument("--out", type=Path, default=Path("experiments/results/drift"))


def settings_for(margin: float, window_s: int, inference: Any) -> DriftSettings:
    d = inference.drift
    return DriftSettings(
        window_s=window_s,
        margin=margin,
        clear_ratio=d.clear_ratio,
        sustain_s=d.sustain_s,
        clear_s=d.clear_s,
        adwin_delta=d.adwin_delta,
    )


def psi_arrays(
    raw: npt.NDArray[np.float64], ref_kind: dict[str, Any], sensors: list[str], window: int
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        ch: window_psi_array(raw[:, sensors.index(ch)], r, window) for ch, r in ref_kind.items()
    }


def run_decider(
    psis: dict[str, npt.NDArray[np.float64]], ref_kind: dict[str, Any], cfg: DriftSettings
) -> tuple[list[int], int]:
    """Episode start rows (index into the PSI arrays) and the number of rows in drift."""
    alert = {ch: float(r["psi_threshold"]) * cfg.margin for ch, r in ref_kind.items()}
    clear = {ch: a * cfg.clear_ratio for ch, a in alert.items()}
    decider = Decider(alert, clear, cfg)
    starts: list[int] = []
    drifted_rows = 0
    n = min(len(v) for v in psis.values())
    for i in range(n):
        if decider.step({ch: float(v[i]) for ch, v in psis.items()}):
            starts.append(i)
        drifted_rows += decider.drifted
    return starts, drifted_rows


def clean_rates(
    series: Series, ref: dict[str, Any], sensors: list[str], cfg: DriftSettings
) -> dict[str, Any]:
    episodes = rows = drifted = 0
    runs_with = 0
    for raw, kind in series:
        if kind not in ref:
            continue
        psis = psi_arrays(raw, ref[kind], sensors, cfg.window_s)
        n = min(len(v) for v in psis.values())
        if n == 0:
            continue
        starts, d_rows = run_decider(psis, ref[kind], cfg)
        episodes += len(starts)
        rows += n
        drifted += d_rows
        runs_with += bool(starts)
    hours = rows / 3600.0
    return {
        "runs": len(series),
        "operating_hours": hours,
        "episodes": episodes,
        "episodes_per_hour": episodes / hours if hours else None,
        "runs_with_drift": runs_with,
        "drifted_row_fraction": drifted / rows if rows else None,
    }


def shifted(
    raw: npt.NDArray[np.float64],
    sensors: list[str],
    channel: str,
    delta: float,
    at: int,
) -> npt.NDArray[np.float64]:
    out = raw.copy()
    out[at:, sensors.index(channel)] += delta
    return out


def detection(
    series: Series,
    ref: dict[str, Any],
    sensors: list[str],
    cfg: DriftSettings,
    channel: str,
    delta_of: Any,
    frac: float,
) -> dict[str, Any]:
    """Detection of one shift over every test series long enough to evaluate it."""
    delays: list[int] = []
    n_series = 0
    outside = detected_outside = 0
    for raw, kind in series:
        at = int(raw.shape[0] * frac)
        if kind not in ref or channel not in ref[kind] or raw.shape[0] - at < cfg.window_s:
            continue
        n_series += 1
        x = shifted(raw, sensors, channel, delta_of(ref[kind][channel]), at)
        # does the shifted condition leave the training support? (a shift inside it is, by
        # design, not drift: the models were trained on those conditions)
        edges = ref[kind][channel]["edges"]
        med = float(np.median(x[at:, sensors.index(channel)]))
        is_outside = med < edges[0] or med > edges[-1]
        outside += is_outside
        psis = psi_arrays(x, ref[kind], sensors, cfg.window_s)
        starts, _ = run_decider(psis, ref[kind], cfg)
        # PSI index i ends at operating row i + window - 1
        after = [s + cfg.window_s - 1 - at for s in starts if s + cfg.window_s - 1 >= at]
        if after:
            delays.append(after[0])
            detected_outside += is_outside
    return {
        "series": n_series,
        "detected": len(delays),
        "detection_rate": len(delays) / n_series if n_series else None,
        "outside_support": outside,
        "detected_outside_support": detected_outside,
        "detection_rate_outside_support": detected_outside / outside if outside else None,
        "delay_median_s": float(np.median(delays)) if delays else None,
        "delay_p90_s": float(np.percentile(delays, 90)) if delays else None,
    }


def dating(
    series: Series,
    ref: dict[str, Any],
    sensors: list[str],
    cfg: DriftSettings,
    channel: str,
    delta: float,
    frac: float,
    limit: int,
) -> dict[str, Any]:
    """Through the full monitor (PSI + ADWIN): how close is ``since`` to the shift?"""
    errors: list[float] = []
    n = 0
    for raw, kind in series:
        at = int(raw.shape[0] * frac)
        if kind not in ref or channel not in ref[kind] or raw.shape[0] - at < cfg.window_s * 2:
            continue
        if n >= limit:
            break
        n += 1
        md = MachineDrift(kind, ref, cfg)
        x = shifted(raw, sensors, channel, delta, at)
        since = None
        for i in range(x.shape[0]):
            report = md.update({c: float(x[i, sensors.index(c)]) for c in md.channels}, float(i))
            if report["drifted"]:
                since = report["since"]
                break
        if since is not None:
            errors.append(float(since) - at)
    return {
        "series": n,
        "dated": len(errors),
        "since_minus_shift_median_s": float(np.median(errors)) if errors else None,
        "since_minus_shift_p90_s": float(np.percentile(errors, 90)) if errors else None,
    }


def evaluate(deploy: Any, inference: Any, root: Path) -> dict[str, Any]:
    from twin_ml.config import load_dataset_config
    from twin_ml.data.dataset import Dataset, resolve_dataset_dir
    from twin_ml.serve.bundle import drift_reference

    d = deploy.drift
    rep = d.report
    ds = Dataset.open(
        resolve_dataset_dir(load_dataset_config(root / deploy.anomaly.dataset_config), root)
    )
    train, val, test = ds.split("train"), ds.split("val"), ds.split("test")
    ref = drift_reference(deploy, train, val)
    sensors = list(train.schema.sensors)
    val_series = operating_series(val, d.states)
    test_series = operating_series(test, d.states)

    grid: list[dict[str, Any]] = []
    for margin in rep.margins:
        cfg = settings_for(margin, d.window_s, inference)
        grid.append(
            {
                "margin": margin,
                "validation": clean_rates(val_series, ref, sensors, cfg),
                "test": clean_rates(test_series, ref, sensors, cfg),
            }
        )
    ok = [
        g
        for g in grid
        if (g["validation"]["episodes_per_hour"] or 0.0) <= rep.max_false_episodes_per_hour
    ]
    chosen = ok[0] if ok else None
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_key": ds.meta["dataset_key"],
        "settings": {
            "channels": d.channels,
            "bins": d.bins,
            "support_quantile": d.support_quantile,
            "window_s": d.window_s,
            "alert_quantile": d.alert_quantile,
            "clear_ratio": inference.drift.clear_ratio,
            "sustain_s": inference.drift.sustain_s,
            "clear_s": inference.drift.clear_s,
            "adwin_delta": inference.drift.adwin_delta,
            "live_margin": inference.drift.threshold_margin,
        },
        "reference": {
            kind: {
                ch: {
                    "training_rows": r["n"],
                    "support": [r["edges"][0], r["edges"][-1]],
                    "psi_alert_level": r["psi_threshold"],
                    "calibration_windows": r["calibration_windows"],
                }
                for ch, r in per.items()
            }
            for kind, per in ref.items()
        },
        "false_drift": grid,
        "rule": {
            "max_false_episodes_per_hour": rep.max_false_episodes_per_hour,
            "chosen_margin": chosen["margin"] if chosen else None,
        },
        "shifts": [],
        "dating": [],
    }
    if chosen is None:
        return result
    cfg = settings_for(chosen["margin"], d.window_s, inference)
    frac = rep.shift_at_fraction
    for channel, deltas in (("load", rep.load_shifts), ("speed", rep.speed_shifts)):
        for delta in deltas:
            if channel == "load":

                def delta_of(r: dict[str, Any], v: float = delta) -> float:
                    return v

            else:

                def delta_of(r: dict[str, Any], v: float = delta) -> float:
                    return v * float(r["mean"])

            res = detection(test_series, ref, sensors, cfg, channel, delta_of, frac)
            result["shifts"].append(
                {
                    "channel": channel,
                    "shift": delta,
                    "unit": "pp" if channel == "load" else "x mean",
                    **res,
                }
            )
    big = max(rep.load_shifts)
    result["dating"].append(
        {
            "channel": "load",
            "shift": big,
            **dating(test_series, ref, sensors, cfg, "load", big, frac, rep.dating_series),
        }
    )
    return result


def _fmt(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def write_report(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    s = result["settings"]
    lines = [
        "# Drift detection on the operating-condition inputs",
        "",
        f"Generated by `uv run poe drift-report` ({result['generated_at']}, dataset "
        f"`{result['dataset_key']}`). Every number comes from running the live detector code on "
        "the simulator's clean runs; see ADR 0031.",
        "",
        "## What is measured",
        "",
        f"- **Inputs:** {', '.join(s['channels'])} (operating-condition inputs; health signals are "
        "not monitored for drift because a developing fault changes them by design).",
        f"- **PSI** of the last {s['window_s']} operating rows against the training distribution "
        f"of the machine's kind ({s['bins']} quantile bins of the training rows inside the "
        f"{s['support_quantile']:.1%} to {1 - s['support_quantile']:.1%} support, plus an overflow "
        "bin on each side; additively smoothed counts).",
        f"- **Alert level per kind and channel:** the {s['alert_quantile']:.1%} quantile of the "
        "PSI "
        "of clean *validation* windows, times the live margin. A machine sits in one load band "
        "for long stretches, so the textbook 0.25 would flag clean runs.",
        f"- **Decision:** PSI at or above the alert level for {s['sustain_s']:g} s raises drift; "
        f"it clears below {s['clear_ratio']:g} x the alert level for {s['clear_s']:g} s. ADWIN "
        f"(delta {s['adwin_delta']:g}) only dates the change.",
        "",
        "## Reference and alert levels (before the margin)",
        "",
        "| Kind | Channel | Training rows | Support | PSI alert level | Validation windows |",
        "|---|---|---|---|---|---|",
    ]
    for kind, per in result["reference"].items():
        for ch, r in per.items():
            lines.append(
                f"| {kind} | {ch} | {r['training_rows']} | {r['support'][0]:.1f} … "
                f"{r['support'][1]:.1f} | {r['psi_alert_level']:.2f} | {r['calibration_windows']} |"
            )
    chosen = result["rule"]["chosen_margin"]
    lines += [
        "",
        "## False drift on clean runs",
        "",
        "Episodes of drift per machine-hour of operating time on runs with no condition shift "
        "(every episode is a false alarm). The live margin is chosen on the validation runs only "
        f"(the smallest margin with at most {result['rule']['max_false_episodes_per_hour']:g} "
        "episodes per hour); the test columns are not used for the choice.",
        "",
        "| Margin | Val episodes / h | Val runs with drift | Test episodes / h | "
        "Test runs with drift | Test rows in drift |",
        "|---|---|---|---|---|---|",
    ]
    for g in result["false_drift"]:
        v, t = g["validation"], g["test"]
        mark = " (chosen)" if g["margin"] == chosen else ""
        lines.append(
            f"| {g['margin']:g}{mark} | {_fmt(v['episodes_per_hour'], 3)} | "
            f"{v['runs_with_drift']} / {v['runs']} | {_fmt(t['episodes_per_hour'], 3)} | "
            f"{t['runs_with_drift']} / {t['runs']} | {_fmt(t['drifted_row_fraction'], 4)} |"
        )
    lines.append("")
    if chosen is None:
        lines += [
            "**No margin on the grid met the rule on the validation runs; no shift evaluation "
            "was run.**",
            "",
        ]
    else:
        lines += [
            f"The live margin in `config/inference.yaml` is **{s['live_margin']:g}**; the rule "
            f"picks **{chosen:g}** (the config must be set to it by hand and committed).",
            "",
            "## Detection of synthetic input shifts (test runs, at the chosen margin)",
            "",
            "The operating-condition inputs of the clean test runs are shifted from the middle of "
            "each run. **This is a synthetic shift of recorded rows, not a physical simulation.** "
            "Delay is in operating rows (1 row = 1 s) after the shift starts; a shift that is "
            "never detected within the run counts as missed. A shift that moves the input to "
            "a value *inside* the range the models were trained on is by design not drift, so "
            "the table also gives the runs whose shifted input left the training support "
            "(median of the shifted rows outside the support) and the detection rate among "
            "those.",
            "",
            "| Input | Shift | Runs | Detected | Rate | Outside support | Detected there "
            "| Rate there | Delay median s | Delay p90 s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for sh in result["shifts"]:
            unit = "pp" if sh["channel"] == "load" else "x mean speed"
            lines.append(
                f"| {sh['channel']} | +{sh['shift']:g} {unit} | {sh['series']} | "
                f"{sh['detected']} | {_fmt(sh['detection_rate'], 3)} | {sh['outside_support']} | "
                f"{sh['detected_outside_support']} | "
                f"{_fmt(sh['detection_rate_outside_support'], 3)} | "
                f"{_fmt(sh['delay_median_s'], 0)} | {_fmt(sh['delay_p90_s'], 0)} |"
            )
        lines.append("")
        for dt in result["dating"]:
            lines += [
                "## ADWIN dating of the change",
                "",
                f"Through the full monitor on {dt['series']} test runs with a +{dt['shift']:g} pp "
                f"load shift: the reported start of the drift (`since`) minus the true shift time, "
                f"median {_fmt(dt['since_minus_shift_median_s'], 0)} s, p90 "
                f"{_fmt(dt['since_minus_shift_p90_s'], 0)} s over {dt['dated']} dated runs "
                "(negative = reported before the shift).",
                "",
            ]
    lines += [
        "## Limits",
        "",
        "- Drift is measured on speed and load only. It says the conditions differ from the "
        "training data, not that the predictions are wrong.",
        "- Windows of one run are strongly correlated, so the false-drift rate is a rate over a "
        "small number of independent runs; read it as an order of magnitude.",
        "- Simulator data only. The shifts are synthetic edits of recorded rows.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> int:
    from twin_common.config.inference import load_inference_config
    from twin_ml.config import load_deploy_config

    deploy = load_deploy_config(args.deploy_config)
    inference = load_inference_config(args.inference_config)
    result = evaluate(deploy, inference, Path())
    write_report(result, args.out)
    print(json.dumps({"chosen_margin": result["rule"]["chosen_margin"]}))
    return 0
