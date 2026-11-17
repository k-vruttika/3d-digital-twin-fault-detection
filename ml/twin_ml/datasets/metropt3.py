"""MetroPT-3 loader and audit (Phase 10, ADR 0027). No simulator import, no ML library.

The CSV holds ~10 s samples of 15 signals of one compressor (one machine, kind ``APU``). It has
no labels: the four air-leak failures come from the company reports in
``config/datasets/metropt3.yaml``. The loader resamples to the configured grid (1-min means),
keeps empty minutes as missing rows, and the build splits series at long gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.datasets.common import (
    EPOCH,
    Event,
    Grid,
    dense,
    files_sha256,
    gap_runs,
    require_files,
    resample_mean,
    segments,
    to_seconds,
)
from twin_ml.datasets.config import DatasetFacts, parse_time


@dataclass(frozen=True)
class RawMetro:
    t: npt.NDArray[np.int64]  # seconds since 1970 (naive clock), sorted
    values: npt.NDArray[np.float64]  # (N, sensors) in facts.sensors order
    sha256: str
    path: Path


def read_raw(facts: DatasetFacts, root: Path = Path(), *, sha: bool = True) -> RawMetro:
    """Read the CSV (pyarrow, imported lazily); columns as named in the facts file."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    path = require_files(facts, root)["telemetry"]
    cols = [facts.timestamp_column, *(s.column for s in facts.sensors)]
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            include_columns=cols,
            column_types={facts.timestamp_column: pa.timestamp("s")}
            | {s.column: pa.float64() for s in facts.sensors},
            timestamp_parsers=[facts.timestamp_format],
        ),
    )
    t = table.column(facts.timestamp_column).to_numpy().astype("datetime64[s]")
    values = np.stack(
        [table.column(s.column).to_numpy(zero_copy_only=False) for s in facts.sensors], axis=1
    ).astype(np.float64)
    secs = (t - EPOCH).astype(np.int64)
    order = np.argsort(secs, kind="stable")
    if np.any(order != np.arange(order.size)):
        secs, values = secs[order], values[order]
    return RawMetro(t=secs, values=values, sha256=files_sha256([path]) if sha else "", path=path)


def origin(raw: RawMetro, step_s: int) -> int:
    """Grid origin: the first timestamp floored to the step (seconds since 1970)."""
    return int(raw.t[0] // step_s * step_s)


def grid(facts: DatasetFacts, raw: RawMetro, step_s: int) -> tuple[int, Grid]:
    """(origin, the one machine on the regular grid of bin means)."""
    t0 = origin(raw, step_s)
    bins, means, _ = resample_mean(raw.t, raw.values, step_s, t0)
    n_bins = int(bins[-1]) + 1
    values, observed = dense(bins, means, n_bins)
    assert facts.kind is not None
    assert facts.machine_id is not None
    g = Grid(
        machine_id=facts.machine_id,
        kind=facts.kind,
        t_s=np.arange(n_bins, dtype=np.float64) * step_s,
        values=values,
        observed=observed,
    )
    return t0, g


def events(facts: DatasetFacts, t0: int) -> list[Event]:
    assert facts.machine_id is not None
    return [
        Event(
            id=str(f.id),
            machine_id=facts.machine_id,
            start_s=float(to_seconds(parse_time(f.start)) - t0),
            end_s=float(to_seconds(parse_time(f.end)) - t0),
            fault_type=f.type,
        )
        for f in facts.failures
    ]


# ---------------------------------------------------------------------- audit


def _iso(seconds: int | float) -> str:
    return str(EPOCH + np.timedelta64(int(seconds), "s")).replace("T", " ")


def coverage(g: Grid, start_s: float, end_s: float, step_s: int) -> float:
    """Share of grid rows in ``[start_s, end_s)`` that hold data (rows outside the grid count
    as empty)."""
    n = max(1, round((end_s - start_s) / step_s))
    sel = (g.t_s >= start_s) & (g.t_s < end_s)
    return float(g.observed[sel].sum() / n)


def audit(
    facts: DatasetFacts,
    raw: RawMetro,
    *,
    step_s: int,
    max_gap_rows: int,
    lookbacks_rows: list[int],
    gap_threshold_s: int = 60,
    pdf_rows: int | None = 15_169_480,
) -> dict[str, Any]:
    """Facts measured from the file: rows, span, sampling, gaps, ranges, missing values and,
    per failure, samples and data coverage of the failure interval and of each lookback."""
    t = raw.t
    d = np.diff(t)
    u, c = np.unique(d, return_counts=True)
    order = np.argsort(-c)
    gaps_at, gaps_len = gap_runs(t, gap_threshold_s)
    longest = np.argsort(-gaps_len)[:10]
    sensors = []
    for j, s in enumerate(facts.sensors):
        x = raw.values[:, j]
        fin = x[np.isfinite(x)]
        entry: dict[str, Any] = {
            "name": s.name,
            "column": s.column,
            "type": s.type,
            "unit": s.unit,
            "missing": int((~np.isfinite(x)).sum()),
            "min": float(fin.min()) if fin.size else None,
            "p1": float(np.percentile(fin, 1)) if fin.size else None,
            "median": float(np.median(fin)) if fin.size else None,
            "p99": float(np.percentile(fin, 99)) if fin.size else None,
            "max": float(fin.max()) if fin.size else None,
        }
        if s.type == "digital":
            vals = np.unique(fin)
            entry["distinct_values"] = [float(v) for v in vals[:10]]
            entry["share_active"] = float((fin > 0.5).mean()) if fin.size else None
        sensors.append(entry)

    t0, g = grid(facts, raw, step_s)
    segs = segments(g.observed, max_gap_rows)
    seg_len = np.array([b - a for a, b in segs])
    failures = []
    for e in events(facts, t0):
        a, b = e.start_s + t0, e.end_s + t0
        inside = int(((t >= a) & (t <= b)).sum())
        lb = {
            str(L): {
                "hours": L * step_s / 3600,
                "raw_samples": int(((t >= a - L * step_s) & (t < a)).sum()),
                "coverage": coverage(g, e.start_s - L * step_s, e.start_s, step_s),
            }
            for L in lookbacks_rows
        }
        failures.append(
            {
                "id": e.id,
                "start": _iso(a),
                "end": _iso(b),
                "duration_h": (b - a) / 3600,
                "type": e.fault_type,
                "raw_samples_inside": int(inside),
                "interval_coverage": coverage(g, e.start_s, e.end_s + step_s, step_s),
                "lookbacks": lb,
            }
        )
    per_day = 86400 // step_s
    n_days = -(-g.t_s.size // per_day)
    obs = np.zeros(n_days * per_day, dtype=bool)
    obs[: g.t_s.size] = g.observed
    daily = obs.reshape(n_days, per_day).mean(axis=1)
    return {
        "dataset": facts.name,
        "origin": facts.origin,
        "file": raw.path.name,
        "daily_coverage": {
            "first_day": _iso(t0)[:10],
            "share": [round(float(x), 4) for x in daily],
        },
        "sha256": raw.sha256,
        "rows": int(t.size),
        "first": _iso(t[0]),
        "last": _iso(t[-1]),
        "span_days": float((t[-1] - t[0]) / 86400),
        "duplicates": int((d == 0).sum()),
        "sorted": bool(np.all(d >= 0)),
        "sampling": {
            "median_s": float(np.median(d)),
            "min_s": int(d.min()),
            "max_s": int(d.max()),
            "top_intervals": [
                {"s": int(u[i]), "count": int(c[i]), "share": float(c[i] / d.size)}
                for i in order[:6]
            ],
            "pdf_rows_1hz": pdf_rows,
            "rows_ratio_pdf_to_file": (pdf_rows / t.size) if pdf_rows else None,
        },
        "gaps": {
            "threshold_s": gap_threshold_s,
            "count": int(gaps_len.size),
            "total_h": float(gaps_len.sum() / 3600),
            "longest": [
                {"start": _iso(gaps_at[i]), "hours": float(gaps_len[i] / 3600)} for i in longest
            ],
        },
        "sensors": sensors,
        "grid": {
            "step_s": step_s,
            "max_gap_rows": max_gap_rows,
            "rows": int(g.t_s.size),
            "observed_rows": int(g.observed.sum()),
            "series": len(segs),
            "series_rows_median": float(np.median(seg_len)) if seg_len.size else 0.0,
            "rows_in_series": int(seg_len.sum()),
        },
        "failures": failures,
        "failure_count": len(failures),
        "failure_source": facts.failure_source,
        "label_meaning": (
            "Failure intervals are company maintenance reports (air leaks), not sensor labels; "
            "the dataset itself is unlabeled."
        ),
        "pdf_check": None if facts.pdf_check is None else facts.pdf_check.model_dump(),
    }
