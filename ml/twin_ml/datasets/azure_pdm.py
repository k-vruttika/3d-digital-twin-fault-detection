"""Azure PdM loader and audit (Phase 10, ADR 0027). No simulator import, no ML library.

Hourly telemetry of 100 machines over 2015, with component failures, error codes, maintenance
records and machine models. A public benchmark whose origin is widely understood to be synthetic
(Microsoft tutorial); described as such everywhere. Machine kind = the machine's ``model``.
"""

from __future__ import annotations

from collections import Counter
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
    require_files,
    resample_mean,
)
from twin_ml.datasets.config import DatasetFacts


@dataclass(frozen=True)
class RawAzure:
    machine: npt.NDArray[np.int64]  # telemetry rows, sorted by (machine, time)
    t: npt.NDArray[np.int64]  # seconds since 1970 (naive clock)
    values: npt.NDArray[np.float64]  # (N, sensors)
    kinds: dict[int, str]  # machine -> model
    ages: dict[int, int]
    fail_machine: npt.NDArray[np.int64]
    fail_t: npt.NDArray[np.int64]
    fail_comp: list[str]
    err_machine: npt.NDArray[np.int64]
    err_t: npt.NDArray[np.int64]
    err_code: list[str]
    maint_machine: npt.NDArray[np.int64]
    maint_t: npt.NDArray[np.int64]
    maint_comp: list[str]
    sha256: str


def _read(path: Path, ts: str, fmt: str, cols: dict[str, Any]) -> Any:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    types = {ts: pa.timestamp("s")} | cols if ts else cols
    return pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            include_columns=list(types), column_types=types, timestamp_parsers=[fmt]
        ),
    )


def _secs(col: Any) -> npt.NDArray[np.int64]:
    out: npt.NDArray[np.int64] = (col.to_numpy().astype("datetime64[s]") - EPOCH).astype(np.int64)
    return out


def read_raw(facts: DatasetFacts, root: Path = Path(), *, sha: bool = True) -> RawAzure:
    import pyarrow as pa

    paths = require_files(facts, root)
    ts, fmt, mc = facts.timestamp_column, facts.timestamp_format, facts.machine_column
    assert mc is not None
    assert facts.kind_column is not None
    tel = _read(
        paths["telemetry"],
        ts,
        fmt,
        {mc: pa.int64()} | {s.column: pa.float64() for s in facts.sensors},
    )
    machine = tel.column(mc).to_numpy()
    t = _secs(tel.column(ts))
    values = np.stack([tel.column(s.column).to_numpy() for s in facts.sensors], axis=1)
    order = np.lexsort((t, machine))
    machine, t, values = machine[order], t[order], values[order]
    mach = _read(
        paths["machines"],
        "",
        fmt,
        {mc: pa.int64(), facts.kind_column: pa.string(), "age": pa.int64()},
    )
    kinds = dict(
        zip(mach.column(mc).to_pylist(), mach.column(facts.kind_column).to_pylist(), strict=True)
    )
    ages = dict(zip(mach.column(mc).to_pylist(), mach.column("age").to_pylist(), strict=True))
    fail = _read(paths["failures"], ts, fmt, {mc: pa.int64(), "failure": pa.string()})
    err = _read(paths["errors"], ts, fmt, {mc: pa.int64(), "errorID": pa.string()})
    mnt = _read(paths["maint"], ts, fmt, {mc: pa.int64(), "comp": pa.string()})
    files = [paths[k] for k in sorted(paths)]
    return RawAzure(
        machine=machine.astype(np.int64),
        t=t,
        values=values.astype(np.float64),
        kinds={int(k): str(v) for k, v in kinds.items()},
        ages={int(k): int(v) for k, v in ages.items()},
        fail_machine=fail.column(mc).to_numpy().astype(np.int64),
        fail_t=_secs(fail.column(ts)),
        fail_comp=[str(x) for x in fail.column("failure").to_pylist()],
        err_machine=err.column(mc).to_numpy().astype(np.int64),
        err_t=_secs(err.column(ts)),
        err_code=[str(x) for x in err.column("errorID").to_pylist()],
        maint_machine=mnt.column(mc).to_numpy().astype(np.int64),
        maint_t=_secs(mnt.column(ts)),
        maint_comp=[str(x) for x in mnt.column("comp").to_pylist()],
        sha256=files_sha256(files) if sha else "",
    )


def machine_id(m: int) -> str:
    return f"M{m:03d}"


def origin(raw: RawAzure, step_s: int) -> int:
    return int(raw.t.min() // step_s * step_s)


def grids(
    facts: DatasetFacts, raw: RawAzure, step_s: int, *, error_channels: bool = False
) -> tuple[int, list[Grid]]:
    """(origin, one grid per machine); with ``error_channels`` the error counts per row (one
    channel per code) are appended to the telemetry channels."""
    t0 = origin(raw, step_s)
    out: list[Grid] = []
    bounds = np.flatnonzero(np.diff(raw.machine)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, raw.machine.size], strict=True):
        m = int(raw.machine[lo])
        bins, means, _ = resample_mean(raw.t[lo:hi], raw.values[lo:hi], step_s, t0)
        first = int(bins[0])
        n = int(bins[-1]) - first + 1
        values, observed = dense(bins - first, means, n)
        if error_channels:
            counts = np.zeros((n, len(facts.error_codes)), dtype=np.float32)
            sel = raw.err_machine == m
            k = (raw.err_t[sel] - t0) // step_s - first
            codes = [raw.err_code[i] for i in np.flatnonzero(sel)]
            for kk, code in zip(k, codes, strict=True):
                if 0 <= kk < n:
                    counts[kk, facts.error_codes.index(code)] += 1.0
            values = np.concatenate([values, np.where(observed[:, None], counts, np.nan)], axis=1)
        out.append(
            Grid(
                machine_id=machine_id(m),
                kind=raw.kinds[m],
                t_s=(first + np.arange(n, dtype=np.float64)) * step_s,
                values=values.astype(np.float32),
                observed=observed,
            )
        )
    return t0, out


def events(raw: RawAzure, t0: int) -> list[Event]:
    """One event per (machine, time); several failed components at once -> ``multi``."""
    groups: dict[tuple[int, int], list[str]] = {}
    for m, t, c in zip(raw.fail_machine, raw.fail_t, raw.fail_comp, strict=True):
        groups.setdefault((int(m), int(t)), []).append(c)
    out = []
    for i, ((m, t), comps) in enumerate(sorted(groups.items())):
        cs = tuple(sorted(set(comps)))
        out.append(
            Event(
                id=f"{machine_id(m)}-{i}",
                machine_id=machine_id(m),
                start_s=float(t - t0),
                end_s=float(t - t0),
                fault_type=cs[0] if len(cs) == 1 else "multi",
                components=cs,
            )
        )
    return out


# ---------------------------------------------------------------------- audit


def _iso(seconds: int | float) -> str:
    return str(EPOCH + np.timedelta64(int(seconds), "s")).replace("T", " ")


def audit(facts: DatasetFacts, raw: RawAzure, *, step_s: int) -> dict[str, Any]:
    t, m = raw.t, raw.machine
    same = np.diff(m) == 0
    d = np.diff(t)[same]
    u, c = np.unique(d, return_counts=True)
    dup = int((d == 0).sum())
    gaps = d[d > step_s]
    sensors = []
    for j, s in enumerate(facts.sensors):
        x = raw.values[:, j]
        fin = x[np.isfinite(x)]
        sensors.append(
            {
                "name": s.name,
                "missing": int((~np.isfinite(x)).sum()),
                "min": float(fin.min()),
                "p1": float(np.percentile(fin, 1)),
                "median": float(np.median(fin)),
                "p99": float(np.percentile(fin, 99)),
                "max": float(fin.max()),
            }
        )
    evs = events(raw, 0)
    hours = Counter(int(x) % 86400 // 3600 for x in raw.fail_t)
    months = Counter(_iso(x)[:7] for x in raw.fail_t)
    comps = Counter(raw.fail_comp)
    per_event_comps = Counter(len(e.components) for e in evs)
    # errors before failures: share of events with >= 1 error on the same machine in the hours
    # before (window excludes the failure hour itself)
    err_by_m: dict[int, npt.NDArray[np.int64]] = {}
    for mm in np.unique(raw.err_machine):
        err_by_m[int(mm)] = np.sort(raw.err_t[raw.err_machine == mm])

    def preceded(h: int) -> float:
        hit = 0
        for e in evs:
            mm = int(e.machine_id[1:])
            ts = err_by_m.get(mm, np.zeros(0, np.int64))
            a, b = e.start_s - h * 3600, e.start_s
            hit += bool(((ts >= a) & (ts < b)).any())
        return hit / len(evs) if evs else float("nan")

    maint = set(zip(raw.maint_machine.tolist(), raw.maint_t.tolist(), raw.maint_comp, strict=True))
    fail_rows = list(
        zip(raw.fail_machine.tolist(), raw.fail_t.tolist(), raw.fail_comp, strict=True)
    )
    return {
        "dataset": facts.name,
        "origin": facts.origin,
        "sha256": raw.sha256,
        "telemetry": {
            "rows": int(t.size),
            "machines": int(np.unique(m).size),
            "first": _iso(t.min()),
            "last": _iso(t.max()),
            "span_days": float((t.max() - t.min()) / 86400),
            "intervals": [
                {"s": int(uu), "count": int(cc)} for uu, cc in zip(u[:6], c[:6], strict=True)
            ],
            "duplicates": dup,
            "gaps": int(gaps.size),
            "gap_total_h": float(gaps.sum() / 3600) if gaps.size else 0.0,
            "missing_values": int((~np.isfinite(raw.values)).sum()),
        },
        "sensors": sensors,
        "machines": {
            "count": len(raw.kinds),
            "by_model": dict(sorted(Counter(raw.kinds.values()).items())),
            "age_min": min(raw.ages.values()),
            "age_max": max(raw.ages.values()),
        },
        "failures": {
            "rows": len(raw.fail_comp),
            "events": len(evs),
            "by_component": dict(sorted(comps.items())),
            "components_per_event": {str(k): v for k, v in sorted(per_event_comps.items())},
            "machines_with_failure": int(np.unique(raw.fail_machine).size),
            "by_hour_of_day": {str(k): v for k, v in sorted(hours.items())},
            "by_month": dict(sorted(months.items())),
            "first": _iso(raw.fail_t.min()),
            "last": _iso(raw.fail_t.max()),
            "with_maintenance_row": sum(r in maint for r in fail_rows),
        },
        "errors": {
            "rows": len(raw.err_code),
            "by_code": dict(sorted(Counter(raw.err_code).items())),
            "events_with_error_in_prev_24h": preceded(24),
            "events_with_error_in_prev_48h": preceded(48),
        },
        "maintenance": {
            "rows": len(raw.maint_comp),
            "first": _iso(raw.maint_t.min()),
            "last": _iso(raw.maint_t.max()),
            "by_component": dict(sorted(Counter(raw.maint_comp).items())),
        },
        "failure_source": facts.failure_source,
        "label_meaning": (
            "A failure row is the hour a component was recorded as failed (and replaced); "
            "failures have no duration. Several rows at the same machine and hour form one event."
        ),
    }
