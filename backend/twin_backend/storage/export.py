"""Export a time range of the event store to CSV or JSONL.

The readings file uses the canonical Reading wire format (camelCase keys; missing sensor values
are JSON ``null`` or an empty CSV cell; the quality flag is kept), so the Phase 2
``FileReplayAdapter`` replays it with the default field mapping (``format: csv`` or ``jsonl``).
Rows are ordered by (ts, line machine order, seq), the order every adapter reproduces.

With ``all_streams`` every other stream is written next to it as ``<stem>.<stream>.jsonl``
(contract JSON per line; quarantine records as stored).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from twin_backend.storage.base import EventStore, Stored, Stream
from twin_common.contracts import Reading

ExportFormat = Literal["csv", "jsonl"]

# CSV columns: every Reading wire key except schemaVersion (defaulted on replay).
CSV_COLUMNS: tuple[str, ...] = (
    "machineId",
    "seq",
    "ts",
    "state",
    "speed",
    "temp",
    "vib",
    "load",
    "current",
    "runTime",
    "quality",
)


@dataclass
class ExportResult:
    readings_path: Path
    rows: dict[Stream, int] = field(default_factory=dict)
    paths: dict[Stream, Path] = field(default_factory=dict)


def _ordered(rows: list[Stored[Any]], machine_order: Sequence[str] | None) -> list[Reading]:
    rank = {m: i for i, m in enumerate(machine_order or ())}
    readings: list[Reading] = [r.item for r in rows]
    readings.sort(key=lambda r: (r.ts, rank.get(r.machine_id, len(rank)), r.machine_id, r.seq))
    return readings


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)  # shortest exact round trip
    return str(value)


def readings_to_csv(readings: Sequence[Reading]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for r in readings:
        wire = r.model_dump(mode="json")
        writer.writerow([_csv_value(wire[c]) for c in CSV_COLUMNS])
    return buf.getvalue()


def readings_to_jsonl(readings: Sequence[Reading]) -> str:
    return "".join(r.model_dump_json() + "\n" for r in readings)


async def export_range(
    store: EventStore,
    t0: datetime | None,
    t1: datetime | None,
    out: Path,
    fmt: ExportFormat,
    *,
    machine_order: Sequence[str] | None = None,
    all_streams: bool = False,
) -> ExportResult:
    """Write readings with ``t0 <= ts < t1`` to ``out`` (and other streams if requested)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    readings = _ordered(await store.fetch(Stream.READINGS, t0, t1), machine_order)
    text = readings_to_csv(readings) if fmt == "csv" else readings_to_jsonl(readings)
    out.write_text(text, encoding="utf-8", newline="")
    result = ExportResult(out, {Stream.READINGS: len(readings)}, {Stream.READINGS: out})
    if all_streams:
        for stream in Stream:
            if stream is Stream.READINGS:
                continue
            rows = await store.fetch(stream, t0, t1)
            path = out.with_name(f"{out.stem}.{stream.value}.jsonl")
            path.write_text(
                "".join(r.item.model_dump_json() + "\n" for r in rows),
                encoding="utf-8",
                newline="",
            )
            result.rows[stream] = len(rows)
            result.paths[stream] = path
    return result
