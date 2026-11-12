"""File replay adapter: replays the simulator's JSONL / CSV exports (or a JSON array).

Minimal mode depends on this adapter, so it uses the standard library only: no broker,
database, ML or network library is imported.

- **Order:** JSONL and JSON are replayed in file order. The simulator CSV is grouped by
  machine, so CSV rows are merged into (ts, line machine order, seq) order.
- **Pacing:** each record is due at ``start + (ts - ts0) / replay_speed`` on the injected clock.
- **start_offset_s:** records earlier than ``ts0 + start_offset_s`` are skipped.
- **loop:** every pass shifts ``ts`` by the recording span and each machine's ``seq`` by its
  maximum, so a looped replay looks like one continuous stream (dedup keeps it).

Payloads go out as raw records (dict, or the raw line when it is not JSON); field mapping and
validation happen in the ingest pipeline.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from twin_backend.adapters.base import AdapterError, BaseAdapter
from twin_backend.clock import Clock
from twin_common.config.line import LineConfig
from twin_common.config.sources import FileSourceConfig
from twin_common.enums import AdapterKind

MISSING_FILE_HINT = (
    "generate it with `uv run poe sim-quick` (full line) or `uv run poe sim-quick-single` "
    "(single station), or point file.path in config/sources.yaml at an existing export"
)


@dataclass(frozen=True, slots=True)
class _Record:
    payload: dict[str, Any] | str
    ts: datetime | None
    machine_id: str | None
    seq: int | None
    line_no: int


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class FileReplayAdapter(BaseAdapter):
    kind = AdapterKind.FILE

    def __init__(
        self,
        cfg: FileSourceConfig,
        line: LineConfig,
        *,
        root: Path,
        clock: Clock,
    ) -> None:
        super().__init__(clock)
        self.cfg = cfg
        self.line = line
        self.path = (root / cfg.path) if not Path(cfg.path).is_absolute() else Path(cfg.path)
        fields = cfg.mapping.fields
        self._ts_key = fields.get("ts", "ts")
        self._seq_key = fields.get("seq", "seq")
        self._machine_key = fields.get("machineId", "machineId")
        self._records: list[_Record] = []
        self.passes = 0

    @property
    def time_scale(self) -> float:
        """Event seconds per wall second: the configured replay speed."""
        return self.cfg.replay_speed

    async def _prepare(self) -> None:
        if not self.path.is_file():
            raise AdapterError(f"replay file not found: {self.path} - {MISSING_FILE_HINT}")
        self._records = self._load()
        if not self._records:
            raise AdapterError(f"replay file is empty: {self.path}")

    # -- loading -----------------------------------------------------------------------------
    def _record(self, payload: dict[str, Any] | str, line_no: int) -> _Record:
        if isinstance(payload, str):
            return _Record(payload, None, None, None, line_no)
        machine = payload.get(self._machine_key)
        return _Record(
            payload,
            _parse_ts(payload.get(self._ts_key)),
            machine if isinstance(machine, str) else None,
            _parse_int(payload.get(self._seq_key)),
            line_no,
        )

    def _load(self) -> list[_Record]:
        text = self.path.read_text(encoding="utf-8")
        records: list[_Record] = []
        if self.cfg.format == "jsonl":
            for i, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = line
                records.append(self._record(obj if isinstance(obj, dict) else line, i))
        elif self.cfg.format == "json":
            try:
                items = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"{self.path}: not valid JSON: {exc}") from exc
            if not isinstance(items, list):
                raise AdapterError(f"{self.path}: expected a JSON array of readings")
            for i, obj in enumerate(items, start=1):
                records.append(self._record(obj if isinstance(obj, dict) else json.dumps(obj), i))
        else:
            reader = csv.DictReader(text.splitlines())
            for i, row in enumerate(reader, start=2):
                records.append(self._record(dict(row), i))
            order = {m.id: k for k, m in enumerate(self.line.machines)}
            last = len(order)
            records.sort(
                key=lambda r: (
                    r.ts is None,
                    r.ts or datetime.min,
                    order.get(r.machine_id or "", last),
                    r.machine_id or "",
                    r.seq if r.seq is not None else -1,
                )
            )
        stamped = [r.ts for r in records if r.ts is not None]
        if stamped and self.cfg.start_offset_s > 0:
            cutoff = stamped[0] + timedelta(seconds=self.cfg.start_offset_s)
            records = [r for r in records if r.ts is None or r.ts >= cutoff]
        return records

    # -- replay ------------------------------------------------------------------------------
    def _loop_shift(self) -> tuple[timedelta, dict[str, int]]:
        stamps = sorted({r.ts for r in self._records if r.ts is not None})
        if len(stamps) >= 2:
            step = min(b - a for a, b in itertools.pairwise(stamps))
            span = stamps[-1] - stamps[0] + step
        else:
            span = timedelta(0)
        max_seq: dict[str, int] = {}
        for r in self._records:
            if r.machine_id is not None and r.seq is not None:
                max_seq[r.machine_id] = max(max_seq.get(r.machine_id, r.seq), r.seq)
        return span, max_seq

    def _shifted(self, rec: _Record, n: int, span: timedelta, max_seq: Mapping[str, int]) -> Any:
        if n == 0 or isinstance(rec.payload, str):
            return rec.payload
        payload = dict(rec.payload)
        if rec.ts is not None:
            payload[self._ts_key] = (rec.ts + n * span).isoformat()
        if rec.seq is not None and rec.machine_id is not None and rec.machine_id in max_seq:
            new_seq = rec.seq + n * max_seq[rec.machine_id]
            payload[self._seq_key] = (
                str(new_seq) if isinstance(rec.payload.get(self._seq_key), str) else new_seq
            )
        return payload

    async def _run(self) -> None:
        self._mark_connected()
        span, max_seq = self._loop_shift()
        first_ts = next((r.ts for r in self._records if r.ts is not None), None)
        start_mono = self.clock.monotonic()
        speed = self.cfg.replay_speed
        n = 0
        while True:
            for rec in self._records:
                if rec.ts is not None and first_ts is not None:
                    offset = (rec.ts + n * span - first_ts).total_seconds() / speed
                    delay = start_mono + offset - self.clock.monotonic()
                    if delay > 0:
                        await self.clock.sleep(delay)
                await self._emit(
                    self._shifted(rec, n, span, max_seq),
                    meta={"file": self.path.name, "line": str(rec.line_no), "pass": str(n)},
                )
            self.passes = n + 1
            if not self.cfg.loop or span <= timedelta(0):
                return
            n += 1
