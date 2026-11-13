"""Quarantine sinks: rejected messages are kept with the reason, never silently dropped.

Phase 3 adds an event-store-backed sink; the memory sink serves tests and minimal mode, the
JSONL sink keeps a file for inspection.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from twin_backend.adapters.base import Payload
from twin_backend.bus.base import EventBus, Topic
from twin_common.config.sources import QuarantineConfig
from twin_common.enums import AdapterKind, QuarantineReason


class QuarantineRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts: datetime
    source: AdapterKind
    reason: QuarantineReason
    stage: str
    detail: str
    machine_id: str | None
    payload: dict[str, Any] | str
    meta: dict[str, str]


def payload_for_record(payload: Payload) -> dict[str, Any] | str:
    """A JSON-safe copy of the raw payload."""
    if isinstance(payload, Mapping):
        copy: dict[str, Any] = json.loads(json.dumps(dict(payload), default=str))
        return copy
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


class QuarantineSink(Protocol):
    async def put(self, record: QuarantineRecord) -> None: ...


class MemoryQuarantine:
    """Bounded in-memory sink (oldest records dropped beyond ``max_records``)."""

    def __init__(self, max_records: int) -> None:
        self.records: deque[QuarantineRecord] = deque(maxlen=max_records)
        self.total = 0

    async def put(self, record: QuarantineRecord) -> None:
        self.records.append(record)
        self.total += 1


class JsonlQuarantine:
    """Appends one JSON line per record. File I/O runs in a worker thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    async def put(self, record: QuarantineRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append, record.model_dump_json())


class BusQuarantine:
    """Keeps the inner sink and also publishes every record on the bus (``Topic.QUARANTINE``)."""

    def __init__(self, inner: QuarantineSink, bus: EventBus) -> None:
        self.inner = inner
        self.bus = bus
        self.total = 0

    async def put(self, record: QuarantineRecord) -> None:
        await self.inner.put(record)
        await self.bus.publish(Topic.QUARANTINE, record)
        self.total += 1


def build_quarantine(cfg: QuarantineConfig, root: Path) -> MemoryQuarantine | JsonlQuarantine:
    if cfg.kind == "jsonl":
        assert cfg.path is not None  # enforced by QuarantineConfig
        return JsonlQuarantine(root / cfg.path)
    return MemoryQuarantine(cfg.max_records)
