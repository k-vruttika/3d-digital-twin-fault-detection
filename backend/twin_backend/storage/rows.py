"""Row codecs shared by the store implementations: contract <-> column tuple per stream.

Timestamps are passed through ``ts_out`` / ``ts_in`` so each backend picks its representation
(SQLite: integer microseconds; PostgreSQL: ``timestamptz``). Nested payloads are JSON text.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from twin_backend.ingest.quarantine import QuarantineRecord
from twin_backend.storage.base import Stored, Stream
from twin_common.contracts import Alert, Command, CommandAck, Prediction, Reading, StateChange

COLUMNS: dict[Stream, tuple[str, ...]] = {
    Stream.READINGS: (
        "ts",
        "machine_id",
        "seq",
        "schema_version",
        "state",
        "speed",
        "temp",
        "vib",
        "load",
        "current",
        "run_time",
        "quality",
        "ingested_at",
    ),
    Stream.STATE_CHANGES: (
        "ts",
        "machine_id",
        "state",
        "previous_state",
        "reason",
        "ingested_at",
    ),
    Stream.PREDICTIONS: (
        "ts",
        "machine_id",
        "horizon_s",
        "model_version",
        "payload",
        "ingested_at",
    ),
    Stream.COMMANDS: (
        "ts",
        "command_id",
        "idempotency_key",
        "machine_id",
        "type",
        "payload",
        "ingested_at",
    ),
    Stream.COMMAND_ACKS: ("ts", "command_id", "status", "reason", "ingested_at"),
    Stream.ALERTS: (
        "ts",
        "alert_id",
        "machine_id",
        "severity",
        "kind",
        "message",
        "source",
        "ingested_at",
    ),
    Stream.QUARANTINE: (
        "ts",
        "source",
        "reason",
        "stage",
        "detail",
        "machine_id",
        "payload",
        "meta",
        "ingested_at",
    ),
}

# Natural keys: duplicates on these are skipped (idempotent inserts). Quarantine has none.
UNIQUE_KEYS: dict[Stream, tuple[str, ...]] = {
    Stream.READINGS: ("machine_id", "seq", "ts"),
    Stream.STATE_CHANGES: ("machine_id", "ts", "state"),
    Stream.PREDICTIONS: ("machine_id", "ts", "horizon_s", "model_version"),
    Stream.COMMANDS: ("command_id", "ts"),
    Stream.COMMAND_ACKS: ("command_id", "status", "ts"),
    Stream.ALERTS: ("alert_id", "ts"),
}

# Deterministic fetch order.
ORDER_BY: dict[Stream, tuple[str, ...]] = {
    Stream.READINGS: ("ts", "machine_id", "seq"),
    Stream.STATE_CHANGES: ("ts", "machine_id", "state"),
    Stream.PREDICTIONS: ("ts", "machine_id", "horizon_s", "model_version"),
    Stream.COMMANDS: ("ts", "command_id"),
    Stream.COMMAND_ACKS: ("ts", "command_id", "status"),
    Stream.ALERTS: ("ts", "alert_id"),
    Stream.QUARANTINE: ("ts", "ingested_at", "source", "reason", "detail"),
}


def has_machine_column(stream: Stream) -> bool:
    return "machine_id" in COLUMNS[stream]


TsOut = Callable[[datetime], Any]
TsIn = Callable[[Any], datetime]


def encode(stream: Stream, stored: Stored[Any], ts_out: TsOut) -> tuple[Any, ...]:
    item, ing = stored.item, ts_out(stored.ingested_at)
    if stream is Stream.READINGS:
        r: Reading = item
        return (
            ts_out(r.ts),
            r.machine_id,
            r.seq,
            r.schema_version,
            r.state.value,
            r.speed,
            r.temp,
            r.vib,
            r.load,
            r.current,
            r.run_time,
            r.quality.value,
            ing,
        )
    if stream is Stream.STATE_CHANGES:
        c: StateChange = item
        prev = c.previous_state.value if c.previous_state is not None else None
        return (ts_out(c.ts), c.machine_id, c.state.value, prev, c.reason, ing)
    if stream is Stream.PREDICTIONS:
        p: Prediction = item
        return (ts_out(p.ts), p.machine_id, p.horizon_s, p.model_version, p.model_dump_json(), ing)
    if stream is Stream.COMMANDS:
        cmd: Command = item
        return (
            ts_out(cmd.ts),
            cmd.command_id,
            cmd.idempotency_key,
            cmd.machine_id,
            cmd.type.value,
            cmd.model_dump_json(),
            ing,
        )
    if stream is Stream.COMMAND_ACKS:
        a: CommandAck = item
        return (ts_out(a.ts), a.command_id, a.status.value, a.reason, ing)
    if stream is Stream.ALERTS:
        al: Alert = item
        return (
            ts_out(al.ts),
            al.alert_id,
            al.machine_id,
            al.severity.value,
            al.kind,
            al.message,
            al.source,
            ing,
        )
    q: QuarantineRecord = item
    return (
        ts_out(q.ts),
        q.source.value,
        q.reason.value,
        q.stage,
        q.detail,
        q.machine_id,
        json.dumps(q.payload),
        json.dumps(q.meta, sort_keys=True),
        ing,
    )


def decode(stream: Stream, row: Mapping[str, Any], ts_in: TsIn) -> Stored[Any]:
    ing = ts_in(row["ingested_at"])
    ts = ts_in(row["ts"])
    item: Any
    if stream is Stream.READINGS:
        item = Reading.model_validate(
            {
                "schema_version": row["schema_version"],
                "machine_id": row["machine_id"],
                "seq": row["seq"],
                "ts": ts,
                "state": row["state"],
                "speed": row["speed"],
                "temp": row["temp"],
                "vib": row["vib"],
                "load": row["load"],
                "current": row["current"],
                "run_time": row["run_time"],
                "quality": row["quality"],
            }
        )
    elif stream is Stream.STATE_CHANGES:
        item = StateChange.model_validate(
            {
                "machine_id": row["machine_id"],
                "state": row["state"],
                "previous_state": row["previous_state"],
                "ts": ts,
                "reason": row["reason"],
            }
        )
    elif stream is Stream.PREDICTIONS:
        item = Prediction.model_validate_json(row["payload"])
    elif stream is Stream.COMMANDS:
        item = Command.model_validate_json(row["payload"])
    elif stream is Stream.COMMAND_ACKS:
        item = CommandAck.model_validate(
            {"command_id": row["command_id"], "status": row["status"], "reason": row["reason"]}
            | {"ts": ts}
        )
    elif stream is Stream.ALERTS:
        item = Alert.model_validate(
            {
                "alert_id": row["alert_id"],
                "machine_id": row["machine_id"],
                "ts": ts,
                "severity": row["severity"],
                "kind": row["kind"],
                "message": row["message"],
                "source": row["source"],
            }
        )
    else:
        item = QuarantineRecord(
            ts=ts,
            source=row["source"],
            reason=row["reason"],
            stage=row["stage"],
            detail=row["detail"],
            machine_id=row["machine_id"],
            payload=json.loads(row["payload"]),
            meta=json.loads(row["meta"]),
        )
    return Stored(item, ing)
