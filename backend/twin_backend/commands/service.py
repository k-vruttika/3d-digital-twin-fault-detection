"""Command path (ADR 0014): validation, idempotency, safety, routing, ack, audit trail.

``POST /commands`` → :meth:`CommandService.submit`:

1. **Idempotency:** a repeated ``idempotencyKey`` with the same request returns the original
   final ack without sending anything again (also while the first request is still in
   flight); the same key with a different request is a conflict. Keys are kept for
   ``idempotency_ttl_s`` (bounded by ``idempotency_max_entries``) and reloaded from the event
   store on startup.
2. **Safety rules** (``safety.py``; STOP is always allowed). A violation is REJECTED with the
   rule's reason.
3. **Capability:** a source without commands (the file replay) gives REJECTED with a clear
   reason.
4. **Routing** to the active adapter, waiting at most ``command_timeout_s`` (TIMEOUT).

**Audit trail:** the Command, an ACCEPTED ack (safety and capability passed) and the final ack
(EXECUTED / REJECTED / TIMEOUT) are published on ``Topic.COMMANDS``, so the event store keeps
every command, rejected ones included. Their ``ts`` is on the event-time axis (ADR 0010): the
target machine's watermark (the line's if the machine has none, wall clock if there is no data
at all). The wall-clock time is kept as the rows' ``ingested_at``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import Field

from twin_backend.adapters.manager import SourceManager
from twin_backend.bus.base import EventBus, Topic
from twin_backend.clock import Clock
from twin_backend.commands.safety import SafetyRules
from twin_backend.engine.line_state import StateModel
from twin_backend.engine.twin import TwinEngine
from twin_backend.storage.base import EventStore, Stream
from twin_common.config.commands import CommandsConfig
from twin_common.config.line import LineConfig
from twin_common.contracts import Command, CommandAck, ParamValue
from twin_common.enums import AckStatus, Capability, CommandType
from twin_common.logging import get_logger

log = get_logger(__name__)

FINAL_STATUSES = frozenset({AckStatus.REJECTED, AckStatus.EXECUTED, AckStatus.TIMEOUT})


class CommandRequest(StateModel):
    """Body of ``POST /commands``. The server adds commandId, issuedBy and ts."""

    machine_id: str = Field(min_length=1)
    type: CommandType
    params: dict[str, ParamValue] = Field(default_factory=dict)
    idempotency_key: str = Field(
        min_length=1, max_length=200, description="repeat it to retry safely"
    )

    def fingerprint(self) -> str:
        body = {"machineId": self.machine_id, "type": self.type.value, "params": self.params}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class CommandRecord(StateModel):
    """A command with every ack it got (the audit view)."""

    command: Command
    acks: list[CommandAck] = Field(default_factory=list)

    @property
    def final(self) -> CommandAck | None:
        finals = [a for a in self.acks if a.status in FINAL_STATUSES]
        return finals[-1] if finals else None


class IdempotencyConflictError(Exception):
    """The idempotency key was used before for a different request."""


class UnknownMachineError(Exception):
    pass


@dataclass
class _Entry:
    fingerprint: str
    record: CommandRecord
    created: datetime
    done: asyncio.Future[CommandAck] = field(repr=False, default_factory=asyncio.Future)


@dataclass(frozen=True, slots=True)
class SubmitResult:
    ack: CommandAck
    replay: bool  # True: an earlier request with this idempotency key answered


class CommandService:
    def __init__(
        self,
        cfg: CommandsConfig,
        line: LineConfig,
        manager: SourceManager,
        engine: TwinEngine,
        bus: EventBus,
        clock: Clock,
        store: EventStore | None,
    ) -> None:
        self.cfg = cfg
        self.kinds = {m.id: m.kind for m in line.machines}
        self.manager = manager
        self.engine = engine
        self.bus = bus
        self.clock = clock
        self.store = store
        self.safety = SafetyRules(cfg)
        self._by_key: OrderedDict[str, _Entry] = OrderedDict()
        self._by_id: OrderedDict[str, CommandRecord] = OrderedDict()

    # -- startup -----------------------------------------------------------------------------
    async def load(self) -> int:
        """Reload idempotency keys of the last ``idempotency_ttl_s`` from the event store."""
        if self.store is None:
            return 0
        horizon = self.clock.now() - timedelta(seconds=self.cfg.idempotency_ttl_s)
        commands = [s for s in await self.store.fetch(Stream.COMMANDS) if s.ingested_at >= horizon]
        if not commands:
            return 0
        acks: dict[str, list[CommandAck]] = {}
        for s in await self.store.fetch(Stream.COMMAND_ACKS, commands[0].item.ts):
            acks.setdefault(s.item.command_id, []).append(s.item)
        n = 0
        for s in commands:
            cmd: Command = s.item
            record = CommandRecord(command=cmd, acks=acks.get(cmd.command_id, []))
            final = record.final
            if final is None:
                continue  # the backend stopped before the ack: a retry is sent again
            req = CommandRequest(
                machine_id=cmd.machine_id,
                type=cmd.type,
                params=cmd.params,
                idempotency_key=cmd.idempotency_key,
            )
            entry = _Entry(req.fingerprint(), record, s.ingested_at)
            entry.done.set_result(final)
            self._remember(cmd.idempotency_key, entry)
            n += 1
        log.info("commands.idempotency_restored", keys=n)
        return n

    # -- submit ------------------------------------------------------------------------------
    async def submit(self, req: CommandRequest, issued_by: str) -> SubmitResult:
        if req.machine_id not in self.kinds:
            raise UnknownMachineError(req.machine_id)
        self._expire()
        entry = self._by_key.get(req.idempotency_key)
        if entry is not None:
            if entry.fingerprint != req.fingerprint():
                raise IdempotencyConflictError(req.idempotency_key)
            log.info("commands.idempotent_replay", key=req.idempotency_key)
            return SubmitResult(await asyncio.shield(entry.done), replay=True)

        cmd = Command(
            command_id=uuid.uuid4().hex,
            idempotency_key=req.idempotency_key,
            machine_id=req.machine_id,
            type=req.type,
            params=req.params,
            issued_by=issued_by,
            ts=self._event_now(req.machine_id),
        )
        record = CommandRecord(command=cmd)
        entry = _Entry(req.fingerprint(), record, self.clock.now())
        self._remember(req.idempotency_key, entry)
        try:
            ack = await self._execute(record)
        except BaseException as exc:
            # never leave waiters hanging; a failed attempt is not remembered
            self._by_key.pop(req.idempotency_key, None)
            if not entry.done.done():
                entry.done.set_exception(exc)
                entry.done.exception()  # mark retrieved
            raise
        entry.done.set_result(ack)
        return SubmitResult(ack, replay=False)

    async def _execute(self, record: CommandRecord) -> CommandAck:
        cmd = record.command
        await self._publish(record, cmd)
        state = self.engine.machine_state(cmd.machine_id)
        verdict = self.safety.check(cmd, state, self.kinds[cmd.machine_id])
        if not verdict.allowed:
            return await self._finish(record, AckStatus.REJECTED, verdict.reason)
        if not self.manager.supports(Capability.COMMANDS):
            kind = self.manager.active_kind
            return await self._finish(
                record,
                AckStatus.REJECTED,
                f"the active source ({kind.value if kind else 'no'} adapter) does not accept "
                "commands; switch to a live source (mock, mqtt, opcua) first",
            )
        await self._publish(
            record, self._ack(cmd, AckStatus.ACCEPTED, "passed the safety rules; sent to machine")
        )
        result = await self.manager.route_command(cmd)
        return await self._finish(record, result.status, result.reason)

    async def _finish(
        self, record: CommandRecord, status: AckStatus, reason: str | None
    ) -> CommandAck:
        ack = self._ack(record.command, status, reason)
        await self._publish(record, ack)
        log.info(
            "commands.ack",
            command_id=ack.command_id,
            machine=record.command.machine_id,
            type=record.command.type.value,
            status=status.value,
            reason=reason,
            by=record.command.issued_by,
        )
        return ack

    def _ack(self, cmd: Command, status: AckStatus, reason: str | None) -> CommandAck:
        ts = max(self._event_now(cmd.machine_id), cmd.ts)
        return CommandAck(command_id=cmd.command_id, status=status, reason=reason, ts=ts)

    async def _publish(self, record: CommandRecord, message: Command | CommandAck) -> None:
        if isinstance(message, CommandAck):
            record.acks.append(message)
        await self.bus.publish(Topic.COMMANDS, message)

    def _event_now(self, machine_id: str) -> datetime:
        """Event time for audit rows: machine watermark, else line watermark, else now."""
        ts = self.engine.machine_watermark(machine_id) or self.engine.state.watermark()
        return ts if ts is not None else self.clock.now()

    # -- bookkeeping -------------------------------------------------------------------------
    def _remember(self, key: str, entry: _Entry) -> None:
        self._by_key[key] = entry
        self._by_key.move_to_end(key)
        self._by_id[entry.record.command.command_id] = entry.record
        while len(self._by_key) > self.cfg.idempotency_max_entries:
            _, old = self._by_key.popitem(last=False)
            self._by_id.pop(old.record.command.command_id, None)

    def _expire(self) -> None:
        horizon = self.clock.now() - timedelta(seconds=self.cfg.idempotency_ttl_s)
        while self._by_key:
            key, entry = next(iter(self._by_key.items()))
            if entry.created >= horizon or not entry.done.done():
                break
            del self._by_key[key]
            self._by_id.pop(entry.record.command.command_id, None)

    # -- audit queries -----------------------------------------------------------------------
    async def get(self, command_id: str) -> CommandRecord | None:
        record = self._by_id.get(command_id)
        if record is not None:
            return record
        if self.store is None:
            return None
        cmds = [
            s.item
            for s in await self.store.fetch(Stream.COMMANDS)
            if s.item.command_id == command_id
        ]
        if not cmds:
            return None
        acks = [
            s.item
            for s in await self.store.fetch(Stream.COMMAND_ACKS, cmds[0].ts)
            if s.item.command_id == command_id
        ]
        return CommandRecord(command=cmds[0], acks=acks)

    async def history(
        self, t0: datetime | None, t1: datetime | None, machine_ids: list[str] | None
    ) -> list[CommandRecord]:
        records: dict[str, CommandRecord] = {}
        if self.store is not None:
            cmds = await self.store.fetch(Stream.COMMANDS, t0, t1, machine_ids=machine_ids)
            if cmds:
                acks: dict[str, list[CommandAck]] = {}
                for s in await self.store.fetch(Stream.COMMAND_ACKS, cmds[0].item.ts):
                    acks.setdefault(s.item.command_id, []).append(s.item)
                for s in cmds:
                    records[s.item.command_id] = CommandRecord(
                        command=s.item, acks=acks.get(s.item.command_id, [])
                    )
        # this process's records are complete and current (the store lags one writer flush)
        records.update(self._by_id)
        out = [
            r
            for r in records.values()
            if (t0 is None or r.command.ts >= t0)
            and (t1 is None or r.command.ts < t1)
            and (machine_ids is None or r.command.machine_id in machine_ids)
        ]
        return sorted(out, key=lambda r: (r.command.ts, r.command.command_id))

    def stats(self) -> dict[str, Any]:
        return {"idempotencyKeys": len(self._by_key)}
