"""Command safety rules (STOP fail-safe) and the CommandService: idempotency, capability,
timeout, audit-trail order and restore from the event store."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from twin_backend.bus import InMemoryEventBus, Topic
from twin_backend.clock import FakeClock
from twin_backend.commands.safety import SafetyRules
from twin_backend.commands.service import (
    CommandRequest,
    CommandService,
    IdempotencyConflictError,
    UnknownMachineError,
)
from twin_backend.storage.base import Stored, Stream
from twin_backend.testing import REPO_ROOT, drain_readings
from twin_common.config import load_commands_config, load_line_config
from twin_common.contracts import Command, CommandAck, ParamValue
from twin_common.enums import AckStatus, Capability, CommandType, MachineKind, MachineState
from twin_common.testing import SAMPLE_TS, sample_command

CMDS = load_commands_config(REPO_ROOT / "config" / "commands.yaml")
LINE = load_line_config(REPO_ROOT / "config" / "line.yaml")


def _cmd(ctype: CommandType, **params: float) -> Command:
    return sample_command(type=ctype, params=params, machine_id="CNC-01")


# ---------------------------------------------------------------- safety rules
@pytest.mark.parametrize("state", [*MachineState, None], ids=str)
def test_stop_is_allowed_in_every_state(state: MachineState | None) -> None:
    assert SafetyRules(CMDS).check(_cmd(CommandType.STOP), state, MachineKind.CNC).allowed


@pytest.mark.parametrize(
    ("ctype", "params", "state", "fragment"),
    [
        (CommandType.SET_SPEED, {"speed": 8000.0}, MachineState.FAULT, "faulted or in maintenance"),
        (CommandType.SET_LOAD, {"load": 50.0}, MachineState.MAINTENANCE, "faulted or in maint"),
        (CommandType.SCHEDULE_MAINTENANCE, {}, MachineState.RUNNING, "STOP it first"),
        (CommandType.START, {}, MachineState.FAULT, "RESET it first"),
        (CommandType.START, {}, MachineState.OFFLINE, "data link is down"),
        (CommandType.RESET, {}, MachineState.STALE, "data link is down"),
        (CommandType.SET_SPEED, {"speed": 20000.0}, MachineState.RUNNING, "outside [1, 12000]"),
        (CommandType.SET_LOAD, {"load": 150.0}, MachineState.IDLE, "outside [0, 120]"),
        (CommandType.START, {}, None, "not known yet"),
    ],
)
def test_rules_reject_with_reason(
    ctype: CommandType, params: dict[str, float], state: MachineState | None, fragment: str
) -> None:
    v = SafetyRules(CMDS).check(_cmd(ctype, **params), state, MachineKind.CNC)
    assert not v.allowed
    assert v.reason is not None
    assert fragment in v.reason


@pytest.mark.parametrize(
    ("ctype", "params", "state"),
    [
        (CommandType.SET_SPEED, {"speed": 9000.0}, MachineState.RUNNING),
        (CommandType.SCHEDULE_MAINTENANCE, {}, MachineState.IDLE),
        (CommandType.SCHEDULE_MAINTENANCE, {}, MachineState.FAULT),
        (CommandType.RESET, {}, MachineState.FAULT),
        (CommandType.START, {}, MachineState.IDLE),
    ],
)
def test_rules_allow(ctype: CommandType, params: dict[str, float], state: MachineState) -> None:
    assert SafetyRules(CMDS).check(_cmd(ctype, **params), state, MachineKind.CNC).allowed


# ---------------------------------------------------------------- service with fakes
class FakeManager:
    def __init__(self, *, commands: bool = True, delay: float = 0.0) -> None:
        self.commands = commands
        self.delay = delay
        self.routed: list[Command] = []
        self.status = AckStatus.EXECUTED
        self.active_kind = None

    def supports(self, capability: Capability) -> bool:
        return self.commands and capability is Capability.COMMANDS

    async def route_command(self, cmd: Command) -> CommandAck:
        self.routed.append(cmd)
        if self.delay:
            await asyncio.sleep(self.delay)
        return CommandAck(command_id=cmd.command_id, status=self.status, reason=None, ts=SAMPLE_TS)


class FakeEngine:
    machine_ids: ClassVar[list[str]] = [m.id for m in LINE.machines]

    def __init__(self, state: MachineState | None = MachineState.RUNNING) -> None:
        self.states: dict[str, MachineState | None] = dict.fromkeys(self.machine_ids, state)

        class _S:
            @staticmethod
            def watermark() -> Any:
                return SAMPLE_TS + timedelta(seconds=10)

        self.state = _S()

    def machine_state(self, machine_id: str) -> MachineState | None:
        return self.states[machine_id]

    def machine_watermark(self, machine_id: str) -> Any:
        return SAMPLE_TS + timedelta(seconds=5)


def _service(
    manager: FakeManager | None = None, engine: FakeEngine | None = None, store: Any = None
) -> tuple[CommandService, FakeManager, InMemoryEventBus, Any]:
    mgr = manager or FakeManager()
    clock = FakeClock()
    bus = InMemoryEventBus(1000, clock=clock)
    sub = bus.subscribe(Topic.COMMANDS)
    svc = CommandService(
        CMDS, LINE, cast(Any, mgr), cast(Any, engine or FakeEngine()), bus, clock, store
    )
    return svc, mgr, bus, sub


def _req(ctype: CommandType = CommandType.STOP, key: str = "k1", **params: float) -> CommandRequest:
    values: dict[str, ParamValue] = dict(params)
    return CommandRequest(machine_id="CNC-01", type=ctype, params=values, idempotency_key=key)


async def test_audit_trail_order_and_event_time() -> None:
    svc, mgr, _bus, sub = _service()
    result = await svc.submit(_req(), "alice")
    assert result.ack.status is AckStatus.EXECUTED
    assert not result.replay
    events = drain_readings(sub)
    assert [type(e).__name__ for e in events] == ["Command", "CommandAck", "CommandAck"]
    cmd, accepted, final = events
    assert cmd.issued_by == "alice"
    assert cmd.ts == SAMPLE_TS + timedelta(seconds=5)  # machine watermark (event time)
    assert accepted.status is AckStatus.ACCEPTED
    assert final.status is AckStatus.EXECUTED
    assert mgr.routed == [cmd]
    record = await svc.get(cmd.command_id)
    assert record is not None
    assert [a.status for a in record.acks] == [AckStatus.ACCEPTED, AckStatus.EXECUTED]


async def test_safety_rejection_is_audited_and_not_sent() -> None:
    svc, mgr, _bus, sub = _service(engine=FakeEngine(MachineState.FAULT))
    result = await svc.submit(_req(CommandType.SET_SPEED, speed=8000.0), "alice")
    assert result.ack.status is AckStatus.REJECTED
    assert "faulted" in (result.ack.reason or "")
    assert [type(e).__name__ for e in drain_readings(sub)] == ["Command", "CommandAck"]
    assert mgr.routed == []
    stop = await svc.submit(_req(CommandType.STOP, key="k2"), "alice")  # fail-safe
    assert stop.ack.status is AckStatus.EXECUTED


async def test_no_command_capability_rejects_with_reason() -> None:
    svc, mgr, _bus, _sub = _service(FakeManager(commands=False))
    result = await svc.submit(_req(), "alice")
    assert result.ack.status is AckStatus.REJECTED
    assert "does not accept commands" in (result.ack.reason or "")
    assert mgr.routed == []


async def test_idempotency_replay_conflict_and_in_flight_dedup() -> None:
    svc, mgr, _bus, _sub = _service(FakeManager(delay=0.05))
    first, second = await asyncio.gather(
        svc.submit(_req(CommandType.SET_SPEED, speed=8000.0), "alice"),
        svc.submit(_req(CommandType.SET_SPEED, speed=8000.0), "bob"),
    )
    assert first.ack == second.ack
    assert sorted([first.replay, second.replay]) == [False, True]
    assert len(mgr.routed) == 1  # sent once
    again = await svc.submit(_req(CommandType.SET_SPEED, speed=8000.0), "alice")
    assert again.replay
    assert again.ack == first.ack
    with pytest.raises(IdempotencyConflictError):
        await svc.submit(_req(CommandType.SET_SPEED, speed=9000.0), "alice")
    with pytest.raises(UnknownMachineError):
        await svc.submit(
            CommandRequest(machine_id="NOPE", type=CommandType.STOP, idempotency_key="x"), "a"
        )
    assert len(mgr.routed) == 1


async def test_keys_expire_after_ttl() -> None:
    svc, mgr, _bus, _sub = _service()
    await svc.submit(_req(), "alice")
    cast(FakeClock, svc.clock).advance(CMDS.idempotency_ttl_s + 1)
    again = await svc.submit(_req(), "alice")
    assert not again.replay
    assert len(mgr.routed) == 2


async def test_timeout_from_the_manager_is_the_final_ack() -> None:
    svc, mgr, _bus, _sub = _service()
    mgr.status = AckStatus.TIMEOUT
    assert (await svc.submit(_req(), "alice")).ack.status is AckStatus.TIMEOUT


async def test_idempotency_keys_are_restored_from_the_store(
    store_factory: Any, tmp_path: Path
) -> None:
    store = await store_factory(tmp_path)
    try:
        svc, _mgr, _bus, sub = _service()
        original = await svc.submit(_req(CommandType.SET_SPEED, speed=8000.0), "alice")
        events = drain_readings(sub)
        now = svc.clock.now()
        for stream, typ in ((Stream.COMMANDS, Command), (Stream.COMMAND_ACKS, CommandAck)):
            rows = [Stored(e, now) for e in events if isinstance(e, typ)]
            await store.insert(stream, rows)
        restarted, mgr2, _b, _s = _service(store=store)
        assert await restarted.load() == 1
        replay = await restarted.submit(_req(CommandType.SET_SPEED, speed=8000.0), "alice")
        assert replay.replay
        assert replay.ack == original.ack
        assert mgr2.routed == []
        record = await restarted.get(original.ack.command_id)
        assert record is not None
        assert [a.status for a in record.acks] == [AckStatus.ACCEPTED, AckStatus.EXECUTED]
        history = await restarted.history(None, None, ["CNC-01"])
        assert [r.command.command_id for r in history] == [original.ack.command_id]
    finally:
        await store.close()
