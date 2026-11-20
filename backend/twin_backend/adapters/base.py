"""Source adapter protocol and shared machinery (see ADR 0007).

An adapter turns one outside system (file, mock server, MQTT, OPC UA) into a stream of
:class:`RawMessage`. Adapters do not validate: the ingest pipeline normalizes, validates,
deduplicates and reorders. Adapters own their connection and reconnect with exponential backoff.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from twin_backend.clock import Clock
from twin_common.config.sources import ReconnectConfig
from twin_common.contracts import Command, CommandAck, ScenarioAck, ScenarioRequest
from twin_common.enums import AckStatus, AdapterKind, AdapterStatus, Capability
from twin_common.logging import get_logger

log = get_logger(__name__)

Payload = bytes | str | Mapping[str, Any]

# Internal buffer between an adapter's I/O task and its consumer. Purely a buffer size: a full
# queue applies backpressure to the reader (file replay, socket reads); it never drops data.
DEFAULT_QUEUE_SIZE = 1024
# How often stop() repeats the cancellation of a task that has not ended yet.
STOP_RECANCEL_S = 0.2


@dataclass(frozen=True, slots=True)
class RawMessage:
    """One message as received from a source, before normalization."""

    source: AdapterKind
    payload: Payload
    received_mono: float
    received_at: datetime
    sent_at: datetime | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def with_payload(self, payload: Payload) -> RawMessage:
        return replace(self, payload=payload)


@dataclass(frozen=True, slots=True)
class AdapterHealth:
    kind: AdapterKind
    status: AdapterStatus
    connected: bool
    messages: int
    reconnects: int
    last_message_at: datetime | None
    last_error: str | None


class AdapterError(RuntimeError):
    """An adapter cannot start (bad config, missing file, ...)."""


class WhatIfUnsupportedError(RuntimeError):
    """The active source cannot fork a simulation (a simulation-only capability)."""

    def __init__(self, kind: str) -> None:
        super().__init__(
            f"the active source ({kind} adapter) does not support what-if projections "
            "(a simulation-only capability; use the mock source)"
        )


class WhatIfError(RuntimeError):
    """A what-if fork failed at the source (not running, timeout, transport)."""


@runtime_checkable
class SourceAdapter(Protocol):
    kind: AdapterKind
    capabilities: frozenset[Capability]

    @property
    def time_scale(self) -> float:
        """Event-time seconds per wall-clock second (replay speed; 1.0 for live sources)."""
        ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def stream(self) -> AsyncIterator[RawMessage]: ...

    def health(self) -> AdapterHealth: ...

    async def send_command(self, cmd: Command) -> CommandAck: ...


def unsupported_ack(cmd: Command, kind: AdapterKind, now: datetime) -> CommandAck:
    return CommandAck(
        command_id=cmd.command_id,
        status=AckStatus.REJECTED,
        reason=f"the {kind.value} adapter does not support commands",
        ts=now,
    )


def scenario_ack(
    req: ScenarioRequest, status: AckStatus, reason: str | None, now: datetime
) -> ScenarioAck:
    return ScenarioAck(request_id=req.request_id, status=status, reason=reason, ts=now)


class Backoff:
    """Exponential backoff with seeded jitter: min(max, initial * mult^n) * (1 +/- jitter)."""

    def __init__(self, cfg: ReconnectConfig) -> None:
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self.attempt = 0

    def next_delay(self) -> float:
        base = min(self.cfg.max_s, self.cfg.initial_s * self.cfg.multiplier**self.attempt)
        self.attempt += 1
        if self.cfg.jitter == 0:
            return base
        return base * (1.0 + self._rng.uniform(-self.cfg.jitter, self.cfg.jitter))

    def reset(self) -> None:
        self.attempt = 0


_EOF = object()


class BaseAdapter:
    """Runs the subclass's ``_run()`` I/O loop in a task and exposes it as ``stream()``.

    ``_run`` calls ``_emit`` for every message and returns at end of data (EOF). Exceptions
    escaping ``_run`` mark the adapter FAILED and end the stream; transport errors that can
    be retried should be handled inside ``_run`` with ``_wait_reconnect``.
    """

    kind: AdapterKind
    capabilities: frozenset[Capability] = frozenset()

    def __init__(
        self,
        clock: Clock,
        reconnect: ReconnectConfig | None = None,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.clock = clock
        self.backoff = Backoff(reconnect) if reconnect is not None else None
        self._queue_size = queue_size
        self._queue: asyncio.Queue[RawMessage | object] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._status = AdapterStatus.STOPPED
        self._connected = False
        self._messages = 0
        self._reconnects = 0
        self._last_message_at: datetime | None = None
        self._last_error: str | None = None
        self._stopping = False

    @property
    def time_scale(self) -> float:
        """Live sources deliver event time at wall-clock pace; replaying adapters override."""
        return 1.0

    # -- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._status = AdapterStatus.STARTING
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        await self._prepare()
        self._task = asyncio.create_task(self._main(), name=f"adapter-{self.kind.value}")

    async def stop(self) -> None:
        self._stopping = True
        task, self._task = self._task, None
        if task is not None:
            # Re-cancel until the task ends: on Python 3.11 ``asyncio.wait_for`` (used inside
            # client libraries, e.g. asyncua's next_event) can swallow a cancellation that
            # arrives in the same loop iteration as the inner result.
            while not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=STOP_RECANCEL_S)
            with contextlib.suppress(asyncio.CancelledError):
                task.result()
        await self._cleanup()
        self._connected = False
        if self._status not in (AdapterStatus.EOF, AdapterStatus.FAILED):
            self._status = AdapterStatus.STOPPED  # keep the more informative terminal status
        self._put_eof()

    async def _prepare(self) -> None:
        """Checks that must fail fast in ``start()`` (e.g. the replay file exists)."""

    async def _cleanup(self) -> None:
        """Release resources after the I/O task stopped."""

    async def _run(self) -> None:
        raise NotImplementedError

    async def _main(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = AdapterStatus.FAILED
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.error("adapter.failed", kind=self.kind.value, error=self._last_error)
        else:
            if not self._stopping:
                self._status = AdapterStatus.EOF
                log.info("adapter.eof", kind=self.kind.value, messages=self._messages)
        self._connected = False
        self._put_eof()

    def _put_eof(self) -> None:
        # Never block here; if the consumer is behind, make room so it still sees the end.
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(_EOF)

    # -- streaming ---------------------------------------------------------------------------
    async def stream(self) -> AsyncIterator[RawMessage]:
        while True:
            item = await self._queue.get()
            if item is _EOF:
                return
            assert isinstance(item, RawMessage)
            yield item

    async def _emit(
        self,
        payload: Payload,
        *,
        sent_at: datetime | None = None,
        meta: Mapping[str, str] | None = None,
    ) -> None:
        now = self.clock.now()
        msg = RawMessage(
            source=self.kind,
            payload=payload,
            received_mono=self.clock.monotonic(),
            received_at=now,
            sent_at=sent_at,
            meta=meta or {},
        )
        self._messages += 1
        self._last_message_at = now
        await self._queue.put(msg)

    # -- health ------------------------------------------------------------------------------
    def _mark_connected(self) -> None:
        self._connected = True
        self._status = AdapterStatus.CONNECTED
        if self.backoff is not None:
            self.backoff.reset()

    def _begin_reconnect(self, exc: BaseException) -> float:
        """Record a transport error; returns the backoff delay before the next attempt."""
        self._connected = False
        self._status = AdapterStatus.RECONNECTING
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._reconnects += 1
        delay = self.backoff.next_delay() if self.backoff is not None else 0.0
        log.warning(
            "adapter.reconnecting", kind=self.kind.value, error=self._last_error, delay_s=delay
        )
        return delay

    async def _wait_reconnect(self, exc: BaseException) -> None:
        """Record a transport error and sleep for the next backoff delay."""
        await self.clock.sleep(self._begin_reconnect(exc))

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            kind=self.kind,
            status=self._status,
            connected=self._connected,
            messages=self._messages,
            reconnects=self._reconnects,
            last_message_at=self._last_message_at,
            last_error=self._last_error,
        )

    async def send_command(self, cmd: Command) -> CommandAck:
        return unsupported_ack(cmd, self.kind, self.clock.now())

    async def send_scenario(self, req: ScenarioRequest) -> ScenarioAck:
        """Simulation only (ADR 0015); adapters with ``Capability.SCENARIO`` override this."""
        return scenario_ack(
            req,
            AckStatus.REJECTED,
            f"the {self.kind.value} adapter does not support scenario injection",
            self.clock.now(),
        )
