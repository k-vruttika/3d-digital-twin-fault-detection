"""SourceManager: runs the active adapter into the ingest pipeline; supports hot switching.

The active adapter comes from ``sources.yaml`` (``active:``). ``switch()`` starts the new
adapter first and only then stops the old one, so a source that cannot start leaves the
current one running. Pipeline state (dedup, reorder, staleness) survives a switch, so data
that both sources deliver is deduplicated. The API exposes ``switch()`` in Phase 4.

With a ``StorageService`` (Phase 3) live ingestion persists automatically: storage starts
before the adapter, quarantined messages are stored too, and ``stop()`` publishes what the
reorder buffer still holds and then drains the writer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from twin_backend.adapters.base import AdapterHealth, SourceAdapter, WhatIfUnsupportedError
from twin_backend.adapters.registry import build_adapter, mappings_from
from twin_backend.bus.base import EventBus, Topic
from twin_backend.clock import Clock, SystemClock
from twin_backend.ingest.metrics import IngestMetrics
from twin_backend.ingest.pipeline import IngestPipeline
from twin_backend.ingest.quarantine import BusQuarantine, QuarantineSink, build_quarantine
from twin_common.config.line import LineConfig
from twin_common.config.sources import SourcesConfig
from twin_common.contracts import Command, CommandAck, ScenarioAck, ScenarioRequest
from twin_common.enums import AckStatus, AdapterKind, AdapterStatus, Capability
from twin_common.logging import get_logger
from twin_common.whatif import WhatIfSimRequest, WhatIfSimResult

if TYPE_CHECKING:
    from twin_backend.storage.service import StorageService

log = get_logger(__name__)

AdapterFactory = Callable[..., SourceAdapter]


class SourceManager:
    def __init__(
        self,
        sources: SourcesConfig,
        line: LineConfig,
        bus: EventBus,
        *,
        root: Path,
        clock: Clock | None = None,
        quarantine: QuarantineSink | None = None,
        metrics: IngestMetrics | None = None,
        factory: AdapterFactory = build_adapter,
        run_ticker: bool = True,
        storage: StorageService | None = None,
    ) -> None:
        self.sources = sources
        self.line = line
        self.bus = bus
        self.root = root
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.storage = storage
        sink: QuarantineSink = (
            quarantine
            if quarantine is not None
            else build_quarantine(sources.ingest.quarantine, root)
        )
        stored: QuarantineSink = storage.quarantine_sink(sink) if storage is not None else sink
        # The engine's data-quality status needs quarantined messages too.
        self.quarantine: QuarantineSink = BusQuarantine(stored, bus)
        self.pipeline = IngestPipeline(
            line,
            sources.ingest,
            bus,
            self.quarantine,
            self.clock,
            mappings_from(sources),
            metrics,
        )
        self._factory = factory
        self._run_ticker = run_ticker
        self._adapter: SourceAdapter | None = None
        self._pump: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.switches = 0

    @property
    def adapter(self) -> SourceAdapter | None:
        return self._adapter

    @property
    def active_kind(self) -> AdapterKind | None:
        return self._adapter.kind if self._adapter is not None else None

    def health(self) -> AdapterHealth | None:
        return self._adapter.health() if self._adapter is not None else None

    # -- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> None:
        async with self._lock:
            if self._adapter is not None:
                return
            if self.storage is not None:
                await self.storage.start()
            adapter = self._build(self.sources.active, self.sources)
            await adapter.start()
            self._attach(adapter)
            if self._run_ticker and self._ticker is None:
                self._ticker = asyncio.create_task(self._tick_loop(), name="ingest-ticker")

    async def stop(self, *, before_storage: Callable[[], Awaitable[None]] | None = None) -> None:
        """Stop the adapter, publish what the reorder buffer holds, then stop storage.

        ``before_storage`` runs after the last readings were published and before the writer
        drains, so consumers (the twin engine) can publish their final events and have them
        stored.
        """
        async with self._lock:
            await self._detach()
            if self._ticker is not None:
                self._ticker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._ticker
                self._ticker = None
            if self.storage is not None:
                await self.pipeline.flush()
            if before_storage is not None:
                await before_storage()
            if self.storage is not None:
                await self.storage.stop()

    async def wait_idle(self) -> None:
        """Wait until the current adapter's stream ended (finite file replay)."""
        if self._pump is not None:
            await asyncio.shield(self._pump)

    async def switch(self, kind: AdapterKind, sources: SourcesConfig | None = None) -> None:
        """Hot-switch to another adapter (optionally with new source settings)."""
        async with self._lock:
            cfg = sources if sources is not None else self.sources
            new = self._build(kind, cfg)
            await new.start()  # raises -> the current adapter keeps running
            old = self.active_kind
            await self._detach()
            if sources is not None:
                self.sources = sources
                mapping = getattr(sources, kind.value).mapping
                self.pipeline.set_mapping(kind, mapping)
            self._attach(new)
            self.switches += 1
            log.info("sources.switched", old=old.value if old else None, new=kind.value)

    def _build(self, kind: AdapterKind, sources: SourcesConfig) -> SourceAdapter:
        return self._factory(kind, sources, self.line, root=self.root, clock=self.clock)

    def _attach(self, adapter: SourceAdapter) -> None:
        self._adapter = adapter
        self.pipeline.set_time_scale(adapter.time_scale)
        self._pump = asyncio.create_task(self._pump_loop(adapter), name=f"pump-{adapter.kind}")

    async def _detach(self) -> None:
        adapter, self._adapter = self._adapter, None
        pump, self._pump = self._pump, None
        if adapter is not None:
            await adapter.stop()
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def _pump_loop(self, adapter: SourceAdapter) -> None:
        async for raw in adapter.stream():
            await self.pipeline.process(raw)
        if adapter.health().status is AdapterStatus.EOF:
            await self.pipeline.flush()

    async def _tick_loop(self) -> None:
        while True:
            await self.clock.sleep(self.sources.ingest.tick_s)
            await self.pipeline.tick()

    # -- commands ----------------------------------------------------------------------------
    async def route_command(self, cmd: Command) -> CommandAck:
        """Send a command to the active adapter and wait for its ack (no bus publishing).

        A source without the commands capability answers REJECTED with a clear reason; no ack
        within ``command_timeout_s`` gives TIMEOUT.
        """
        adapter = self._adapter
        if adapter is None or Capability.COMMANDS not in adapter.capabilities:
            kind = adapter.kind.value if adapter is not None else "no"
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.REJECTED,
                reason=f"the active source ({kind} adapter) does not accept commands",
                ts=self.clock.now(),
            )
        try:
            return await asyncio.wait_for(
                adapter.send_command(cmd), timeout=self.sources.command_timeout_s
            )
        except TimeoutError:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.TIMEOUT,
                reason=f"no ack within {self.sources.command_timeout_s:g} s",
                ts=self.clock.now(),
            )

    async def send_command(self, cmd: Command) -> CommandAck:
        """Route a command and publish it and its ack on the bus (``Topic.COMMANDS``)."""
        await self.bus.publish(Topic.COMMANDS, cmd)
        ack = await self.route_command(cmd)
        await self.bus.publish(Topic.COMMANDS, ack)
        return ack

    def supports(self, capability: Capability) -> bool:
        adapter = self._adapter
        return adapter is not None and capability in adapter.capabilities

    async def whatif(self, req: WhatIfSimRequest, timeout_s: float) -> WhatIfSimResult:
        """Simulation only (ADR 0031): fork the simulator behind the active source."""
        adapter = self._adapter
        fork = getattr(adapter, "whatif", None)
        if adapter is None or fork is None or Capability.WHATIF not in adapter.capabilities:
            raise WhatIfUnsupportedError(adapter.kind.value if adapter is not None else "no")
        result: WhatIfSimResult = await fork(req, timeout_s)
        return result

    async def send_scenario(self, req: ScenarioRequest) -> ScenarioAck:
        """Simulation only (ADR 0015): forward to the active adapter (no bus publishing)."""
        adapter = self._adapter
        send = getattr(adapter, "send_scenario", None)
        if adapter is None or send is None or Capability.SCENARIO not in adapter.capabilities:
            kind = adapter.kind.value if adapter is not None else "no"
            return ScenarioAck(
                request_id=req.request_id,
                status=AckStatus.REJECTED,
                reason=f"the active source ({kind} adapter) does not support scenario injection "
                "(a simulation-only capability)",
                ts=self.clock.now(),
            )
        try:
            ack: ScenarioAck = await asyncio.wait_for(
                send(req), timeout=self.sources.command_timeout_s
            )
        except TimeoutError:
            ack = ScenarioAck(
                request_id=req.request_id,
                status=AckStatus.TIMEOUT,
                reason=f"no ack within {self.sources.command_timeout_s:g} s",
                ts=self.clock.now(),
            )
        return ack
