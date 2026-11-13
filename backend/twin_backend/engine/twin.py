"""TwinEngine: live line state plus everything derived from it (ADR 0012).

The engine *observes* the LineState fold (``LiveStateTracker``) and never changes how it folds,
so the live state still equals ``reconstruct(t)``. On top of the fold it keeps:

- a state machine per machine (allowed transitions from ``config/engine.yaml``);
- line propagation: the root cause of every STARVED/BLOCKED machine (``propagation.py``);
- per-machine data quality (``data_quality.py``);
- alerts (``alerts.py``), published on ``Topic.ALERTS``.

When a *reading* changes a machine's effective state, the engine publishes a ``StateChange``
with a reason (e.g. ``"BLOCKED by CNC-01"``). It has the reading's ts and state, so folding it
changes nothing, and a duplicate of a pipeline StateChange (same machine, ts, state) collapses
into one stored row. StateChanges from the pipeline (STALE, OFFLINE, recovery) are not
re-emitted.

Derived views (line view, data quality) are not bus contracts; WebSocket clients get them
through ``add_listener``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge
from pydantic import BaseModel

from twin_backend.bus.base import EventBus, Subscription, Topic
from twin_backend.clock import Clock
from twin_backend.engine.alerts import AckResult, AlertManager
from twin_backend.engine.data_quality import DataQualityMonitor, DataQualityReport
from twin_backend.engine.line_state import LineState, LiveStateTracker, StateEvent, StateModel
from twin_backend.engine.propagation import LineStatus, LineView, line_view
from twin_backend.engine.state_machine import StateMachines
from twin_backend.engine.topology import Topology
from twin_backend.ingest.quarantine import QuarantineRecord
from twin_common.config.engine import AlertKind, EngineConfig
from twin_common.config.line import LineConfig
from twin_common.contracts import Alert, Reading, StateChange
from twin_common.enums import DataQuality, MachineState
from twin_common.logging import get_logger

log = get_logger(__name__)

_LINK_DOWN = (MachineState.STALE, MachineState.OFFLINE)
_STATE_ALERTS: dict[AlertKind, MachineState] = {
    AlertKind.MACHINE_WARNING: MachineState.WARNING,
    AlertKind.MACHINE_FAULT: MachineState.FAULT,
    AlertKind.MACHINE_OFFLINE: MachineState.OFFLINE,
}

Listener = Callable[[str, BaseModel], None]


class RootCauseView(StateModel):
    cause: str | None = None
    unknown: bool = False


class LineViewModel(StateModel):
    """Serialisable line view (API ``/machines``, WS channel ``line``)."""

    status: LineStatus
    root_causes: dict[str, RootCauseView]
    impacted: dict[str, list[str]]

    @classmethod
    def of(cls, view: LineView) -> LineViewModel:
        return cls(
            status=view.status,
            root_causes={
                m: RootCauseView(cause=rc.cause, unknown=rc.unknown)
                for m, rc in view.root_causes.items()
            },
            impacted=view.impacted,
        )


class EngineMetrics:
    def __init__(self, registry: CollectorRegistry | None) -> None:
        self.alert_events = Counter(
            "linetwin_engine_alert_events",
            "Alert lifecycle events published by the engine",
            ["kind", "transition"],
            registry=registry,
        )
        self.alerts_unresolved = Gauge(
            "linetwin_engine_alerts_unresolved",
            "Alerts that are active or not yet acknowledged",
            registry=registry,
        )
        self.state_changes = Counter(
            "linetwin_engine_state_changes",
            "StateChange events published by the engine",
            registry=registry,
        )
        self.degraded = Gauge(
            "linetwin_engine_data_quality_degraded",
            "1 if the machine's data quality is DEGRADED",
            ["machine"],
            registry=registry,
        )


class TwinEngine:
    def __init__(
        self,
        line: LineConfig,
        cfg: EngineConfig,
        bus: EventBus,
        clock: Clock,
        *,
        registry: CollectorRegistry | None = None,
        queue_size: int | None = None,
    ) -> None:
        self.line = line
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.topology = Topology.from_line(line)
        self.machine_ids = list(self.topology.machine_ids)
        self.state_machines = StateMachines(self.machine_ids, cfg, registry)
        self.dq = DataQualityMonitor(self.machine_ids, cfg.data_quality, clock)
        self.alerts = AlertManager(cfg.alerts)
        self.metrics = EngineMetrics(registry)
        self.tracker = LiveStateTracker(
            bus, self.machine_ids, queue_size=queue_size, on_apply=self._on_apply
        )
        self._quarantine: Subscription = bus.subscribe(Topic.QUARANTINE, maxsize=queue_size)
        self._outbox: list[tuple[Topic, BaseModel]] = []
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._listeners: list[Listener] = []
        self._line = line_view({}, self.topology)
        self._dq: dict[str, DataQualityReport] = {
            m: DataQualityReport(machine_id=m) for m in self.machine_ids
        }
        # Machines with a reading or StateChange since this engine started. Alert conditions
        # (including clearing a restored alert) are only evaluated on fresh state.
        self._known: set[str] = set()

    # -- views -------------------------------------------------------------------------------
    @property
    def state(self) -> LineState:
        return self.tracker.state

    def snapshot(self) -> LineState:
        return self.tracker.snapshot()

    def line_view(self) -> LineView:
        return self._line

    def line_model(self) -> LineViewModel:
        return LineViewModel.of(self._line)

    def data_quality(self, machine_id: str) -> DataQualityReport:
        return self._dq[machine_id]

    def data_quality_all(self) -> dict[str, DataQualityReport]:
        return dict(self._dq)

    def machine_state(self, machine_id: str) -> MachineState | None:
        view = self.state.machines.get(machine_id)
        return view.state if view is not None else None

    def machine_watermark(self, machine_id: str) -> datetime | None:
        view = self.state.machines.get(machine_id)
        if view is None:
            return None
        stamps = [e.ts for e in (view.last_reading, view.last_state_change) if e is not None]
        return max(stamps) if stamps else None

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    # -- lifecycle ---------------------------------------------------------------------------
    def load_alerts(self, events: list[Alert]) -> None:
        n = self.alerts.load(events)
        self.metrics.alerts_unresolved.set(n)
        if n:
            log.info("engine.alerts_restored", unresolved=n)

    def start(self) -> None:
        if self._tasks:
            return
        self.tracker.start()
        self._tasks = [
            asyncio.create_task(self._publish_loop(), name="engine-publish"),
            asyncio.create_task(self._quarantine_loop(), name="engine-quarantine"),
            asyncio.create_task(self._tick_loop(), name="engine-tick"),
        ]

    async def stop(self) -> None:
        """Apply and publish everything still queued, then stop (before storage drains)."""
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.drain()  # the engine's own StateChanges come back and are folded too
        await self.tracker.stop()
        self._quarantine.close()
        self._drain_quarantine()
        await self.flush()

    async def drain(self, max_rounds: int = 10) -> None:
        """Apply everything queued now and publish the results until nothing is left
        (deterministic tests; shutdown)."""
        for _ in range(max_rounds):
            n = self.tracker.drain()
            self._drain_quarantine()
            pending = len(self._outbox)
            await self.flush()
            if n == 0 and pending == 0:
                break

    async def flush(self) -> None:
        out, self._outbox = self._outbox, []
        for topic, message in out:
            await self.bus.publish(topic, message)

    async def _publish_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await self.flush()

    async def _quarantine_loop(self) -> None:
        async for record in self._quarantine:
            self._on_quarantine(record)

    def _drain_quarantine(self) -> None:
        while (record := self._quarantine.get_nowait()) is not None:
            self._on_quarantine(record)

    def _on_quarantine(self, record: BaseModel) -> None:
        if isinstance(record, QuarantineRecord):
            self.dq.add_quarantined(record.machine_id)

    async def _tick_loop(self) -> None:
        while True:
            await self.clock.sleep(self.cfg.data_quality.evaluate_every_s)
            self.tick()

    # -- core --------------------------------------------------------------------------------
    def _emit(self, topic: Topic, message: BaseModel) -> None:
        self._outbox.append((topic, message))
        self._wake.set()

    def _notify(self, channel: str, payload: BaseModel) -> None:
        for listener in list(self._listeners):
            try:
                listener(channel, payload)
            except Exception as exc:  # a broken listener must not stop the engine
                log.error("engine.listener_failed", channel=channel, error=repr(exc))

    def _on_apply(self, event: StateEvent, state: LineState) -> None:
        machine_id = event.machine_id
        if isinstance(event, Reading):
            self.dq.add_reading(event)
        if isinstance(event, Reading | StateChange) and machine_id in self.machine_ids:
            self._known.add(machine_id)
        view = state.machines.get(machine_id)
        if view is None:
            return
        transition = self.state_machines.observe(machine_id, view.state)
        if transition is not None:
            self.dq.set_link_down(machine_id, transition.state in _LINK_DOWN)
            new_line = line_view({m: v.state for m, v in state.machines.items()}, self.topology)
            if new_line != self._line:
                self._line = new_line
                self._notify("line", self.line_model())
            if isinstance(event, Reading) and view.state_ts is not None:
                self._emit(
                    Topic.STATES,
                    StateChange(
                        machine_id=machine_id,
                        state=transition.state,
                        previous_state=transition.previous,
                        ts=view.state_ts,
                        reason=self._reason(machine_id, transition.state),
                    ),
                )
                self.metrics.state_changes.inc()
        if isinstance(event, Reading | StateChange):
            self._evaluate_alerts(state)

    def _reason(self, machine_id: str, state: MachineState) -> str:
        rc = self._line.root_causes.get(machine_id)
        if state in (MachineState.STARVED, MachineState.BLOCKED) and rc is not None and rc.cause:
            suffix = " (its state is unknown)" if rc.unknown else ""
            return f"{state.value} by {rc.cause}{suffix}"
        return "reported by source"

    def _evaluate_alerts(self, state: LineState) -> None:
        """Re-evaluate the state-based alert conditions on fresh state only.

        A machine's own alerts wait until that machine's state is known since startup. A line
        stoppage depends on the neighbours along the propagation chain, so it waits until every
        machine's state is known. This keeps alerts restored after a restart (ADR 0011) active,
        with the same id, until fresh data confirms or ends their condition.
        """
        events: list[Alert] = []
        watermark = state.as_of
        line_known = self._known.issuperset(self.machine_ids)
        for m in self.machine_ids:
            view = state.machines[m]
            if m not in self._known or view.state is None or view.state_ts is None:
                continue
            for kind, alert_state in _STATE_ALERTS.items():
                events += self.alerts.update(
                    m,
                    kind,
                    view.state is alert_state,
                    view.state_ts,
                    f"{m} is {alert_state.value}",
                )
            impacted = self._line.impacted.get(m, [])
            if line_known and watermark is not None:
                events += self.alerts.update(
                    m,
                    AlertKind.LINE_STOPPAGE,
                    bool(impacted),
                    watermark,
                    self._stoppage_message(m, view.state, impacted, state),
                )
        self._publish_alerts(events)

    def _stoppage_message(
        self, machine_id: str, st: MachineState, impacted: list[str], state: LineState
    ) -> str:
        parts = []
        for m in impacted:
            ms = state.machines[m].state
            parts.append(f"{m} ({ms.value if ms is not None else '?'})")
        return f"{machine_id} is {st.value} and stops {', '.join(parts)}"

    def _publish_alerts(self, events: list[Alert], transition: str | None = None) -> None:
        for a in events:
            label = transition or ("raise" if a.active else "clear")
            self.metrics.alert_events.labels(kind=a.kind, transition=label).inc()
            self._emit(Topic.ALERTS, a)
        if events:
            self.metrics.alerts_unresolved.set(len(self.alerts.current()))

    def tick(self) -> None:
        """Re-evaluate data quality (wall clock) and its alerts."""
        reports = self.dq.evaluate()
        events: list[Alert] = []
        for m, report in reports.items():
            if report != self._dq.get(m):
                self._dq[m] = report
                self._notify("dataQuality", report)
            degraded = report.status is DataQuality.DEGRADED
            self.metrics.degraded.labels(machine=m).set(1 if degraded else 0)
            ts = self.machine_watermark(m)
            # Judge only once enough fresh data was seen since startup: right after a restart
            # the empty window reads "OK" and must not clear a restored data-quality alert.
            judgeable = report.link_down or self.dq.seen(m) >= self.cfg.data_quality.min_samples
            if m not in self._known or ts is None or not (judgeable or degraded):
                continue
            events += self.alerts.update(
                m,
                AlertKind.DATA_QUALITY,
                degraded,
                ts,
                f"{m} data quality DEGRADED: {'; '.join(report.reasons) or 'recovering'}",
            )
        self._publish_alerts(events)

    def acknowledge(self, alert_id: str, by: str) -> AckResult:
        alert = self.alerts.get(alert_id)
        if alert is None:
            return AckResult(None, False)
        ts = self.machine_watermark(alert.machine_id) or alert.ts
        result = self.alerts.acknowledge(alert_id, by, ts, self.clock.now())
        if result.changed and result.alert is not None:
            self._publish_alerts([result.alert], "ack")
        return result

    def is_known(self, machine_id: str) -> bool:
        """True once the machine has had a reading or StateChange since startup."""
        return machine_id in self._known

    def debug_counts(self) -> dict[str, Any]:
        return {
            "invalidTransitions": self.state_machines.invalid_count(),
            "trackerDropped": self.tracker.dropped,
        }
