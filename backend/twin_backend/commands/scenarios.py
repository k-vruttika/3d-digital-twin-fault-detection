"""Scenario injection path (simulation only, ADR 0015).

Real equipment would not offer this. ``POST /scenario`` asks the active simulator publisher to
inject a fault now or to schedule a scenario file's events. Every request and its ack are
published on ``Topic.SCENARIOS`` and stored (stream ``scenario_requests``), rejected ones
included, so injected faults can be told apart from real ones in replay and experiments.
``ts`` is on the event-time axis (the target machine's or the line's watermark).
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from prometheus_client import CollectorRegistry, Counter
from pydantic import Field

from twin_backend.adapters.manager import SourceManager
from twin_backend.bus.base import EventBus, Topic
from twin_backend.clock import Clock
from twin_backend.engine.line_state import StateModel
from twin_backend.engine.twin import TwinEngine
from twin_backend.storage.base import EventStore, Stream
from twin_common.contracts import ScenarioAck, ScenarioRequest
from twin_common.enums import Capability, ScenarioAction
from twin_common.logging import get_logger

log = get_logger(__name__)

RECENT_MAX = 1000  # in-memory buffer size, not a domain threshold


class ScenarioPayload(StateModel):
    """Body of ``POST /scenario``. The server adds requestId, requestedBy, source and ts."""

    action: ScenarioAction
    machine_id: str | None = Field(default=None, description="INJECT_FAULT: target machine")
    fault_type: str | None = Field(
        default=None, description="INJECT_FAULT: simulator fault, e.g. bearing_wear, belt_jam"
    )
    progression_s: float | None = Field(
        default=None, gt=0, description="INJECT_FAULT: gradual fault duration (default fastest)"
    )
    scenario: str | None = Field(
        default=None, description="LOAD_SCENARIO: scenario file name without .yaml"
    )
    seed: int | None = Field(default=None, ge=0, description="LOAD_SCENARIO: plan seed")


class ScenarioRecord(StateModel):
    request: ScenarioRequest
    acks: list[ScenarioAck] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    request: ScenarioRequest
    ack: ScenarioAck
    supported: bool  # False: the active source has no scenario capability


class ScenarioService:
    def __init__(
        self,
        manager: SourceManager,
        engine: TwinEngine,
        bus: EventBus,
        clock: Clock,
        store: EventStore | None,
        registry: CollectorRegistry | None,
    ) -> None:
        self.manager = manager
        self.engine = engine
        self.bus = bus
        self.clock = clock
        self.store = store
        self.requests = Counter(
            "linetwin_scenario_requests",
            "Scenario injection requests (simulation only)",
            ["source", "action", "status"],
            registry=registry,
        )
        # Requests of this process (the store lags by up to one writer flush).
        self._recent: OrderedDict[str, ScenarioRecord] = OrderedDict()

    def build(self, payload: ScenarioPayload, requested_by: str) -> ScenarioRequest:
        """Raises ``ValueError`` (pydantic) for an inconsistent request."""
        kind = self.manager.active_kind
        return ScenarioRequest(
            request_id=uuid.uuid4().hex,
            action=payload.action,
            machine_id=payload.machine_id,
            fault_type=payload.fault_type,
            progression_s=payload.progression_s,
            scenario=payload.scenario,
            seed=payload.seed,
            requested_by=requested_by,
            source=kind.value if kind is not None else "none",
            ts=self._event_now(payload.machine_id),
        )

    async def submit(self, req: ScenarioRequest) -> ScenarioResult:
        await self.bus.publish(Topic.SCENARIOS, req)
        record = ScenarioRecord(request=req)
        self._recent[req.request_id] = record
        while len(self._recent) > RECENT_MAX:
            self._recent.popitem(last=False)
        supported = self.manager.supports(Capability.SCENARIO)
        result = await self.manager.send_scenario(req)
        ack = result.model_copy(update={"ts": max(self._event_now(req.machine_id), req.ts)})
        await self.bus.publish(Topic.SCENARIOS, ack)
        record.acks.append(ack)
        self.requests.labels(
            source=req.source, action=req.action.value, status=ack.status.value
        ).inc()
        log.info(
            "scenario.request",
            request_id=req.request_id,
            action=req.action.value,
            machine=req.machine_id,
            fault=req.fault_type,
            scenario=req.scenario,
            status=ack.status.value,
            reason=ack.reason,
            by=req.requested_by,
        )
        return ScenarioResult(req, ack, supported)

    def _event_now(self, machine_id: str | None) -> datetime:
        ts = (
            self.engine.machine_watermark(machine_id)
            if machine_id is not None and machine_id in self.engine.machine_ids
            else None
        ) or self.engine.state.watermark()
        return ts if ts is not None else self.clock.now()

    async def history(
        self, t0: datetime | None, t1: datetime | None, machine_id: str | None = None
    ) -> list[ScenarioRecord]:
        records: dict[str, ScenarioRecord] = {}
        if self.store is not None:
            rows = [s.item for s in await self.store.fetch(Stream.SCENARIO_REQUESTS, t0, None)]
            acks: dict[str, list[ScenarioAck]] = {}
            for r in rows:
                if isinstance(r, ScenarioAck):
                    acks.setdefault(r.request_id, []).append(r)
            for r in rows:
                if isinstance(r, ScenarioRequest):
                    records[r.request_id] = ScenarioRecord(
                        request=r, acks=acks.get(r.request_id, [])
                    )
        records.update(self._recent)  # this process's records are complete and current
        out = [
            r
            for r in records.values()
            if (t0 is None or r.request.ts >= t0)
            and (t1 is None or r.request.ts < t1)
            and (machine_id is None or r.request.machine_id == machine_id)
        ]
        return sorted(out, key=lambda r: (r.request.ts, r.request.request_id))

    async def get(self, request_id: str) -> ScenarioRecord | None:
        if request_id in self._recent:
            return self._recent[request_id]
        for record in await self.history(None, None):
            if record.request.request_id == request_id:
                return record
        return None
