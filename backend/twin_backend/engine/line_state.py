"""Line state as a pure, order-independent fold over events (ADR 0010).

The same fold builds the live state (``LiveStateTracker``, from the event bus) and the
reconstructed state (``twin_backend.storage.replay.reconstruct``, from the event store), so the
two agree by construction once they have seen the same events, whatever the arrival order.

Rules, per machine:

- **last reading:** the reading with the largest ``(ts, seq)`` among readings whose quality is
  not LATE. LATE readings are stored but never applied to live state (ADR 0007).
- **last state change:** the StateChange with the largest ``(ts, state)``.
- **effective state:** whichever of the two is newer by ts; on a tie the StateChange wins
  (a recovery change carries the resuming reading's ts and the same state anyway).
- **predictions:** the newest prediction per horizon, by ``(ts, modelVersion)``.

Every rule takes a maximum, so applying events in any order gives the same state.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from twin_backend.bus.base import EventBus, Subscription, Topic
from twin_common.contracts import Prediction, Reading, StateChange
from twin_common.enums import MachineState, Quality


class _StateModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
    )


class MachineView(_StateModel):
    machine_id: str
    state: MachineState | None = None
    state_ts: datetime | None = None
    reported_state: MachineState | None = None
    last_reading: Reading | None = None
    last_state_change: StateChange | None = None


class LineState(_StateModel):
    """Full line state. ``as_of`` is set by whoever built it (reconstruct time, live watermark)."""

    as_of: datetime | None = None
    machines: dict[str, MachineView] = Field(default_factory=dict)
    # machine id -> horizon key (``f"{horizon_s:g}"``) -> newest prediction
    predictions: dict[str, dict[str, Prediction]] = Field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        """Everything except ``as_of``: what live and reconstructed states must agree on."""
        return self.model_dump(mode="json", exclude={"as_of"})

    def watermark(self) -> datetime | None:
        """Newest event ts applied to this state."""
        stamps: list[datetime] = []
        for view in self.machines.values():
            if view.last_reading is not None:
                stamps.append(view.last_reading.ts)
            if view.last_state_change is not None:
                stamps.append(view.last_state_change.ts)
        for per_horizon in self.predictions.values():
            stamps.extend(p.ts for p in per_horizon.values())
        return max(stamps) if stamps else None


StateEvent = Reading | StateChange | Prediction


def new_line_state(machine_ids: Iterable[str]) -> LineState:
    return LineState(machines={m: MachineView(machine_id=m) for m in machine_ids})


def ensure_machines(state: LineState, machine_ids: Iterable[str]) -> None:
    for m in machine_ids:
        if m not in state.machines:
            state.machines[m] = MachineView(machine_id=m)


def horizon_key(horizon_s: float) -> str:
    return f"{horizon_s:g}"


def _view(state: LineState, machine_id: str) -> MachineView:
    view = state.machines.get(machine_id)
    if view is None:
        view = state.machines[machine_id] = MachineView(machine_id=machine_id)
    return view


def _refresh_effective(view: MachineView) -> None:
    reading, change = view.last_reading, view.last_state_change
    view.reported_state = reading.state if reading is not None else None
    if change is not None and (reading is None or change.ts >= reading.ts):
        view.state, view.state_ts = change.state, change.ts
    elif reading is not None:
        view.state, view.state_ts = reading.state, reading.ts
    else:
        view.state, view.state_ts = None, None


def apply_event(state: LineState, event: StateEvent) -> bool:
    """Fold one event into ``state`` (in place). Returns True if the state changed."""
    if isinstance(event, Reading):
        if event.quality is Quality.LATE:
            return False
        view = _view(state, event.machine_id)
        last = view.last_reading
        if last is not None and (event.ts, event.seq) <= (last.ts, last.seq):
            return False
        view.last_reading = event
        _refresh_effective(view)
        return True
    if isinstance(event, StateChange):
        view = _view(state, event.machine_id)
        prev = view.last_state_change
        if prev is not None and (event.ts, event.state.value) <= (prev.ts, prev.state.value):
            return False
        view.last_state_change = event
        _refresh_effective(view)
        return True
    per_horizon = state.predictions.setdefault(event.machine_id, {})
    key = horizon_key(event.horizon_s)
    old = per_horizon.get(key)
    if old is not None and (event.ts, event.model_version) <= (old.ts, old.model_version):
        return False
    per_horizon[key] = event
    return True


def fold(state: LineState, events: Iterable[StateEvent]) -> LineState:
    for event in events:
        apply_event(state, event)
    return state


STATE_TOPICS = (Topic.READINGS, Topic.STATES, Topic.PREDICTIONS)


class LiveStateTracker:
    """Keeps the live LineState from the event bus (readings, state changes, predictions).

    ``drain()`` applies everything queued right now (deterministic tests); ``start()`` runs a
    background task that applies events as they arrive. ``on_apply`` is called after every
    applied event.
    """

    def __init__(
        self,
        bus: EventBus,
        machine_ids: Iterable[str],
        *,
        queue_size: int | None = None,
        on_apply: Callable[[StateEvent, LineState], None] | None = None,
    ) -> None:
        self.state = new_line_state(machine_ids)
        self._subs: list[Subscription] = [
            bus.subscribe(topic, maxsize=queue_size) for topic in STATE_TOPICS
        ]
        self._on_apply = on_apply
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def dropped(self) -> int:
        return sum(s.dropped for s in self._subs)

    def _apply(self, message: BaseModel) -> None:
        if isinstance(message, Reading | StateChange | Prediction):
            apply_event(self.state, message)
            self.state.as_of = self.state.watermark()
            if self._on_apply is not None:
                self._on_apply(message, self.state)

    def drain(self) -> int:
        n = 0
        for sub in self._subs:
            while (message := sub.get_nowait()) is not None:
                self._apply(message)
                n += 1
        return n

    async def _pump(self, sub: Subscription) -> None:
        async for message in sub:
            self._apply(message)

    def start(self) -> None:
        if not self._tasks:
            self._tasks = [asyncio.create_task(self._pump(s)) for s in self._subs]

    async def stop(self) -> None:
        for sub in self._subs:
            sub.close()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.drain()

    def snapshot(self) -> LineState:
        return self.state.model_copy(deep=True)
