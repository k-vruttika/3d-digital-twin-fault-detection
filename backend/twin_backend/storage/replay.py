"""Time travel: rebuild the full line state at any event time from the store (ADR 0010).

``reconstruct(t)`` = the newest snapshot at or before ``t`` (if any), then every stored
reading, state change and prediction with ``snapshot.ts < ts <= t`` folded in with the same
reducer the live tracker uses (``twin_backend.engine.line_state``). LATE readings are stored
but skipped by the reducer, exactly as live state skips them (ADR 0007).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta

from twin_backend.engine.line_state import (
    LineState,
    apply_event,
    ensure_machines,
    new_line_state,
)
from twin_backend.storage.base import STATE_STREAMS, EventStore


async def reconstruct(
    store: EventStore,
    machine_ids: Iterable[str],
    t: datetime,
    *,
    use_snapshots: bool = True,
) -> LineState:
    machines = list(machine_ids)
    snap = await store.latest_snapshot(t) if use_snapshots else None
    if snap is not None:
        state = LineState.model_validate_json(snap.state_json)
        ensure_machines(state, machines)
        after: datetime | None = snap.ts
    else:
        state = new_line_state(machines)
        after = None
    for stored in await store.fetch_replay(STATE_STREAMS, after, t):
        apply_event(state, stored.item)
    state.as_of = t
    return state


async def iter_states(
    store: EventStore,
    machine_ids: Iterable[str],
    t0: datetime,
    t1: datetime,
    step: timedelta,
) -> AsyncIterator[LineState]:
    """States at ``t0, t0 + step, ... <= t1``; each step only replays the events since the last."""
    if step <= timedelta(0):
        raise ValueError("step must be positive")
    state = await reconstruct(store, machine_ids, t0)
    yield state.model_copy(deep=True)
    t = t0
    while t + step <= t1:
        nxt = t + step
        for stored in await store.fetch_replay(STATE_STREAMS, t, nxt):
            apply_event(state, stored.item)
        state.as_of = nxt
        yield state.model_copy(deep=True)
        t = nxt
