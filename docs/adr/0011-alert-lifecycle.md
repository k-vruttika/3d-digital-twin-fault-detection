# ADR 0011: Alert lifecycle (ISA-18.2) as append-only events

- **Status:** Accepted
- **Date:** 2026-11-13
- **Phase:** 4 (twin engine, API, commands)

## Context

Phase 3 stored `Alert` events, but nothing produced them, and the contract had no way to say that an alert had ended or that an operator had seen it. Phase 4 needs alerts that are raised, deduplicated, cleared when the condition ends, and acknowledged by operators. They must be stored like every other event and survive a backend restart.

## Decision

### Two independent states (ISA-18.2)

`Alert` gains `active: bool`, `acknowledged: bool`, `ackedBy`, `ackedAt` and `rev: int`. Active and acknowledged are independent, as in ISA-18.2 alarm management:

- an alert can be acknowledged while it is active, or cleared before it is acknowledged;
- a cleared but unacknowledged alert stays listed until someone acknowledges it;
- an alert is **resolved** (no longer listed) once it is both cleared and acknowledged.

### Every transition is a new event

Raise, clear, acknowledge and re-raise each publish a new `Alert` with the same `alertId` and the next `rev`. The current alert is the event with the highest `rev`. This keeps the event store append-only and makes "current alerts" a fold over events, like the line state (ADR 0010). `rev` makes that fold unambiguous when two transitions share a timestamp.

- **Unique key:** `(alert_id, rev, ts)`. TimescaleDB requires the partitioning column `ts` in every unique key. Migrations: SQLite `0002_alert_lifecycle` (table rebuilt) and TimescaleDB `0004_alert_lifecycle`.
- **Timestamps:** `ts` is event time (ADR 0010). For an acknowledgement it is the machine's watermark at that moment, and `ackedAt` records the wall-clock time.
- **Deterministic ids:** `{machine}:{kind}:{raise ts in µs}`, so a replay of the same data raises the same ids.
- **Re-raise:** if the condition returns while the alert is cleared but not yet acknowledged, the same alert becomes active again (no debounce: the condition was real). After clear and acknowledge, a new occurrence gets a new id.

### Engine alerts

`AlertManager` (`twin_backend/engine/alerts.py`) keeps one condition per (machine, kind) and returns the events; the twin engine publishes them on `Topic.ALERTS`. Kinds, severities and the event-time debounce `raise_after_s` are set per kind in `config/engine.yaml`: `machine_warning`, `machine_fault`, `machine_offline`, `data_quality` and `line_stoppage`. `line_stoppage` is raised on the root-cause machine of STARVED/BLOCKED neighbours (ADR 0012) and is debounced, because short buffer hiccups are normal.

### Restart

On startup the runtime folds the stored alert events of source `engine` and restores the unresolved ones. A restored alert is **not re-evaluated until the machine's state is known again**, meaning a reading or StateChange has arrived since startup. For `line_stoppage`, which depends on the neighbours along the propagation chain, the whole line must be known. For `data_quality`, at least `min_samples` messages must have been seen. Otherwise the empty state after a restart would clear the alert spuriously and raise it again with a new id.

## Consequences

- Alert history is complete and replayable: `GET /alerts/history` returns every transition.
- `GET /alerts` and the WebSocket snapshot show only unresolved alerts, so the operator list is short.
- Producers other than the engine (the inference service in Phase 11) use the same contract and lifecycle with their own `source`.
