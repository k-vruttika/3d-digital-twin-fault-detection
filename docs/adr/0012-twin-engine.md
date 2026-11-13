# ADR 0012: Twin engine as an observer of the line-state fold

- **Status:** Accepted
- **Date:** 2026-11-13
- **Phase:** 4 (twin engine, API, commands)

## Context

Phase 3 built the line state as an order-independent fold that the live tracker and `reconstruct(t)` share, so the two agree by construction (ADR 0010). Phase 4 adds per-machine state machines, line propagation with root causes, data-quality status and alerts. None of these may break that guarantee.

## Decision

### The engine observes; it never changes the fold

`TwinEngine` (`twin_backend/engine/twin.py`) wraps `LiveStateTracker` through its `on_apply` hook. Everything it derives sits beside the fold:

- **State machine** (`state_machine.py`): allowed transitions for all nine `MachineState`s come from `config/engine.yaml`, and the loader requires an entry for every state. A transition that is not allowed is applied anyway, logged (`engine.invalid_transition`) and counted (`linetwin_engine_invalid_transitions_total{machine,from_state,to_state}`). The data is the truth, and at 1 Hz sampling an operator's RESET + START legitimately shows as FAULT → RUNNING. The shipped table accepts every transition in simulated runs (full line with operator stop and faults, and the single station). The test suite checks this.
- **Topology** (`topology.py`): upstream/downstream come from the line config's buffers. Without buffers, the machine order is the flow order, which is how `line_single.yaml` works.
- **Propagation** (`propagation.py`): the root cause of a STARVED machine follows the upstream links, and of a BLOCKED machine the downstream links. It stops at a RUNNING machine (the bottleneck), at a machine stopped for another reason (the cause), or at `SOURCE` / `SINK`. This is the simulator's rule, and the test suite checks that both agree on every sample of generated runs. The twin adds one case: a neighbour whose link is down (STALE, OFFLINE) or not yet known is reported as the cause with `unknown: true`.
- **Data quality** (`data_quality.py`): OK or DEGRADED per machine. It is computed over a rolling wall-clock window of late %, quarantine % and partial-reading %, and reports DEGRADED while the machine's link is down. Thresholds, `min_samples` and the recovery hysteresis come from config. Quarantined messages reach the engine on a new bus topic `quarantine`. The store keeps getting them through its quarantine sink, not from that topic.
- **Alerts** (ADR 0011).

### Engine StateChanges

When a **reading** changes a machine's effective state, the engine publishes a `StateChange` with a reason, for example `"BLOCKED by CNC-01"` or `"reported by source"`. It carries the reading's ts and state, so folding it changes nothing (a tie goes to the StateChange, with the same state). A duplicate of a pipeline StateChange has the same `(machine, ts, state)` key and collapses into one stored row. StateChanges that the pipeline caused (STALE, OFFLINE, recovery) are not re-emitted. Live state and `reconstruct` still agree. This is tested on both store backends with the engine running.

### Shutdown and restart

- **Shutdown:** `SourceManager.stop(before_storage=...)` lets the engine apply and publish its last events, including its own StateChanges coming back through the bus, before the writer drains. Otherwise the stored and live states would differ at the end.
- **Restart:** the live state is **not** seeded from the store, because a restarted file replay would deliver older timestamps that the max-fold ignores. Only alerts are restored (ADR 0011), and they wait for fresh data.

## Consequences

- `reconstruct(t)` stays exact. Derived views (root causes, data quality) are live-only and are recomputed from the state. They are not stored, except through the StateChange reasons and the alerts.
- A WebSocket client gets the line view and data quality through engine listeners, because they are not bus contracts.
