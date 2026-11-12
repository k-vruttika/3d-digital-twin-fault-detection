# LineTwin architecture

This is the Mermaid version of the diagram in `CLAUDE.md` section 3. The services (TimescaleDB, Mosquitto) run via Docker Compose (`uv run poe up`), with native installs as a fallback (see [ADR 0005](adr/0005-docker-compose-services.md) and [setup.md](setup.md)). Minimal mode never needs them.

## Data path

```mermaid
flowchart LR
    subgraph SIM["Simulator (4-machine line, physics-based degradation,<br/>snapshot / restore / fast-forward)"]
        FILE["File export<br/>CSV / JSON / Parquet"]
        MOCK["Mock REST/WS<br/>machine server"]
        MQTTP["MQTT publisher<br/>(Mosquitto, Docker Compose)"]
        OPCUA["OPC UA server<br/>(asyncua)"]
    end

    subgraph ING["Ingest"]
        ADAPT["Adapter layer<br/>(+ chaos wrapper)"]
        NORM["Normalizer"]
        VAL["Validator"]
        DEDUP["Dedup / reorder"]
        QUAR[("Quarantine")]
    end

    BUS(["Event bus<br/>(in-process async pub/sub)"])

    STORE[("Event store<br/>TimescaleDB | SQLite")]
    ENGINE["Twin engine<br/>state machines, line propagation"]
    INFER["Inference service<br/>models, calibration, conformal,<br/>anomaly, attributions, drift"]
    WS["WS gateway"]
    FE["Frontend<br/>TwinEngine (Three.js) + React dashboard"]

    FILE --> ADAPT
    MOCK --> ADAPT
    MQTTP --> ADAPT
    OPCUA --> ADAPT
    ADAPT --> NORM --> VAL --> DEDUP --> BUS
    VAL -- invalid --> QUAR
    BUS --> STORE
    BUS --> ENGINE
    BUS --> INFER
    BUS --> WS --> FE
```

## Ingest (Phase 2)

Implemented in `backend/twin_backend/adapters/`, `ingest/` and `bus/`. The semantics are described in [ADR 0007](adr/0007-adapters-and-ingest.md), the bus in [ADR 0006](adr/0006-event-bus-interface.md), and the wire formats in [api.md](api.md).

```mermaid
flowchart LR
    SM["SourceManager<br/>active adapter from sources.yaml,<br/>hot switch"] --> AD["Adapter<br/>file | mock | mqtt | opcua<br/>(optional ChaosAdapter)"]
    AD -- RawMessage --> N["Normalizer<br/>field mapping, units"]
    N --> V["Validator<br/>schema, machine, missing values, ranges"]
    V -- reject + reason --> Q[("Quarantine sink")]
    V --> DD["Dedup<br/>(machineId, seq)"] --> RB["Reorder buffer<br/>lateness window; LATE flag"]
    RB -- readings --> BUS(["Event bus"])
    ST["Staleness monitor<br/>STALE / OFFLINE"] -- states --> BUS
    TICK["tick (ingest.tick_s)"] --> RB & ST
```

- **Adapters** reconnect with exponential backoff. Every stage is driven by an injectable clock, so tests advance a fake clock instead of sleeping.
- **Metrics** are Prometheus counters and an end-to-end latency histogram, kept per pipeline registry. They will be served at `/metrics` from Phase 4.
- **Commands:** `SourceManager.send_command` forwards a command to the active adapter if that adapter has the `commands` capability. Commands and acks are also published on the `commands` topic.
- **Event loop:** all async entry points run on `twin_common.aio.run`, which uses a selector loop on Windows ([ADR 0008](adr/0008-windows-selector-event-loop.md)).

## Storage and event sourcing (Phase 3)

Implemented in `backend/twin_backend/storage/` and `engine/line_state.py`. The decisions are in [ADR 0009](adr/0009-event-store-and-migrations.md) (store, TimescaleDB features, migrations) and [ADR 0010](adr/0010-event-time-replay-and-snapshots.md) (event-time axis, line-state fold, snapshots, reconstruct).

```mermaid
flowchart LR
    BUS(["Event bus"]) -- "readings, states, predictions,<br/>commands, alerts" --> W["StorageWriter<br/>own bounded queues, batches,<br/>retries, dropped-writes metric"]
    Q["Quarantine tee"] --> W
    W --> ES[("EventStore<br/>SQLite (minimal mode) |<br/>TimescaleDB hypertables")]
    ES --> AGG["History<br/>1 s raw | 1 min | 10 min<br/>(continuous aggregates)"]
    ES --> SNAP["Snapshotter<br/>every interval_s of event time"]
    SNAP --> ES
    ES --> REC["reconstruct(t)<br/>snapshot + event replay"]
    ES --> EXP["Export CSV / JSONL<br/>(replayable by the file adapter)"]
    BUS --> LIVE["LiveStateTracker<br/>(same fold as reconstruct)"]
```

- **Backend selection:** `config/storage.yaml`, `backend: sqlite` (the default, needs no service) or `timescale`. `twin_backend.runtime.build_runtime()` loads `line.yaml`, `sources.yaml` and `storage.yaml` and wires the bus, storage and `SourceManager`. With storage present, live ingestion persists automatically.
- **Never blocking:** the writer reads from its own bus subscriptions. When it falls behind, the oldest rows are dropped and counted (`linetwin_storage_dropped_writes_total`). Ingestion never waits for the database.
- **One time axis:** every stored event carries its event time `ts`. Staleness state changes are stamped on the same axis (scaled by the replay speed). `ingested_at` (wall clock) is metadata only.
- **Time travel:** `reconstruct(t)` folds snapshot + events with the same order-independent reducer the live tracker uses, so it equals the live state for the same data. LATE readings are stored but never applied.
- **Metrics:** the storage metrics share the ingest metrics registry (served at `/metrics` from Phase 4).
- **Measured write throughput:** see `experiments/results/storage_bench.md` (`uv run poe storage-bench`).

## Command path

```mermaid
flowchart LR
    D["Dashboard"] -->|POST /commands| S["Safety rules"] --> CB["Command bus"] --> A["Active adapter<br/>send_command"] --> M["Machine"]
    M -->|CommandAck| CB
```

## What-if path

```mermaid
flowchart LR
    D["Dashboard"] -->|POST /whatif| SN["Simulator snapshot"] --> FF["Fast-forward copy"] --> MDL["Model"] --> T["Projected trajectory"]
```

## Minimal mode

The minimal path is `File export` or `Mock server` → adapter → ingest → bus → twin engine → WS gateway → frontend, with the SQLite event store (`config/storage.yaml` ships with `backend: sqlite`) and no broker, TimescaleDB, Docker or ML. It must keep working in every phase (CLAUDE.md section 1).

## Shared contracts

All wire formats are defined once in `common/twin_common/contracts.py`:

- `Reading` (schemaVersion 2; sensor values nullable, `null` = missing)
- `StateChange` (bus topic `states`; `ts` on the event-time axis)
- `Prediction`
- `Attribution`
- `Command`
- `CommandAck`
- `Alert` (bus topic `alerts`; stored from Phase 3, produced from the inference phases)

JSON is camelCase and Python attributes are snake_case. Timestamps must be timezone-aware and are normalized to UTC. Unknown fields are rejected.

Line layout, data sources and storage come from `config/line.yaml`, `config/sources.yaml` and `config/storage.yaml`, which are validated by `twin_common.config`.
