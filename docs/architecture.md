# LineTwin architecture

This is the Mermaid version of the diagram in `CLAUDE.md` section 3. All services run natively on Windows; there is no Docker (see [ADR 0002](adr/0002-native-services-no-docker.md)).

## Data path

```mermaid
flowchart LR
    subgraph SIM["Simulator (4-machine line, physics-based degradation,<br/>snapshot / restore / fast-forward)"]
        FILE["File export<br/>CSV / JSON / Parquet"]
        MOCK["Mock REST/WS<br/>machine server"]
        MQTTP["MQTT publisher<br/>(native Mosquitto)"]
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

    STORE[("Event store<br/>PostgreSQL | SQLite")]
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

The minimal path is `File export` or `Mock server` → adapter → ingest → bus → twin engine → WS gateway → frontend, with the SQLite (or no) event store and no broker, PostgreSQL or ML. It must keep working in every phase (CLAUDE.md section 1).

## Shared contracts

All wire formats are defined once in `common/twin_common/contracts.py`:

- `Reading` (schemaVersion 2)
- `Prediction`
- `Attribution`
- `Command`
- `CommandAck`

JSON is camelCase and Python attributes are snake_case. Timestamps must be timezone-aware and are normalized to UTC. Unknown fields are rejected.

Line layout and data sources come from `config/line.yaml` and `config/sources.yaml`, which are validated by `twin_common.config`.
