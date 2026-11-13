# LineTwin interfaces

This page covers the backend API (Phase 4: REST, WebSocket stream, commands, alerts), what the simulator publishes and what the backend adapters consume (Phase 2), and the event-store export and CLI (Phase 3).

All payloads use the canonical contracts from `twin_common.contracts`: camelCase JSON and UTC ISO-8601 timestamps.

## Backend API (Phase 4)

### Running it

```text
uv run poe sim-quick; uv run poe minimal     # file replay + SQLite: no Docker, broker or ML
uv run poe up; uv run poe dev                # source from sources.yaml + TimescaleDB, with the
                                             # matching simulator publisher started alongside
uv run poe dev --source opcua --speed 5      # override source / simulator speed
uv run python -m twin_backend serve --help   # all options (--mode, --line, --source, --port, ...)
```

The server binds to `api.yaml` `host`/`port` (default `127.0.0.1:8000`) and runs uvicorn inside `twin_common.aio.run()`, so on Windows it uses a selector event loop (`GET /health` reports `eventLoop`). CORS allows the Vite dev server (`http://localhost:5173`, `http://127.0.0.1:5173`); add origins in `api.yaml`.

### Authentication and roles

- API keys are set in `.env` (never committed; see `.env.example`), one list per role: `LINETWIN_API_KEYS_OPERATOR=alice:<key>,bob:<key>` and `LINETWIN_API_KEYS_VIEWER=...`. The name before the colon is recorded as `issuedBy` / `ackedBy`.
- REST clients send `X-API-Key: <key>`. **Viewers** can read everything. **Operators** can also send commands, switch sources, acknowledge alerts and inject scenarios. A missing or wrong key is `401`, a viewer doing an operator action `403`.
- **No keys configured:** reads work without a key and writes are refused (`403`, with a hint), but only on a loopback address. Binding to any other address without keys refuses to start. A warning is logged whenever the server runs without keys.
- Keys never belong in a URL. They are compared in constant time and redacted from every log line (structlog fields and uvicorn access logs, including a key wrongly passed as `?apiKey=`).
- `GET /health` needs no key.

### WebSocket stream (`/stream`)

All frames are JSON text: `{"type": "...", "data": ...}`.

**1. Authenticate with the first message** (within `api.yaml` `ws.auth_timeout_s`, 2 s):

```json
{"type": "auth", "apiKey": "<key>"}
```

The server answers `{"type": "auth", "data": {"principal": "victor", "role": "viewer"}}`. A wrong key, or a first message that is not `auth`, gets an `error` message and close code **4401**. Without keys configured (loopback only), `{"type": "auth"}` without a key, or silence until the timeout, grants anonymous viewer access; otherwise the connection is closed with 4401.

**2. Subscribe** (repeat to change the selection):

```json
{"type": "subscribe", "machines": ["CNC-01", "INSP-01"], "channels": ["readings", "states", "line", "alerts"]}
```

`machines: null` (or omitted) means every machine; `channels` defaults to all of `readings`, `states`, `line`, `dataQuality`, `alerts`, `commands`, `predictions`. The server answers `subscribed` and then a `snapshot`:

```json
{"type": "snapshot", "data": {"asOf": "...", "machines": {"CNC-01": {"machineId": "CNC-01", "state": "RUNNING", "...": "..."}},
 "line": {"status": "RUNNING", "rootCauses": {...}, "impacted": {}}, "dataQuality": {...}, "alerts": [...]}}
```

**3. Stream messages**

| `type` | Channel | `data` | Delivery for slow clients |
|---|---|---|---|
| `reading` | readings | Reading | coalesced: newest per machine |
| `state` | states | StateChange (with `reason`, e.g. `"BLOCKED by CNC-01"`) | ordered, bounded |
| `line` | line | line view: `status`, `rootCauses` per machine, `impacted` (root cause → machines) | coalesced |
| `dataQuality` | dataQuality | per-machine report: `status` OK/DEGRADED, `latePct`, `quarantinePct`, `partialPct`, `linkDown`, `reasons` | coalesced: newest per machine |
| `alert` | alerts | Alert lifecycle event (see Alert below) | ordered, bounded |
| `command` / `ack` | commands | Command / CommandAck | ordered, bounded |
| `prediction` | predictions | Prediction (Phase 11) | ordered, bounded |
| `dropped` | (any) | `{"count": n}`: ordered messages lost because this client was slow | |
| `error` | | `{"reason": "..."}` | |

Clients may also send `{"type": "ping"}` (answer `pong`).

**Backpressure:** the server never waits for a client. Each client has its own buffer: coalesced channels keep only the newest value per machine until the client takes it, ordered channels keep at most `ws.client_queue_size` (256) messages and drop the oldest, reported with `dropped`. A client whose send takes longer than `ws.send_timeout_s` (5 s) is closed with code **4408**. A slow client therefore never slows down other clients or ingestion (tested in `backend/tests/test_api_ws_backpressure.py`).

### Commands

`POST /commands` (operator) with `{"machineId", "type", "params", "idempotencyKey"}`; the server sets `commandId`, `issuedBy` and `ts`. It answers with the **final** `CommandAck` (200 for every outcome):

| Outcome | `status` | Example `reason` |
|---|---|---|
| executed by the machine | `EXECUTED` | |
| a safety rule (`config/commands.yaml`) forbids it | `REJECTED` | `setpoints cannot change while the machine is faulted or in maintenance (machine is FAULT)` |
| the active source cannot take commands (file replay) | `REJECTED` | `the active source (file adapter) does not accept commands; switch to a live source ...` |
| the machine refused it | `REJECTED` | `machine is already stopped` |
| no ack within `sources.yaml` `command_timeout_s` | `TIMEOUT` | `no ack within 5 s` |

- **STOP is always allowed** by the safety rules, in every state including OFFLINE, STALE and FAULT (fail-safe principle, [ADR 0014](adr/0014-command-path.md)). The config loader rejects rules that name STOP.
- **Idempotency:** repeating an `idempotencyKey` with the same request returns the original ack and sends nothing (header `Idempotent-Replay: true`); the same key with a different request is `409`. Keys are kept for `idempotency_ttl_s` and survive a restart.
- **Audit trail:** the Command, an `ACCEPTED` ack (safety and capability passed) and the final ack are stored, rejected commands included. `GET /commands/{id}` and `GET /commands?from=&to=&machines=` show them. Audit `ts` values are on the event-time axis (ADR 0010); the wall-clock time is the rows' `ingested_at`.

### Alerts

Engine alerts (`source: "engine"`): `machine_warning` (WARNING), `machine_fault` (CRITICAL), `machine_offline` (CRITICAL), `data_quality` (WARNING) and `line_stoppage` (WARNING: the root-cause machine of STARVED/BLOCKED neighbours, after `raise_after_s` of event time). Kinds, severities and the debounce are in `config/engine.yaml`.

The lifecycle follows ISA-18.2 ([ADR 0011](adr/0011-alert-lifecycle.md)): an alert is raised once (deduplicated), cleared when its condition ends, and stays listed in `GET /alerts` until it is acknowledged (`POST /alerts/{id}/ack`); acknowledging is possible before or after clearing. Every transition is a new stored event (`GET /alerts/history`). Alerts survive a backend restart and are only re-evaluated once fresh data for the machine (the whole line for `line_stoppage`) has arrived.

### Scenario injection (simulation only)

Real equipment would not offer this. When the active source is a simulator publisher (mock, MQTT, OPC UA), an operator can inject a fault now or schedule a scenario file's events ([ADR 0015](adr/0015-scenario-injection.md)):

```json
{"action": "INJECT_FAULT", "machineId": "CNV-01", "faultType": "belt_jam"}
{"action": "INJECT_FAULT", "machineId": "CNC-01", "faultType": "bearing_wear", "progressionS": 300}
{"action": "LOAD_SCENARIO", "scenario": "cnc_stop", "seed": 3}
```

Fault types: `bearing_wear`, `overheating`, `spindle_imbalance` (CNC only), `belt_jam`, `lubrication_loss`. The simulator's verdict is in the ack (`EXECUTED`, `REJECTED` with a reason, `TIMEOUT`). With the file source the request is `409`. Every request and its ack are stored (table `scenario_requests`, `GET /scenario`), rejected ones included, so injected faults can be told apart from real ones in replay and experiments.

Transport per publisher: mock `POST /scenario`; MQTT `{prefix}/scenario` with the ack on `{prefix}/scenario/ack`; OPC UA method `InjectScenario` on the line object (one String argument with the ScenarioRequest JSON, returns the ScenarioAck JSON).

<!-- BEGIN GENERATED: uv run poe api-docs -->

### REST API reference (OpenAPI 3.1.0, LineTwin API 0.4.0)

Generated from the FastAPI app; do not edit by hand. Timestamps are ISO-8601 with a zone. Errors are `{"detail": "..."}`. A live schema is served at `/openapi.json` and interactive docs at `/docs`.

#### alerts

##### `GET /alerts`

Unresolved alerts (active, or cleared but not yet acknowledged). Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `active` | query | no | boolean \| null | only active / only cleared |
| `acknowledged` | query | no | boolean \| null | filter by acknowledgement |
| `machine` | query | no | string \| null | machine id |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [AlertsResponse](#alertsresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `GET /alerts/history`

Stored alert lifecycle events in a time range (every raise, clear, ack). Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `from` | query | no | string (date-time) \| null |  |
| `to` | query | no | string (date-time) \| null |  |
| `machine` | query | no | string \| null | machine id |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [AlertsResponse](#alertsresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `POST /alerts/{alert_id}/ack`

Acknowledge an alert (operator); repeating it changes nothing. Role: **operator**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `alert_id` | path | yes | string |  |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [Alert](#alert) |
| 404 | Unknown or already resolved alert |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### commands

##### `GET /commands`

Audit trail: commands (and their acks) in an event-time range. Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `from` | query | no | string (date-time) \| null |  |
| `to` | query | no | string (date-time) \| null |  |
| `machines` | query | no | array of string \| null | machine ids |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [CommandsResponse](#commandsresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `POST /commands`

Send a command (operator); answers with the final ack. Role: **operator**.

Every outcome of the command itself is a 200 with the final `CommandAck`: `EXECUTED`, `REJECTED` (safety rule, unsupported by the active source, or refused by the machine; see `reason`) or `TIMEOUT`. STOP is always allowed by the safety rules. Repeating an `idempotencyKey` with the same request returns the original ack without sending again (response header `Idempotent-Replay: true`).

Request body: [CommandRequest](#commandrequest)

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [CommandAck](#commandack) |
| 403 | Not an operator |  |
| 404 | Unknown machine |  |
| 409 | idempotencyKey already used for a different request |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `GET /commands/{command_id}`

A command and every ack it got (audit trail). Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `command_id` | path | yes | string |  |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [CommandRecord](#commandrecord) |
| 404 | Unknown command |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### history

##### `GET /export`

Readings in a time range as CSV or JSONL (replayable by the file adapter). Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `from` | query | yes | string (date-time) | inclusive, ISO-8601 with zone |
| `to` | query | yes | string (date-time) | exclusive, ISO-8601 with zone |
| `format` | query | no | `csv` \| `jsonl` | file format |
| `machines` | query | no | array of string \| null | machine ids (repeat the parameter or separate with commas); all if omitted |

| Status | Description | Body |
|---|---|---|
| 200 | Canonical Reading rows ordered by (ts, line order, seq) | `text/csv`, `application/x-ndjson` |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `GET /history`

Readings in a time range: raw, or 1 s / 1 min / 10 min buckets. Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `from` | query | yes | string (date-time) | inclusive, ISO-8601 with zone |
| `to` | query | yes | string (date-time) | exclusive, ISO-8601 with zone |
| `resolution` | query | no | `raw` \| `1s` \| `1m` \| `10m` | raw rows or bucket width |
| `machines` | query | no | array of string \| null | machine ids (repeat the parameter or separate with commas); all if omitted |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [HistoryResponse](#historyresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `GET /replay`

Line state at event time t (snapshot + replay, equals reconstruct(t)). Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `t` | query | yes | string (date-time) | event time, ISO-8601 with zone |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [LineState](#linestate) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### machines

##### `GET /machines`

Live state of every machine. Role: **viewer**.

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [MachinesResponse](#machinesresponse) |

##### `GET /machines/{machine_id}/state`

Live state of one machine. Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `machine_id` | path | yes | string |  |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [MachineStateResponse](#machinestateresponse) |
| 404 | Unknown machine |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### scenario (simulation only)

##### `GET /scenario`

Audit trail of scenario requests in an event-time range. Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `from` | query | no | string (date-time) \| null |  |
| `to` | query | no | string (date-time) \| null |  |
| `machine` | query | no | string \| null | machine id |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [ScenariosResponse](#scenariosresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `POST /scenario`

Inject a fault or load a scenario (operator; simulation only). Role: **operator**.

Simulation-only capability: asks the simulator behind the active source to start a fault now (`INJECT_FAULT`) or to schedule a scenario file's events from now on (`LOAD_SCENARIO`). The request and its ack are stored, so injected faults are distinguishable from real ones. The simulator's verdict is in the ack (`EXECUTED`, `REJECTED` with a reason, `TIMEOUT`).

Request body: [ScenarioPayload](#scenariopayload)

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [ScenarioRecord](#scenariorecord) |
| 403 | Not an operator |  |
| 409 | The active source does not support scenario injection |  |
| 422 | Inconsistent request |  |

##### `GET /scenario/{request_id}`

One scenario request and its ack. Role: **viewer**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `request_id` | path | yes | string |  |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [ScenarioRecord](#scenariorecord) |
| 404 | Unknown request |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### sources

##### `GET /sources`

Active source and its health. Role: **viewer**.

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [SourcesResponse](#sourcesresponse) |

##### `POST /sources/switch`

Hot-switch to another source (operator); a failing source keeps the current one. Role: **operator**.

Request body: [SwitchRequest](#switchrequest)

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [SourcesResponse](#sourcesresponse) |
| 409 | The new source could not start |  |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

#### system

##### `GET /health`

Liveness and wiring (no key needed). Role: **none (no key needed)**.

| Parameter | In | Required | Type | Description |
|---|---|---|---|---|
| `modules` | query | no | boolean | also list which optional heavy modules are loaded |

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | [HealthResponse](#healthresponse) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

##### `GET /metrics`

Prometheus metrics (text exposition format). Role: **viewer**.

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | `text/plain` |

##### `GET /predict`

Fault predictions (Phase 11; returns 501 until then). Role: **viewer**.

| Status | Description | Body |
|---|---|---|
| 501 | available from Phase 11 (inference service); not implemented yet |  |

##### `POST /whatif`

What-if simulation (Phase 11; returns 501 until then). Role: **viewer**.

| Status | Description | Body |
|---|---|---|
| 501 | available from Phase 11 (inference service); not implemented yet |  |

#### Schemas

##### AckStatus

One of: `ACCEPTED`, `REJECTED`, `EXECUTED`, `TIMEOUT`

##### AdapterHealthModel

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | [AdapterKind](#adapterkind) | yes |  |
| `status` | [AdapterStatus](#adapterstatus) | yes |  |
| `connected` | boolean | yes |  |
| `messages` | integer | yes |  |
| `reconnects` | integer | yes |  |
| `lastMessageAt` | string (date-time) \| null | no |  |
| `lastError` | string \| null | no |  |

##### AdapterKind

One of: `file`, `mock`, `mqtt`, `opcua`

##### AdapterStatus

Connection status of a source adapter.

One of: `STARTING`, `CONNECTED`, `RECONNECTING`, `EOF`, `STOPPED`, `FAILED`

##### Alert

One lifecycle event of an operator-facing alert (bus topic ``alerts``, ADR 0011).  Alarm states follow ISA-18.2: ``active`` (the condition is present) and ``acknowledged`` (an operator has seen it) are independent. Every transition (raise, clear, acknowledge, re-raise) is a new append-only event with the same ``alertId`` and the next ``rev``; the current alert is the event with the highest ``rev``. An alert is resolved (no longer shown) once it is both cleared and acknowledged.  ``ts`` is on the event-time axis (ADR 0010); ``ackedAt`` is the wall-clock time of the acknowledgement. ``kind`` is a short machine-readable code (e.g. ``machine_fault``), ``source`` names the producer (e.g. ``engine``).

| Field | Type | Required | Description |
|---|---|---|---|
| `alertId` | string | yes |  |
| `machineId` | string | yes |  |
| `ts` | string (date-time) | yes |  |
| `severity` | [AlertSeverity](#alertseverity) | yes |  |
| `kind` | string | yes |  |
| `message` | string | yes |  |
| `source` | string | yes |  |
| `active` | boolean | no |  |
| `acknowledged` | boolean | no |  |
| `ackedBy` | string \| null | no |  |
| `ackedAt` | string (date-time) \| null | no |  |
| `rev` | integer | no | per-alert revision, +1 per transition |

##### AlertSeverity

How urgent an alert is.

One of: `INFO`, `WARNING`, `CRITICAL`

##### AlertsResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `alerts` | array of [Alert](#alert) | yes |  |

##### Attribution

Contribution of one sensor (and the physical part it maps to) to a prediction.

| Field | Type | Required | Description |
|---|---|---|---|
| `sensor` | [SensorName](#sensorname) | yes |  |
| `part` | string | yes |  |
| `weight` | number | yes |  |

##### AuthInfo

| Field | Type | Required | Description |
|---|---|---|---|
| `keysConfigured` | boolean | yes |  |
| `anonymousRead` | boolean | yes |  |
| `operatorKeys` | boolean | yes |  |

##### Capability

Optional features a source adapter supports beyond streaming readings.

One of: `commands`, `whatif`, `history`, `scenario`

##### Command

Operator command sent towards a machine.

| Field | Type | Required | Description |
|---|---|---|---|
| `commandId` | string | yes |  |
| `idempotencyKey` | string | yes |  |
| `machineId` | string | yes |  |
| `type` | [CommandType](#commandtype) | yes |  |
| `params` | map of boolean \| integer \| number \| string | no |  |
| `issuedBy` | string | yes |  |
| `ts` | string (date-time) | yes |  |

##### CommandAck

Acknowledgement of a command by the command bus or the machine.

| Field | Type | Required | Description |
|---|---|---|---|
| `commandId` | string | yes |  |
| `status` | [AckStatus](#ackstatus) | yes |  |
| `reason` | string \| null | no |  |
| `ts` | string (date-time) | yes |  |

##### CommandRecord

A command with every ack it got (the audit view).

| Field | Type | Required | Description |
|---|---|---|---|
| `command` | [Command](#command) | yes |  |
| `acks` | array of [CommandAck](#commandack) | no |  |

##### CommandRequest

Body of ``POST /commands``. The server adds commandId, issuedBy and ts.

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `type` | [CommandType](#commandtype) | yes |  |
| `params` | map of boolean \| integer \| number \| string | no |  |
| `idempotencyKey` | string | yes | repeat it to retry safely |

##### CommandType

One of: `START`, `STOP`, `SET_SPEED`, `SET_LOAD`, `SCHEDULE_MAINTENANCE`, `RESET`

##### CommandsResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `commands` | array of [CommandRecord](#commandrecord) | yes |  |

##### DataQuality

Per-machine data-quality status derived by the twin engine (late, quarantine, partial readings, staleness). Separate from ``Reliability``, which describes model output.

One of: `OK`, `DEGRADED`

##### DataQualityReport

What the API and WS stream show per machine.

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `status` | [DataQuality](#dataquality) | no |  |
| `samples` | integer | no |  |
| `latePct` | number | no |  |
| `quarantinePct` | number | no |  |
| `partialPct` | number | no |  |
| `linkDown` | boolean | no |  |
| `reasons` | array of string | no |  |

##### HTTPValidationError

| Field | Type | Required | Description |
|---|---|---|---|
| `detail` | array of [ValidationError](#validationerror) | no |  |

##### HealthClass

ML target classes for machine health.

One of: `NORMAL`, `WARNING`, `FAULT`

##### HealthResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | `ok` \| `degraded` | yes |  |
| `lineId` | string | yes |  |
| `machines` | integer | yes |  |
| `source` | [AdapterHealthModel](#adapterhealthmodel) \| null | no |  |
| `store` | string \| null | no |  |
| `eventLoop` | string | yes |  |
| `auth` | [AuthInfo](#authinfo) | yes |  |
| `loadedOptionalModules` | array of string \| null | no |  |

##### HistoryBucketModel

| Field | Type | Required | Description |
|---|---|---|---|
| `bucket` | string (date-time) | yes |  |
| `machineId` | string | yes |  |
| `n` | integer | yes |  |
| `nLate` | integer | yes |  |
| `nSuspect` | integer | yes |  |
| `sensors` | map of [SensorStatsModel](#sensorstatsmodel) | yes |  |

##### HistoryResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `resolution` | `raw` \| `1s` \| `1m` \| `10m` | yes |  |
| `start` | string (date-time) | yes |  |
| `end` | string (date-time) | yes |  |
| `machines` | array of string \| null | no |  |
| `readings` | array of [Reading](#reading) \| null | no |  |
| `buckets` | array of [HistoryBucketModel](#historybucketmodel) \| null | no |  |

##### LineState

Full line state. ``as_of`` is set by whoever built it (reconstruct time, live watermark).

| Field | Type | Required | Description |
|---|---|---|---|
| `asOf` | string (date-time) \| null | no |  |
| `machines` | map of [MachineView](#machineview) | no |  |
| `predictions` | map of map of [Prediction](#prediction) | no |  |

##### LineStatus

One of: `RUNNING`, `DEGRADED`, `STOPPED`, `UNKNOWN`

##### LineViewModel

Serialisable line view (API ``/machines``, WS channel ``line``).

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | [LineStatus](#linestatus) | yes |  |
| `rootCauses` | map of [RootCauseView](#rootcauseview) | yes |  |
| `impacted` | map of array of string | yes |  |

##### MachineKind

One of: `CONVEYOR`, `CNC`, `INSPECTION`

##### MachineState

Operational state of a single machine.

One of: `RUNNING`, `IDLE`, `WARNING`, `FAULT`, `STARVED`, `BLOCKED`, `MAINTENANCE`, `OFFLINE`, `STALE`

##### MachineStateResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `kind` | [MachineKind](#machinekind) | yes |  |
| `name` | string | yes |  |
| `state` | [MachineState](#machinestate) \| null | no |  |
| `stateTs` | string (date-time) \| null | no |  |
| `reportedState` | [MachineState](#machinestate) \| null | no |  |
| `rootCause` | [RootCauseView](#rootcauseview) | yes |  |
| `dataQuality` | [DataQualityReport](#dataqualityreport) | yes |  |
| `lastReading` | [Reading](#reading) \| null | no |  |
| `lastStateChange` | [StateChange](#statechange) \| null | no |  |
| `predictions` | map of [Prediction](#prediction) | no |  |
| `alerts` | array of [Alert](#alert) | no |  |

##### MachineView

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `state` | [MachineState](#machinestate) \| null | no |  |
| `stateTs` | string (date-time) \| null | no |  |
| `reportedState` | [MachineState](#machinestate) \| null | no |  |
| `lastReading` | [Reading](#reading) \| null | no |  |
| `lastStateChange` | [StateChange](#statechange) \| null | no |  |

##### MachinesResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `lineId` | string | yes |  |
| `asOf` | string (date-time) \| null | no |  |
| `line` | [LineViewModel](#lineviewmodel) | yes |  |
| `machines` | array of [MachineStateResponse](#machinestateresponse) | yes |  |

##### Prediction

Model output for one machine at one horizon.

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `ts` | string (date-time) | yes |  |
| `horizonS` | number | yes |  |
| `probs` | map of number | yes |  |
| `calibrated` | boolean | yes |  |
| `conformalSet` | array of [HealthClass](#healthclass) | yes |  |
| `ttfS` | number \| null | no | None: no fault expected |
| `ttfInterval` | array of any \| null | no |  |
| `anomalyScore` | number | yes |  |
| `attributions` | array of [Attribution](#attribution) | no |  |
| `reliability` | [Reliability](#reliability) | yes |  |
| `modelVersion` | string | yes |  |

##### Quality

Data-quality flag attached to each reading by the ingest pipeline.

One of: `OK`, `FILLED`, `LATE`, `SUSPECT`

##### Reading

One telemetry sample from one machine.  Sensor values are nullable: ``null`` (or an absent key) means the sensor value is missing for this sample. The ingest pipeline keeps partially missing readings with ``quality=SUSPECT`` and quarantines readings without any sensor value (ADR 0007).

| Field | Type | Required | Description |
|---|---|---|---|
| `schemaVersion` | `2` | no |  |
| `machineId` | string | yes |  |
| `seq` | integer | yes | Monotonic per-machine sequence number |
| `ts` | string (date-time) | yes |  |
| `state` | [MachineState](#machinestate) | yes |  |
| `speed` | number \| null | no | RPM; null = missing |
| `temp` | number \| null | no | degrees C; null = missing |
| `vib` | number \| null | no | mm/s RMS; null = missing |
| `load` | number \| null | no | percent; null = missing |
| `current` | number \| null | no | A; null = missing |
| `runTime` | number | yes | seconds since start |
| `quality` | [Quality](#quality) | no |  |

##### Reliability

One of: `OK`, `DEGRADED`

##### RootCauseView

| Field | Type | Required | Description |
|---|---|---|---|
| `cause` | string \| null | no |  |
| `unknown` | boolean | no |  |

##### ScenarioAck

The simulator's answer to a ScenarioRequest (status EXECUTED / REJECTED / TIMEOUT).

| Field | Type | Required | Description |
|---|---|---|---|
| `requestId` | string | yes |  |
| `status` | [AckStatus](#ackstatus) | yes |  |
| `reason` | string \| null | no |  |
| `ts` | string (date-time) | yes |  |

##### ScenarioAction

Simulation-only scenario injection (ADR 0015). Real equipment would not offer this.

One of: `INJECT_FAULT`, `LOAD_SCENARIO`

##### ScenarioPayload

Body of ``POST /scenario``. The server adds requestId, requestedBy, source and ts.

| Field | Type | Required | Description |
|---|---|---|---|
| `action` | [ScenarioAction](#scenarioaction) | yes |  |
| `machineId` | string \| null | no | INJECT_FAULT: target machine |
| `faultType` | string \| null | no | INJECT_FAULT: simulator fault, e.g. bearing_wear, belt_jam |
| `progressionS` | number \| null | no | INJECT_FAULT: gradual fault duration (default fastest) |
| `scenario` | string \| null | no | LOAD_SCENARIO: scenario file name without .yaml |
| `seed` | integer \| null | no | LOAD_SCENARIO: plan seed |

##### ScenarioRecord

| Field | Type | Required | Description |
|---|---|---|---|
| `request` | [ScenarioRequest](#scenariorequest) | yes |  |
| `acks` | array of [ScenarioAck](#scenarioack) | no |  |

##### ScenarioRequest

A simulation-only request to inject a fault or load a scenario (ADR 0015).  Stored in the event store (stream ``scenario_requests``) so injected faults can be told apart from real ones in replay and experiments. ``ts`` is on the event-time axis (the source's watermark when the request was made). ``faultType`` names a simulator fault (e.g. ``bearing_wear``); the simulator validates it.

| Field | Type | Required | Description |
|---|---|---|---|
| `requestId` | string | yes |  |
| `action` | [ScenarioAction](#scenarioaction) | yes |  |
| `machineId` | string \| null | no |  |
| `faultType` | string \| null | no |  |
| `progressionS` | number \| null | no |  |
| `scenario` | string \| null | no |  |
| `seed` | integer \| null | no |  |
| `requestedBy` | string | yes |  |
| `source` | string | yes |  |
| `ts` | string (date-time) | yes |  |

##### ScenariosResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `requests` | array of [ScenarioRecord](#scenariorecord) | yes |  |

##### SensorName

Sensor channels carried by a Reading (field names in Python and in YAML config).

One of: `speed`, `temp`, `vib`, `load`, `current`

##### SensorStatsModel

| Field | Type | Required | Description |
|---|---|---|---|
| `count` | integer | yes |  |
| `mean` | number \| null | no |  |
| `min` | number \| null | no |  |
| `max` | number \| null | no |  |

##### SourcesResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `active` | [AdapterKind](#adapterkind) \| null | no |  |
| `capabilities` | array of [Capability](#capability) | no |  |
| `health` | [AdapterHealthModel](#adapterhealthmodel) \| null | no |  |
| `configured` | array of [AdapterKind](#adapterkind) | yes |  |
| `switches` | integer | yes |  |

##### StateChange

A machine's effective state changed (bus topic ``states``).  Emitted by the ingest pipeline when the data link degrades (STALE, OFFLINE) or recovers, and by the twin engine (later phases) for machine state transitions.

| Field | Type | Required | Description |
|---|---|---|---|
| `machineId` | string | yes |  |
| `state` | [MachineState](#machinestate) | yes |  |
| `previousState` | [MachineState](#machinestate) \| null | no |  |
| `ts` | string (date-time) | yes |  |
| `reason` | string \| null | no |  |

##### SwitchRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | [AdapterKind](#adapterkind) | yes |  |

##### ValidationError

| Field | Type | Required | Description |
|---|---|---|---|
| `loc` | array of string \| integer | yes |  |
| `msg` | string | yes |  |
| `type` | string | yes |  |
| `input` | any | no |  |
| `ctx` | object | no |  |

<!-- END GENERATED -->

## Reading (schemaVersion 2)

```json
{"schemaVersion": 2, "machineId": "CNC-01", "seq": 42, "ts": "2026-01-01T00:00:42Z",
 "state": "RUNNING", "speed": 7949.97, "temp": 30.9, "vib": 1.29, "load": 33.2,
 "current": 10.01, "runTime": 42.0, "quality": "OK"}
```

- **Sensor values** (`speed`, `temp`, `vib`, `load`, `current`) are nullable: `null` or an absent key means the value is missing. The ingest pipeline keeps partially missing readings with `quality: "SUSPECT"` and quarantines readings that have no sensor value at all ([ADR 0007](adr/0007-adapters-and-ingest.md)).
- **Required:** `machineId`, `seq`, `ts`, `state` and `runTime`.
- **`quality`:** `OK`, `FILLED`, `LATE` or `SUSPECT`, set by the ingest pipeline.

## Mock machine server (`uv run poe sim-mock`)

Base URL: `mock.base_url` in `config/sources.yaml` (default `http://127.0.0.1:8100`).

| Method | Path | Description |
|---|---|---|
| GET | `/info` | `{"lineId": "line1", "machines": ["CNV-01", ...]}` |
| GET | `/state` | Latest Reading per machine (array) |
| GET | `/history?machineId=&sinceSeq=0&limit=1000` | Readings with `seq > sinceSeq`, oldest first; all machines when `machineId` is omitted. The last `--history-size` readings per machine are kept. |
| WS | `/stream` | One text frame per Reading: `{"reading": {...}, "sentAt": "<wall-clock ISO>"}`. If a client falls behind, it loses messages and should backfill from `/history`. |
| POST | `/command` | Body: `Command`. Response: `CommandAck` from the simulator (`EXECUTED` / `REJECTED`, or `TIMEOUT` when the simulation is not running). 422 on invalid commands. |

## MQTT (`uv run poe up`, then `uv run poe sim-mqtt`)

Broker and prefix: `mqtt:` in `config/sources.yaml` (default prefix `plant/line1`).

| Topic | Direction | Payload | QoS / retain |
|---|---|---|---|
| `{prefix}/{machineId}/telemetry` | sim → backend | Reading JSON; MQTT v5 user property `sentAt` | `mqtt.qos` (1), not retained |
| `{prefix}/{machineId}/state` | sim → any | `{"machineId", "state", "seq", "ts"}`, published on every state change | same QoS, **retained** |
| `{prefix}/{machineId}/cmd` | backend → sim | `Command` JSON | 1 |
| `{prefix}/{machineId}/cmd/ack` | sim → backend | `CommandAck` JSON | 1 |

The backend connects with a persistent MQTT v5 session (`clean_start=False`, session expiry `mqtt.session_expiry_s`), so the broker queues telemetry while the backend reconnects.

## OPC UA (`uv run poe sim-opcua`)

Endpoint and namespace: `opcua:` in `config/sources.yaml` (default `opc.tcp://127.0.0.1:4840/linetwin/`, namespace `urn:linetwin:sim`). No security; the server listens on localhost only.

```
Objects/
  {line_id}/                                  object
    {machineId}/                              object, EventNotifier = SubscribeToEvents
      Speed, Temperature, Vibration, Load, Current, RunTime    Double (NaN = missing)
      State                                   String (MachineState)
      Seq                                     Int64
      Ts                                      String (ISO-8601 UTC)
      Start(commandId, idempotencyKey, issuedBy) -> (status, reason)
      Stop(...), ScheduleMaintenance(...), Reset(...)          same signature
      SetSpeed(commandId, idempotencyKey, issuedBy, value: Double) -> (status, reason)
      SetLoad(commandId, idempotencyKey, issuedBy, value: Double) -> (status, reason)
Types/EventTypes/BaseEventType/ReadingEventType
      fields: MachineId, Seq, Ts, State, Speed, Temperature, Vibration, Load, Current, RunTime
```

- **One `ReadingEventType` event per reading** is fired from the machine node. The backend adapter subscribes to these events, which arrive complete and in order. The variables always hold the latest values (for browsing and HMIs).
- **`status`** is a `CommandAck` status (`EXECUTED`, `REJECTED`, `TIMEOUT`), and `reason` is an empty string when there is none.
- **Shared names:** the browse names are defined once in `twin_common.opcua_names`. The backend maps them to Reading fields through `opcua.mapping` in `sources.yaml`.

## File replay

`file:` in `config/sources.yaml`. Supported formats:

- `csv`: the simulator export; columns are mapped through `file.mapping`, and label columns are dropped. An event-store export (also CSV) needs the default mapping instead; see [Replaying an export](#replaying-an-export).
- `jsonl`: one Reading per line (simulator or event-store export; the default mapping works for both).
- `json`: an array of Readings.

Options: `replay_speed`, `loop` (seq and ts continue across passes) and `start_offset_s`. Minimal mode uses this adapter and needs no service.

## Alert

One lifecycle event of an alert ([ADR 0011](adr/0011-alert-lifecycle.md)):

```json
{"alertId": "CNC-01:line_stoppage:1767225732000000", "machineId": "CNC-01",
 "ts": "2026-01-01T00:02:12Z", "severity": "WARNING", "kind": "line_stoppage",
 "message": "CNC-01 is IDLE and stops CNV-01 (BLOCKED), INSP-01 (STARVED), CNV-02 (STARVED)",
 "source": "engine", "active": true, "acknowledged": false, "ackedBy": null, "ackedAt": null,
 "rev": 0}
```

- **`active` / `acknowledged`:** independent ISA-18.2 states. The current alert is the event with the highest `rev`; it is resolved once cleared and acknowledged.
- **`ts`:** event time; **`ackedAt`:** the wall-clock time of the acknowledgement.
- **`alertId`:** `{machine}:{kind}:{raise ts in µs}`, deterministic for a replay of the same data.
- **`severity`:** `INFO`, `WARNING` or `CRITICAL`. **`kind`:** a short machine-readable code. **`source`:** the producer (`engine`; `inference` from Phase 11).
- Published on the bus topic `alerts` and stored in the `alerts` table (unique per `alertId`, `rev`, `ts`).

## Event store CLI and export (Phase 3)

The store comes from `config/storage.yaml`, by default SQLite at `data/store/linetwin.sqlite`. Connection settings for TimescaleDB come from `.env` (`LINETWIN_PG_*`).

```text
uv run poe db-migrate                     # apply pending migrations
uv run poe db-info                        # row counts, time span, snapshots
uv run poe db-export --format jsonl --out data/export/run.jsonl
uv run poe db-export --format csv --out data/export/run.csv --from 2026-01-01T00:00:00Z --to 2026-01-01T00:10:00Z
uv run poe db-export --format jsonl --out data/export/run.jsonl --all   # plus every other stream
```

- **Range:** `--from` is inclusive and `--to` exclusive (ISO-8601 with a timezone). Either can be omitted.
- **Order:** rows are ordered by (ts, line machine order, seq).
- **Readings file:** written in the canonical Reading wire format:
  - **JSONL:** one Reading JSON per line, exactly as on the bus. Missing sensor values are `null`.
  - **CSV:** header `machineId,seq,ts,state,speed,temp,vib,load,current,runTime,quality`. A missing sensor value is an empty cell, floats are written with full precision, and `schemaVersion` is left out because it defaults to 2.
  - **Quality:** the `quality` flag is kept (`OK`, `SUSPECT`, `LATE`, ...), so a replay stores the same events.
- **`--all`:** also writes `<stem>.<stream>.jsonl` for state changes, predictions, commands, acks, alerts and quarantine. These are for inspection only; the file adapter replays readings.

### Replaying an export

**Exported files replay with the default (empty) field mapping, not with the `file.mapping` that ships in `config/sources.yaml`.** The shipped mapping exists for the *simulator's* CSV export, whose columns are snake_case (`machine_id`, `run_time`) and which carries extra label columns (`drop_unmapped: true`). An export already uses the canonical camelCase keys. With the shipped mapping, the adapter looks for `machine_id` and `run_time`, finds neither, and every row is quarantined as `SCHEMA_ERROR` ("machineId: Field required").

So configure the file adapter for an export with an empty mapping, in the `file:` section of `config/sources.yaml`. Keep the shipped mapping when you go back to the simulator CSV:

```yaml
active: file
file:
  path: data/export/run.jsonl       # or data/export/run.csv
  format: jsonl                     # csv for a CSV export
  replay_speed: 1.0
  loop: false                       # true works too; seq and ts continue across passes
  start_offset_s: 0.0
  mapping: {}                       # default mapping: the export uses canonical keys
```

In code, the same thing is:

```python
from twin_common.config.sources import FieldMapping

file_cfg = sources.file.model_copy(
    update={"path": "data/export/run.csv", "format": "csv", "mapping": FieldMapping()}
)
sources = sources.model_copy(update={"file": file_cfg})
```

The round trip export → `FileReplayAdapter` → ingest → a fresh store gives identical readings, including null sensor values, SUSPECT and LATE flags. `tests/test_phase3_integration.py` checks this for CSV and JSONL on both store backends.

Replaying an export into the *same* store it came from adds nothing, because inserts are idempotent on `(machineId, seq, ts)`.
