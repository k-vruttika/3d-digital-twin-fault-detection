# LineTwin interfaces

This page covers what the simulator publishes and what the backend adapters consume (Phase 2), plus the event-store export and CLI (Phase 3). The backend REST/WebSocket API follows in Phase 4.

All payloads use the canonical contracts from `twin_common.contracts`: camelCase JSON and UTC ISO-8601 timestamps.

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

```json
{"alertId": "alert-0001", "machineId": "CNC-01", "ts": "2026-01-15T08:30:00Z",
 "severity": "WARNING", "kind": "fault_predicted", "message": "FAULT likely within 30 s",
 "source": "inference"}
```

- **`severity`:** `INFO`, `WARNING` or `CRITICAL`.
- **`kind`:** a short machine-readable code.
- **`source`:** the producer.
- Published on the bus topic `alerts` and stored in the `alerts` table. Nothing produces alerts yet; the inference phases will.

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
