# LineTwin interfaces

This page covers what the simulator publishes and what the backend adapters consume (Phase 2). The backend REST/WebSocket API follows in Phase 4.

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

- `csv`: the simulator export; columns are mapped through `file.mapping`, and label columns are dropped.
- `jsonl`: one Reading per line.
- `json`: an array of Readings.

Options: `replay_speed`, `loop` (seq and ts continue across passes) and `start_offset_s`. Minimal mode uses this adapter and needs no service.
