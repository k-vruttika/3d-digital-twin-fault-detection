# ADR 0015: Scenario injection (simulation only) and its audit stream

- **Status:** Accepted
- **Date:** 2026-11-13
- **Phase:** 4 (twin engine, API, commands)

## Context

Demos, tests and the robustness experiments need to make a fault happen now, or to play a scenario's events against a running simulated line, through the same live sources the twin normally uses. Real equipment would never offer this. Faults caused this way must also be distinguishable from real ones in the store, otherwise replays and experiment labels would be misleading.

## Decision

### Simulation-only capability

`Capability.SCENARIO` is a new adapter capability. All three simulator publishers support it, since they share `SimSession`:

| Publisher | Request | Answer |
|---|---|---|
| mock server | `POST /scenario` with a ScenarioRequest | the ScenarioAck |
| MQTT | `{prefix}/scenario` (QoS 1) | `{prefix}/scenario/ack`, matched by `requestId` |
| OPC UA | method `InjectScenario` on the line object, one String argument with the request JSON | the ack JSON |

The mock, MQTT and OPC UA adapters declare the capability. The file adapter does not, and `POST /scenario` answers 409 with a reason when it is active. The endpoint requires the operator role. The docs and the API description state that this is a simulation feature.

### Actions

- **`INJECT_FAULT`** (`machineId`, `faultType`, optional `progressionS`): `LineSimulator.inject_fault` queues a fault event at the current simulation time. It goes through the same event queue as planned faults, so snapshot/restore stays exact, and it is logged in `fault_log`. The simulator rejects unknown machines, faults that do not apply to the machine kind (for example `spindle_imbalance` on a conveyor), a second active fault, maintenance, and a jam on a machine that is not running. Gradual faults default to the fastest configured progression.
- **`LOAD_SCENARIO`** (`scenario`, optional `seed`): a file in the session's scenario directory, named without a path (the name is validated by a pattern, so no path traversal). A run plan is made from it, and its fault, sensor-fault and command events are scheduled relative to now (`LineSimulator.schedule_plan`). Loads, thresholds and the source keep the current run's values. Scenarios that do not fit the line are rejected.

Both actions run at a tick boundary through `RealtimeRunner.submit_call`, the same mechanism as commands. An exception in a request reaches its caller and does not stop the simulation. The ack carries simulation time, like the simulator's command acks.

### Audit stream

- **Contracts:** new `ScenarioRequest` (`requestId`, `action`, target fields, `requestedBy`, `source`, `ts`) and `ScenarioAck` (`requestId`, `status`, `reason`, `ts`) in `twin_common.contracts`.
- **Storage:** both are published on `Topic.SCENARIOS` and stored in the append-only table `scenario_requests`, one row per request and one per ack, unique on `(request_id, kind, ts)`, with the full JSON payload. Migrations: SQLite `0003`, TimescaleDB `0005` (a hypertable). Requests refused because the source lacks the capability are stored too. `ts` is on the event-time axis (the target machine's or the line's watermark), so a replay shows the request before the WARNING/FAULT it caused.
- **Queries:** `GET /scenario` and `GET /scenario/{id}` show the trail.
- **Metric:** `linetwin_scenario_requests_total{source,action,status}`.

Scenario requests are deliberately not `Command`s. They target the simulator, not a machine, and they need different fields.

## Consequences

- Experiments can join `scenario_requests` to separate injected faults from planned or real ones.
- A real deployment simply has no adapter with `Capability.SCENARIO`, so the endpoint answers 409 there.
- The TimescaleDB settings fingerprint now includes the list of streams, so a table added by a migration also gets the configured chunk interval and policies.
