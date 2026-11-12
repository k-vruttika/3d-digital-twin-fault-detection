# ADR 0007: Source adapters and ingest semantics

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 2 (adapters and ingestion)

## Context

The twin must receive the same machine data from interchangeable outside systems (file replay, a mock REST/WebSocket server, MQTT, OPC UA) and survive messy transport: duplicates, reordering, delays, corruption, missing values and outages. Minimal mode (file only) must keep working without any service.

## Decision

### Adapters

- **Protocol:** `SourceAdapter` (`twin_backend.adapters.base`):
  - `start()`, `stop()`, `stream() -> AsyncIterator[RawMessage]`, `health() -> AdapterHealth`;
  - `send_command(Command) -> CommandAck`, which answers REJECTED when the adapter lacks the capability;
  - `capabilities`: a subset of `{commands, whatif, history}`.
- **No validation in adapters.** They deliver `RawMessage(source, payload, received_mono, received_at, sent_at, meta)`. The payload is the raw dict, string or bytes, so even garbage reaches the pipeline and is quarantined with a reason.
- **BaseAdapter** runs the I/O loop in a task and feeds a bounded queue. That gives backpressure and never drops. `stop()` repeats the cancellation until the task has ended, because on Python 3.11 `asyncio.wait_for` inside client libraries can swallow a cancellation.
- **Reconnect:** every network adapter reconnects with exponential backoff and seeded jitter (`reconnect:` in `sources.yaml`). asyncua's own reconnect supervisor is not used, so all adapters behave and report health the same way.
- **Per adapter:**

  | Adapter | Transport | Gap recovery | sentAt from | Commands |
  |---|---|---|---|---|
  | File | JSONL/CSV/JSON, stdlib only | n/a | n/a | no |
  | Mock | WebSocket `{reading, sentAt}` | `GET /history?sinceSeq` after every (re)connect; REST polling while the WS is down | envelope | `POST /command` |
  | MQTT | `{prefix}/+/telemetry`, MQTT v5 | persistent session (`clean_start=False`, session expiry): the broker queues QoS 1 telemetry while disconnected | user property `sentAt` | `cmd` / `cmd/ack` topics, TIMEOUT if no ack |
  | OPC UA | `ReadingEventType` events per machine node | none (events fired while disconnected are lost and show up as a gap) | event `Time` | method calls |

- **OPC UA streams events, not variables.** A subscription to variables only delivers value changes, merges updates, and gives no atomic set of sensor values for one sample. One event per reading is complete, ordered and queued on the server. The variables still exist for browsing and HMIs.
- **MQTT QoS:** telemetry uses QoS 1 by default (`mqtt.qos: 1`).
  - QoS 0 has the least overhead but may legitimately lose messages on reconnects or broker load, so the identical-sequence integration test would be flaky.
  - QoS 1 is at least once; the resulting duplicates are removed by dedup on (machineId, seq).
  - QoS 2 (exactly once) costs two extra round trips per message for a guarantee dedup already gives.
- **Selection and hot switch:** the active adapter comes from `sources.yaml`. `SourceManager.switch()` starts the new adapter before stopping the old one, so a failing source leaves the current one running. Pipeline state survives the switch, so overlapping data is deduplicated.
- **ChaosAdapter** wraps any adapter and injects seeded drop, duplicate, delay, reorder and corrupt faults, for tests and demos.

### Ingest pipeline

Normalize → validate → dedup → reorder → event bus. Anything rejected goes to a quarantine sink with a reason, stage, detail, raw payload and source.

- **Normalizer:**
  - The per-adapter `mapping.fields` maps canonical camelCase keys to source keys; `drop_unmapped` drops extra keys such as the simulator CSV's label columns.
  - `mapping.units` converts from a fixed table (`twin_common.units`) into the line catalog units.
- **Validator:**
  - **Missing sensor values:** an absent key, `null`, NaN, ±inf or an empty string means the value is missing (`None`). A non-numeric string is a schema error, not "missing".
  - **Partially missing readings are kept.** Sensor fields in `Reading` are nullable (`null` = missing). A reading with some of its machine's sensors missing gets `quality=SUSPECT`, and the ingest metrics count it (`partial`, plus `missing_sensor{sensor}`).
  - **Readings without any valid sensor value** are quarantined (`NO_SENSOR_DATA`).
  - **Always quarantined:** decode errors, schema errors (including a missing or non-numeric `seq`, `ts`, `state` or `runTime`), a non-finite `runTime` (`NON_FINITE`) and unknown machines (`UNKNOWN_MACHINE`, checked against the loaded line config).
  - **Out of the catalog range:** controlled by `ingest.out_of_range`:
    - `suspect` (default) keeps the reading flagged `SUSPECT`;
    - `quarantine` rejects it (`OUT_OF_RANGE`).
    
    Everything the simulator can legitimately produce lies inside the ranges (`simulator/tests/test_sim_sensor_ranges.py`). For that reason the load maximum was raised from 120 to 130 %, above `max_load_pct` plus measurement noise.
- **Dedup** on (machineId, seq), remembering `dedup_window` seqs per machine. A seq more than `seq_reset_threshold` below the highest seen means the source restarted: that machine's dedup and reorder state is reset. The config enforces `seq_reset_threshold <= dedup_window`.
- **Reorder buffer**, per machine, using arrival time on an injectable clock:
  - The expected seq is released at once, together with any held messages that are now contiguous.
  - A message ahead of a gap waits until the gap fills or it has waited `lateness_window_s`. Then the gap is given up and counted.
  - A message whose seq has already been passed is published with `quality=LATE`. It is stored, but live-state consumers must not apply it. LATE data also does not count as a sign of life for staleness.
  - A machine's first messages (at start, or after a source restart) are held for one window, so early out-of-order messages are not wrongly marked LATE.
- **Staleness:** a machine with no live data for `stale_after_s` gets a `StateChange` to STALE, and after `offline_after_s` (default 5 s) to OFFLINE. It changes back to the reported state when data resumes. Machines that never report count from the start.
- **Metrics:** Prometheus, one registry per pipeline:
  - counters for messages in, published, quarantined{reason}, duplicates, late, reordered, gap seqs, suspect, partial, missing_sensor{sensor} and source resets;
  - an end-to-end latency histogram measured to the bus publish, from the transport's send time (`basis="source"`) or else from receipt (`basis="receive"`).
- **Loops:** a looped file replay shifts `ts` by the recording span and `seq` by each machine's maximum, so the replay looks like one continuous stream instead of being deduplicated away.

## Consequences

- One canonical Reading sequence from any source. The integration test proves that File (JSONL and CSV), Mock, MQTT and OPC UA deliver identical sequences for the same simulated run.
- **Contract change:** the sensor fields of `Reading` are now `float | None` while `schemaVersion` stays 2. Every consumer must handle `None`. Today the only one is the validator's range check; later phases (features, models, frontend) must treat `None` as missing, for example via imputation or masking. Payloads that send all values are unaffected.
- **The CSV export** keeps sensor-dropout rows with every sensor NaN; they are quarantined as `NO_SENSOR_DATA`. The JSONL export leaves them out. Both deliver the same canonical sequence.
- **Startup delay:** each machine's first readings wait up to `lateness_window_s` (2 s by default).
- **OPC UA outages** lose the readings fired while disconnected; they appear as gaps. The mock server and MQTT do not lose data across short outages.
- **Looped file replay** restarts `runTime` on every pass.

## Amendment (Phase 3, 2026-11-12): staleness timestamps on the event-time axis

- `StateChange.ts` from the staleness monitor is now event time, not wall time:
  - **STALE / OFFLINE:** the machine's last live reading ts + wall-clock silence × the active adapter's `time_scale`.
  - **Recovery:** the ts of the reading that resumed the data.
  - **A machine that never reported:** the line-wide watermark extrapolated the same way.
- `SourceAdapter.time_scale` is the file adapter's `replay_speed`, and 1.0 for live sources. `SourceManager` passes it to the pipeline (`IngestPipeline.set_time_scale`).
- Silence thresholds (`stale_after_s`, `offline_after_s`) are still measured in wall-clock seconds.
- Why: readings and state changes must share one time axis so stored events can be replayed together ([ADR 0010](0010-event-time-replay-and-snapshots.md)). For live sources nothing changes in practice.
- **LATE readings** are stored by the Phase 3 writer and never applied by the line-state fold. This holds both live and in `reconstruct`.
- **`SourceManager.stop()`** now also releases what the reorder buffer still holds before draining the storage writer, but only when storage is wired in. Nothing held is lost on shutdown.
