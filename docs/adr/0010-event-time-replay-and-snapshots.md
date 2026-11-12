# ADR 0010: Event-time axis, order-independent line state, snapshots and reconstruct

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 3 (storage and event sourcing)

## Context

`reconstruct(t)` must return the full line state at time `t`, and it must equal the live state the twin showed when it got the same data. Phase 2 left two complications:

1. **Two clocks.** Readings carry event time. A file replay of a simulator export is stamped in sim time, months away from the wall clock. Staleness `StateChange`s were stamped with the wall clock. In minimal-mode replays the two axes did not line up, so "state at t" was not defined.
2. **No order between topics.** Readings, state changes and predictions travel on separate bus topics. Their arrival order differs between live consumers and is not stored.

## Decision

### Everything on the event-time axis

- `StalenessMonitor` stamps STALE and OFFLINE with the machine's last live reading ts plus the wall-clock silence multiplied by the active adapter's `time_scale`. A recovery change gets the ts of the reading that resumed the data. See the amendment in [ADR 0007](0007-adapters-and-ingest.md).
- `SourceAdapter.time_scale` gives the event seconds per wall second: the replay speed for the file adapter, 1.0 for live sources. `SourceManager` passes it to the pipeline on every attach or switch. Timestamps are exact at any replay speed.
- A machine that never reported uses the line-wide watermark (the newest reading ts of any machine), extrapolated the same way. Only before the first reading of all does the wall clock stand in.
- Snapshots, `reconstruct` and history all use event time. `ingested_at` (wall clock) is stored as metadata only (ADR 0009).

### Line state as an order-independent fold

`twin_backend.engine.line_state` defines `LineState`, which has per-machine `MachineView`s and the newest prediction per (machine, horizon). It also defines `apply_event`, built only from maxima:

- **Last reading:** the largest `(ts, seq)` among readings whose quality is not LATE. LATE readings are stored but never applied, the same rule live state follows ([ADR 0007](0007-adapters-and-ingest.md)).
- **Last state change:** the largest `(ts, state)`.
- **Effective state:** the newer by ts of the last reading's state and the last state change. A tie goes to the state change.
- **Prediction:** the largest `(ts, modelVersion)` per horizon.

Because each rule is a maximum, the result does not depend on the order in which events are applied. The live tracker (`LiveStateTracker`, fed from the bus) and `reconstruct` (fed from the store) use the same fold, so they agree once they have seen the same events, whatever the arrival or cross-topic order. The Phase 4 twin engine builds on the same fold.

### Snapshots and reconstruct

- **`reconstruct(store, machines, t)`** takes the newest snapshot at or before `t` (if any), then folds in every stored reading, state change and prediction with `snapshot.ts < ts ≤ t`.
- **Snapshot times.** `Snapshotter` saves snapshots at epoch-aligned multiples of `snapshots.interval_s`, so the same data always gives the same snapshot times. It only saves up to the store watermark minus `settle_s`. The config requires `settle_s ≥ ingest.lateness_window_s + writer.flush_interval_s`, so data still held by the reorder buffer or the writer is normally included.
- **Snapshots equal replay by construction.** Each snapshot is itself computed with `reconstruct` from the previous snapshot.
- **Late data deletes affected snapshots.** If a committed batch contains non-LATE readings, state changes or predictions at or before the newest snapshot (e.g. an MQTT backlog after an outage), every snapshot at or after the oldest such event is deleted and rebuilt on the next run. A lock shared with the snapshotter keeps a snapshot computed before such a deletion from being saved after it.
- **`iter_states(t0, t1, step)`** replays forward step by step for time travel.

### Old data and the aggregates (amended 2026-11-12, before the Phase 3 commit)

Event-time history must work for data far in the past, such as replayed simulator recordings stamped months back or backfills. The TimescaleDB refresh policies only cover recent wall-clock time, and real-time aggregation only covers buckets above the materialization watermark. Old buckets are therefore served from the materialization alone, and new rows there stay invisible until someone refreshes that range.

The first version kept the ranges to refresh in process memory. They were lost on restart and invisible to other processes. Now:

- **Recording.** When the store inserts readings older than the refresh-policy window (the shorter `start_offset_s` minus one 10-minute bucket), it adds a row `(lo, hi)` to the table `aggregate_refresh_queue`. The row is written **in the same transaction** as the readings, so committed data always has its refresh queued. Migration `0003_aggregate_refresh_queue.sql` creates the table.
- **Consuming.** Before a 1 min or 10 min history query, `refresh_pending(t0, t1)`:
  1. takes a database advisory lock, so one consumer at a time across all processes;
  2. reads the queued ranges that overlap the query;
  3. refreshes both aggregates over their span (1 min first);
  4. deletes exactly those rows once the refresh succeeded.

  A second consumer waits for the lock and then finds nothing left.
- **Failure handling.**
  - If a background policy job is refreshing an overlapping window at that moment (`LockNotAvailableError`), the rows stay queued, the query returns what is materialized, and the next query retries.
  - Any other failure leaves the rows queued as well.
- **Test.** One store instance writes old sim-time data and closes. A new instance then returns the data in its 1 min and 10 min history, and the queue is empty afterwards. A control row written without a queue entry stays invisible, which proves the queue is what makes old data appear.

## Consequences

- **Tested equality.** `tests/test_phase3_integration.py` runs on both backends with a seeded run that contains a 16 s outage (STALE, OFFLINE, then recovery) and one LATE reading. `reconstruct(t)` equals the recorded live state at 25 random settled times, with and without snapshots.
- **Live state can lag replay.** While the reorder buffer holds a message, live state is behind what replay will show for that time. That is why the test only samples settled times.
- **Backlog after STALE.** Readings that arrive in order but late (a backlog) after a STALE mark leave the machine STALE until a reading newer than the STALE ts arrives. This is deterministic and the same for live state and replay.
- **Network adapters assume `time_scale = 1`.** A simulator publisher run with `--speed ≠ 1` behind MQTT, OPC UA or the mock server therefore gets approximate staleness timestamps.
- **Changed contract meaning.** The meaning of `StateChange.ts` changed from wall time to event time. For live sources the two are equal.
