# ADR 0009: Event store, TimescaleDB features and plain-SQL migrations

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 3 (storage and event sourcing)

## Context

Every event must be stored so any past moment can be replayed. The store needs two backends with the same semantics:

- TimescaleDB, the Docker Compose service (ADR 0005);
- SQLite, for minimal mode and tests. Minimal mode must never need Docker, a broker or ML (CLAUDE.md §1), and its test forbids importing `asyncpg` or `sqlalchemy`.

Ingestion must never wait for the database.

## Decision

### One interface, two implementations

`twin_backend.storage.base.EventStore` (Protocol) has these parts:

- **Streams.** Seven append-only tables, one per `Stream`: `readings`, `state_changes`, `predictions`, `commands`, `command_acks`, `alerts`, `quarantine`. A separate `snapshots` table holds line snapshots (ADR 0010).
- **Rows.** Every row has the event's own `ts` (event time) and `ingested_at` (the wall-clock time the message was published on the bus).
  - `ingested_at` is metadata for latency measurement and debugging only. Replay and history never use it.
- **Sensor values.** They are nullable columns, so `null` means missing. The `quality` flag (`OK`, `FILLED`, `LATE`, `SUSPECT`) is stored as is.
- **Idempotent inserts.** Each table has a natural key, and duplicates are skipped:
  - readings: `(machine_id, seq, ts)`;
  - state changes: `(machine_id, ts, state)`;
  - predictions: `(machine_id, ts, horizon_s, model_version)`;
  - commands: `(command_id, ts)`;
  - acks: `(command_id, status, ts)`;
  - alerts: `(alert_id, ts)`;
  - quarantine has no natural key, so every record is kept.

  A retried batch, or a replayed export written into the same store, therefore never duplicates rows.
- **Ranges.** `fetch` covers `[t0, t1)`. `fetch_replay` covers `(after, until]` across several streams, in a deterministic order: ts, then stream rank (reading < state change < prediction), then machine.
- **History.**
  - The 1 s, 1 min and 10 min buckets include every bucket whose start lies in `[t0, t1)`.
  - Averages, minima and maxima ignore null values, and per-sensor counts report how many values each statistic is based on.
  - LATE and SUSPECT readings are included and counted (`n_late`, `n_suspect`). They are real measurements at their own ts.

**SQLite** (`storage/sqlite.py`):
- Uses the standard library `sqlite3` only, with one dedicated worker thread that owns the connection, so the event loop never blocks.
- WAL mode; one transaction per batch.
- Timestamps are stored as integer microseconds since the epoch, which round-trip exactly.
- Downsampling is computed in SQL (`AVG`/`MIN`/`MAX`/`COUNT(col)` skip NULL).

**TimescaleDB** (`storage/timescale.py`, asyncpg; imported only when selected):
- **Hypertables** on `ts` for all seven event tables. Chunk intervals come from `storage.yaml`.
- **Continuous aggregates:**
  - `readings_1m` stores sum, count, min and max per sensor, plus `n`, `n_late` and `n_suspect`.
  - `readings_10m` is a hierarchical aggregate on top of it. Means are `sum / count`, so rolling up stays exact and ignores nulls.
  - Both use real-time aggregation (`materialized_only = false`), so recent buckets are computed from raw data at query time.
  - 1 s history is computed from raw readings with `time_bucket`.
- **Late data reaches the aggregates in two ways:**
  1. **Refresh policies** re-materialize the last `start_offset_s` of wall-clock time every `schedule_s`. The config requires `start_offset_s ≥ max_late_arrival_s + end_offset_s`, so a LATE reading up to `max_late_arrival_s` old (default 3600 s, more than the MQTT session's 300 s) is re-aggregated by the next policy run. A test proves this with the real policy job (`CALL run_job`).
  2. **Older data**, such as a replayed recording whose event time lies months back or a backfill, falls outside every policy window. The store records the batch's old time range in the table `aggregate_refresh_queue`, in the same transaction as the readings. Before a 1 min or 10 min history query, the store refreshes the queued ranges that overlap it, then deletes them. Because the queue is in the database, this works across processes and restarts (see ADR 0010, "Old data and the aggregates").
- **Optional compression and retention policies** (`compression.enabled`, `retention.enabled`; both off by default). Retention must keep more than the longest refresh window.
- **Applying settings.** Chunk intervals and policies are applied by `migrate()` from config. A fingerprint in `linetwin_settings` skips the work when nothing changed, so restarts do not reschedule jobs. A new refresh policy first runs one interval after it is created.
- **Batch insert.** `COPY` into a temporary staging table, then `INSERT … SELECT … ON CONFLICT DO NOTHING`. In `experiments/results/storage_bench.md` this path is about 2.6× faster than `executemany` at batch size 1000.
- **Text order.** Text is ordered with `COLLATE "C"`, so the row order matches SQLite.
- **JSON payloads are `text`, not `jsonb`.** jsonb cannot hold `\u0000`, which quarantined garbage payloads legitimately contain. A NUL character inside a plain text column (e.g. a quarantine `detail`) is stored as U+FFFD.

### Writer: batched, bounded, never blocking

`StorageWriter` subscribes to the bus with its own queue bound (`writer.queue_size`):
- Every `flush_interval_s` it writes everything queued, in transactions of up to `batch_size` rows.
- A failed batch is retried with doubling backoff, then dropped.
- Drops are counted in `linetwin_storage_dropped_writes_total{stream, reason=queue_full|write_error}`.
- Quarantine records (not on the bus) reach it through a tee sink with the same bound.

### Migrations: versioned plain SQL, not Alembic

Files are `storage/migrations/<dialect>/NNNN_name.sql`, run by a small runner (`storage/migrator.py`) and recorded in `schema_migrations(version, name, checksum, applied_at)`:
- **Ordering and checks.** Files run in version order. An applied file whose content changed, or a version in the database without a file, is an error: migrations are append-only.
- **Checksums** are taken over the text with `\n` line endings, so Windows checkouts with CRLF agree with Linux and macOS.
- **Transactions.** Each file runs in one transaction with its `schema_migrations` row. A file whose first line is `-- linetwin: no-transaction` runs statement by statement instead, because continuous aggregates cannot be created inside a transaction block.
- **Locking.** On TimescaleDB an advisory lock serializes concurrent migrators.

Why not Alembic:
- The two dialects differ substantially. The TimescaleDB DDL (`create_hypertable`, continuous aggregates, policies) is raw SQL anyway.
- There are no ORM models to autogenerate migrations from.
- Alembic would pull SQLAlchemy into minimal mode, which its test forbids.

The cost is no automatic downgrade scripts, which this project does not need.

### Test isolation

Database tests never touch the configured database:
- The session fixture `timescale_test_db` creates `linetwin_test_<random hex>` next to it, migrates it, and drops it at session end with `DROP DATABASE … WITH (FORCE)`.
- Each test starts from an emptied store.
- The benchmark uses its own `linetwin_bench_<hex>` database the same way.

## Consequences

- Minimal mode persists to SQLite with no extra dependency. The minimal-mode test checks that no driver beyond `sqlite3` is loaded.
- Every store-level test runs on both backends (`STORE_BACKENDS`). The TimescaleDB variants carry the `services` marker and cannot be skipped silently when `LINETWIN_REQUIRE_SERVICES=1`.
- **Aggregates can be briefly stale.** Between a late insert into an already materialized bucket and the next policy run (≤ `schedule_s`), the 1 min and 10 min aggregates are stale. Raw data and `reconstruct` are always current.
- **Old data is refreshed lazily.** It reaches the materialized aggregates on the first 1 min or 10 min history query that covers it, in any process; that query pays for the refresh. `refresh_pending()` refreshes everything queued, for example from a maintenance job.
- With retention enabled, `reconstruct(t)` for times before the retention horizon becomes incomplete.
