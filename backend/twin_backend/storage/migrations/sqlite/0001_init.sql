-- LineTwin event store, SQLite dialect (minimal mode and tests). See ADR 0009.
-- Timestamps are INTEGER microseconds since 1970-01-01 UTC (exact, sortable, cheap bucketing).
-- ingested_at is wall-clock metadata only. Nested payloads are JSON text.

CREATE TABLE readings (
    ts             INTEGER NOT NULL,
    machine_id     TEXT    NOT NULL,
    seq            INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    state          TEXT    NOT NULL,
    speed          REAL,
    temp           REAL,
    vib            REAL,
    load           REAL,
    current        REAL,
    run_time       REAL    NOT NULL,
    quality        TEXT    NOT NULL,
    ingested_at    INTEGER NOT NULL,
    UNIQUE (machine_id, seq, ts)
);
CREATE INDEX readings_ts ON readings (ts, machine_id);

CREATE TABLE state_changes (
    ts             INTEGER NOT NULL,
    machine_id     TEXT    NOT NULL,
    state          TEXT    NOT NULL,
    previous_state TEXT,
    reason         TEXT,
    ingested_at    INTEGER NOT NULL,
    UNIQUE (machine_id, ts, state)
);
CREATE INDEX state_changes_ts ON state_changes (ts);

CREATE TABLE predictions (
    ts            INTEGER NOT NULL,
    machine_id    TEXT    NOT NULL,
    horizon_s     REAL    NOT NULL,
    model_version TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    ingested_at   INTEGER NOT NULL,
    UNIQUE (machine_id, ts, horizon_s, model_version)
);
CREATE INDEX predictions_ts ON predictions (ts);

CREATE TABLE commands (
    ts              INTEGER NOT NULL,
    command_id      TEXT    NOT NULL,
    idempotency_key TEXT    NOT NULL,
    machine_id      TEXT    NOT NULL,
    type            TEXT    NOT NULL,
    payload         TEXT    NOT NULL,
    ingested_at     INTEGER NOT NULL,
    UNIQUE (command_id, ts)
);
CREATE INDEX commands_ts ON commands (ts);

CREATE TABLE command_acks (
    ts          INTEGER NOT NULL,
    command_id  TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    reason      TEXT,
    ingested_at INTEGER NOT NULL,
    UNIQUE (command_id, status, ts)
);
CREATE INDEX command_acks_ts ON command_acks (ts);

CREATE TABLE alerts (
    ts          INTEGER NOT NULL,
    alert_id    TEXT    NOT NULL,
    machine_id  TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    ingested_at INTEGER NOT NULL,
    UNIQUE (alert_id, ts)
);
CREATE INDEX alerts_ts ON alerts (ts);

-- Rejected messages have no natural key: every quarantine event is kept.
CREATE TABLE quarantine (
    ts          INTEGER NOT NULL,
    source      TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    stage       TEXT    NOT NULL,
    detail      TEXT    NOT NULL,
    machine_id  TEXT,
    payload     TEXT    NOT NULL,
    meta        TEXT    NOT NULL,
    ingested_at INTEGER NOT NULL
);
CREATE INDEX quarantine_ts ON quarantine (ts);

-- Full line state at event time ts (JSON of twin_backend.engine.line_state.LineState).
CREATE TABLE snapshots (
    ts          INTEGER PRIMARY KEY,
    state       TEXT    NOT NULL,
    ingested_at INTEGER NOT NULL
);
