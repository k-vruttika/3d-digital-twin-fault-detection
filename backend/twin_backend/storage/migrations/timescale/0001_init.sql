-- LineTwin event store, TimescaleDB dialect (PostgreSQL 16 + timescaledb). See ADR 0009.
-- Every event table is a hypertable on ts (event time). Chunk intervals, compression and
-- retention come from config/storage.yaml and are applied at startup, not here.
-- ingested_at is wall-clock metadata only.
-- JSON payloads are stored as text, not jsonb: jsonb cannot hold \u0000, which raw quarantined
-- payloads (e.g. garbage bytes) legitimately contain.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE readings (
    ts             timestamptz      NOT NULL,
    machine_id     text             NOT NULL,
    seq            bigint           NOT NULL,
    schema_version smallint         NOT NULL,
    state          text             NOT NULL,
    speed          double precision,
    temp           double precision,
    vib            double precision,
    load           double precision,
    current        double precision,
    run_time       double precision NOT NULL,
    quality        text             NOT NULL,
    ingested_at    timestamptz      NOT NULL,
    UNIQUE (machine_id, seq, ts)
);
SELECT create_hypertable('readings', by_range('ts', INTERVAL '1 day'));
CREATE INDEX readings_machine_ts ON readings (machine_id, ts DESC);

CREATE TABLE state_changes (
    ts             timestamptz NOT NULL,
    machine_id     text        NOT NULL,
    state          text        NOT NULL,
    previous_state text,
    reason         text,
    ingested_at    timestamptz NOT NULL,
    UNIQUE (machine_id, ts, state)
);
SELECT create_hypertable('state_changes', by_range('ts', INTERVAL '7 days'));

CREATE TABLE predictions (
    ts            timestamptz      NOT NULL,
    machine_id    text             NOT NULL,
    horizon_s     double precision NOT NULL,
    model_version text             NOT NULL,
    payload       text             NOT NULL,
    ingested_at   timestamptz      NOT NULL,
    UNIQUE (machine_id, ts, horizon_s, model_version)
);
SELECT create_hypertable('predictions', by_range('ts', INTERVAL '7 days'));

CREATE TABLE commands (
    ts              timestamptz NOT NULL,
    command_id      text        NOT NULL,
    idempotency_key text        NOT NULL,
    machine_id      text        NOT NULL,
    type            text        NOT NULL,
    payload         text        NOT NULL,
    ingested_at     timestamptz NOT NULL,
    UNIQUE (command_id, ts)
);
SELECT create_hypertable('commands', by_range('ts', INTERVAL '7 days'));

CREATE TABLE command_acks (
    ts          timestamptz NOT NULL,
    command_id  text        NOT NULL,
    status      text        NOT NULL,
    reason      text,
    ingested_at timestamptz NOT NULL,
    UNIQUE (command_id, status, ts)
);
SELECT create_hypertable('command_acks', by_range('ts', INTERVAL '7 days'));

CREATE TABLE alerts (
    ts          timestamptz NOT NULL,
    alert_id    text        NOT NULL,
    machine_id  text        NOT NULL,
    severity    text        NOT NULL,
    kind        text        NOT NULL,
    message     text        NOT NULL,
    source      text        NOT NULL,
    ingested_at timestamptz NOT NULL,
    UNIQUE (alert_id, ts)
);
SELECT create_hypertable('alerts', by_range('ts', INTERVAL '7 days'));

-- Rejected messages have no natural key: every quarantine event is kept.
CREATE TABLE quarantine (
    ts          timestamptz NOT NULL,
    source      text        NOT NULL,
    reason      text        NOT NULL,
    stage       text        NOT NULL,
    detail      text        NOT NULL,
    machine_id  text,
    payload     text        NOT NULL,
    meta        text        NOT NULL,
    ingested_at timestamptz NOT NULL
);
SELECT create_hypertable('quarantine', by_range('ts', INTERVAL '7 days'));

-- Full line state at event time ts. A plain table: few rows, looked up by ts.
CREATE TABLE snapshots (
    ts          timestamptz PRIMARY KEY,
    state       text        NOT NULL,
    ingested_at timestamptz NOT NULL
);

-- Fingerprint of the config-driven settings (chunk intervals, policies) last applied.
CREATE TABLE linetwin_settings (
    key   text PRIMARY KEY,
    value text NOT NULL
);
