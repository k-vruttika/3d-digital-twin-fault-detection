-- Alert lifecycle (ADR 0011, ISA-18.2): every transition is a new row with the next rev.
-- SQLite cannot change a UNIQUE constraint in place, so the table is rebuilt.

CREATE TABLE alerts_new (
    ts           INTEGER NOT NULL,
    alert_id     TEXT    NOT NULL,
    machine_id   TEXT    NOT NULL,
    severity     TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    message      TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    active       INTEGER NOT NULL DEFAULT 1,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    acked_by     TEXT,
    acked_at     INTEGER,
    rev          INTEGER NOT NULL DEFAULT 0,
    ingested_at  INTEGER NOT NULL,
    UNIQUE (alert_id, rev, ts)
);
INSERT INTO alerts_new (ts, alert_id, machine_id, severity, kind, message, source, ingested_at)
    SELECT ts, alert_id, machine_id, severity, kind, message, source, ingested_at FROM alerts;
DROP TABLE alerts;
ALTER TABLE alerts_new RENAME TO alerts;
CREATE INDEX alerts_ts ON alerts (ts);
CREATE INDEX alerts_id ON alerts (alert_id, rev);
