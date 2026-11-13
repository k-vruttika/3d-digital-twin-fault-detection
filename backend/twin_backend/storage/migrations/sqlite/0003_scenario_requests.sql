-- Scenario injection audit (simulation only, ADR 0015). One row per request (kind 'request')
-- and one per acknowledgement (kind 'ack'), so injected faults can be told apart from real
-- ones in replay and experiments. ts is event time; the JSON payload is the full contract.

CREATE TABLE scenario_requests (
    ts          INTEGER NOT NULL,
    request_id  TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    action      TEXT,
    machine_id  TEXT,
    status      TEXT,
    payload     TEXT    NOT NULL,
    ingested_at INTEGER NOT NULL,
    UNIQUE (request_id, kind, ts)
);
CREATE INDEX scenario_requests_ts ON scenario_requests (ts);
CREATE INDEX scenario_requests_id ON scenario_requests (request_id);
