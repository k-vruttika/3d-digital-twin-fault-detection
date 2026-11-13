-- Scenario injection audit (simulation only, ADR 0015). One row per request (kind 'request')
-- and one per acknowledgement (kind 'ack'), so injected faults can be told apart from real
-- ones in replay and experiments. ts is event time; the JSON payload is the full contract.

CREATE TABLE scenario_requests (
    ts          timestamptz NOT NULL,
    request_id  text        NOT NULL,
    kind        text        NOT NULL,
    action      text,
    machine_id  text,
    status      text,
    payload     text        NOT NULL,
    ingested_at timestamptz NOT NULL,
    UNIQUE (request_id, kind, ts)
);
SELECT create_hypertable('scenario_requests', by_range('ts', INTERVAL '7 days'));
CREATE INDEX scenario_requests_id ON scenario_requests (request_id);
