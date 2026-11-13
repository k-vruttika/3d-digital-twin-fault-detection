-- Alert lifecycle (ADR 0011, ISA-18.2): every transition is a new row with the next rev.
-- The unique key must contain the hypertable's partitioning column ts.

ALTER TABLE alerts
    ADD COLUMN active       boolean     NOT NULL DEFAULT true,
    ADD COLUMN acknowledged boolean     NOT NULL DEFAULT false,
    ADD COLUMN acked_by     text,
    ADD COLUMN acked_at     timestamptz,
    ADD COLUMN rev          integer     NOT NULL DEFAULT 0;

DO $$
DECLARE
    c text;
BEGIN
    FOR c IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'alerts'::regclass AND contype = 'u'
    LOOP
        EXECUTE format('ALTER TABLE alerts DROP CONSTRAINT %I', c);
    END LOOP;
END
$$;

ALTER TABLE alerts ADD CONSTRAINT alerts_alert_id_rev_ts_key UNIQUE (alert_id, rev, ts);
CREATE INDEX alerts_id ON alerts (alert_id, rev);
