-- Pending continuous-aggregate refreshes for readings written outside the refresh-policy window
-- (replayed recordings, backfills). See ADR 0009 and ADR 0010.
--
-- The store inserts one row per batch, in the same transaction as the readings, covering
-- [lo, hi] of the batch's old readings. Before a 1 min / 10 min history query, any process
-- refreshes the pending ranges that overlap the query and deletes them after the refresh
-- succeeded. Because the queue lives in the database, ranges written by another process, or
-- before a restart, are not lost.

CREATE TABLE aggregate_refresh_queue (
    id         bigserial   PRIMARY KEY,
    lo         timestamptz NOT NULL,
    hi         timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (lo <= hi)
);
CREATE INDEX aggregate_refresh_queue_range ON aggregate_refresh_queue (lo, hi);
