-- linetwin: no-transaction
-- Continuous aggregates over readings (continuous aggregates cannot be created in a
-- transaction block, hence the marker above). See ADR 0009.
--
-- readings_1m keeps sum, count, min and max per sensor so that means stay exact when rolled up:
-- mean = sum / count, where count(col) and sum(col) ignore NULL (missing) values. readings_10m
-- is a hierarchical aggregate on top of readings_1m. LATE and SUSPECT readings are included and
-- counted. Both use real-time aggregation (materialized_only = false): buckets newer than the
-- materialization watermark are computed from raw readings at query time. Refresh policies are
-- applied from config/storage.yaml at startup.

CREATE MATERIALIZED VIEW readings_1m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 minute', ts) AS bucket,
    machine_id,
    count(*) AS n,
    count(*) FILTER (WHERE quality = 'LATE') AS n_late,
    count(*) FILTER (WHERE quality = 'SUSPECT') AS n_suspect,
    sum(speed) AS speed_sum, count(speed) AS speed_count,
    min(speed) AS speed_min, max(speed) AS speed_max,
    sum(temp) AS temp_sum, count(temp) AS temp_count,
    min(temp) AS temp_min, max(temp) AS temp_max,
    sum(vib) AS vib_sum, count(vib) AS vib_count,
    min(vib) AS vib_min, max(vib) AS vib_max,
    sum(load) AS load_sum, count(load) AS load_count,
    min(load) AS load_min, max(load) AS load_max,
    sum(current) AS current_sum, count(current) AS current_count,
    min(current) AS current_min, max(current) AS current_max
FROM readings
GROUP BY 1, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW readings_10m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '10 minutes', bucket) AS bucket,
    machine_id,
    sum(n)::bigint AS n,
    sum(n_late)::bigint AS n_late,
    sum(n_suspect)::bigint AS n_suspect,
    sum(speed_sum) AS speed_sum, sum(speed_count)::bigint AS speed_count,
    min(speed_min) AS speed_min, max(speed_max) AS speed_max,
    sum(temp_sum) AS temp_sum, sum(temp_count)::bigint AS temp_count,
    min(temp_min) AS temp_min, max(temp_max) AS temp_max,
    sum(vib_sum) AS vib_sum, sum(vib_count)::bigint AS vib_count,
    min(vib_min) AS vib_min, max(vib_max) AS vib_max,
    sum(load_sum) AS load_sum, sum(load_count)::bigint AS load_count,
    min(load_min) AS load_min, max(load_max) AS load_max,
    sum(current_sum) AS current_sum, sum(current_count)::bigint AS current_count,
    min(current_min) AS current_min, max(current_max) AS current_max
FROM readings_1m
GROUP BY 1, 2
WITH NO DATA;
