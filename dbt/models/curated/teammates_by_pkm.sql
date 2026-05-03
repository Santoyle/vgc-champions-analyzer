{{ config(materialized='table') }}
SELECT
    regulation_id,
    pokemon,
    NULL::VARCHAR AS teammate,
    NULL::FLOAT   AS avg_correlation,
    0             AS n_months_seen
FROM raw_usage
WHERE 1=0
