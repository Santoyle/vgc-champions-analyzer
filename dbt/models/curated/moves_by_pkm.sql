{{ config(materialized='table') }}
SELECT
    regulation_id,
    pokemon,
    NULL::VARCHAR AS move,
    NULL::FLOAT   AS avg_pct,
    0             AS n_months_seen
FROM raw_usage
WHERE 1=0
