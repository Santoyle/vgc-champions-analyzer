{{ config(materialized='table') }}
SELECT
    regulation_id,
    pokemon,
    AVG(usage_pct)    AS avg_usage_pct,
    COUNT(*)          AS n_months,
    MAX(usage_pct)    AS max_usage_pct,
    MIN(usage_pct)    AS min_usage_pct
FROM raw_usage
WHERE source = 'pikalytics'
  AND pokemon IS NOT NULL
GROUP BY regulation_id, pokemon
ORDER BY avg_usage_pct DESC
