{{ config(materialized='table') }}

SELECT
    regulation_id,
    pokemon,
    year_month,
    teammates_json
FROM raw_usage
WHERE teammates_json IS NOT NULL
  AND teammates_json != '{}'
