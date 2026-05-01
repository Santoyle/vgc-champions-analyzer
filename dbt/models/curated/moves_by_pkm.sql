{{ config(materialized='table') }}

SELECT
    regulation_id,
    pokemon,
    year_month,
    moves_json
FROM raw_usage
WHERE moves_json IS NOT NULL
  AND moves_json != '{}'
