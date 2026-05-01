{{ config(materialized='table') }}

SELECT
    regulation_id,
    pokemon,
    year_month,
    items_json
FROM raw_usage
WHERE items_json IS NOT NULL
  AND items_json != '{}'
