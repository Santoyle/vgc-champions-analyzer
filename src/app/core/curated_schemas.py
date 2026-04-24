"""
Schemas pandera para validar las tablas curadas del pipeline.

Cada schema corresponde a una capa de datos del proyecto:

- RAW_USAGE_SCHEMA: valida los Parquets crudos de data/raw/
  producidos por prior_ingest.py desde los chaos JSON de Smogon.

- USAGE_BY_REG_SCHEMA: valida el output de create_usage_by_reg(),
  la tabla de uso agregado por Pokémon y regulación.

- MOVES_BY_PKM_SCHEMA: valida el output de create_moves_by_pkm(),
  los moves más usados por Pokémon.

- ITEMS_BY_PKM_SCHEMA: valida el output de create_items_by_pkm(),
  los ítems más usados por Pokémon.

- TEAMMATES_BY_PKM_SCHEMA: valida el output de
  create_teammates_by_pkm(), los compañeros más frecuentes.

Todos los schemas usan strict=False para tolerar columnas extra
(reg, source) añadidas por el particionado hive de DuckDB, y
coerce=True para convertir tipos numéricos compatibles sin fallar.
"""
from __future__ import annotations

import pandera.pandas as pa  # noqa: F401  (re-exported via __all__)
from pandera.pandas import Check, Column, DataFrameSchema

RAW_USAGE_SCHEMA = DataFrameSchema(
    {
        "regulation_id": Column(
            str,
            checks=Check.isin(["I", "H", "F", "M-A"]),
            nullable=False,
        ),
        "format_id": Column(str, nullable=False),
        "year_month": Column(
            str,
            checks=Check.str_matches(r"^\d{4}-\d{2}$"),
            nullable=False,
        ),
        "cutoff": Column(
            int,
            checks=Check.isin([0, 1500, 1630, 1760]),
            nullable=False,
        ),
        "total_battles": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
        "pokemon": Column(str, nullable=False),
        "raw_count": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "usage_pct": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(100.0),
            ],
            nullable=False,
        ),
        "abilities_json": Column(str, nullable=False),
        "items_json": Column(str, nullable=False),
        "moves_json": Column(str, nullable=False),
        "teammates_json": Column(str, nullable=False),
        "spreads_json": Column(str, nullable=False),
    },
    coerce=True,
    strict=False,
)

USAGE_BY_REG_SCHEMA = DataFrameSchema(
    {
        "regulation_id": Column(str, nullable=False),
        "pokemon": Column(str, nullable=False),
        "avg_usage_pct": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(100.0),
            ],
            nullable=False,
        ),
        "total_raw_count": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
        "n_months": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
        "max_usage_pct": Column(float, nullable=False),
        "min_usage_pct": Column(float, nullable=False),
    },
    coerce=True,
    strict=False,
)

MOVES_BY_PKM_SCHEMA = DataFrameSchema(
    {
        "regulation_id": Column(str, nullable=False),
        "pokemon": Column(str, nullable=False),
        "move": Column(str, nullable=False),
        "avg_pct": Column(
            float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "n_months_seen": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,
)

ITEMS_BY_PKM_SCHEMA = DataFrameSchema(
    {
        "regulation_id": Column(str, nullable=False),
        "pokemon": Column(str, nullable=False),
        "item": Column(str, nullable=False),
        "avg_pct": Column(
            float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "n_months_seen": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,
)

TEAMMATES_BY_PKM_SCHEMA = DataFrameSchema(
    {
        "regulation_id": Column(str, nullable=False),
        "pokemon": Column(str, nullable=False),
        "teammate": Column(str, nullable=False),
        "avg_correlation": Column(
            float,
            nullable=False,
        ),
        "n_months_seen": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,
)

__all__ = [
    "RAW_USAGE_SCHEMA",
    "USAGE_BY_REG_SCHEMA",
    "MOVES_BY_PKM_SCHEMA",
    "ITEMS_BY_PKM_SCHEMA",
    "TEAMMATES_BY_PKM_SCHEMA",
]
