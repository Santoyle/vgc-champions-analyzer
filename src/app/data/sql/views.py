"""
Vistas DuckDB sobre los Parquets crudos y tablas curadas.

Este módulo crea una vista DuckDB unificada (raw_usage) sobre todos
los Parquets hive-particionados de data/raw/ y proporciona funciones
para computar tablas analíticas agregadas (uso, moves, ítems,
teammates).

Flujo de uso:
    1. Llamar register_raw_view(con) antes de cualquier query
       sobre raw_usage — registra la vista de lectura de Parquets.
    2. Llamar create_usage_by_reg / create_moves_by_pkm / etc.
       para obtener DataFrames listos para la UI.
    3. Llamar materialize_all_views(con) para pre-computar y
       persistir todos los DataFrames curados como Parquets en
       data/curated/, de modo que la UI de Streamlit los lea sin
       recalcular.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
# Forward slashes: DuckDB los acepta en Windows dentro de strings SQL
_RAW_GLOB = (_DATA_DIR / "raw" / "**" / "*.parquet").as_posix()


def register_raw_view(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Registra una vista DuckDB sobre todos los Parquets
    crudos en data/raw/ usando glob recursivo.

    La vista se llama 'raw_usage' y expone todas las
    columnas de los Parquets incluyendo las columnas
    hive (reg, source).

    Idempotente: usa CREATE OR REPLACE VIEW.

    Args:
        con: Conexión DuckDB activa.
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_usage AS
        SELECT *
        FROM read_parquet(
            '{_RAW_GLOB}',
            hive_partitioning = true,
            union_by_name = true
        )
    """)
    log.info("Vista 'raw_usage' registrada sobre %s", _RAW_GLOB)


def create_usage_by_reg(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str | None = None,
) -> pd.DataFrame:
    """
    Calcula estadísticas de uso agregadas por Pokémon
    y regulación, promediando sobre todos los meses
    disponibles del período.

    Si regulation_id es None, calcula para todas las
    regulaciones disponibles.

    Columnas del resultado:
        regulation_id: ID de la regulación
        pokemon: nombre del Pokémon
        avg_usage_pct: uso promedio sobre los meses
        total_raw_count: suma de raw_count
        n_months: número de meses con datos
        max_usage_pct: pico máximo de uso
        min_usage_pct: mínimo de uso

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Filtrar por reg específica.
                       None = todas.

    Returns:
        DataFrame ordenado por avg_usage_pct desc.
    """
    where_clause = (
        f"WHERE regulation_id = '{regulation_id}'" if regulation_id else ""
    )

    query = f"""
        SELECT
            regulation_id,
            pokemon,
            AVG(usage_pct)    AS avg_usage_pct,
            SUM(raw_count)    AS total_raw_count,
            COUNT(*)          AS n_months,
            MAX(usage_pct)    AS max_usage_pct,
            MIN(usage_pct)    AS min_usage_pct
        FROM raw_usage
        {where_clause}
        GROUP BY regulation_id, pokemon
        ORDER BY avg_usage_pct DESC
    """
    return con.execute(query).df()


def create_moves_by_pkm(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extrae los moves más usados por cada Pokémon,
    parseando el JSON string moves_json.

    Para cada Pokémon retorna los top_n moves con
    mayor porcentaje de uso promedio sobre los meses.

    Columnas del resultado:
        regulation_id, pokemon, move, avg_pct,
        n_months_seen

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Filtrar por reg específica.
        top_n: Número de moves top por Pokémon.

    Returns:
        DataFrame con moves por Pokémon.
    """
    _EMPTY_COLS_MOVES = [
        "regulation_id",
        "pokemon",
        "move",
        "avg_pct",
        "n_months_seen",
    ]

    where_clause = (
        f"WHERE regulation_id = '{regulation_id}'" if regulation_id else ""
    )

    raw_query = f"""
        SELECT regulation_id, pokemon, year_month,
               moves_json
        FROM raw_usage
        {where_clause}
    """
    df_raw = con.execute(raw_query).df()

    if df_raw.empty:
        return pd.DataFrame(columns=_EMPTY_COLS_MOVES)

    records = []
    for _, row in df_raw.iterrows():
        moves: dict[str, float] = json.loads(str(row["moves_json"]))
        for move, pct in moves.items():
            records.append(
                {
                    "regulation_id": row["regulation_id"],
                    "pokemon": row["pokemon"],
                    "year_month": row["year_month"],
                    "move": move,
                    "pct": float(pct),
                }
            )

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLS_MOVES)

    df = pd.DataFrame(records)
    result = (
        df.groupby(["regulation_id", "pokemon", "move"])
        .agg(
            avg_pct=("pct", "mean"),
            n_months_seen=("pct", "count"),
        )
        .reset_index()
        .sort_values(
            ["regulation_id", "pokemon", "avg_pct"],
            ascending=[True, True, False],
        )
    )

    return (
        result.groupby(["regulation_id", "pokemon"])
        .head(top_n)
        .reset_index(drop=True)
    )


def create_items_by_pkm(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extrae los ítems más usados por cada Pokémon,
    parseando el JSON string items_json.

    Misma estructura que create_moves_by_pkm pero
    para ítems.

    Columnas del resultado:
        regulation_id, pokemon, item, avg_pct,
        n_months_seen

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Filtrar por reg específica.
        top_n: Número de ítems top por Pokémon.

    Returns:
        DataFrame con ítems por Pokémon.
    """
    _EMPTY_COLS_ITEMS = [
        "regulation_id",
        "pokemon",
        "item",
        "avg_pct",
        "n_months_seen",
    ]

    where_clause = (
        f"WHERE regulation_id = '{regulation_id}'" if regulation_id else ""
    )

    raw_query = f"""
        SELECT regulation_id, pokemon, year_month,
               items_json
        FROM raw_usage
        {where_clause}
    """
    df_raw = con.execute(raw_query).df()

    if df_raw.empty:
        return pd.DataFrame(columns=_EMPTY_COLS_ITEMS)

    records = []
    for _, row in df_raw.iterrows():
        items: dict[str, float] = json.loads(str(row["items_json"]))
        for item, pct in items.items():
            records.append(
                {
                    "regulation_id": row["regulation_id"],
                    "pokemon": row["pokemon"],
                    "year_month": row["year_month"],
                    "item": item,
                    "pct": float(pct),
                }
            )

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLS_ITEMS)

    df = pd.DataFrame(records)
    result = (
        df.groupby(["regulation_id", "pokemon", "item"])
        .agg(
            avg_pct=("pct", "mean"),
            n_months_seen=("pct", "count"),
        )
        .reset_index()
        .sort_values(
            ["regulation_id", "pokemon", "avg_pct"],
            ascending=[True, True, False],
        )
    )

    return (
        result.groupby(["regulation_id", "pokemon"])
        .head(top_n)
        .reset_index(drop=True)
    )


def create_teammates_by_pkm(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extrae los compañeros más frecuentes por Pokémon,
    parseando teammates_json.

    Captura la correlación de co-uso: qué Pokémon
    aparecen más frecuentemente en el mismo equipo.

    Columnas del resultado:
        regulation_id, pokemon, teammate,
        avg_correlation, n_months_seen

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Filtrar por reg específica.
        top_n: Número de teammates top por Pokémon.

    Returns:
        DataFrame con teammates por Pokémon.
    """
    _EMPTY_COLS_TM = [
        "regulation_id",
        "pokemon",
        "teammate",
        "avg_correlation",
        "n_months_seen",
    ]

    where_clause = (
        f"WHERE regulation_id = '{regulation_id}'" if regulation_id else ""
    )

    raw_query = f"""
        SELECT regulation_id, pokemon, year_month,
               teammates_json
        FROM raw_usage
        {where_clause}
    """
    df_raw = con.execute(raw_query).df()

    if df_raw.empty:
        return pd.DataFrame(columns=_EMPTY_COLS_TM)

    records = []
    for _, row in df_raw.iterrows():
        teammates: dict[str, float] = json.loads(str(row["teammates_json"]))
        for teammate, corr in teammates.items():
            records.append(
                {
                    "regulation_id": row["regulation_id"],
                    "pokemon": row["pokemon"],
                    "year_month": row["year_month"],
                    "teammate": teammate,
                    "correlation": float(corr),
                }
            )

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLS_TM)

    df = pd.DataFrame(records)
    result = (
        df.groupby(["regulation_id", "pokemon", "teammate"])
        .agg(
            avg_correlation=("correlation", "mean"),
            n_months_seen=("correlation", "count"),
        )
        .reset_index()
        .sort_values(
            ["regulation_id", "pokemon", "avg_correlation"],
            ascending=[True, True, False],
        )
    )

    return (
        result.groupby(["regulation_id", "pokemon"])
        .head(top_n)
        .reset_index(drop=True)
    )


def _write_curated(
    df: pd.DataFrame,
    reg_id: str,
    tabla: str,
) -> None:
    """
    Escribe un DataFrame como Parquet en la ruta
    curated correspondiente.

    Ruta: data/curated/reg={reg_id}/{tabla}/data.parquet

    Args:
        df: DataFrame a escribir.
        reg_id: ID de la regulación.
        tabla: Nombre de la tabla curada.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.app.utils.db import get_curated_path

    curated_dir = get_curated_path(reg_id, tabla)
    output_path = curated_dir / "data.parquet"

    if df.empty:
        log.warning(
            "DataFrame vacío para %s/%s — no se escribe Parquet.",
            reg_id,
            tabla,
        )
        return

    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(output_path), compression="snappy")
    log.info("Curated: %s (%d filas).", output_path, len(df))


def materialize_all_views(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str | None = None,
) -> dict[str, int]:
    """
    Ejecuta todas las vistas y materializa los
    resultados como Parquets curados en:
        data/curated/reg={reg_id}/{tabla}/

    Si regulation_id es None, materializa para todas
    las regulaciones disponibles en raw_usage.

    Esto pre-computa los DataFrames para que la UI
    de Streamlit pueda leerlos sin re-calcular.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Materializar solo esta reg.
                       None = todas las disponibles.

    Returns:
        Dict {tabla: n_filas} con conteo de filas
        por tabla materializada.
    """
    register_raw_view(con)

    if regulation_id:
        reg_ids: list[str] = [regulation_id]
    else:
        regs_df = con.execute(
            "SELECT DISTINCT regulation_id "
            "FROM raw_usage ORDER BY regulation_id"
        ).df()
        reg_ids = regs_df["regulation_id"].tolist()

    log.info(
        "Materializando vistas para %d regulaciones: %s",
        len(reg_ids),
        reg_ids,
    )

    row_counts: dict[str, int] = {}

    for reg_id in reg_ids:
        df_usage = create_usage_by_reg(con, reg_id)
        _write_curated(df_usage, reg_id, "usage_by_reg")
        row_counts[f"{reg_id}/usage_by_reg"] = len(df_usage)

        df_moves = create_moves_by_pkm(con, reg_id)
        _write_curated(df_moves, reg_id, "moves_by_pkm")
        row_counts[f"{reg_id}/moves_by_pkm"] = len(df_moves)

        df_items = create_items_by_pkm(con, reg_id)
        _write_curated(df_items, reg_id, "items_by_pkm")
        row_counts[f"{reg_id}/items_by_pkm"] = len(df_items)

        df_tm = create_teammates_by_pkm(con, reg_id)
        _write_curated(df_tm, reg_id, "teammates_by_pkm")
        row_counts[f"{reg_id}/teammates_by_pkm"] = len(df_tm)

    return row_counts


__all__ = [
    "register_raw_view",
    "create_usage_by_reg",
    "create_moves_by_pkm",
    "create_items_by_pkm",
    "create_teammates_by_pkm",
    "materialize_all_views",
]
