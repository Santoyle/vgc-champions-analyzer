"""
Conexiones cacheadas a bases de datos y helpers de rutas.

Separación de motores por diseño:
  DuckDB  → análisis OLAP: queries analíticas sobre archivos Parquet.
            Optimizado para lecturas masivas, columnar, sin concurrencia
            de escritura simultánea.
  SQLite  → estado OLTP del usuario: equipos guardados, roster personal,
            notas. Optimizado para escrituras frecuentes y pequeñas.

NUNCA mezclar queries de DuckDB y SQLite en la misma operación.
Si un módulo necesita datos de ambas fuentes, consulta cada motor
por separado y combina los resultados en Python.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

# db.py está en src/app/utils/ — 4 niveles abajo de la raíz del proyecto:
# raíz ← src/ ← src/app/ ← src/app/utils/ ← src/app/utils/db.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Conexiones cacheadas
# ---------------------------------------------------------------------------


def get_duckdb() -> duckdb.DuckDBPyConnection:
    """
    Retorna una conexión DuckDB cacheada como recurso compartido.

    Usa @st.cache_resource porque DuckDBPyConnection no es pickleable
    (no puede usarse con cache_data). Una sola instancia se comparte
    entre todos los reruns de la misma sesión.

    La base de datos se crea en data/vgc.duckdb.
    No registra views ni crea tablas — eso es responsabilidad de los
    módulos que usan datos reales. Este módulo solo gestiona la conexión.

    Returns:
        Conexión DuckDB activa en lectura/escritura.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _get_duckdb_cached() -> duckdb.DuckDBPyConnection:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = _DATA_DIR / "vgc.duckdb"
        log.info("Abriendo DuckDB en %s", db_path)
        return duckdb.connect(str(db_path))

    return _get_duckdb_cached()


def get_sqlite() -> sqlite3.Connection:
    """
    Retorna una conexión SQLite cacheada para el estado mutable del usuario.

    Separada de DuckDB por diseño: SQLite gestiona equipos guardados,
    notas y roster del usuario. DuckDB gestiona analytics sobre Parquet.
    NUNCA mezclar queries entre ambas.

    Aplica WAL mode para mejor concurrencia con múltiples lectores.

    Crea las tablas necesarias si no existen:
        saved_teams: equipos guardados por el usuario.
        user_roster: Pokémon del roster personal del usuario.
        user_notes: notas libres asociadas a cualquier entidad.

    Returns:
        Conexión SQLite activa con WAL mode habilitado.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _get_sqlite_cached() -> sqlite3.Connection:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = _DATA_DIR / "app.sqlite"
        log.info("Abriendo SQLite en %s", db_path)

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_teams (
                team_id        TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                regulation_id  TEXT NOT NULL,
                paste_showdown TEXT,
                created_at     TEXT NOT NULL,
                notes          TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_roster (
                dex_id      INTEGER PRIMARY KEY,
                species     TEXT NOT NULL,
                ability     TEXT,
                stat_points TEXT,
                nature      TEXT,
                moves       TEXT,
                item        TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_notes (
                note_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                text        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)

        conn.commit()
        return conn

    return _get_sqlite_cached()


# ---------------------------------------------------------------------------
# Helpers de rutas hive-particionadas
# ---------------------------------------------------------------------------


def get_parquet_path(reg_id: str, source: str, fecha: str) -> Path:
    """
    Retorna la ruta hive-particionada para un archivo Parquet de datos raw.

    Convención de particionado:
        data/raw/reg={reg_id}/source={source}/{fecha}.parquet

    Ejemplos:
        get_parquet_path("M-A", "pikalytics", "2026-05-01")
        → data/raw/reg=M-A/source=pikalytics/2026-05-01.parquet

        get_parquet_path("M-A", "showdown", "2026-05-01")
        → data/raw/reg=M-A/source=showdown/2026-05-01.parquet

    NO crea el archivo ni el directorio padre. Los scrapers son
    responsables de crear los directorios antes de escribir.

    Args:
        reg_id: ID de la regulación. NUNCA hardcodear — siempre
                recibirlo como parámetro.
        source: Nombre de la fuente de datos. Valores esperados:
                "pikalytics", "showdown", "limitless", "rk9".
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        Path absoluto al archivo Parquet.
    """
    return _DATA_DIR / "raw" / f"reg={reg_id}" / f"source={source}" / f"{fecha}.parquet"


def get_curated_path(reg_id: str, tabla: str) -> Path:
    """
    Retorna el directorio para datos curados de una tabla específica.

    Convención:
        data/curated/reg={reg_id}/{tabla}/

    Ejemplo:
        get_curated_path("M-A", "usage_stats")
        → data/curated/reg=M-A/usage_stats/

    A diferencia de get_parquet_path, crea el directorio si no existe
    — los módulos de transformación pueden escribir directamente sin
    crear el directorio previamente.

    Args:
        reg_id: ID de la regulación.
        tabla: Nombre de la tabla curada. Valores esperados:
               "usage_stats", "moves_by_pkm", "items_by_pkm",
               "teammates_by_pkm", "counters_by_pkm".

    Returns:
        Path absoluto al directorio de la tabla (creado si no existía).
    """
    path = _DATA_DIR / "curated" / f"reg={reg_id}" / tabla
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Helper de conveniencia para queries
# ---------------------------------------------------------------------------


def query_duckdb(
    sql: str,
    params: list[object] | None = None,
) -> list[tuple[object, ...]]:
    """
    Ejecuta una query SQL en DuckDB y retorna los resultados como
    lista de tuplas.

    Helper de conveniencia para queries simples que no necesitan
    DataFrame. Para análisis más complejos, usar get_duckdb()
    directamente con .df() o .pl().

    Args:
        sql: Query SQL a ejecutar.
        params: Parámetros posicionales opcionales para la query
                (evita SQL injection).

    Returns:
        Lista de tuplas con los resultados. Lista vacía si no hay
        resultados.
    """
    conn = get_duckdb()
    if params:
        result = conn.execute(sql, params)
    else:
        result = conn.execute(sql)
    rows: list[tuple[object, ...]] = result.fetchall()
    return rows


__all__ = [
    "get_duckdb",
    "get_sqlite",
    "get_parquet_path",
    "get_curated_path",
    "query_duckdb",
]
