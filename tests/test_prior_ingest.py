"""
Tests de integración para el pipeline prior_ingest y las vistas DuckDB.

Valida cuatro aspectos del pipeline:
  1. Existencia y formato de los Parquets crudos en data/raw/.
  2. Conformidad de los datos crudos con RAW_USAGE_SCHEMA.
  3. Corrección del output de las funciones de vistas curadas.
  4. Propiedades semánticas de los datos (sanity checks).

Los tests se parametrizan sobre KNOWN_REGS_WITH_DATA = ["I", "H"],
regulaciones con Parquets confirmados. La Reg F se excluye porque
sus meses retornaron 404 en Smogon al momento de la ingesta.

La fixture duckdb_con usa scope="module" para crear la conexión
una sola vez y reutilizarla en todos los tests del módulo.
"""
from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.app.core.curated_schemas import (
    ITEMS_BY_PKM_SCHEMA,
    MOVES_BY_PKM_SCHEMA,
    RAW_USAGE_SCHEMA,
    TEAMMATES_BY_PKM_SCHEMA,
    USAGE_BY_REG_SCHEMA,
)
from src.app.data.sql.views import (
    create_items_by_pkm,
    create_moves_by_pkm,
    create_teammates_by_pkm,
    create_usage_by_reg,
    register_raw_view,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_CURATED_DIR = _PROJECT_ROOT / "data" / "curated"

KNOWN_REGS_WITH_DATA = ["I", "H"]


@pytest.fixture(scope="module")
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Conexión DuckDB en memoria para los tests.
    No usa la DB de producción."""
    con = duckdb.connect(":memory:")
    register_raw_view(con)
    return con


class TestRawParquets:
    """Verifica que los Parquets crudos existen
    y tienen el formato correcto."""

    def test_raw_parquets_exist(self) -> None:
        """Hay al menos un Parquet en data/raw/."""
        parquets = glob.glob(
            str(_RAW_DIR / "**" / "*.parquet"),
            recursive=True,
        )
        assert len(parquets) > 0, (
            "No hay Parquets en data/raw/. "
            "Ejecuta: python -m src.app.data.pipelines"
            ".prior_ingest --regs I H --cutoff 1760"
        )

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_raw_parquets_exist_per_reg(self, reg_id: str) -> None:
        """Hay Parquets para cada regulación conocida."""
        reg_dir = _RAW_DIR / f"reg={reg_id}"
        parquets = glob.glob(
            str(reg_dir / "**" / "*.parquet"),
            recursive=True,
        )
        assert len(parquets) > 0, (
            f"No hay Parquets para Reg {reg_id}. "
            f"Directorio esperado: {reg_dir}"
        )

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_raw_parquet_columns(self, reg_id: str) -> None:
        """Los Parquets tienen las columnas esperadas."""
        reg_dir = _RAW_DIR / f"reg={reg_id}"
        parquets = glob.glob(
            str(reg_dir / "**" / "*.parquet"),
            recursive=True,
        )
        if not parquets:
            pytest.skip(f"Sin Parquets para Reg {reg_id}")

        table = pq.read_table(parquets[0])
        columns = set(table.column_names)
        required = {
            "regulation_id",
            "pokemon",
            "raw_count",
            "usage_pct",
            "moves_json",
            "items_json",
            "teammates_json",
        }
        missing = required - columns
        assert not missing, (
            f"Columnas faltantes en {parquets[0]}: {missing}"
        )

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_raw_parquet_not_empty(self, reg_id: str) -> None:
        """Los Parquets tienen al menos 50 filas."""
        reg_dir = _RAW_DIR / f"reg={reg_id}"
        parquets = glob.glob(
            str(reg_dir / "**" / "*.parquet"),
            recursive=True,
        )
        if not parquets:
            pytest.skip(f"Sin Parquets para Reg {reg_id}")

        table = pq.read_table(parquets[0])
        assert len(table) >= 50, (
            f"Parquet con muy pocas filas: {len(table)}"
        )


class TestRawSchemaValidation:
    """Valida los Parquets crudos contra RAW_USAGE_SCHEMA."""

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_raw_schema_valid(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Los datos crudos pasan el schema pandera."""
        df = duckdb_con.execute(
            "SELECT * FROM raw_usage WHERE regulation_id = ? LIMIT 200",
            [reg_id],
        ).df()

        if df.empty:
            pytest.skip(f"Sin datos en raw_usage para {reg_id}")

        RAW_USAGE_SCHEMA.validate(df)

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_raw_usage_pct_range(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """usage_pct está entre 0 y 100."""
        df = duckdb_con.execute(
            "SELECT usage_pct FROM raw_usage WHERE regulation_id = ?",
            [reg_id],
        ).df()

        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")

        assert df["usage_pct"].min() >= 0.0
        assert df["usage_pct"].max() <= 100.0


class TestCuratedViews:
    """Valida el output de las funciones de vistas."""

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_by_reg_not_empty(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_usage_by_reg retorna DataFrame no vacío."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        assert not df.empty, f"usage_by_reg vacío para {reg_id}"

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_by_reg_schema(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_usage_by_reg pasa el schema pandera."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        USAGE_BY_REG_SCHEMA.validate(df)

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_moves_by_pkm_schema(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_moves_by_pkm pasa el schema pandera."""
        df = create_moves_by_pkm(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        MOVES_BY_PKM_SCHEMA.validate(df)

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_items_by_pkm_schema(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_items_by_pkm pasa el schema pandera."""
        df = create_items_by_pkm(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        ITEMS_BY_PKM_SCHEMA.validate(df)

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_teammates_by_pkm_schema(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_teammates_by_pkm pasa el schema."""
        df = create_teammates_by_pkm(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        TEAMMATES_BY_PKM_SCHEMA.validate(df)


class TestDataProperties:
    """Verifica propiedades semánticas de los datos."""

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_incineroar_in_reg_i(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Incineroar aparece en los datos de Reg I.
        Es el Pokémon más usado históricamente en VGC."""
        if reg_id != "I":
            pytest.skip("Solo relevante para Reg I")

        df = create_usage_by_reg(duckdb_con, "I")
        pokemon_names = df["pokemon"].str.lower().tolist()
        assert "incineroar" in pokemon_names, (
            "Incineroar no encontrado en Reg I — "
            "los datos pueden estar incompletos"
        )

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_by_reg_has_expected_columns(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """usage_by_reg tiene todas las columnas esperadas."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        expected = {
            "regulation_id",
            "pokemon",
            "avg_usage_pct",
            "total_raw_count",
            "n_months",
            "max_usage_pct",
            "min_usage_pct",
        }
        assert expected.issubset(set(df.columns))

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_moves_top_n_per_pokemon(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Cada Pokémon tiene como máximo 10 moves
        en moves_by_pkm (top_n default)."""
        df = create_moves_by_pkm(duckdb_con, reg_id, top_n=10)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")

        counts = df.groupby(["regulation_id", "pokemon"]).size()
        assert counts.max() <= 10, (
            f"Algún Pokémon tiene más de 10 moves: "
            f"{counts.idxmax()} = {counts.max()}"
        )
