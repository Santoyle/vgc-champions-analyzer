"""
Tests para el Meta Overview y el módulo de rating Glicko-2.

Grupos:
  1 — Propiedades de datos (Parquets reales). Usa pytest.skip()
      automáticamente si no hay datos disponibles localmente.
  2 — Tests unitarios puros de PlayerRating (dataclass).
  3 — Tests unitarios puros de update_rating (algoritmo Glicko-2).
  4 — Tests de update_ratings_from_replays (procesamiento por lotes).
  5 — Tests de persist_ratings (persistencia SQLite).

Los Grupos 2-5 no dependen de archivos externos y siempre se ejecutan.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.app.data.sql.views import (
    create_items_by_pkm,
    create_moves_by_pkm,
    create_usage_by_reg,
    register_raw_view,
)
from src.app.modules.rating import (
    DEFAULT_RATING,
    DEFAULT_RD,
    SCALE,
    PlayerRating,
    persist_ratings,
    update_rating,
    update_ratings_from_replays,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
KNOWN_REGS_WITH_DATA = ["I", "H"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Conexión DuckDB en memoria con vista raw registrada."""
    con = duckdb.connect(":memory:")
    register_raw_view(con)
    return con


@pytest.fixture(scope="module")
def sqlite_con() -> sqlite3.Connection:
    """Conexión SQLite en memoria para tests de rating."""
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA journal_mode=WAL")
    return con


# ---------------------------------------------------------------------------
# Dataclass helper para tests de replays
# ---------------------------------------------------------------------------


@dataclass
class MockReplay:
    """Replay sintético para tests de rating (duck-typed, sin importar ParsedReplay)."""

    p1: str
    p2: str
    winner: str | None


# ---------------------------------------------------------------------------
# Grupo 1 — Tests de propiedades de datos
# ---------------------------------------------------------------------------


class TestUsageDataProperties:
    """
    Verifica propiedades semánticas de los datos de uso cargados desde
    los Parquets reales. Salta automáticamente si no hay datos.
    """

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_dataframe_not_empty(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """create_usage_by_reg retorna DF no vacío."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        assert len(df) > 0

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_has_required_columns(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """DataFrame tiene las columnas esperadas."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        required = {
            "regulation_id",
            "pokemon",
            "avg_usage_pct",
            "total_raw_count",
            "n_months",
        }
        assert required.issubset(set(df.columns))

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_usage_pct_in_valid_range(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """avg_usage_pct está entre 0 y 100."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        assert df["avg_usage_pct"].min() >= 0.0
        assert df["avg_usage_pct"].max() <= 100.0

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_pokemon_names_are_non_empty_strings(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Todos los nombres de Pokémon son strings no vacíos."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        assert df["pokemon"].notna().all()
        assert (df["pokemon"].str.len() > 0).all()

    @pytest.mark.parametrize("reg_id", KNOWN_REGS_WITH_DATA)
    def test_no_duplicate_pokemon_per_reg(
        self,
        reg_id: str,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """No hay Pokémon duplicados en usage_by_reg para una regulación."""
        df = create_usage_by_reg(duckdb_con, reg_id)
        if df.empty:
            pytest.skip(f"Sin datos para {reg_id}")
        df_reg = df[df["regulation_id"] == reg_id]
        assert df_reg["pokemon"].nunique() == len(df_reg), (
            "Hay Pokémon duplicados en usage_by_reg"
        )

    def test_incineroar_top_usage_reg_i(
        self,
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Incineroar aparece en el top-5 de uso de Reg I (es el Pokémon
        más usado históricamente en VGC)."""
        df = create_usage_by_reg(duckdb_con, "I")
        if df.empty:
            pytest.skip("Sin datos para Reg I")
        top5 = df.head(5)["pokemon"].str.lower().tolist()
        assert "incineroar" in top5, (
            f"Incineroar no está en top-5 de Reg I. Top-5 actual: {top5}"
        )


# ---------------------------------------------------------------------------
# Grupo 2 — Tests de PlayerRating
# ---------------------------------------------------------------------------


class TestPlayerRating:
    """Tests unitarios del dataclass PlayerRating."""

    def test_default_rating_is_1500(self) -> None:
        """Un jugador nuevo tiene rating ELO 1500."""
        player = PlayerRating(player_name="Ash", regulation_id="M-A")
        assert player.rating == pytest.approx(DEFAULT_RATING, abs=0.01)

    def test_default_rd_is_350(self) -> None:
        """Un jugador nuevo tiene RD de 350."""
        player = PlayerRating(player_name="Ash", regulation_id="M-A")
        assert player.rd == pytest.approx(DEFAULT_RD, abs=1.0)

    def test_mu_zero_gives_rating_1500(self) -> None:
        """mu=0 corresponde exactamente a 1500 ELO."""
        player = PlayerRating(player_name="Test", regulation_id="M-A", mu=0.0)
        assert player.rating == pytest.approx(1500.0, abs=0.001)

    def test_confidence_interval_width(self) -> None:
        """El intervalo de confianza tiene ancho 4 * rd (±2σ)."""
        player = PlayerRating(player_name="Test", regulation_id="M-A")
        low, high = player.confidence_interval
        assert high - low == pytest.approx(4 * player.rd, abs=0.01)

    def test_rating_scales_with_mu(self) -> None:
        """Mayor mu implica mayor rating ELO."""
        p_low = PlayerRating("Low", "M-A", mu=-1.0)
        p_mid = PlayerRating("Mid", "M-A", mu=0.0)
        p_high = PlayerRating("High", "M-A", mu=1.0)
        assert p_low.rating < p_mid.rating < p_high.rating


# ---------------------------------------------------------------------------
# Grupo 3 — Tests de update_rating
# ---------------------------------------------------------------------------


class TestUpdateRating:
    """Tests del algoritmo Glicko-2 update_rating."""

    def test_winner_rating_increases(self) -> None:
        """El rating del ganador aumenta después de una victoria."""
        player = PlayerRating("Winner", "M-A")
        opponent = PlayerRating("Loser", "M-A")
        updated = update_rating(player, [opponent], [1.0])
        assert updated.rating > player.rating

    def test_loser_rating_decreases(self) -> None:
        """El rating del perdedor disminuye."""
        player = PlayerRating("Loser", "M-A")
        opponent = PlayerRating("Winner", "M-A")
        updated = update_rating(player, [opponent], [0.0])
        assert updated.rating < player.rating

    def test_draw_minimal_change(self) -> None:
        """Un empate entre jugadores iguales produce cambio mínimo."""
        player = PlayerRating("P1", "M-A")
        opponent = PlayerRating("P2", "M-A")
        updated = update_rating(player, [opponent], [0.5])
        assert abs(updated.rating - player.rating) < 10

    def test_n_games_increases(self) -> None:
        """n_games aumenta correctamente."""
        player = PlayerRating("P1", "M-A")
        opponent = PlayerRating("P2", "M-A")
        updated = update_rating(player, [opponent], [1.0])
        assert updated.n_games == 1

    def test_rd_decreases_with_games(self) -> None:
        """La incertidumbre (RD/phi) disminuye con más partidas."""
        player = PlayerRating("P1", "M-A")
        opponent = PlayerRating("P2", "M-A")
        updated = update_rating(player, [opponent], [1.0])
        assert updated.rd < player.rd

    def test_raises_on_empty_opponents(self) -> None:
        """ValueError si opponents está vacío."""
        player = PlayerRating("P1", "M-A")
        with pytest.raises(ValueError):
            update_rating(player, [], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        """ValueError si len(opponents) != len(scores)."""
        player = PlayerRating("P1", "M-A")
        opponent = PlayerRating("P2", "M-A")
        with pytest.raises(ValueError):
            update_rating(player, [opponent], [1.0, 0.0])

    def test_regulation_id_preserved(self) -> None:
        """El regulation_id se preserva en el resultado."""
        player = PlayerRating("P1", "M-A")
        opponent = PlayerRating("P2", "M-A")
        updated = update_rating(player, [opponent], [1.0])
        assert updated.regulation_id == "M-A"

    def test_player_name_preserved(self) -> None:
        """El player_name se preserva."""
        player = PlayerRating("SantoPlayer", "M-A")
        opponent = PlayerRating("Rival", "M-A")
        updated = update_rating(player, [opponent], [1.0])
        assert updated.player_name == "SantoPlayer"


# ---------------------------------------------------------------------------
# Grupo 4 — Tests de update_ratings_from_replays
# ---------------------------------------------------------------------------


class TestUpdateRatingsFromReplays:
    """Tests de la función que procesa múltiples replays."""

    def test_creates_ratings_for_all_players(self) -> None:
        """Crea ratings para todos los jugadores en los replays."""
        replays = [
            MockReplay("Alice", "Bob", "Alice"),
            MockReplay("Carol", "Dave", "Dave"),
        ]
        ratings = update_ratings_from_replays(
            replays,  # type: ignore[arg-type]
            "M-A",
        )
        assert "Alice" in ratings
        assert "Bob" in ratings
        assert "Carol" in ratings
        assert "Dave" in ratings

    def test_winner_has_higher_rating_than_loser(self) -> None:
        """Después de varios juegos, el ganador constante tiene rating mayor."""
        replays = [
            MockReplay("Champion", "Victim", "Champion") for _ in range(5)
        ]
        ratings = update_ratings_from_replays(
            replays,  # type: ignore[arg-type]
            "M-A",
        )
        assert ratings["Champion"].rating > ratings["Victim"].rating

    def test_handles_none_winner_as_draw(self) -> None:
        """winner=None se trata como empate."""
        replays = [MockReplay("A", "B", None)]
        ratings = update_ratings_from_replays(
            replays,  # type: ignore[arg-type]
            "M-A",
        )
        assert abs(ratings["A"].rating - ratings["B"].rating) < 5

    def test_skips_replays_with_empty_players(self) -> None:
        """Replays con p1 o p2 vacío se ignoran sin error."""
        replays = [
            MockReplay("", "Bob", "Bob"),
            MockReplay("Alice", "Carol", "Alice"),
        ]
        ratings = update_ratings_from_replays(
            replays,  # type: ignore[arg-type]
            "M-A",
        )
        assert "Alice" in ratings
        assert "Carol" in ratings
        assert "" not in ratings

    def test_with_existing_ratings(self) -> None:
        """Usa ratings existentes como punto de partida."""
        existing = {"Expert": PlayerRating("Expert", "M-A", mu=2.0)}
        replays = [MockReplay("Expert", "Newbie", "Expert")]
        ratings = update_ratings_from_replays(
            replays,  # type: ignore[arg-type]
            "M-A",
            existing_ratings=existing,
        )
        assert ratings["Expert"].mu > 1.5


# ---------------------------------------------------------------------------
# Grupo 5 — Tests de persist_ratings
# ---------------------------------------------------------------------------


class TestPersistRatings:
    """Tests de persistencia en SQLite."""

    def test_persist_creates_table(
        self,
        sqlite_con: sqlite3.Connection,
    ) -> None:
        """persist_ratings crea la tabla si no existe."""
        ratings = {"TestPlayer": PlayerRating("TestPlayer", "M-A")}
        persist_ratings(ratings, sqlite_con)
        cursor = sqlite_con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='player_ratings'"
        )
        assert cursor.fetchone() is not None

    def test_persist_saves_correct_rating_elo(
        self,
        sqlite_con: sqlite3.Connection,
    ) -> None:
        """Los valores ELO persistidos son correctos."""
        player = PlayerRating("EloTest", "M-A", mu=0.0)
        persist_ratings({"EloTest": player}, sqlite_con)
        cursor = sqlite_con.execute(
            "SELECT rating_elo FROM player_ratings "
            "WHERE player_name = 'EloTest'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert abs(row[0] - 1500.0) < 0.01

    def test_persist_upsert_updates_existing(
        self,
        sqlite_con: sqlite3.Connection,
    ) -> None:
        """persist_ratings actualiza un registro existente (upsert, sin duplicar)."""
        player_v1 = PlayerRating("UpsertTest", "M-A", mu=0.0)
        persist_ratings({"UpsertTest": player_v1}, sqlite_con)

        player_v2 = PlayerRating("UpsertTest", "M-A", mu=1.0)
        persist_ratings({"UpsertTest": player_v2}, sqlite_con)

        cursor = sqlite_con.execute(
            "SELECT COUNT(*) FROM player_ratings "
            "WHERE player_name = 'UpsertTest'"
        )
        count = cursor.fetchone()[0]
        assert count == 1

        cursor = sqlite_con.execute(
            "SELECT mu FROM player_ratings WHERE player_name = 'UpsertTest'"
        )
        mu_stored = cursor.fetchone()[0]
        assert mu_stored == pytest.approx(1.0, abs=0.001)

    def test_persist_returns_count(
        self,
        sqlite_con: sqlite3.Connection,
    ) -> None:
        """persist_ratings retorna el número de ratings insertados/actualizados."""
        ratings = {
            f"Player{i}": PlayerRating(f"Player{i}", "M-A") for i in range(5)
        }
        count = persist_ratings(ratings, sqlite_con)
        assert count == 5
