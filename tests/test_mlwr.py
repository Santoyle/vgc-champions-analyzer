"""
Tests unitarios para MLWR nivel 1 (``compute_mlwr_level1``).

Los tests sustituyen la conexión DuckDB mediante ``MagicMock``:
``mock_con.execute(...).df()`` devuelve DataFrames sintéticos en memoria.
Se vacía explícitamente la caché de Streamlit de ``compute_mlwr_level1``
(cada ejecución vía pytest) para que ``@st.cache_data`` no reuse resultados.

Ningún test abre disco ni DuckDB real. ``test_baseline_deduped_per_battle``
comprueba el invariante principal del baseline: repetir el mismo Pokémon en
varios turnos dentro de una batalla cuenta como una sola batalla al agrupar
por ``(replay_id, player, pokemon_slug)``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import numpy as np  # noqa: F401
import pytest

from src.app.metrics.mlwr import (
    compute_mlwr_level1,
    _resolve_winner_column,
)


@pytest.fixture(autouse=True)
def _clear_streamlit_mlwr_cache() -> Generator[None, None, None]:
    """
    Vacía la caché de ``compute_mlwr_level1``.

    Streamlit marca la función con ``@st.cache_data``; cuando el segundo
    argumento es un ``MagicMock``, Streamlit suele usar la misma clave entre
    pruebas y devuelve resultados de otros tests.
    """
    cast(Any, compute_mlwr_level1).clear()
    yield
    cast(Any, compute_mlwr_level1).clear()


def _make_mock_con(df: pd.DataFrame) -> Any:
    """
    Crea un mock de conexión DuckDB que retorna
    el DataFrame dado cuando se llama execute().df().
    """
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.df.return_value = df
    mock_con.execute.return_value = mock_result
    return mock_con


def _make_replay_df(
    n_replays: int = 20,
    win_rate: float = 0.6,
    pokemon: str = "garchomp",
    move: str = "earthquake",
    regulation_id: str = "TEST",
) -> pd.DataFrame:
    """
    Genera un DataFrame sintético de replay_turns
    con señal controlada para testing.

    Cada replay tiene:
    - player p1 usando el move especificado
    - winner p1 con probabilidad win_rate
    """
    rows = []
    for i in range(n_replays):
        winner = "p1" if i < int(
            n_replays * win_rate
        ) else "p2"
        rows.append({
            "regulation_id": regulation_id,
            "replay_id": f"replay-{i}",
            "player": "p1",
            "pokemon_slug": pokemon,
            "move_slug": move,
            "winner": winner,
        })
    return pd.DataFrame(rows)


class TestResolveWinnerColumn:
    """Tests para la normalización de winner."""

    def test_p1_p2_winners_resolved(self) -> None:
        """p1/p2 en winner se resuelven
        correctamente."""
        df = pd.DataFrame({
            "player": ["p1", "p2", "p1"],
            "winner": ["p1", "p1", "p2"],
        })
        result = _resolve_winner_column(df)
        assert result.iloc[0] == True   # noqa: E712
        assert result.iloc[1] == False   # noqa: E712
        assert result.iloc[2] == False   # noqa: E712

    def test_username_winners_return_nan(
        self,
    ) -> None:
        """Usernames en winner retornan NaN."""
        df = pd.DataFrame({
            "player": ["p1", "p2"],
            "winner": ["trainername", "otherone"],
        })
        result = _resolve_winner_column(df)
        assert result.isna().all()

    def test_mixed_majority_p1p2(self) -> None:
        """Si mayoría son p1/p2, usa esos y
        NaN para los usernames."""
        df = pd.DataFrame({
            "player": ["p1", "p1", "p1", "p2"],
            "winner": ["p1", "p2", "p1", "trainer"],
        })
        result = _resolve_winner_column(df)
        assert result.iloc[0] == True  # noqa: E712
        assert result.iloc[1] == False  # noqa: E712
        assert result.iloc[2] == True  # noqa: E712
        assert pd.isna(result.iloc[3])

    def test_all_p1_wins(self) -> None:
        """Todos p1 ganan."""
        df = pd.DataFrame({
            "player": ["p1"] * 5,
            "winner": ["p1"] * 5,
        })
        result = _resolve_winner_column(df)
        assert result.all()


class TestComputeMLWRLevel1:
    """Tests para compute_mlwr_level1."""

    def test_returns_dataframe(self) -> None:
        """Retorna DataFrame."""
        df = _make_replay_df(n_replays=20)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        """DataFrame tiene las columnas esperadas."""
        df = _make_replay_df(n_replays=20)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        required = {
            "pokemon_slug", "move_slug", "n_uses",
            "wr_with_move", "wr_baseline", "mlwr",
            "wilson_low", "wilson_high",
            "regulation_id",
        }
        assert required.issubset(set(result.columns))

    def test_mlwr_in_range(self) -> None:
        """mlwr está en [-1, 1]."""
        df = _make_replay_df(n_replays=30)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        if not result.empty:
            assert (result["mlwr"] >= -1.0).all()
            assert (result["mlwr"] <= 1.0).all()

    def test_positive_mlwr_for_winning_move(
        self,
    ) -> None:
        """Move usado siempre en victorias tiene
        MLWR positivo."""
        rows = []
        # 10 replays donde p1 usa EQ y gana
        for i in range(10):
            rows.append({
                "regulation_id": "TEST",
                "replay_id": f"win-{i}",
                "player": "p1",
                "pokemon_slug": "garchomp",
                "move_slug": "earthquake",
                "winner": "p1",
            })
        # 5 replays donde p1 no usa EQ y pierde
        for i in range(5):
            rows.append({
                "regulation_id": "TEST",
                "replay_id": f"lose-{i}",
                "player": "p1",
                "pokemon_slug": "garchomp",
                "move_slug": "protect",
                "winner": "p2",
            })
        df = pd.DataFrame(rows)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        eq_rows = result[
            result["move_slug"] == "earthquake"
        ]
        assert not eq_rows.empty
        assert eq_rows.iloc[0]["mlwr"] > 0

    def test_sorted_by_mlwr_descending(
        self,
    ) -> None:
        """Resultado ordenado por mlwr DESC."""
        df = _make_replay_df(n_replays=30)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        if len(result) >= 2:
            assert all(
                result["mlwr"].iloc[i]
                >= result["mlwr"].iloc[i+1]
                for i in range(len(result)-1)
            )

    def test_min_n_filter(self) -> None:
        """min_n filtra pares con pocos usos."""
        df = _make_replay_df(
            n_replays=5, move="rare-move"
        )
        con = _make_mock_con(df)
        result_n1 = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        result_n10 = compute_mlwr_level1(
            "TEST", con, min_n=10
        )
        assert len(result_n1) >= len(result_n10)

    def test_empty_df_returns_empty(self) -> None:
        """DataFrame vacío retorna DataFrame vacío
        con columnas correctas."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        assert result.empty
        assert "pokemon_slug" in result.columns
        assert "mlwr" in result.columns

    def test_username_winners_returns_empty(
        self,
    ) -> None:
        """Si winner son usernames (no p1/p2),
        retorna DataFrame vacío."""
        df = pd.DataFrame({
            "regulation_id": ["TEST"] * 5,
            "replay_id": [f"r{i}" for i in range(5)],
            "player": ["p1"] * 5,
            "pokemon_slug": ["garchomp"] * 5,
            "move_slug": ["earthquake"] * 5,
            "winner": ["trainer1"] * 5,
        })
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        assert result.empty

    def test_wilson_ci_brackets_mlwr(
        self,
    ) -> None:
        """wilson_low <= mlwr <= wilson_high."""
        df = _make_replay_df(n_replays=30)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        if not result.empty:
            assert (
                result["wilson_low"]
                <= result["mlwr"]
            ).all()
            assert (
                result["mlwr"]
                <= result["wilson_high"]
            ).all()

    def test_regulation_id_in_result(self) -> None:
        """regulation_id del input aparece en
        resultado."""
        df = _make_replay_df(
            n_replays=10, regulation_id="M-A"
        )
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "M-A", con, min_n=1
        )
        if not result.empty:
            assert (
                result["regulation_id"] == "M-A"
            ).all()

    def test_baseline_deduped_per_battle(
        self,
    ) -> None:
        """El baseline no se infla por múltiples
        turnos del mismo Pokémon en la misma
        batalla."""
        rows = []
        for i in range(5):
            winner = "p1" if i < 3 else "p2"
            for turn in range(5):
                rows.append({
                    "regulation_id": "TEST",
                    "replay_id": f"r{i}",
                    "player": "p1",
                    "pokemon_slug": "incineroar",
                    "move_slug": "fake-out",
                    "winner": winner,
                })
        df = pd.DataFrame(rows)
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        inc = result[
            result["pokemon_slug"] == "incineroar"
        ]
        if not inc.empty:
            assert inc.iloc[0]["wr_baseline"] == pytest.approx(
                0.6, abs=0.01
            )