"""
Tests unitarios para LRE (Lead Recommendation Engine).

Los grupos 1–2 ejercitan helpers sin DuckDB de aplicación (extract / cold start).
El grupo 3 usa ``tmp_path`` y un DuckDB temporal para ``replay_turns`` y
``populate_lead_pair_stats``. El grupo 4 sustituye la conexión por
``MagicMock`` y vacía la caché de Streamlit en cada test para
``lead_recommendation``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest

from src.app.core.schema import SPSpread  # noqa: F401
from src.app.metrics.lre import (
    LEAD_PAIR_STATS_DDL,  # noqa: F401
    _cold_start_score,
    _extract_leads_from_replay_turns,
    lead_recommendation,
    populate_lead_pair_stats,
)


@pytest.fixture(autouse=True)
def _clear_streamlit_lre_cache() -> Generator[None, None, None]:
    """
    Vacía la caché de ``lead_recommendation`` (``@st.cache_data``) entre tests.
    """
    cast(Any, lead_recommendation).clear()
    yield
    cast(Any, lead_recommendation).clear()


def _make_mock_con(df: pd.DataFrame) -> Any:
    """Mock de conexión DuckDB para lead_pair_stats."""
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.df.return_value = df
    mock_con.execute.return_value = mock_result
    return mock_con


def _make_replay_turns_df(
    n_replays: int = 10,
    win_rate: float = 0.6,
) -> pd.DataFrame:
    """
    Genera replay_turns sintético con leads en T1.
    Cada replay tiene 2 Pokémon por jugador en T1.
    """
    rows = []
    for i in range(n_replays):
        winner = "p1" if i < int(
            n_replays * win_rate
        ) else "p2"
        # p1 leads: garchomp + incineroar
        for slot, pkm in enumerate(
            ["garchomp", "incineroar"]
        ):
            rows.append({
                "regulation_id": "TEST",
                "replay_id": f"r{i}",
                "turn": 1,
                "player": "p1",
                "pokemon_slug": pkm,
                "winner": winner,
            })
        # p2 leads: rillaboom + urshifu
        for slot, pkm in enumerate(
            ["rillaboom", "urshifu-rapid-strike"]
        ):
            rows.append({
                "regulation_id": "TEST",
                "replay_id": f"r{i}",
                "turn": 1,
                "player": "p2",
                "pokemon_slug": pkm,
                "winner": winner,
            })
    return pd.DataFrame(rows)


class TestExtractLeads:
    """Tests del extractor de leads del turno 1."""

    def test_extracts_two_leads_per_player(
        self,
    ) -> None:
        """Extrae exactamente 2 leads por jugador
        por replay."""
        df = _make_replay_turns_df(n_replays=5)
        result = _extract_leads_from_replay_turns(df)
        assert not result.empty
        # Cada (replay_id, player) tiene 1 fila
        counts = result.groupby(
            ["replay_id", "player"]
        ).size()
        assert (counts == 1).all()

    def test_leads_are_sorted_alphabetically(
        self,
    ) -> None:
        """lead_a <= lead_b alfabéticamente."""
        df = _make_replay_turns_df(n_replays=5)
        result = _extract_leads_from_replay_turns(df)
        for _, row in result.iterrows():
            assert row["lead_a"] <= row["lead_b"]

    def test_empty_df_returns_empty(self) -> None:
        """DataFrame vacío retorna vacío."""
        result = _extract_leads_from_replay_turns(
            pd.DataFrame(columns=[
                "regulation_id",
                "replay_id",
                "turn",
                "player",
                "pokemon_slug",
                "winner",
            ])
        )
        assert result.empty

    def test_skip_player_with_one_pokemon(
        self,
    ) -> None:
        """Jugador con solo 1 Pokémon en T1
        se omite."""
        df = pd.DataFrame([{
            "regulation_id": "TEST",
            "replay_id": "r0",
            "turn": 1,
            "player": "p1",
            "pokemon_slug": "garchomp",
            "winner": "p1",
        }])
        result = _extract_leads_from_replay_turns(df)
        assert result.empty


class TestColdStartScore:
    """Tests del cold start fallback."""

    def test_returns_float_in_range(self) -> None:
        """Retorna float en [0.35, 0.65]."""
        from src.app.core.champions_calc import (
            _load_pokemon_master,
        )

        pm = _load_pokemon_master()
        score = _cold_start_score(
            "garchomp", "incineroar",
            "rillaboom", "urshifu-rapid-strike",
            pm,
        )
        assert 0.35 <= score <= 0.65

    def test_empty_pokemon_master_returns_half(
        self,
    ) -> None:
        """Sin datos de tipos retorna 0.5."""
        score = _cold_start_score(
            "fakemon", "fakemon2",
            "fakemon3", "fakemon4",
            {},
        )
        assert score == 0.5


class TestPopulateLeadPairStats:
    """Tests de populate_lead_pair_stats con
    DuckDB :memory:."""

    def test_creates_table_and_inserts(
        self,
        tmp_path: Any,
    ) -> None:
        """Crea tabla y popula filas."""
        db_path = str(tmp_path / "test.duckdb")
        # Crear replay_turns en memoria
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE replay_turns (
                regulation_id VARCHAR,
                replay_id     VARCHAR,
                turn          SMALLINT,
                action_idx    SMALLINT,
                player        VARCHAR,
                slot          VARCHAR,
                pokemon_slug  VARCHAR,
                move_slug     VARCHAR,
                winner        VARCHAR
            )
        """)
        # Insertar datos sintéticos
        df = _make_replay_turns_df(n_replays=10)
        con.register("df_rt", df)
        con.execute("""
            INSERT INTO replay_turns
            SELECT
                regulation_id,
                replay_id,
                turn,
                0 AS action_idx,
                player,
                'a' AS slot,
                pokemon_slug,
                NULL AS move_slug,
                winner
            FROM df_rt
        """)
        con.close()

        populate_lead_pair_stats("TEST", db_path)

        con2 = duckdb.connect(
            db_path, read_only=True
        )
        n = con2.execute("""
            SELECT COUNT(1) FROM lead_pair_stats
            WHERE regulation_id='TEST'
        """).fetchone()[0]
        con2.close()
        assert n > 0

    def test_win_rate_between_0_and_1(
        self,
        tmp_path: Any,
    ) -> None:
        """win_rate en [0, 1]."""
        db_path = str(tmp_path / "test2.duckdb")
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE replay_turns (
                regulation_id VARCHAR,
                replay_id     VARCHAR,
                turn          SMALLINT,
                action_idx    SMALLINT,
                player        VARCHAR,
                slot          VARCHAR,
                pokemon_slug  VARCHAR,
                move_slug     VARCHAR,
                winner        VARCHAR
            )
        """)
        df = _make_replay_turns_df(n_replays=10)
        con.register("df_rt", df)
        con.execute("""
            INSERT INTO replay_turns
            SELECT regulation_id, replay_id, turn,
                   0, player, 'a', pokemon_slug,
                   NULL, winner
            FROM df_rt
        """)
        con.close()

        populate_lead_pair_stats("TEST", db_path)

        con2 = duckdb.connect(
            db_path, read_only=True
        )
        df_lps = con2.execute("""
            SELECT win_rate FROM lead_pair_stats
            WHERE regulation_id='TEST'
        """).df()
        con2.close()
        assert (
            df_lps["win_rate"].between(0, 1)
        ).all()


class TestLeadRecommendation:
    """Tests de lead_recommendation con mock."""

    def _empty_lps(self) -> pd.DataFrame:
        """lead_pair_stats vacío (cold start)."""
        return pd.DataFrame(columns=[
            "my_lead_a", "my_lead_b",
            "opp_lead_a", "opp_lead_b",
            "n_matches", "win_rate",
        ])

    def test_returns_15_rows_for_6_pokemon(
        self,
    ) -> None:
        """6 Pokémon → C(6,2) = 15 filas."""
        con = _make_mock_con(self._empty_lps())
        my_team = (
            "garchomp", "incineroar", "primarina",
            "archaludon", "sneasler", "flutter-mane",
        )
        opp_team = (
            "rillaboom", "urshifu-rapid-strike",
            "calyrex-shadow", "landorus-therian",
            "amoonguss", "miraidon",
        )
        df = lead_recommendation(
            "TEST", my_team, opp_team, con,
        )
        assert len(df) == 15

    def test_expected_win_prob_in_range(
        self,
    ) -> None:
        """expected_win_prob en [0, 1]."""
        con = _make_mock_con(self._empty_lps())
        my_team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
            "sneasler", "flutter-mane",
        )
        opp_team = (
            "rillaboom", "urshifu-rapid-strike",
            "calyrex-shadow", "landorus-therian",
            "amoonguss", "miraidon",
        )
        df = lead_recommendation(
            "TEST", my_team, opp_team, con,
        )
        assert (
            df["expected_win_prob"]
            .between(0, 1)
        ).all()

    def test_sorted_by_expected_win_prob_desc(
        self,
    ) -> None:
        """Ordenado por expected_win_prob DESC."""
        con = _make_mock_con(self._empty_lps())
        my_team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
            "sneasler", "flutter-mane",
        )
        opp_team = (
            "rillaboom", "urshifu-rapid-strike",
            "calyrex-shadow", "landorus-therian",
            "amoonguss", "miraidon",
        )
        df = lead_recommendation(
            "TEST", my_team, opp_team, con,
        )
        probs = df["expected_win_prob"].tolist()
        assert all(
            probs[i] >= probs[i + 1]
            for i in range(len(probs) - 1)
        )

    def test_cold_start_score_in_valid_range(
        self,
    ) -> None:
        """Con cold start EV en [0.35, 0.65]."""
        con = _make_mock_con(self._empty_lps())
        my_team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
        )
        opp_team = (
            "rillaboom", "urshifu-rapid-strike",
            "calyrex-shadow", "landorus-therian",
        )
        df = lead_recommendation(
            "TEST", my_team, opp_team, con,
        )
        assert (
            df["expected_win_prob"]
            .between(0.35, 0.65)
        ).all()

    def test_empty_team_returns_empty(
        self,
    ) -> None:
        """Equipo con menos de 2 Pokémon retorna
        DataFrame vacío."""
        con = _make_mock_con(self._empty_lps())
        df = lead_recommendation(
            "TEST", ("garchomp",), ("rillaboom",),
            con,
        )
        assert df.empty

    def test_regulation_id_in_result(
        self,
    ) -> None:
        """regulation_id del input aparece en
        resultado."""
        con = _make_mock_con(self._empty_lps())
        my_team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
            "sneasler", "flutter-mane",
        )
        opp_team = (
            "rillaboom", "urshifu-rapid-strike",
            "calyrex-shadow", "landorus-therian",
            "amoonguss", "miraidon",
        )
        df = lead_recommendation(
            "M-A", my_team, opp_team, con,
        )
        assert (df["regulation_id"] == "M-A").all()
