"""
Tests unitarios para las métricas V2 del Bloque 15.

Cubren MLWR, METI, MEIT, TAI, Shapley Slot Value (SV), SPDO y LRE usando
``MagicMock`` para DuckDB y datos sintéticos en memoria (sin acceso real a
disco en los tests que mockean la conexión).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.app.core.schema import SPSpread
from src.app.metrics.advanced_metrics import (
    compute_meit,
    compute_tai,
    shapley_slot_value,
)
from src.app.metrics.lre import lead_recommendation
from src.app.metrics.mlwr import (
    compute_mlwr_level1,
    compute_meti,
)
from src.app.metrics.spdo import (
    optimize_defensive_sp,
)


def _make_mock_con(df: pd.DataFrame) -> Any:
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.df.return_value = df
    mock_con.execute.return_value = mock_result
    return mock_con


def _make_replay_turns_df(
    n_replays: int = 20,
    include_mega: bool = True,
) -> pd.DataFrame:
    """
    DataFrame sintético de replay_turns para
    tests de METI y MLWR.
    """
    rows = []
    for i in range(n_replays):
        winner = "p1" if i % 2 == 0 else "p2"
        mega_turn = 1 if include_mega and i < 5 else None
        for turn in range(1, 4):
            mega_p1 = (
                include_mega
                and i < 5
                and turn >= 1
            )
            rows.append({
                "regulation_id": "TEST",
                "replay_id":     f"r{i}",
                "turn":          turn,
                "player":        "p1",
                "pokemon_slug":  "garchomp",
                "move_slug":     "earthquake",
                "mega_used_p1":  mega_p1,
                "mega_used_p2":  False,
                "mega_pokemon_slug": (
                    "garchomp"
                    if mega_p1 and turn == 1
                    else None
                ),
                "winner":        winner,
            })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _clear_caches() -> Generator[None, None, None]:
    """Limpia todas las cachés de st.cache_data
    entre tests."""

    def _clear() -> None:
        for fn in [
            compute_mlwr_level1,
            compute_meti,
            compute_meit,
            compute_tai,
            shapley_slot_value,
        ]:
            try:
                cast(Any, fn).clear()
            except Exception:
                pass

    _clear()
    yield
    _clear()


class TestComputeMLWRLevel1:
    """Tests de smoke para MLWR L1."""

    def test_returns_dataframe(self) -> None:
        """Retorna DataFrame."""
        df = _make_replay_turns_df()
        df["won"] = df["winner"] == "p1"
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_mlwr_column(self) -> None:
        """Tiene columna mlwr."""
        df = _make_replay_turns_df()
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        if not result.empty:
            assert "mlwr" in result.columns

    def test_mlwr_in_range(self) -> None:
        """mlwr en [-1, 1]."""
        df = _make_replay_turns_df()
        con = _make_mock_con(df)
        result = compute_mlwr_level1(
            "TEST", con, min_n=1
        )
        if not result.empty:
            assert (result["mlwr"] >= -1).all()
            assert (result["mlwr"] <= 1).all()


class TestComputeMeti:
    """Tests de compute_meti."""

    def test_mega_enabled_false_returns_empty(
        self,
    ) -> None:
        """mega_enabled=False retorna vacío."""
        con = _make_mock_con(
            pd.DataFrame()
        )
        result = compute_meti(
            "TEST", con, mega_enabled=False
        )
        assert result.empty
        assert "pokemon_slug" in result.columns

    def test_returns_dataframe_with_data(
        self,
    ) -> None:
        """Con datos retorna DataFrame."""
        df = _make_replay_turns_df(
            n_replays=20, include_mega=True
        )
        con = _make_mock_con(df)
        result = compute_meti(
            "TEST", con, mega_enabled=True
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        """Tiene columnas requeridas."""
        df = _make_replay_turns_df()
        con = _make_mock_con(df)
        result = compute_meti(
            "TEST", con, mega_enabled=True
        )
        required = {
            "pokemon_slug", "turn_bucket",
            "n_matches", "win_rate",
            "baseline_wr", "meti_delta",
            "regulation_id",
        }
        assert required.issubset(
            set(result.columns)
        )

    def test_turn_buckets_valid(self) -> None:
        """Buckets son T1/T2/T3/T4+/never."""
        df = _make_replay_turns_df()
        con = _make_mock_con(df)
        result = compute_meti(
            "TEST", con, mega_enabled=True
        )
        valid = {"T1", "T2", "T3", "T4+", "never"}
        if not result.empty:
            assert result[
                "turn_bucket"
            ].isin(valid).all()


class TestComputeMeit:
    """Tests de compute_meit."""

    def test_mega_enabled_false(self) -> None:
        """mega_enabled=False retorna not_applicable."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_meit(
            "TEST", "garchomp", con,
            mega_enabled=False,
        )
        assert result["not_applicable"] is True
        assert result["found"] is False

    def test_returns_dict(self) -> None:
        """Retorna dict."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_meit(
            "TEST", "garchomp", con,
            mega_enabled=False,
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        """Dict tiene keys requeridas."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_meit(
            "TEST", "garchomp", con,
            mega_enabled=False,
        )
        required = {
            "found", "meit_score",
            "activation_rate",
            "sustained_damage_ratio",
            "opportunity_cost",
            "not_applicable",
        }
        assert required.issubset(
            set(result.keys())
        )

    def test_meit_score_float(self) -> None:
        """meit_score es float."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_meit(
            "TEST", "garchomp", con,
            mega_enabled=False,
        )
        assert isinstance(
            result["meit_score"], float
        )

    def test_custom_coefficients(self) -> None:
        """Coeficientes custom se aplican."""
        df = _make_replay_turns_df()
        con = _make_mock_con(df)
        result = compute_meit(
            "TEST", "garchomp", con,
            mega_enabled=True,
            coefficients={
                "alpha": 0.6,
                "beta": 0.3,
                "gamma": 0.1,
            },
        )
        assert result["alpha"] == 0.6
        assert result["beta"] == 0.3
        assert result["gamma"] == 0.1


class TestComputeTai:
    """Tests de compute_tai."""

    def test_short_team_returns_not_found(
        self,
    ) -> None:
        """Equipo < 2 Pokémon retorna found=False."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_tai(
            "TEST", ("garchomp",), con
        )
        assert result["found"] is False

    def test_returns_dict(self) -> None:
        """Retorna dict."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_tai(
            "TEST",
            ("garchomp", "incineroar"),
            con,
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        """Dict tiene keys requeridas."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_tai(
            "TEST",
            ("garchomp", "incineroar"),
            con,
        )
        required = {
            "tai_score", "weighted_wr",
            "wr_variance", "n_archetypes",
            "archetype_breakdown",
            "regulation_id", "team", "found",
        }
        assert required.issubset(
            set(result.keys())
        )

    def test_no_clusters_returns_not_found(
        self,
    ) -> None:
        """Sin clusters retorna found=False."""
        con = _make_mock_con(pd.DataFrame())
        result = compute_tai(
            "TEST",
            ("garchomp", "incineroar",
             "primarina"),
            con,
        )
        assert result["found"] is False


class TestShapleySlotValue:
    """Tests de shapley_slot_value."""

    def test_empty_team_returns_empty(
        self,
    ) -> None:
        """Equipo < 2 retorna dict vacío."""
        con = _make_mock_con(pd.DataFrame())
        result = shapley_slot_value(
            "TEST", ("garchomp",), con
        )
        assert result == {}

    def test_returns_dict_with_6_keys(
        self,
    ) -> None:
        """6 Pokémon → 6 keys."""
        con = _make_mock_con(pd.DataFrame())
        team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
            "sneasler", "flutter-mane",
        )
        result = shapley_slot_value(
            "TEST", team, con, wp_model=None
        )
        assert len(result) == 6

    def test_sum_equals_wr_minus_half(
        self,
    ) -> None:
        """Σφᵢ ≈ WR_full - 0.5."""
        con = _make_mock_con(pd.DataFrame())
        team = (
            "garchomp", "incineroar",
            "primarina", "archaludon",
        )
        result = shapley_slot_value(
            "TEST", team, con, wp_model=None
        )
        # Sin modelo WR_full=0.5 → Σ=0.0
        total = sum(result.values())
        assert abs(total) < 0.01

    def test_all_slugs_in_result(self) -> None:
        """Todos los slugs del equipo en result."""
        con = _make_mock_con(pd.DataFrame())
        team = (
            "garchomp", "incineroar",
            "primarina",
        )
        result = shapley_slot_value(
            "TEST", team, con, wp_model=None
        )
        for pkm in team:
            assert pkm in result


class TestOptimizeDefensiveSp:
    """Tests de smoke para SPDO Modo 2."""

    MOCK_PM = {
        1: {
            "name": "garchomp",
            "types": ["Dragon", "Ground"],
            "base_stats": {
                "hp": 108, "attack": 130,
                "defense": 95,
                "special-attack": 80,
                "special-defense": 85,
                "speed": 102,
            },
        },
        2: {
            "name": "incineroar",
            "types": ["Fire", "Dark"],
            "base_stats": {
                "hp": 95, "attack": 115,
                "defense": 90,
                "special-attack": 70,
                "special-defense": 90,
                "speed": 60,
            },
        },
    }

    def test_returns_result_or_none(self) -> None:
        """Retorna DefensiveOptResult o None."""
        from src.app.metrics.spdo import (
            DefensiveOptResult,
        )
        g_sp = SPSpread(
            hp=2, atk=32, **{"def": 0},
            spa=0, spd=0, spe=32,
        )
        move = {
            "power": 100, "type": "Ground",
            "category": "Physical",
            "name": "Earthquake",
        }
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move,
            attacker_spread=g_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=self.MOCK_PM,
        )
        assert result is None or isinstance(
            result, DefensiveOptResult
        )

    def test_spread_total_le_66(self) -> None:
        """Spread resultado tiene suma <= 66."""
        g_sp = SPSpread(
            hp=2, atk=32, **{"def": 0},
            spa=0, spd=0, spe=32,
        )
        move = {
            "power": 100, "type": "Ground",
            "category": "Physical",
            "name": "Earthquake",
        }
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move,
            attacker_spread=g_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=self.MOCK_PM,
        )
        if result and result.found:
            assert result.spread.total() <= 66


class TestLeadRecommendation:
    """Tests de smoke para LRE."""

    def _empty_lps(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "my_lead_a", "my_lead_b",
            "opp_lead_a", "opp_lead_b",
            "n_matches", "win_rate",
        ])

    def test_returns_15_rows(self) -> None:
        """6 Pokémon → 15 filas."""
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
            "TEST", my_team, opp_team, con
        )
        assert len(df) == 15

    def test_no_hardcoded_regulation(
        self,
    ) -> None:
        """No hay 'M-A' hardcodeado en el módulo
        lre.py."""
        import inspect
        from src.app.metrics import lre
        src = inspect.getsource(lre)
        assert '"M-A"' not in src
        assert "'M-A'" not in src
