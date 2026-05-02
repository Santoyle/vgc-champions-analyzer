"""
Tests unitarios SPDO (Modos 2, 3 y 4) en ``src/app.metrics.spdo``.

**Grupo 1:** ``find_spe_sp_to_outspeed`` — aritmética pura, sin lectura de disco.

**Grupos 2–3:** ``optimize_defensive_sp`` / ``optimize_offensive_sp`` con
``MOCK_PM`` inyectado como ``pokemon_master`` (sin depender del JSON real).

**Grupo 4:** ``build_speed_tier_table`` — misma tabla de speed tiers usando
``MOCK_PM`` inyectado.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.app.core.schema import SPSpread
from src.app.metrics.spdo import (
    DefensiveOptResult,
    OffensiveOptResult,
    build_speed_tier_table,
    find_spe_sp_to_outspeed,
    optimize_defensive_sp,
    optimize_offensive_sp,
)

MOCK_PM: dict[int, dict[str, Any]] = {
    1: {
        "name": "garchomp",
        "types": ["Dragon", "Ground"],
        "base_stats": {
            "hp": 108,
            "attack": 130,
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
            "hp": 95,
            "attack": 115,
            "defense": 90,
            "special-attack": 70,
            "special-defense": 90,
            "speed": 60,
        },
    },
    3: {
        "name": "sneasler",
        "types": ["Fighting", "Poison"],
        "base_stats": {
            "hp": 80,
            "attack": 130,
            "defense": 60,
            "special-attack": 40,
            "special-defense": 65,
            "speed": 120,
        },
    },
    4: {
        "name": "flutter-mane",
        "types": ["Ghost", "Fairy"],
        "base_stats": {
            "hp": 55,
            "attack": 55,
            "defense": 55,
            "special-attack": 135,
            "special-defense": 135,
            "speed": 135,
        },
    },
}


@pytest.fixture(scope="module")
def garchomp_sp() -> SPSpread:
    return SPSpread(
        hp=2,
        atk=32,
        **{"def": 0},
        spa=0,
        spd=0,
        spe=32,
    )


@pytest.fixture(scope="module")
def incineroar_sp() -> SPSpread:
    return SPSpread(
        hp=28,
        **{"def": 4},
        atk=0,
        spa=0,
        spd=20,
        spe=0,
    )


@pytest.fixture(scope="module")
def move_eq() -> dict[str, Any]:
    return {
        "power": 100,
        "type": "Ground",
        "category": "Physical",
        "name": "Earthquake",
    }


class TestFindSpeSpToOutspeed:
    """Tests matemáticos puros — sin I/O."""

    def test_returns_int_when_possible(self) -> None:
        """Retorna int cuando es posible."""
        result = find_spe_sp_to_outspeed(
            my_base_speed=102,
            target_speed=80,
        )
        assert isinstance(result, int)
        assert 0 <= result <= 32

    def test_returns_none_when_impossible(self) -> None:
        """Retorna None cuando es imposible con 32 SP máximo."""
        result = find_spe_sp_to_outspeed(
            my_base_speed=60,
            target_speed=220,
        )
        assert result is None

    def test_zero_sp_when_already_faster(self) -> None:
        """SP=0 cuando ya es más rápido."""
        result = find_spe_sp_to_outspeed(
            my_base_speed=135,
            target_speed=50,
        )
        assert result == 0

    def test_trick_room_slower(self) -> None:
        """En TR, busca ser MÁS LENTO."""
        result = find_spe_sp_to_outspeed(
            my_base_speed=60,
            target_speed=100,
            under_trick_room=True,
        )
        assert result is not None
        assert result == 0

    def test_trick_room_fast_pokemon_needs_sp(
        self,
    ) -> None:
        """Pokémon rápido necesita SP para ser lento en TR."""
        result_tr = find_spe_sp_to_outspeed(
            my_base_speed=135,
            target_speed=200,
            under_trick_room=True,
        )
        assert result_tr is not None

    def test_outspeed_is_strictly_greater(
        self,
    ) -> None:
        """Outspeed requiere ser ESTRICTAMENTE más rápido (no igual)."""
        result = find_spe_sp_to_outspeed(
            my_base_speed=102,
            target_speed=154,
        )
        if result is not None:
            from src.app.metrics.spdo import (
                _calc_speed_stat,
            )

            my_speed = _calc_speed_stat(
                102, result, "Hardy"
            )
            assert my_speed > 154


class TestOptimizeDefensiveSp:
    """Tests de SPDO Modo 2."""

    def test_returns_defensive_opt_result(
        self,
        garchomp_sp: SPSpread,
        move_eq: dict[str, Any],
    ) -> None:
        """Retorna DefensiveOptResult."""
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move_eq,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.95,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        assert isinstance(result, DefensiveOptResult)

    def test_found_true_for_achievable_target(
        self,
        garchomp_sp: SPSpread,
        move_eq: dict[str, Any],
    ) -> None:
        """found=True cuando hay spread válido."""
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move_eq,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        assert result is not None
        assert result.found is True

    def test_spread_respects_sp_cap(
        self,
        garchomp_sp: SPSpread,
        move_eq: dict[str, Any],
    ) -> None:
        """El spread resultado tiene suma <= 66."""
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move_eq,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        if result and result.found:
            assert result.spread.total() <= 66

    def test_returns_none_for_unknown_pokemon(
        self,
        garchomp_sp: SPSpread,
        move_eq: dict[str, Any],
    ) -> None:
        """None si el Pokémon no existe."""
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="fakemon999",
            attacker_slug="garchomp",
            attacker_move=move_eq,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            pokemon_master=MOCK_PM,
        )
        assert result is None

    def test_found_false_for_impossible_target(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """found=False cuando target es imposible."""
        move_strong = {
            "power": 250,
            "type": "Ground",
            "category": "Physical",
            "name": "SuperMove",
        }
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move_strong,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            survival_target_pct=1.0,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        assert result is not None
        assert result.found is False

    def test_physical_uses_def_stat(
        self,
        garchomp_sp: SPSpread,
        move_eq: dict[str, Any],
    ) -> None:
        """Move físico optimiza Def, no SpD."""
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="incineroar",
            attacker_slug="garchomp",
            attacker_move=move_eq,
            attacker_spread=garchomp_sp,
            attacker_nature="Jolly",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        if result and result.found:
            assert result.stat_used == "def"

    def test_special_uses_spd_stat(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """Move especial optimiza SpD, no Def."""
        move_special = {
            "power": 90,
            "type": "Fairy",
            "category": "Special",
            "name": "Moonblast",
        }
        primarina_sp = SPSpread(
            hp=0,
            atk=0,
            **{"def": 4},
            spa=32,
            spd=0,
            spe=28,
        )
        result = optimize_defensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_slug="sneasler",
            attacker_move=move_special,
            attacker_spread=primarina_sp,
            attacker_nature="Modest",
            survival_target_pct=0.50,
            fixed_offensive_sp=0,
            pokemon_master=MOCK_PM,
        )
        if result and result.found:
            assert result.stat_used == "spd"


class TestOptimizeOffensiveSp:
    """Tests de SPDO Modo 3."""

    def test_returns_offensive_opt_result(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """Retorna OffensiveOptResult."""
        result = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            pokemon_master=MOCK_PM,
        )
        assert isinstance(result, OffensiveOptResult)

    def test_min_atk_sp_in_range(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """min_atk_sp en [0, 32]."""
        result = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            pokemon_master=MOCK_PM,
        )
        if result and result.found:
            assert 0 <= result.min_atk_sp <= 32

    def test_physical_uses_atk_stat(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """Move físico optimiza Atk."""
        result = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            pokemon_master=MOCK_PM,
        )
        assert result is not None
        assert result.stat_used == "atk"

    def test_returns_none_for_unknown_pokemon(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """None si el Pokémon no existe."""
        result = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="fakemon999",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            pokemon_master=MOCK_PM,
        )
        assert result is None

    def test_with_target_spreads(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """Funciona con target_spreads explícito."""
        target = {
            "spread": SPSpread(
                hp=0,
                atk=0,
                **{"def": 0},
                spa=0,
                spd=0,
                spe=0,
            ),
            "nature": "Hardy",
            "usage_pct": 100.0,
            "target_slug": "incineroar",
        }
        result = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            target_spreads=[target],
            ko_probability_target=0.50,
            pokemon_master=MOCK_PM,
        )
        assert result is not None

    def test_higher_target_needs_more_sp(
        self,
        move_eq: dict[str, Any],
    ) -> None:
        """Mayor probabilidad objetivo requiere mayor o igual Atk_sp."""
        target = {
            "spread": SPSpread(
                hp=0,
                atk=0,
                **{"def": 0},
                spa=0,
                spd=0,
                spe=0,
            ),
            "nature": "Hardy",
            "usage_pct": 100.0,
            "target_slug": "incineroar",
        }
        r_low = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            target_spreads=[target],
            ko_probability_target=0.30,
            pokemon_master=MOCK_PM,
        )
        r_high = optimize_offensive_sp(
            reg_id="TEST",
            pokemon_slug="garchomp",
            attacker_move=move_eq,
            attacker_nature="Jolly",
            target_spreads=[target],
            ko_probability_target=0.50,
            pokemon_master=MOCK_PM,
        )
        if (
            r_low
            and r_low.found
            and r_high
            and r_high.found
        ):
            assert r_low.min_atk_sp <= r_high.min_atk_sp


class TestBuildSpeedTierTable:
    """Tests de SPDO Modo 4."""

    def test_returns_dataframe(self) -> None:
        """Retorna DataFrame."""
        df = build_speed_tier_table(
            pokemon_slug="garchomp",
            reg_id="TEST",
            top_n=5,
            pokemon_master=MOCK_PM,
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        """DataFrame tiene las columnas esperadas."""
        df = build_speed_tier_table(
            pokemon_slug="garchomp",
            reg_id="TEST",
            top_n=5,
            pokemon_master=MOCK_PM,
        )
        required = {
            "target_pokemon",
            "target_speed",
            "spe_sp_to_outspeed",
            "my_speed_at_that_sp",
            "spe_sp_under_tr",
        }
        assert required.issubset(set(df.columns))

    def test_row_count_equals_top_n(self) -> None:
        """Número de filas; ver docstring restricciones."""
        # ``build_speed_tier_table`` toma los ``top_n`` más rápidos del
        # maestro incluyendo al propio Pokémon si califica — no hay exclusión.
        n = len(MOCK_PM)
        df = build_speed_tier_table(
            pokemon_slug="garchomp",
            reg_id="TEST",
            top_n=n,
            pokemon_master=MOCK_PM,
        )
        assert len(df) == n

    def test_sorted_by_target_speed_desc(self) -> None:
        """Ordenado por target_speed DESC."""
        df = build_speed_tier_table(
            pokemon_slug="garchomp",
            reg_id="TEST",
            top_n=5,
            pokemon_master=MOCK_PM,
        )
        if len(df) >= 2:
            speeds = df["target_speed"].tolist()
            assert all(
                speeds[i] >= speeds[i + 1]
                for i in range(len(speeds) - 1)
            )

    def test_unknown_pokemon_returns_empty(self) -> None:
        """Pokémon desconocido retorna vacío."""
        df = build_speed_tier_table(
            pokemon_slug="fakemon999",
            reg_id="TEST",
            top_n=5,
            pokemon_master=MOCK_PM,
        )
        assert df.empty

    def test_faster_pokemon_outspeed_sp_zero(
        self,
    ) -> None:
        """SP=0 outspeed cuando el spread ya corre más rápido."""
        df = build_speed_tier_table(
            pokemon_slug="flutter-mane",
            reg_id="TEST",
            top_n=4,
            my_nature="Timid",
            pokemon_master=MOCK_PM,
        )
        if not df.empty:
            slower = df[df["target_speed"] < 135]
            for _, row in slower.iterrows():
                sp = row["spe_sp_to_outspeed"]
                assert sp == 0 or sp is None

    def test_trick_room_sp_incineroar(self) -> None:
        """Incineroar lento: TR vs rivales rápidos con spe_sp_under_tr 0."""
        df = build_speed_tier_table(
            pokemon_slug="incineroar",
            reg_id="TEST",
            top_n=3,
            pokemon_master=MOCK_PM,
        )
        if not df.empty:
            faster = df[df["target_speed"] > 80]
            for _, row in faster.iterrows():
                assert row["spe_sp_under_tr"] == 0
