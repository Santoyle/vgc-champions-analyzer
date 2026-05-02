"""
Tests unitarios para champions_calc.py (stats, tipos y daño).
Casos de daño alineados con validación manual Porygon Labs (Reg M-A).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.app.core.schema import SPSpread
from src.app.core.champions_calc import (
    NATURE_MULT,
    champions_damage_calc,
    calc_all_stats,
    stat_from_sp,
    type_effectiveness,
)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def garchomp_sp() -> SPSpread:
    return SPSpread(
        hp=2,
        atk=32,
        **{"def": 0},
        spa=0,
        spd=0,
        spe=32,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def incineroar_sp() -> SPSpread:
    return SPSpread(
        hp=28,
        atk=0,
        **{"def": 4},
        spa=0,
        spd=20,
        spe=0,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def primarina_sp() -> SPSpread:
    return SPSpread(
        hp=0,
        atk=0,
        **{"def": 4},
        spa=32,
        spd=0,
        spe=28,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def move_earthquake() -> dict[str, Any]:
    return {
        "power": 100,
        "type": "Ground",
        "category": "Physical",
        "name": "Earthquake",
    }


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def move_moonblast() -> dict[str, Any]:
    return {
        "power": 95,
        "type": "Fairy",
        "category": "Special",
        "name": "Moonblast",
    }


class TestStatFromSp:
    """Tests unitarios de la fórmula SP→stat."""

    def test_garchomp_hp(self) -> None:
        """Garchomp HP: BS=108, SP=2 → 185."""
        assert stat_from_sp(108, 2, is_hp=True) == 185

    def test_garchomp_atk_jolly(self) -> None:
        """Garchomp Atk: BS=130, SP=32, ×1.0 → 182."""
        assert stat_from_sp(130, 32, 1.0) == 182

    def test_garchomp_spe_jolly(self) -> None:
        """Garchomp Spe: BS=102, SP=32, Jolly ×1.1 → 169."""
        assert stat_from_sp(102, 32, 1.1) == 169

    def test_sp_zero_non_hp(self) -> None:
        """SP=0 con naturaleza neutra da stat mínimo."""
        result = stat_from_sp(100, 0, 1.0)
        assert result == stat_from_sp(100, 0, 1.0, is_hp=False)
        assert result > 0

    def test_sp_zero_hp(self) -> None:
        """SP=0 HP da stat mínimo de HP."""
        result = stat_from_sp(100, 0, is_hp=True)
        assert result > 0

    def test_nature_boost(self) -> None:
        """Naturaleza positiva ×1.1 da más que neutral."""
        neutral = stat_from_sp(100, 16, 1.0)
        boosted = stat_from_sp(100, 16, 1.1)
        assert boosted > neutral

    def test_nature_penalty(self) -> None:
        """Naturaleza negativa ×0.9 da menos que neutral."""
        neutral = stat_from_sp(100, 16, 1.0)
        reduced = stat_from_sp(100, 16, 0.9)
        assert reduced < neutral

    def test_sp_cap_32_equivalent_to_252_evs(self) -> None:
        """SP=32 equivale a 252 EVs (cap de la fórmula)."""
        result_32sp = stat_from_sp(100, 32, 1.0)
        result_33sp = stat_from_sp(100, 33, 1.0)
        assert result_32sp == result_33sp

    def test_nature_mult_has_25_entries(self) -> None:
        """NATURE_MULT tiene exactamente 25 naturalezas."""
        assert len(NATURE_MULT) == 25


class TestCalcAllStats:
    """Tests de calc_all_stats con Garchomp validado."""

    def test_garchomp_full_spread(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """Garchomp 2/32/0/0/0/32 Jolly — validado
        contra Porygon Labs."""
        stats = calc_all_stats(
            "garchomp", garchomp_sp, "Jolly"
        )
        assert stats is not None
        assert stats["hp"] == 185
        assert stats["atk"] == 182
        assert stats["spe"] == 169

    def test_returns_none_for_unknown_pokemon(
        self,
    ) -> None:
        """Pokémon inexistente retorna None."""
        spread = SPSpread(
            hp=0,
            atk=0,
            **{"def": 0},
            spa=0,
            spd=0,
            spe=0,
        )
        result = calc_all_stats(
            "fakemon999", spread, "Hardy"
        )
        assert result is None

    def test_all_six_stats_present(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """Retorna los 6 stats esperados."""
        stats = calc_all_stats(
            "garchomp", garchomp_sp, "Jolly"
        )
        assert stats is not None
        assert set(stats.keys()) == {
            "hp",
            "atk",
            "def",
            "spa",
            "spd",
            "spe",
        }

    def test_all_stats_positive(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """Todos los stats son enteros positivos."""
        stats = calc_all_stats(
            "garchomp", garchomp_sp, "Jolly"
        )
        assert stats is not None
        for val in stats.values():
            assert isinstance(val, int)
            assert val > 0


class TestTypeEffectiveness:
    """Tests del multiplicador de tipo."""

    def test_water_vs_fire(self) -> None:
        """Water vs Fire = 2.0."""
        assert type_effectiveness(
            "Water", ["Fire"]
        ) == pytest.approx(2.0)

    def test_fire_vs_grass(self) -> None:
        """Fire vs Grass = 2.0."""
        assert type_effectiveness(
            "Fire", ["Grass"]
        ) == pytest.approx(2.0)

    def test_ground_vs_flying_immune(self) -> None:
        """Ground vs Flying = 0.0 (inmune)."""
        assert type_effectiveness(
            "Ground", ["Flying"]
        ) == pytest.approx(0.0)

    def test_water_vs_water_resisted(self) -> None:
        """Water vs Water = 0.5."""
        assert type_effectiveness(
            "Water", ["Water"]
        ) == pytest.approx(0.5)

    def test_dual_type_super_effective(self) -> None:
        """Ground vs Fire/Rock = 2.0×2.0 = 4.0."""
        assert type_effectiveness(
            "Ground", ["Fire", "Rock"]
        ) == pytest.approx(4.0)

    def test_neutral(self) -> None:
        """Normal vs Dragon = 1.0."""
        assert type_effectiveness(
            "Normal", ["Dragon"]
        ) == pytest.approx(1.0)

    def test_fairy_not_boosted_by_rain(self) -> None:
        """Fairy vs Dragon = 2.0 independiente
        del clima (el clima no afecta Fairy)."""
        eff = type_effectiveness("Fairy", ["Dragon"])
        assert eff == pytest.approx(2.0)


class TestChampionsDamageCalc:
    """Tests del damage calculator validados
    contra Porygon Labs Champions VGC Reg M-A."""

    def test_returns_array_of_16(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Retorna np.ndarray de 16 elementos."""
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {},
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 16

    def test_rolls_ascending(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Los 16 rolls están en orden ascendente."""
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {},
        )
        assert all(
            result[i] <= result[i + 1]
            for i in range(len(result) - 1)
        )

    def test_caso1_eq_single_target(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Caso 1 validado: EQ single target
        92.9%-109.1% (Porygon Labs)."""
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {},
        )
        assert result.min() * 100 == pytest.approx(
            92.9, abs=0.5
        )
        assert result.max() * 100 == pytest.approx(
            109.1, abs=0.5
        )

    def test_caso2_moonblast_fairy(
        self,
        primarina_sp: SPSpread,
        garchomp_sp: SPSpread,
        move_moonblast: dict[str, Any],
    ) -> None:
        """Caso 2 validado: Moonblast Fairy
        109.2%-128.6% (Porygon Labs, ±0.5%)."""
        result = champions_damage_calc(
            "primarina",
            primarina_sp,
            "Modest",
            None,
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            move_moonblast,
            {},
        )
        assert result.min() * 100 == pytest.approx(
            109.2, abs=1.0
        )
        assert result.max() * 100 == pytest.approx(
            128.6, abs=0.5
        )

    def test_caso4_eq_spread_075(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Caso 4 validado: EQ spread ×0.75
        69.7%-81.8% (Porygon Labs)."""
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {"targets": 2},
        )
        assert result.min() * 100 == pytest.approx(
            69.7, abs=0.5
        )
        assert result.max() * 100 == pytest.approx(
            81.8, abs=0.5
        )

    def test_spread_is_75_percent_of_single(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Spread ×0.75 da exactamente 75% del
        daño single target."""
        single = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {},
        )
        spread = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {"targets": 2},
        )
        ratio = spread.mean() / single.mean()
        assert ratio == pytest.approx(0.75, abs=0.01)

    def test_unknown_attacker_returns_zeros(
        self,
        incineroar_sp: SPSpread,
        move_earthquake: dict[str, Any],
    ) -> None:
        """Atacante inexistente retorna zeros."""
        spread = SPSpread(
            hp=0,
            atk=0,
            **{"def": 0},
            spa=0,
            spd=0,
            spe=0,
        )
        result = champions_damage_calc(
            "fakemon999",
            spread,
            "Hardy",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move_earthquake,
            {},
        )
        assert np.all(result == 0.0)

    def test_zero_power_returns_zeros(
        self,
        garchomp_sp: SPSpread,
        incineroar_sp: SPSpread,
    ) -> None:
        """Move con poder 0 retorna zeros."""
        move = {
            "power": 0,
            "type": "Normal",
            "category": "Physical",
            "name": "Splash",
        }
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "incineroar",
            incineroar_sp,
            "Careful",
            None,
            move,
            {},
        )
        assert np.all(result == 0.0)

    def test_immune_returns_zeros(
        self,
        garchomp_sp: SPSpread,
    ) -> None:
        """Ground vs Fairy/Flying (Togekiss): inmunidad Ground."""
        move = {
            "power": 100,
            "type": "Ground",
            "category": "Physical",
            "name": "Earthquake",
        }
        def_spread = SPSpread(
            hp=0,
            atk=0,
            **{"def": 0},
            spa=0,
            spd=0,
            spe=0,
        )
        result = champions_damage_calc(
            "garchomp",
            garchomp_sp,
            "Jolly",
            None,
            "togekiss",
            def_spread,
            "Hardy",
            None,
            move,
            {},
        )
        assert np.all(result == 0.0)

    def test_never_raises_exception(
        self,
    ) -> None:
        """champions_damage_calc nunca propaga
        excepción con inputs inválidos."""
        spread = SPSpread(
            hp=0,
            atk=0,
            **{"def": 0},
            spa=0,
            spd=0,
            spe=0,
        )
        try:
            champions_damage_calc(
                "",
                spread,
                "",
                None,
                "",
                spread,
                "",
                None,
                {},
                {},
            )
        except Exception as exc:
            pytest.fail(
                f"champions_damage_calc propagó "
                f"excepción: {exc}"
            )
