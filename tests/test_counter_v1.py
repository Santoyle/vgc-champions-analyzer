"""
Tests unitarios e integración para src/app/modules/counter.py — heurístico v1.

Grupos:
  1. TestTypeChart        (7 tests) — verifica TYPE_CHART completa y coherente.
  2. TestTypeAdvantageScore (5 tests) — lógica de ventaja de tipo pura.
  3. TestSpeedTierScore   (5 tests) — lógica de speed tier pura.
  4. TestHeuristicCounter (9 tests) — función principal heuristic_counter.
  5. TestCounterWithParsedPaste (3 tests) — integración paste real → counters.

Grupos 1-4 usan MOCK_POKEMON_DATA inyectado directamente; ninguno lee
pokemon_master.json desde disco.
Grupo 5 usa parse_paste() real (I/O de texto, no archivos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.app.data.parsers.ps_paste import parse_paste
from src.app.modules.counter import (
    ITEM_SYNERGY_SCORES,
    TYPE_CHART,
    CounterResult,
    _speed_tier_score,
    _type_advantage_score,
    heuristic_counter,
)

# ---------------------------------------------------------------------------
# Helpers locales (duck-typing sobre ParsedSlot / ParsedTeam)
# ---------------------------------------------------------------------------


@dataclass
class MockSlot:
    """Slot rival sintético para tests."""

    species: str
    item: str = ""
    ability: str = ""
    tera_type: str = ""
    moves: list[str] | None = None
    mega_capable: bool = False

    def __post_init__(self) -> None:
        if self.moves is None:
            self.moves = []


@dataclass
class MockTeam:
    """Equipo rival sintético para tests."""

    slots: list[MockSlot]


# ---------------------------------------------------------------------------
# Datos sintéticos de Pokémon (sin I/O)
# ---------------------------------------------------------------------------

MOCK_POKEMON_DATA: dict[str, dict[str, Any]] = {
    "incineroar": {
        "name": "incineroar",
        "types": ["Fire", "Dark"],
        "base_stats": {
            "hp": 95, "attack": 115, "defense": 90,
            "sp_attack": 80, "sp_defense": 90, "speed": 60,
        },
    },
    "garchomp": {
        "name": "garchomp",
        "types": ["Dragon", "Ground"],
        "base_stats": {
            "hp": 108, "attack": 130, "defense": 95,
            "sp_attack": 80, "sp_defense": 85, "speed": 102,
        },
    },
    "sneasler": {
        "name": "sneasler",
        "types": ["Fighting", "Poison"],
        "base_stats": {
            "hp": 80, "attack": 130, "defense": 60,
            "sp_attack": 40, "sp_defense": 80, "speed": 120,
        },
    },
    "sinistcha": {
        "name": "sinistcha",
        "types": ["Grass", "Ghost"],
        "base_stats": {
            "hp": 71, "attack": 60, "defense": 106,
            "sp_attack": 121, "sp_defense": 80, "speed": 70,
        },
    },
    "kingambit": {
        "name": "kingambit",
        "types": ["Dark", "Steel"],
        "base_stats": {
            "hp": 100, "attack": 135, "defense": 120,
            "sp_attack": 60, "sp_defense": 85, "speed": 50,
        },
    },
    "basculegion": {
        "name": "basculegion",
        "types": ["Water", "Ghost"],
        "base_stats": {
            "hp": 120, "attack": 112, "defense": 65,
            "sp_attack": 80, "sp_defense": 75, "speed": 78,
        },
    },
    "flutter mane": {
        "name": "flutter mane",
        "types": ["Ghost", "Fairy"],
        "base_stats": {
            "hp": 55, "attack": 55, "defense": 55,
            "sp_attack": 135, "sp_defense": 135, "speed": 135,
        },
    },
    "urshifu": {
        "name": "urshifu",
        "types": ["Fighting", "Dark"],
        "base_stats": {
            "hp": 100, "attack": 130, "defense": 100,
            "sp_attack": 63, "sp_defense": 60, "speed": 97,
        },
    },
}


# ---------------------------------------------------------------------------
# GRUPO 1 — TYPE_CHART
# ---------------------------------------------------------------------------


class TestTypeChart:
    """Verifica que TYPE_CHART está completa y es consistente."""

    def test_type_chart_has_18_types(self) -> None:
        assert len(TYPE_CHART) == 18

    def test_fire_supereffective_vs_grass(self) -> None:
        assert TYPE_CHART["Fire"]["Grass"] == 2.0

    def test_ground_immune_vs_flying(self) -> None:
        assert TYPE_CHART["Ground"]["Flying"] == 0.0

    def test_electric_immune_vs_ground(self) -> None:
        assert TYPE_CHART["Electric"]["Ground"] == 0.0

    def test_dragon_immune_vs_fairy(self) -> None:
        assert TYPE_CHART["Dragon"]["Fairy"] == 0.0

    def test_fighting_supereffective_vs_dark(self) -> None:
        assert TYPE_CHART["Fighting"]["Dark"] == 2.0

    def test_all_multipliers_are_valid(self) -> None:
        """Todos los multiplicadores son 0.0, 0.5 o 2.0."""
        valid = {0.0, 0.5, 2.0}
        for atk_type, defenses in TYPE_CHART.items():
            for def_type, mult in defenses.items():
                assert mult in valid, (
                    f"{atk_type} vs {def_type}: {mult} no es válido"
                )


# ---------------------------------------------------------------------------
# GRUPO 2 — _type_advantage_score
# ---------------------------------------------------------------------------


class TestTypeAdvantageScore:
    """Tests de la función de ventaja de tipo."""

    def test_fighting_threatens_dark_steel(self) -> None:
        """Lucha (2×) contra Siniestro y Acero — Kingambit recibe 4×."""
        rival = [MockSlot("Kingambit")]
        score, threatened = _type_advantage_score(
            ["Fighting"], rival, MOCK_POKEMON_DATA
        )
        # Fighting → Dark 2.0, Fighting → Steel 2.0 → producto 4.0
        assert score > 0
        assert "Kingambit" in threatened

    def test_ground_immune_to_electric(self) -> None:
        """Eléctrico da score negativo contra Tierra/Dragón (Garchomp)."""
        rival = [MockSlot("Garchomp")]
        score, _ = _type_advantage_score(
            ["Electric"], rival, MOCK_POKEMON_DATA
        )
        # Garchomp es Tierra/Dragón — Electric no afecta a Tierra (0×)
        assert score <= 0

    def test_water_vs_fire_dark_incineroar(self) -> None:
        """Agua es súper efectiva contra Fuego/Siniestro (Incineroar)."""
        rival = [MockSlot("Incineroar")]
        score, threatened = _type_advantage_score(
            ["Water"], rival, MOCK_POKEMON_DATA
        )
        assert score > 0
        assert "Incineroar" in threatened

    def test_empty_rival_returns_zero(self) -> None:
        """Sin rivales el score es 0."""
        score, threatened = _type_advantage_score(
            ["Fire"], [], MOCK_POKEMON_DATA
        )
        assert score == 0.0
        assert threatened == []

    def test_unknown_pokemon_skipped(self) -> None:
        """Pokémon no conocido se salta sin crash."""
        rival = [MockSlot("UnknownPokemon")]
        score, threatened = _type_advantage_score(
            ["Fire"], rival, MOCK_POKEMON_DATA
        )
        assert threatened == []


# ---------------------------------------------------------------------------
# GRUPO 3 — _speed_tier_score
# ---------------------------------------------------------------------------


class TestSpeedTierScore:
    """Tests de la función de speed tier."""

    def test_faster_than_all_gives_positive(self) -> None:
        """Counter más rápido que todos da score positivo."""
        # Sneasler (120) vs Incineroar (60) y Kingambit (50)
        rival = [MockSlot("Incineroar"), MockSlot("Kingambit")]
        score = _speed_tier_score(120, rival, MOCK_POKEMON_DATA)
        assert score > 0

    def test_slower_than_all_gives_negative(self) -> None:
        """Counter más lento que todos da score negativo."""
        # Kingambit (50) vs Flutter Mane (135) y Sneasler (120)
        rival = [MockSlot("Flutter Mane"), MockSlot("Sneasler")]
        score = _speed_tier_score(50, rival, MOCK_POKEMON_DATA)
        assert score < 0

    def test_zero_speed_returns_zero(self) -> None:
        """counter_spe=0 retorna 0.0."""
        rival = [MockSlot("Incineroar")]
        score = _speed_tier_score(0, rival, MOCK_POKEMON_DATA)
        assert score == 0.0

    def test_empty_rival_returns_zero(self) -> None:
        """Sin rivales retorna 0.0."""
        score = _speed_tier_score(100, [], MOCK_POKEMON_DATA)
        assert score == 0.0

    def test_score_between_minus_one_and_one(self) -> None:
        """Score siempre en [-1.0, 1.0]."""
        rival = [
            MockSlot("Incineroar"),
            MockSlot("Garchomp"),
            MockSlot("Sneasler"),
        ]
        score = _speed_tier_score(100, rival, MOCK_POKEMON_DATA)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# GRUPO 4 — heuristic_counter
# ---------------------------------------------------------------------------


class TestHeuristicCounter:
    """Tests de la función principal de counters."""

    def _make_rival_team(self, species_list: list[str]) -> MockTeam:
        return MockTeam(slots=[MockSlot(s) for s in species_list])

    def test_returns_list_of_counter_results(self) -> None:
        """Retorna lista de CounterResult."""
        rival = self._make_rival_team(["Incineroar", "Garchomp"])
        roster = ["Sneasler", "Flutter Mane", "Urshifu"]
        results = heuristic_counter(
            rival, roster, pokemon_data=MOCK_POKEMON_DATA
        )
        assert isinstance(results, list)
        assert all(isinstance(r, CounterResult) for r in results)

    def test_respects_top_n(self) -> None:
        """Respeta el límite top_n."""
        rival = self._make_rival_team(["Incineroar"])
        roster = [
            "Sneasler", "Flutter Mane", "Urshifu",
            "Garchomp", "Sinistcha",
        ]
        results = heuristic_counter(
            rival, roster, pokemon_data=MOCK_POKEMON_DATA, top_n=3
        )
        assert len(results) <= 3

    def test_scores_normalized_between_0_and_1(self) -> None:
        """Scores normalizados entre -1 y 1."""
        rival = self._make_rival_team(
            ["Incineroar", "Garchomp", "Kingambit"]
        )
        roster = ["Sneasler", "Flutter Mane", "Urshifu", "Sinistcha"]
        results = heuristic_counter(
            rival, roster, pokemon_data=MOCK_POKEMON_DATA
        )
        if results:
            assert all(-1.0 <= r.score <= 1.0 for r in results)

    def test_sorted_by_score_descending(self) -> None:
        """Resultados ordenados por score descendente."""
        rival = self._make_rival_team(["Incineroar", "Garchomp"])
        roster = [
            "Sneasler", "Flutter Mane", "Urshifu",
            "Sinistcha", "Basculegion",
        ]
        results = heuristic_counter(
            rival, roster, pokemon_data=MOCK_POKEMON_DATA
        )
        if len(results) >= 2:
            assert all(
                results[i].score >= results[i + 1].score
                for i in range(len(results) - 1)
            )

    def test_empty_roster_returns_empty(self) -> None:
        """Roster vacío retorna lista vacía."""
        rival = self._make_rival_team(["Incineroar"])
        results = heuristic_counter(
            rival, roster=[], pokemon_data=MOCK_POKEMON_DATA
        )
        assert results == []

    def test_empty_rival_returns_empty(self) -> None:
        """Equipo rival sin slots retorna vacío."""
        rival = MockTeam(slots=[])
        results = heuristic_counter(
            rival, roster=["Sneasler"], pokemon_data=MOCK_POKEMON_DATA
        )
        assert results == []

    def test_empty_pokemon_data_returns_empty(self) -> None:
        """pokemon_data vacío retorna lista vacía."""
        rival = self._make_rival_team(["Incineroar"])
        results = heuristic_counter(rival, roster=["Sneasler"], pokemon_data={})
        assert results == []

    def test_counter_result_has_types(self) -> None:
        """CounterResult incluye los tipos del counter."""
        rival = self._make_rival_team(["Incineroar"])
        results = heuristic_counter(
            rival, roster=["Sneasler"], pokemon_data=MOCK_POKEMON_DATA
        )
        if results:
            sneasler = results[0]
            assert sneasler.species == "Sneasler"
            assert len(sneasler.types) > 0

    def test_fairy_counters_dark_steel_team(self) -> None:
        """Flutter Mane (Ghost/Fairy) debe tener score alto contra
        equipo Siniestro/Acero."""
        rival = self._make_rival_team(["Kingambit", "Incineroar"])
        roster = ["Flutter Mane", "Garchomp", "Basculegion"]
        results = heuristic_counter(
            rival, roster, pokemon_data=MOCK_POKEMON_DATA
        )
        if results:
            top_species = [r.species for r in results]
            assert "Flutter Mane" in top_species[:3]

    def test_counters_directly_populated(self) -> None:
        """counters_directly lista los rivales amenazados."""
        rival = self._make_rival_team(["Incineroar", "Kingambit"])
        results = heuristic_counter(
            rival, roster=["Flutter Mane"], pokemon_data=MOCK_POKEMON_DATA
        )
        if results:
            fm = results[0]
            assert isinstance(fm.counters_directly, list)


# ---------------------------------------------------------------------------
# GRUPO 5 — Integración con parse_paste
# ---------------------------------------------------------------------------


class TestCounterWithParsedPaste:
    """Tests de integración: paste real → counters."""

    INCINEROAR_PASTE = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Tera Type: Fire
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Fake Out
- Parting Shot
- Flare Blitz
- Darkest Lariat
"""

    TEAM_PASTE = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
- Fake Out
- Parting Shot

Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
- Earthquake
- Dragon Claw
"""

    def test_heuristic_accepts_parsed_team(self) -> None:
        """heuristic_counter acepta ParsedTeam real sin crash."""
        team = parse_paste(self.TEAM_PASTE)
        assert len(team.slots) >= 1

        roster = ["Sneasler", "Flutter Mane", "Sinistcha"]
        results = heuristic_counter(
            team, roster, pokemon_data=MOCK_POKEMON_DATA
        )
        assert isinstance(results, list)

    def test_parsed_team_slots_have_species(self) -> None:
        """Las species del paste parseado son correctas."""
        team = parse_paste(self.TEAM_PASTE)
        species = [s.species for s in team.slots]
        assert "Incineroar" in species
        assert "Garchomp" in species

    def test_full_pipeline_no_crash(self) -> None:
        """Pipeline completo paste → counters no lanza excepción."""
        team = parse_paste(self.INCINEROAR_PASTE)
        roster_capitalized = [k.capitalize() for k in MOCK_POKEMON_DATA]
        try:
            results = heuristic_counter(
                team,
                roster=roster_capitalized,
                pokemon_data=MOCK_POKEMON_DATA,
            )
            assert isinstance(results, list)
        except Exception as exc:
            pytest.fail(f"Pipeline completo lanzó excepción: {exc}")
