"""
Tests unitarios para las nuevas funciones del Counter v2 (Bloque 11).

Grupos:
  1. TestWPCounterResult    (3 tests)  — dataclass WPCounterResult (campos, defaults).
  2. TestWPCounter          (5 tests)  — wp_counter con predict_proba y
                                         extract_features mockeados; nunca usa modelo real.
  3. TestTeamMatchupResult  (3 tests)  — dataclass TeamMatchupResult (campos, None values).
  4. TestTopTeamsVsRival    (6 tests)  — top_teams_vs_rival con _load_recent_teams y
                                         load_model/predict_proba mockeados.
  5. TestLoadRecentTeams    (3 tests)  — _load_recent_teams: I/O con fallback graceful;
                                         los tests de regulación ficticia garantizan vacío
                                         sin Parquets.

Estrategia de mocks:
  - predict_proba y load_model se importan de forma lazy dentro de las funciones de
    counter.py, por lo que se parchean en su módulo de origen:
      src.app.modules.wp_train.predict_proba
      src.app.modules.wp_train.load_model
  - extract_features se importa a nivel de módulo en counter.py, por lo que se
    parchea como: src.app.modules.counter.extract_features
  - _load_recent_teams es función de módulo en counter.py:
      src.app.modules.counter._load_recent_teams
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app.modules.counter import (
    TeamMatchupResult,
    WPCounterResult,
    _load_recent_teams,
    heuristic_counter,
    top_teams_vs_rival,
    wp_counter,
)


# ---------------------------------------------------------------------------
# Helpers locales
# ---------------------------------------------------------------------------


@dataclass
class MockSlot:
    """Slot rival sintético (duck-typed)."""

    species: str
    item: str = ""
    moves: list[str] | None = None
    mega_capable: bool = False

    def __post_init__(self) -> None:
        if self.moves is None:
            self.moves = []


@dataclass
class MockTeam:
    """Equipo rival sintético (duck-typed)."""

    slots: list[MockSlot]


MOCK_POKEMON_DATA: dict[str, dict[str, Any]] = {
    "incineroar": {
        "name": "incineroar",
        "types": ["Fire", "Dark"],
        "base_stats": {"speed": 60},
    },
    "garchomp": {
        "name": "garchomp",
        "types": ["Dragon", "Ground"],
        "base_stats": {"speed": 102},
    },
    "sneasler": {
        "name": "sneasler",
        "types": ["Fighting", "Poison"],
        "base_stats": {"speed": 120},
    },
    "flutter mane": {
        "name": "flutter mane",
        "types": ["Ghost", "Fairy"],
        "base_stats": {"speed": 135},
    },
    "kingambit": {
        "name": "kingambit",
        "types": ["Dark", "Steel"],
        "base_stats": {"speed": 50},
    },
    "urshifu": {
        "name": "urshifu",
        "types": ["Fighting", "Dark"],
        "base_stats": {"speed": 97},
    },
}


def _make_rival(species_list: list[str]) -> MockTeam:
    return MockTeam(slots=[MockSlot(s) for s in species_list])


def _make_candidate_teams(n: int = 5) -> list[dict[str, Any]]:
    """Genera candidatos sintéticos con el mismo equipo de 6."""
    pool = ["Incineroar", "Garchomp", "Sneasler", "Flutter Mane", "Kingambit", "Urshifu"]
    return [
        {"team": pool[:6], "replay_id": f"replay-{i}", "rating": 1700 + i * 10}
        for i in range(n)
    ]


def _make_dummy_replay_features() -> Any:
    """Genera un ReplayFeatures sintético para inyectar en mocks."""
    from src.app.modules.wp import ReplayFeatures

    return ReplayFeatures(
        replay_id="test",
        regulation_id="M-A",
        label=None,
        rating_norm=0.5,
        month_idx=0,
        p1_type_coverage=[0.0] * 18,
        p2_type_coverage=[0.0] * 18,
        p1_has_fake_out=0.0,
        p2_has_fake_out=0.0,
        p1_has_trick_room=0.0,
        p2_has_trick_room=0.0,
        p1_has_intimidate=0.0,
        p2_has_intimidate=0.0,
        p1_has_redirection=0.0,
        p2_has_redirection=0.0,
        p1_has_tailwind=0.0,
        p2_has_tailwind=0.0,
        p1_speed_control=0.0,
        p2_speed_control=0.0,
        p1_team_size=1.0,
        p2_team_size=2.0,
    )


# ---------------------------------------------------------------------------
# GRUPO 1 — WPCounterResult dataclass
# ---------------------------------------------------------------------------


class TestWPCounterResult:
    """Tests del dataclass WPCounterResult."""

    def test_fields_exist(self) -> None:
        """WPCounterResult tiene los campos esperados con los valores correctos."""
        result = WPCounterResult(
            species="Sneasler",
            wp_score=0.72,
            heuristic_score=0.85,
            types=["Fighting", "Poison"],
            counters_directly=["Incineroar"],
        )
        assert result.species == "Sneasler"
        assert result.wp_score == pytest.approx(0.72, abs=0.001)
        assert result.heuristic_score == pytest.approx(0.85, abs=0.001)
        assert result.types == ["Fighting", "Poison"]
        assert result.counters_directly == ["Incineroar"]

    def test_wp_score_range(self) -> None:
        """wp_score puede ser cualquier float en [0, 1] — no validado en el dataclass."""
        result = WPCounterResult(species="Test", wp_score=0.5, heuristic_score=0.0)
        assert 0.0 <= result.wp_score <= 1.0

    def test_default_empty_lists(self) -> None:
        """types y counters_directly tienen default_factory=list."""
        result = WPCounterResult(species="Test", wp_score=0.5, heuristic_score=0.0)
        assert result.types == []
        assert result.counters_directly == []


# ---------------------------------------------------------------------------
# GRUPO 2 — wp_counter con modelo mockeado
# ---------------------------------------------------------------------------


class TestWPCounter:
    """Tests para wp_counter con predict_proba y extract_features mockeados."""

    def test_returns_list_of_wp_counter_results(self) -> None:
        """Con modelo disponible retorna lista de WPCounterResult."""
        rival = _make_rival(["Incineroar", "Garchomp"])
        roster = ["Sneasler", "Flutter Mane"]
        dummy_feat = _make_dummy_replay_features()

        with patch(
            "src.app.modules.counter.extract_features",
            return_value=dummy_feat,
        ):
            with patch(
                "src.app.modules.wp_train.predict_proba",
                return_value=np.array([0.65, 0.72]),
            ):
                results = wp_counter(
                    rival,
                    roster=roster,
                    regulation_id="M-A",
                    pokemon_data=MOCK_POKEMON_DATA,
                )

        assert isinstance(results, list)
        assert all(isinstance(r, WPCounterResult) for r in results)

    def test_returns_empty_for_empty_roster(self) -> None:
        """Lista vacía para roster vacío."""
        rival = _make_rival(["Incineroar"])
        results = wp_counter(
            rival,
            roster=[],
            regulation_id="M-A",
            pokemon_data=MOCK_POKEMON_DATA,
        )
        assert results == []

    def test_returns_empty_for_empty_rival(self) -> None:
        """Lista vacía para equipo rival sin slots."""
        rival = MockTeam(slots=[])
        results = wp_counter(
            rival,
            roster=["Sneasler"],
            regulation_id="M-A",
            pokemon_data=MOCK_POKEMON_DATA,
        )
        assert results == []

    def test_returns_empty_when_model_unavailable(self) -> None:
        """Lista vacía cuando predict_proba retorna None (modelo no entrenado)."""
        rival = _make_rival(["Incineroar"])
        roster = ["Sneasler", "Flutter Mane"]
        dummy_feat = _make_dummy_replay_features()

        with patch(
            "src.app.modules.counter.extract_features",
            return_value=dummy_feat,
        ):
            with patch(
                "src.app.modules.wp_train.predict_proba",
                return_value=None,
            ):
                results = wp_counter(
                    rival,
                    roster=roster,
                    regulation_id="M-A",
                    pokemon_data=MOCK_POKEMON_DATA,
                )

        assert results == []

    def test_sorted_by_wp_score_descending(self) -> None:
        """Resultados ordenados por wp_score descendente."""
        rival = _make_rival(["Incineroar", "Kingambit"])
        roster = ["Sneasler", "Flutter Mane", "Urshifu"]
        dummy_feat = _make_dummy_replay_features()

        with patch(
            "src.app.modules.counter.extract_features",
            return_value=dummy_feat,
        ):
            with patch(
                "src.app.modules.wp_train.predict_proba",
                return_value=np.array([0.6, 0.8, 0.5]),
            ):
                results = wp_counter(
                    rival,
                    roster=roster,
                    regulation_id="M-A",
                    pokemon_data=MOCK_POKEMON_DATA,
                )

        if len(results) >= 2:
            assert all(
                results[i].wp_score >= results[i + 1].wp_score
                for i in range(len(results) - 1)
            )

    def test_respects_top_n(self) -> None:
        """top_n limita el número de resultados retornados."""
        rival = _make_rival(["Incineroar"])
        roster = ["Sneasler", "Flutter Mane", "Urshifu", "Garchomp", "Kingambit"]
        dummy_feat = _make_dummy_replay_features()

        with patch(
            "src.app.modules.counter.extract_features",
            return_value=dummy_feat,
        ):
            with patch(
                "src.app.modules.wp_train.predict_proba",
                return_value=np.array([0.6, 0.7, 0.5, 0.4, 0.3]),
            ):
                results = wp_counter(
                    rival,
                    roster=roster,
                    regulation_id="M-A",
                    pokemon_data=MOCK_POKEMON_DATA,
                    top_n=3,
                )

        assert len(results) <= 3


# ---------------------------------------------------------------------------
# GRUPO 3 — TeamMatchupResult dataclass
# ---------------------------------------------------------------------------


class TestTeamMatchupResult:
    """Tests del dataclass TeamMatchupResult."""

    def test_fields_exist(self) -> None:
        """TeamMatchupResult tiene todos los campos esperados."""
        result = TeamMatchupResult(
            team=[
                "Incineroar", "Garchomp", "Sneasler",
                "Flutter Mane", "Kingambit", "Urshifu",
            ],
            wp_score=0.68,
            heuristic_score=0.75,
            source_replay_id="gen9champs-123",
            regulation_id="M-A",
        )
        assert len(result.team) == 6
        assert result.wp_score == pytest.approx(0.68, abs=0.001)
        assert result.heuristic_score == pytest.approx(0.75, abs=0.001)
        assert result.source_replay_id == "gen9champs-123"
        assert result.regulation_id == "M-A"

    def test_wp_score_can_be_none(self) -> None:
        """wp_score puede ser None cuando se usa el fallback heurístico."""
        result = TeamMatchupResult(
            team=["Incineroar"],
            wp_score=None,
            heuristic_score=0.5,
            source_replay_id=None,
            regulation_id="M-A",
        )
        assert result.wp_score is None

    def test_source_replay_id_can_be_none(self) -> None:
        """source_replay_id puede ser None para equipos generados."""
        result = TeamMatchupResult(
            team=["Incineroar"],
            wp_score=0.5,
            heuristic_score=0.5,
            source_replay_id=None,
            regulation_id="M-A",
        )
        assert result.source_replay_id is None


# ---------------------------------------------------------------------------
# GRUPO 4 — top_teams_vs_rival
# ---------------------------------------------------------------------------


class TestTopTeamsVsRival:
    """Tests para top_teams_vs_rival con _load_recent_teams y WP model mockeados."""

    def test_returns_list_of_team_matchup_results(self) -> None:
        """Retorna lista de TeamMatchupResult cuando hay candidatos."""
        rival = _make_rival(["Incineroar", "Garchomp"])
        with patch(
            "src.app.modules.counter._load_recent_teams",
            return_value=_make_candidate_teams(5),
        ):
            results = top_teams_vs_rival(
                rival,
                regulation_id="M-A",
                pokemon_data=MOCK_POKEMON_DATA,
                top_n=3,
            )

        assert isinstance(results, list)
        assert all(isinstance(r, TeamMatchupResult) for r in results)

    def test_returns_empty_for_empty_rival(self) -> None:
        """Lista vacía para equipo rival sin slots."""
        rival = MockTeam(slots=[])
        results = top_teams_vs_rival(rival, regulation_id="M-A")
        assert results == []

    def test_returns_empty_when_no_candidates(self) -> None:
        """Lista vacía cuando _load_recent_teams retorna lista vacía."""
        rival = _make_rival(["Incineroar"])
        with patch(
            "src.app.modules.counter._load_recent_teams",
            return_value=[],
        ):
            results = top_teams_vs_rival(rival, regulation_id="M-A")
        assert results == []

    def test_respects_top_n(self) -> None:
        """Retorna como máximo top_n equipos."""
        rival = _make_rival(["Incineroar"])
        with patch(
            "src.app.modules.counter._load_recent_teams",
            return_value=_make_candidate_teams(10),
        ):
            results = top_teams_vs_rival(
                rival,
                regulation_id="M-A",
                pokemon_data=MOCK_POKEMON_DATA,
                top_n=3,
            )
        assert len(results) <= 3

    def test_fallback_heuristic_when_no_wp_model(self) -> None:
        """Usa heurístico como fallback cuando load_model retorna None; wp_score es None."""
        rival = _make_rival(["Incineroar", "Garchomp"])
        with patch(
            "src.app.modules.counter._load_recent_teams",
            return_value=_make_candidate_teams(3),
        ):
            with patch(
                "src.app.modules.wp_train.load_model",
                return_value=None,
            ):
                results = top_teams_vs_rival(
                    rival,
                    regulation_id="M-A",
                    pokemon_data=MOCK_POKEMON_DATA,
                    top_n=3,
                )

        for r in results:
            assert r.wp_score is None
            assert r.heuristic_score >= 0.0

    def test_no_exception_on_any_input(self) -> None:
        """top_teams_vs_rival es resiliente a errores de I/O en _load_recent_teams."""
        rival = _make_rival(["Incineroar"])
        with patch(
            "src.app.modules.counter._load_recent_teams",
            side_effect=Exception("IO error"),
        ):
            try:
                results = top_teams_vs_rival(rival, regulation_id="M-A")
                assert isinstance(results, list)
            except Exception as exc:
                pytest.fail(f"top_teams_vs_rival propagó excepción: {exc}")


# ---------------------------------------------------------------------------
# GRUPO 5 — _load_recent_teams
# ---------------------------------------------------------------------------


class TestLoadRecentTeams:
    """Tests para la carga de equipos desde Parquets con fallback graceful."""

    def test_returns_empty_when_no_parquets(self) -> None:
        """Lista vacía cuando la regulación ficticia no tiene directorio en data/raw/."""
        result = _load_recent_teams("NONEXISTENT-REG-XYZ", max_teams=10)
        assert result == []

    def test_returns_list_of_dicts_with_correct_keys(self) -> None:
        """Si hay datos reales, cada elemento tiene team, replay_id y rating."""
        result = _load_recent_teams("I", max_teams=5)
        for item in result:
            assert "team" in item
            assert "replay_id" in item
            assert "rating" in item
            assert isinstance(item["team"], list)

    def test_never_raises_exception(self) -> None:
        """_load_recent_teams nunca propaga excepción para ninguna regulación."""
        try:
            result = _load_recent_teams("FAKE-REG-99", max_teams=1)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"_load_recent_teams propagó excepción: {exc}")
