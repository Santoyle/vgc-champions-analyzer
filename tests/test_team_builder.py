"""
Tests para los módulos de lógica de Team Builder.

Cubre:
  Grupo 1 — validate_team(): validación de equipo (species, item, mega,
             stat points). Lógica de dominio pura sin Streamlit.
  Grupo 2 — validate_team_legality(): checks de legalidad contra las listas
             del JSON de regulación.
  Grupo 3 — compute_pmi_from_teammates(): cálculo PMI sobre DataFrames
             sintéticos en memoria.
  Grupo 4 — get_top_teammates(): función de sugerencias con filtros.

Ningún test importa Streamlit ni lee archivos de disco ni hace requests HTTP.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.app.core.schema import RegulationConfig
from src.app.modules.pmi import (
    PMIPair,
    build_pmi_matrix,
    compute_pmi_from_teammates,
    get_top_teammates,
)
from src.app.modules.validate import (
    ValidationError,
    format_errors_for_ui,
    validate_team,
    validate_team_legality,
)

# ---------------------------------------------------------------------------
# Fixtures de RegulationConfig sintéticas
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reg_with_mega() -> RegulationConfig:
    """RegulationConfig con mega y stat points habilitados para tests."""
    from src.app.core.checksum import rehash_dict

    data: dict[str, object] = {
        "regulation_id": "TEST-MEGA",
        "game": "pokemon_champions",
        "date_start": "2026-04-08",
        "date_end": "2026-12-31",
        "battle_format": {
            "team_size": 6,
            "bring": 6,
            "pick": 4,
            "level_cap": 50,
            "best_of_swiss": 1,
            "best_of_topcut": 3,
            "team_preview_sec": 90,
            "turn_sec": 45,
            "player_timer_sec": 420,
            "game_timer_sec": 1200,
        },
        "mechanics": {
            "mega_enabled": True,
            "mega_max_per_battle": 1,
            "tera_enabled": True,
            "z_moves_enabled": False,
            "dynamax_enabled": False,
            "stat_points_system": True,
            "stat_points_total": 66,
            "stat_points_cap_per_stat": 32,
            "iv_system": False,
        },
        "clauses": {
            "species_clause": True,
            "item_clause": True,
            "legendary_ban": True,
            "restricted_ban": False,
            "open_team_list": True,
        },
        "pokemon_legales": [1, 2, 3, 4, 5, 6],
        "mega_evolutions_disponibles": [
            {
                "species": "Charizard",
                "mega_item": "Charizardite X",
                "mega_ability": "Tough Claws",
            }
        ],
        "items_legales": [
            "Sitrus Berry",
            "Choice Scarf",
            "Charizardite X",
            "Venusaurite",
        ],
        "moves_legales": [1, 2, 3, 4, 5],
        "checksum_sha256": "a" * 64,
        "last_verified": "2026-04-24",
        "schema_version": "1.0.0",
        "source_urls": {},
        "transition_window_days": 7,
    }
    return RegulationConfig.model_validate(rehash_dict(data))


@pytest.fixture(scope="module")
def reg_no_mega() -> RegulationConfig:
    """RegulationConfig sin mega ni stat points para tests."""
    from src.app.core.checksum import rehash_dict

    data: dict[str, object] = {
        "regulation_id": "TEST-NOMEGA",
        "game": "scarlet_violet",
        "date_start": "2025-09-01",
        "date_end": "2025-11-30",
        "battle_format": {
            "team_size": 6,
            "bring": 6,
            "pick": 4,
            "level_cap": 50,
            "best_of_swiss": 1,
            "best_of_topcut": 3,
            "team_preview_sec": 90,
            "turn_sec": 45,
            "player_timer_sec": 420,
            "game_timer_sec": 1200,
        },
        "mechanics": {
            "mega_enabled": False,
            "mega_max_per_battle": 0,
            "tera_enabled": True,
            "z_moves_enabled": False,
            "dynamax_enabled": False,
            "stat_points_system": False,
            "stat_points_total": 0,
            "stat_points_cap_per_stat": 0,
            "iv_system": True,
        },
        "clauses": {
            "species_clause": True,
            "item_clause": False,
            "legendary_ban": False,
            "restricted_ban": False,
            "open_team_list": True,
        },
        "pokemon_legales": [1, 2, 3, 4, 5, 6],
        "mega_evolutions_disponibles": [],
        "items_legales": ["Sitrus Berry", "Choice Scarf"],
        "moves_legales": [1, 2, 3, 4, 5],
        "checksum_sha256": "a" * 64,
        "last_verified": "2026-04-24",
        "schema_version": "1.0.0",
        "source_urls": {},
        "transition_window_days": 7,
    }
    return RegulationConfig.model_validate(rehash_dict(data))


# ---------------------------------------------------------------------------
# Helper de slots
# ---------------------------------------------------------------------------


def _slot(
    species: str = "",
    item: str = "",
    mega_capable: bool = False,
    stat_points: dict[str, int] | None = None,
) -> dict[str, object]:
    """Helper para crear un slot de equipo en tests."""
    return {
        "species": species,
        "item": item,
        "ability": "",
        "tera_type": "",
        "nature": "Hardy",
        "moves": ["", "", "", ""],
        "mega_capable": mega_capable,
        "stat_points": stat_points or {},
    }


# ---------------------------------------------------------------------------
# Grupo 1 — Tests de validate_team
# ---------------------------------------------------------------------------


class TestValidateTeam:
    """Tests para validate_team() — función de dominio puro."""

    def test_empty_team_returns_no_errors(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Equipo vacío no genera errores."""
        assert validate_team([], reg_with_mega) == []

    def test_all_empty_slots_returns_no_errors(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """6 slots vacíos no generan errores."""
        assert validate_team([_slot() for _ in range(6)], reg_with_mega) == []

    def test_valid_team_returns_no_errors(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Equipo con 3 Pokémon distintos sin ítems duplicados es válido."""
        team = [
            _slot("Bulbasaur", "Sitrus Berry"),
            _slot("Charmander", "Choice Scarf"),
            _slot("Squirtle", ""),
        ]
        assert validate_team(team, reg_with_mega) == []

    def test_species_clause_detects_duplicate(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Species clause detecta dos Pokémon iguales."""
        team = [_slot("Bulbasaur", "Sitrus Berry"), _slot("Bulbasaur", "Choice Scarf")]
        errors = validate_team(team, reg_with_mega)
        assert any(e.code == "species_clause" for e in errors)

    def test_species_clause_error_has_slot_idx(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """El error de species clause tiene slot_idx."""
        team = [_slot("Bulbasaur"), _slot("Bulbasaur")]
        errors = validate_team(team, reg_with_mega)
        species_errors = [e for e in errors if e.code == "species_clause"]
        assert len(species_errors) > 0
        assert species_errors[0].slot_idx is not None

    def test_item_clause_detects_duplicate(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Item clause detecta ítems duplicados."""
        team = [_slot("Bulbasaur", "Sitrus Berry"), _slot("Charmander", "Sitrus Berry")]
        errors = validate_team(team, reg_with_mega)
        assert any(e.code == "item_clause" for e in errors)

    def test_item_clause_ignores_empty_items(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Item clause ignora slots sin ítem."""
        team = [_slot("Bulbasaur", ""), _slot("Charmander", ""), _slot("Squirtle", "")]
        errors = validate_team(team, reg_with_mega)
        assert not any(e.code == "item_clause" for e in errors)

    def test_item_clause_inactive_when_disabled(
        self, reg_no_mega: RegulationConfig
    ) -> None:
        """Item clause no se aplica cuando está desactivada en la regulación."""
        team = [_slot("Bulbasaur", "Sitrus Berry"), _slot("Charmander", "Sitrus Berry")]
        errors = validate_team(team, reg_no_mega)
        assert not any(e.code == "item_clause" for e in errors)

    def test_mega_clause_detects_two_megas(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Mega clause detecta más de 1 Mega Stone."""
        team = [
            _slot("Charizard", "Charizardite X", mega_capable=True),
            _slot("Venusaur", "Venusaurite", mega_capable=True),
        ]
        errors = validate_team(team, reg_with_mega)
        assert any(e.code == "mega_clause" for e in errors)

    def test_mega_clause_allows_one_mega(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Un solo Mega Stone es válido."""
        team = [
            _slot("Charizard", "Charizardite X", mega_capable=True),
            _slot("Bulbasaur", "Sitrus Berry", mega_capable=False),
        ]
        errors = validate_team(team, reg_with_mega)
        assert not any(e.code == "mega_clause" for e in errors)

    def test_mega_clause_inactive_when_disabled(
        self, reg_no_mega: RegulationConfig
    ) -> None:
        """Mega clause no se aplica cuando mega está desactivado."""
        team = [
            _slot("Charizard", "", mega_capable=True),
            _slot("Venusaur", "", mega_capable=True),
        ]
        errors = validate_team(team, reg_no_mega)
        assert not any(e.code == "mega_clause" for e in errors)

    def test_stat_points_total_exceeded(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Stat points totales superando el límite genera error."""
        team = [
            _slot(
                "Bulbasaur",
                stat_points={"hp": 32, "atk": 32, "def": 10, "spa": 0, "spd": 0, "spe": 0},
            )
        ]  # total = 74 > 66
        errors = validate_team(team, reg_with_mega)
        assert any(e.code == "stat_points" for e in errors)

    def test_stat_points_per_stat_exceeded(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Un stat individual superando el cap genera error."""
        team = [
            _slot(
                "Bulbasaur",
                stat_points={"hp": 40, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            )
        ]  # hp=40 > 32 (cap)
        errors = validate_team(team, reg_with_mega)
        assert any(e.code == "stat_points" for e in errors)

    def test_stat_points_inactive_when_disabled(
        self, reg_no_mega: RegulationConfig
    ) -> None:
        """Stat points no se validan cuando el sistema está desactivado."""
        team = [_slot("Bulbasaur", stat_points={"hp": 999})]
        errors = validate_team(team, reg_no_mega)
        assert not any(e.code == "stat_points" for e in errors)

    def test_validation_error_is_frozen(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """ValidationError es inmutable (frozen=True)."""
        err = ValidationError(code="test", message="test msg", slot_idx=0)
        with pytest.raises((AttributeError, TypeError)):
            err.code = "modified"  # type: ignore[misc]

    def test_format_errors_for_ui_prefixes_emoji(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """format_errors_for_ui agrega ❌ al inicio de cada mensaje."""
        errors = [ValidationError("species_clause", "Test error")]
        formatted = format_errors_for_ui(errors)
        assert len(formatted) == 1
        assert formatted[0].startswith("❌")

    def test_multiple_errors_all_returned(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Se retornan todos los errores de todos los checks, no solo el primero."""
        team = [
            _slot("Bulbasaur", "Sitrus Berry", mega_capable=True),
            _slot("Bulbasaur", "Sitrus Berry", mega_capable=True),
        ]
        errors = validate_team(team, reg_with_mega)
        codes = {e.code for e in errors}
        assert "species_clause" in codes
        assert "item_clause" in codes
        assert "mega_clause" in codes


# ---------------------------------------------------------------------------
# Grupo 2 — Tests de validate_team_legality
# ---------------------------------------------------------------------------


class TestValidateTeamLegality:
    """Tests para validate_team_legality()."""

    def test_legal_item_passes(self, reg_with_mega: RegulationConfig) -> None:
        """Ítem en items_legales no genera error."""
        team = [_slot("Bulbasaur", "Sitrus Berry")]
        errors = validate_team_legality(team, reg_with_mega)
        assert not any(e.code == "illegal_item" for e in errors)

    def test_illegal_item_detected(self, reg_with_mega: RegulationConfig) -> None:
        """Ítem no en items_legales genera error."""
        team = [_slot("Bulbasaur", "Life Orb")]
        errors = validate_team_legality(team, reg_with_mega)
        assert any(e.code == "illegal_item" for e in errors)

    def test_empty_item_not_flagged(self, reg_with_mega: RegulationConfig) -> None:
        """Slot sin ítem no genera error de item."""
        team = [_slot("Bulbasaur", "")]
        errors = validate_team_legality(team, reg_with_mega)
        assert not any(e.code == "illegal_item" for e in errors)

    def test_illegal_item_error_has_slot_idx(
        self, reg_with_mega: RegulationConfig
    ) -> None:
        """Error de ítem ilegal tiene slot_idx correcto."""
        team = [
            _slot("Bulbasaur", "Sitrus Berry"),
            _slot("Charmander", "Life Orb"),
        ]
        errors = validate_team_legality(team, reg_with_mega)
        item_errors = [e for e in errors if e.code == "illegal_item"]
        assert len(item_errors) > 0
        assert item_errors[0].slot_idx == 1


# ---------------------------------------------------------------------------
# Grupo 3 — Tests de compute_pmi_from_teammates
# ---------------------------------------------------------------------------


class TestComputePMI:
    """Tests para el cálculo de PMI."""

    def _make_teammates_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "regulation_id": ["TEST"] * 4,
                "pokemon": ["Incineroar", "Incineroar", "Garchomp", "Garchomp"],
                "teammate": ["Garchomp", "Rillaboom", "Incineroar", "Sinistcha"],
                "avg_correlation": [45.0, 20.0, 45.0, 15.0],
                "n_months_seen": [4, 4, 4, 4],
            }
        )

    def _make_usage_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "regulation_id": ["TEST"] * 4,
                "pokemon": ["Incineroar", "Garchomp", "Rillaboom", "Sinistcha"],
                "avg_usage_pct": [54.0, 37.0, 25.0, 34.0],
            }
        )

    def test_returns_non_empty_df(self) -> None:
        """Con datos válidos retorna DataFrame no vacío."""
        df_pmi = compute_pmi_from_teammates(
            self._make_teammates_df(), self._make_usage_df()
        )
        assert not df_pmi.empty

    def test_has_required_columns(self) -> None:
        """DataFrame de PMI tiene las columnas esperadas."""
        df_pmi = compute_pmi_from_teammates(
            self._make_teammates_df(), self._make_usage_df()
        )
        required = {"pokemon", "teammate", "pmi", "ppmi", "co_usage_pct"}
        assert required.issubset(set(df_pmi.columns))

    def test_ppmi_is_non_negative(self) -> None:
        """PPMI siempre es >= 0."""
        df_pmi = compute_pmi_from_teammates(
            self._make_teammates_df(), self._make_usage_df()
        )
        assert (df_pmi["ppmi"] >= 0).all()

    def test_empty_teammates_returns_empty(self) -> None:
        """DataFrame vacío de teammates retorna DataFrame vacío."""
        df_pmi = compute_pmi_from_teammates(pd.DataFrame(), self._make_usage_df())
        assert df_pmi.empty

    def test_empty_usage_returns_empty(self) -> None:
        """DataFrame vacío de usage retorna DataFrame vacío."""
        df_pmi = compute_pmi_from_teammates(self._make_teammates_df(), pd.DataFrame())
        assert df_pmi.empty

    def test_high_co_usage_has_positive_pmi(self) -> None:
        """Par con alta co-uso (45%) entre Pokémon con uso moderado
        debe tener PMI positivo."""
        df_pmi = compute_pmi_from_teammates(
            self._make_teammates_df(), self._make_usage_df()
        )
        pair = df_pmi[
            (df_pmi["pokemon"] == "Incineroar") & (df_pmi["teammate"] == "Garchomp")
        ]
        if not pair.empty:
            assert float(pair.iloc[0]["pmi"]) > 0


# ---------------------------------------------------------------------------
# Grupo 4 — Tests de get_top_teammates
# ---------------------------------------------------------------------------


class TestGetTopTeammates:
    """Tests para la función de sugerencias."""

    @pytest.fixture
    def sample_pmi_df(self) -> pd.DataFrame:
        """DataFrame PMI sintético para tests."""
        return pd.DataFrame(
            {
                "pokemon": ["Incineroar"] * 4,
                "teammate": ["Garchomp", "Rillaboom", "Sinistcha", "Sneasler"],
                "pmi": [1.2, 0.8, 0.5, 0.3],
                "ppmi": [1.2, 0.8, 0.5, 0.3],
                "co_usage_pct": [45.0, 30.0, 20.0, 15.0],
                "n_months_seen": [4, 4, 4, 4],
            }
        )

    def test_returns_list_of_pmi_pairs(self, sample_pmi_df: pd.DataFrame) -> None:
        """Retorna lista de PMIPair."""
        result = get_top_teammates(sample_pmi_df, "Incineroar")
        assert isinstance(result, list)
        assert all(isinstance(p, PMIPair) for p in result)

    def test_respects_top_n(self, sample_pmi_df: pd.DataFrame) -> None:
        """Respeta el límite top_n."""
        result = get_top_teammates(sample_pmi_df, "Incineroar", top_n=2)
        assert len(result) <= 2

    def test_excludes_specified_pokemon(self, sample_pmi_df: pd.DataFrame) -> None:
        """Excluye Pokémon de la lista exclude."""
        result = get_top_teammates(sample_pmi_df, "Incineroar", exclude=["Garchomp"])
        assert "Garchomp" not in [p.teammate for p in result]

    def test_excludes_self(self, sample_pmi_df: pd.DataFrame) -> None:
        """El Pokémon de referencia no aparece en sus propias sugerencias."""
        df_with_self = pd.concat(
            [
                sample_pmi_df,
                pd.DataFrame(
                    [
                        {
                            "pokemon": "Incineroar",
                            "teammate": "Incineroar",
                            "pmi": 2.0,
                            "ppmi": 2.0,
                            "co_usage_pct": 100.0,
                            "n_months_seen": 4,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        result = get_top_teammates(df_with_self, "Incineroar")
        assert "Incineroar" not in [p.teammate for p in result]

    def test_unknown_pokemon_returns_empty(self, sample_pmi_df: pd.DataFrame) -> None:
        """Pokémon sin datos retorna lista vacía."""
        assert get_top_teammates(sample_pmi_df, "Pikachu") == []

    def test_sorted_by_ppmi_descending(self, sample_pmi_df: pd.DataFrame) -> None:
        """Resultados ordenados por ppmi descendente."""
        result = get_top_teammates(sample_pmi_df, "Incineroar")
        if len(result) >= 2:
            assert all(
                result[i].ppmi >= result[i + 1].ppmi for i in range(len(result) - 1)
            )
