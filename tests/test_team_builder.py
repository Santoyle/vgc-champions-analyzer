"""
Tests unitarios para la lógica del Team Builder.

Grupos:
  1 — TestPasteImport: tests de parse_paste() con strings reales.
      No usa mocks — la función es pura.
  2 — TestTeamValidation: tests de validación de equipos.
      La lógica de validate_team_ui() se reimplementa aquí como
      función helper pura para evitar importar la página Streamlit
      (que ejecuta código de UI al nivel de módulo).
  3 — TestPokemonMasterLoading: tests de lectura de pokemon_master.json.
      Lee el archivo directamente sin st.cache_data.
      Usa pytest.skip() si el archivo no existe o está vacío.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.app.data.parsers.ps_paste import parse_paste

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_POKEMON_MASTER_PATH = _PROJECT_ROOT / "data" / "pokemon_master.json"

# ---------------------------------------------------------------------------
# Pastes sintéticos para TestPasteImport
# ---------------------------------------------------------------------------

_PASTE_6_SLOTS = """\
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

Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
Tera Type: Steel
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Stone Edge
- Protect

Sneasler @ Focus Sash
Ability: Unburden
Level: 50
Tera Type: Poison
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Close Combat
- Dire Claw
- Fake Out
- Protect

Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Sassy Nature
- Spore
- Pollen Puff
- Rage Powder
- Protect

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Protect

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Grass
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Grassy Glide
- Wood Hammer
- U-turn
- Fake Out
"""

_PASTE_MEGA_STONE = """\
Charizard @ Charizardite X
Ability: Tough Claws
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Flare Blitz
- Dragon Claw
- Protect
- Earthquake
"""

_PASTE_NICKNAME = """\
Cinder (Incineroar) @ Sitrus Berry
Ability: Intimidate
Level: 50
- Fake Out
- Parting Shot
"""


# ---------------------------------------------------------------------------
# Helper local de validación (replica validate_team_ui sin Streamlit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Clauses:
    item_clause: bool


@dataclass(frozen=True)
class _Mechanics:
    mega_enabled: bool


@dataclass(frozen=True)
class _MockRegConfig:
    clauses: _Clauses
    mechanics: _Mechanics


def _validate(
    slots: list[dict[str, Any]],
    cfg: _MockRegConfig,
) -> list[str]:
    """
    Reimplementación de validate_team_ui() como función pura local.
    Misma lógica: species clause, item clause, mega clause.
    """
    errors: list[str] = []
    filled = [s for s in slots if s.get("species")]

    if not filled:
        return errors

    # Species clause (siempre activa)
    species_list = [s["species"] for s in filled]
    if len(set(species_list)) != len(species_list):
        dupes = [s for s in set(species_list) if species_list.count(s) > 1]
        errors.append(f"❌ Species clause: {dupes} aparece más de una vez.")

    # Item clause
    if cfg.clauses.item_clause:
        items = [s.get("item", "") for s in filled if s.get("item")]
        if len(set(items)) != len(items):
            dup_items = [i for i in set(items) if items.count(i) > 1]
            errors.append(
                f"❌ Item clause: {dup_items} aparece en más de un slot."
            )

    # Mega clause
    if cfg.mechanics.mega_enabled:
        mega_count = sum(1 for s in filled if s.get("mega_capable", False))
        if mega_count > 1:
            errors.append(
                f"❌ Solo puede haber 1 Pokémon con Mega Stone. "
                f"Tienes {mega_count}."
            )

    return errors


# ---------------------------------------------------------------------------
# Helper local de carga de pokemon_master.json
# ---------------------------------------------------------------------------


def _load_master() -> dict[int, str]:
    """Lee pokemon_master.json directamente sin st.cache_data."""
    if not _POKEMON_MASTER_PATH.exists():
        return {}
    try:
        raw = json.loads(_POKEMON_MASTER_PATH.read_text(encoding="utf-8"))
        pokemon_data = raw.get("pokemon", {})
        return {
            int(dex_id): entry["name"].capitalize()
            for dex_id, entry in pokemon_data.items()
            if isinstance(entry, dict) and "name" in entry
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Grupo 1 — TestPasteImport
# ---------------------------------------------------------------------------


class TestPasteImport:
    """Tests de parse_paste() con strings reales. Sin mocks."""

    def test_parse_valid_paste_returns_6_slots(self) -> None:
        """Un paste de 6 Pokémon produce exactamente 6 slots."""
        team = parse_paste(_PASTE_6_SLOTS)
        assert len(team.slots) == 6

    def test_parse_paste_with_mega_stone_sets_mega_capable(self) -> None:
        """Un ítem tipo Mega Stone activa mega_capable=True."""
        team = parse_paste(_PASTE_MEGA_STONE)
        assert len(team.slots) == 1
        assert team.slots[0].mega_capable is True

    def test_parse_paste_preserves_item(self) -> None:
        """El ítem se parsea y preserva correctamente."""
        team = parse_paste(_PASTE_6_SLOTS)
        incineroar = next(
            (s for s in team.slots if s.species == "Incineroar"), None
        )
        assert incineroar is not None
        assert incineroar.item == "Sitrus Berry"

    def test_parse_paste_preserves_moves(self) -> None:
        """Los 4 movimientos se parsean correctamente."""
        team = parse_paste(_PASTE_MEGA_STONE)
        slot = team.slots[0]
        assert len(slot.moves) == 4
        assert "Flare Blitz" in slot.moves
        assert "Dragon Claw" in slot.moves
        assert "Protect" in slot.moves
        assert "Earthquake" in slot.moves

    def test_parse_paste_with_nickname(self) -> None:
        """El nickname se separa de la especie correctamente."""
        team = parse_paste(_PASTE_NICKNAME)
        assert len(team.slots) == 1
        assert team.slots[0].species == "Incineroar"
        assert team.slots[0].nickname == "Cinder"

    def test_parse_empty_paste_returns_empty_team(self) -> None:
        """Paste vacío retorna equipo sin slots y con warnings."""
        team = parse_paste("")
        assert len(team.slots) == 0
        assert len(team.parse_warnings) > 0


# ---------------------------------------------------------------------------
# Grupo 2 — TestTeamValidation
# ---------------------------------------------------------------------------


class TestTeamValidation:
    """
    Tests de lógica de validación de equipos.
    Usa _validate() — reimplementación local de validate_team_ui().
    """

    _cfg_base = _MockRegConfig(
        clauses=_Clauses(item_clause=True),
        mechanics=_Mechanics(mega_enabled=True),
    )
    _cfg_no_mega = _MockRegConfig(
        clauses=_Clauses(item_clause=False),
        mechanics=_Mechanics(mega_enabled=False),
    )

    def test_empty_team_returns_no_errors(self) -> None:
        """Un equipo sin slots completados no genera errores."""
        slots: list[dict[str, Any]] = [
            {"species": "", "item": "", "mega_capable": False}
            for _ in range(6)
        ]
        assert _validate(slots, self._cfg_base) == []

    def test_team_with_duplicate_species_returns_error(self) -> None:
        """Dos Incineroar en el equipo → error de species clause."""
        slots: list[dict[str, Any]] = [
            {"species": "Incineroar", "item": "Sitrus Berry", "mega_capable": False},
            {"species": "Incineroar", "item": "Rocky Helmet", "mega_capable": False},
            {"species": "Garchomp", "item": "Choice Scarf", "mega_capable": False},
        ]
        errors = _validate(slots, self._cfg_base)
        assert len(errors) == 1
        assert "Species clause" in errors[0]

    def test_team_with_duplicate_items_returns_error(self) -> None:
        """Dos slots con Sitrus Berry → error de item clause."""
        slots: list[dict[str, Any]] = [
            {"species": "Incineroar", "item": "Sitrus Berry", "mega_capable": False},
            {"species": "Garchomp", "item": "Sitrus Berry", "mega_capable": False},
        ]
        errors = _validate(slots, self._cfg_base)
        assert any("Item clause" in e for e in errors)

    def test_team_with_two_megas_returns_error(self) -> None:
        """Dos Pokémon con Mega Stone → error de mega clause."""
        slots: list[dict[str, Any]] = [
            {"species": "Charizard", "item": "Charizardite X", "mega_capable": True},
            {"species": "Venusaur", "item": "Venusaurite", "mega_capable": True},
            {"species": "Incineroar", "item": "Sitrus Berry", "mega_capable": False},
        ]
        errors = _validate(slots, self._cfg_base)
        assert any("Mega Stone" in e for e in errors)

    def test_valid_team_returns_no_errors(self) -> None:
        """Un equipo bien formado no produce errores."""
        slots: list[dict[str, Any]] = [
            {"species": "Incineroar", "item": "Sitrus Berry", "mega_capable": False},
            {"species": "Garchomp", "item": "Choice Scarf", "mega_capable": False},
            {"species": "Sneasler", "item": "Focus Sash", "mega_capable": False},
            {"species": "Amoonguss", "item": "Rocky Helmet", "mega_capable": False},
            {"species": "Flutter Mane", "item": "Choice Specs", "mega_capable": False},
            {"species": "Rillaboom", "item": "Assault Vest", "mega_capable": False},
        ]
        assert _validate(slots, self._cfg_base) == []

    def test_single_pokemon_team_is_valid(self) -> None:
        """Un equipo de 1 Pokémon es válido (sin duplicados posibles)."""
        slots: list[dict[str, Any]] = [
            {"species": "Incineroar", "item": "Sitrus Berry", "mega_capable": False},
        ]
        assert _validate(slots, self._cfg_base) == []


# ---------------------------------------------------------------------------
# Grupo 3 — TestPokemonMasterLoading
# ---------------------------------------------------------------------------


class TestPokemonMasterLoading:
    """
    Tests de carga de data/pokemon_master.json.
    Usa pytest.skip() si el archivo no existe o está vacío.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _require_master(self) -> None:
        """Salta toda la clase si pokemon_master.json no existe o está vacío."""
        if not _POKEMON_MASTER_PATH.exists():
            pytest.skip("pokemon_master.json no existe")
        master = _load_master()
        if not master:
            pytest.skip("pokemon_master.json existe pero está vacío o sin datos")

    def test_pokemon_master_loads_1025_entries(self) -> None:
        """pokemon_master.json contiene exactamente 1025 entradas (Gen 1-9)."""
        master = _load_master()
        assert len(master) == 1025, (
            f"Se esperaban 1025 entradas, se encontraron {len(master)}"
        )

    def test_pokemon_master_has_bulbasaur_at_1(self) -> None:
        """El dex_id 1 corresponde a 'Bulbasaur'."""
        master = _load_master()
        assert 1 in master, "dex_id 1 no encontrado en pokemon_master"
        assert master[1] == "Bulbasaur", (
            f"dex_id 1 es '{master[1]}', se esperaba 'Bulbasaur'"
        )

    def test_pokemon_master_has_incineroar(self) -> None:
        """Incineroar (dex_id 727) está en el master."""
        master = _load_master()
        assert 727 in master, "dex_id 727 (Incineroar) no encontrado"
        assert master[727] == "Incineroar", (
            f"dex_id 727 es '{master[727]}', se esperaba 'Incineroar'"
        )

    def test_fallback_name_for_unknown_id(self) -> None:
        """Para un dex_id fuera del master, el fallback es 'Pokemon #dex_id'."""
        master = _load_master()
        unknown_id = 99999
        assert unknown_id not in master
        fallback = master.get(unknown_id, f"Pokemon #{unknown_id}")
        assert fallback == "Pokemon #99999"
