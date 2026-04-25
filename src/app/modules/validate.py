"""
Lógica de validación de equipos VGC — módulo de dominio puro.

No importa Streamlit ni hace I/O. Diseñado para ser testeable con
pytest sin levantar la app.

Dos funciones principales con propósitos distintos:

  validate_team()
    Checks rápidos de cláusulas y mecánicas (species, item, mega,
    stat points). Se llama en cada rerun de la UI para feedback
    instantáneo.

  validate_team_legality()
    Verifica legalidad de especies, ítems y moves contra las listas
    del JSON de regulación. Más costosa — se llama solo cuando el
    usuario hace click en "Validar" o al guardar/exportar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.app.core.schema import RegulationConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass de error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """
    Un error de validación del equipo.

    Attributes:
        code: Código del error para identificarlo programáticamente.
              Valores: "species_clause", "item_clause", "mega_clause",
              "illegal_species", "illegal_item", "illegal_move",
              "stat_points", "team_size".
        message: Mensaje humano-legible para la UI.
        slot_idx: Índice del slot donde ocurrió el error (0-5).
                  None si aplica al equipo completo.
    """

    code: str
    message: str
    slot_idx: int | None = None


# ---------------------------------------------------------------------------
# Función principal: validación rápida de cláusulas
# ---------------------------------------------------------------------------


def validate_team(
    team: list[dict[str, Any]],
    reg: RegulationConfig,
) -> list[ValidationError]:
    """
    Valida un equipo completo contra las reglas de una regulación.

    El equipo se representa como lista de dicts con estos campos opcionales:
        species (str): Nombre de la especie.
        item (str): Ítem equipado.
        moves (list[str]): Lista de moves.
        mega_capable (bool): True si tiene Mega Stone.
        stat_points (dict[str, int]): SP por stat.

    Los slots con species="" o species ausente se consideran vacíos y se
    ignoran para la mayoría de los checks.

    Checks implementados en orden:
      1. species_clause — no repetir especie (si activa en reg).
      2. item_clause    — no repetir ítem (si activa en reg).
      3. mega_clause    — máximo mega_max_per_battle Mega Stones.
      4. stat_points    — total y por stat dentro de los caps.

    Args:
        team: Lista de dicts representando los slots (0-6 elementos).
        reg: Configuración de la regulación activa.

    Returns:
        Lista de ValidationError en el orden de los checks.
        Vacía si el equipo es válido o está completamente vacío.
    """
    errors: list[ValidationError] = []

    filled_slots = [
        (idx, slot)
        for idx, slot in enumerate(team)
        if slot.get("species", "").strip()
    ]

    if not filled_slots:
        return errors

    # ── Check 1: Species clause ──────────────────────────────────────────

    if reg.clauses.species_clause:
        seen_species: dict[str, int] = {}
        for idx, slot in filled_slots:
            sp = slot.get("species", "").strip()
            if sp in seen_species:
                errors.append(
                    ValidationError(
                        code="species_clause",
                        message=(
                            f"Species clause: '{sp}' aparece en los slots "
                            f"{seen_species[sp] + 1} y {idx + 1}."
                        ),
                        slot_idx=idx,
                    )
                )
            else:
                seen_species[sp] = idx

    # ── Check 2: Item clause ─────────────────────────────────────────────

    if reg.clauses.item_clause:
        seen_items: dict[str, int] = {}
        for idx, slot in filled_slots:
            item = slot.get("item", "").strip()
            if not item:
                continue
            if item in seen_items:
                errors.append(
                    ValidationError(
                        code="item_clause",
                        message=(
                            f"Item clause: '{item}' aparece en los slots "
                            f"{seen_items[item] + 1} y {idx + 1}."
                        ),
                        slot_idx=idx,
                    )
                )
            else:
                seen_items[item] = idx

    # ── Check 3: Mega clause ─────────────────────────────────────────────

    if reg.mechanics.mega_enabled:
        mega_slots = [
            idx
            for idx, slot in filled_slots
            if slot.get("mega_capable", False)
        ]
        if len(mega_slots) > reg.mechanics.mega_max_per_battle:
            extras = mega_slots[reg.mechanics.mega_max_per_battle :]
            errors.append(
                ValidationError(
                    code="mega_clause",
                    message=(
                        f"Solo {reg.mechanics.mega_max_per_battle} "
                        f"Pokémon puede(n) tener Mega Stone. "
                        f"Slots extra con Mega: {[i + 1 for i in extras]}."
                    ),
                    slot_idx=extras[0] if extras else None,
                )
            )

    # ── Check 4: Stat Points ─────────────────────────────────────────────

    if reg.mechanics.stat_points_system:
        total_cap = reg.mechanics.stat_points_total
        per_stat_cap = reg.mechanics.stat_points_cap_per_stat
        for idx, slot in filled_slots:
            sp_dict = slot.get("stat_points", {})
            if not sp_dict or not isinstance(sp_dict, dict):
                continue

            total = sum(
                int(v) for v in sp_dict.values() if isinstance(v, (int, float))
            )
            if total > total_cap:
                errors.append(
                    ValidationError(
                        code="stat_points",
                        message=(
                            f"Slot {idx + 1}: Stat Points totales ({total}) "
                            f"superan el límite ({total_cap})."
                        ),
                        slot_idx=idx,
                    )
                )
                continue

            for stat, value in sp_dict.items():
                if not isinstance(value, (int, float)):
                    continue
                if int(value) > per_stat_cap:
                    errors.append(
                        ValidationError(
                            code="stat_points",
                            message=(
                                f"Slot {idx + 1}: {stat.upper()} SP "
                                f"({int(value)}) supera el límite por stat "
                                f"({per_stat_cap})."
                            ),
                            slot_idx=idx,
                        )
                    )

    return errors


# ---------------------------------------------------------------------------
# Función separada: validación de legalidad (costosa)
# ---------------------------------------------------------------------------


def validate_team_legality(
    team: list[dict[str, Any]],
    reg: RegulationConfig,
    pokemon_name_to_dex: dict[str, int] | None = None,
) -> list[ValidationError]:
    """
    Verifica legalidad de especies, ítems y moves contra las listas del
    JSON de regulación.

    Separada de validate_team() porque implica lookups en listas del JSON
    que pueden ser costosos. La UI llama validate_team() en cada rerun y
    esta función solo cuando el usuario hace click en "Validar" o al
    guardar/exportar.

    Args:
        team: Lista de dicts del equipo.
        reg: Configuración de la regulación.
        pokemon_name_to_dex: Mapeo nombre_lower → dex_id para verificar
            legalidad de especies. Si None, el check de especie se salta.

    Returns:
        Lista de ValidationError de legalidad.
    """
    errors: list[ValidationError] = []

    legal_items = {item.lower() for item in reg.items_legales}
    legal_moves = set(reg.moves_legales)
    legal_dex_ids = set(reg.pokemon_legales)

    filled_slots = [
        (idx, slot)
        for idx, slot in enumerate(team)
        if slot.get("species", "").strip()
    ]

    for idx, slot in filled_slots:
        # Check especie
        if pokemon_name_to_dex is not None:
            species = slot.get("species", "").strip()
            dex_id = pokemon_name_to_dex.get(species.lower())
            if dex_id is None or dex_id not in legal_dex_ids:
                errors.append(
                    ValidationError(
                        code="illegal_species",
                        message=(
                            f"Slot {idx + 1}: '{species}' no es legal "
                            f"en {reg.regulation_id}."
                        ),
                        slot_idx=idx,
                    )
                )

        # Check ítem (case-insensitive)
        item = slot.get("item", "").strip()
        if item and item.lower() not in legal_items:
            errors.append(
                ValidationError(
                    code="illegal_item",
                    message=(
                        f"Slot {idx + 1}: '{item}' no es un ítem legal "
                        f"en {reg.regulation_id}."
                    ),
                    slot_idx=idx,
                )
            )

        # Check moves — solo IDs enteros (los strings de nombres se omiten
        # para evitar falsos positivos contra moves_legales que son IDs)
        moves = slot.get("moves", [])
        for move in moves:
            if not move:
                continue
            if isinstance(move, int) and move not in legal_moves:
                errors.append(
                    ValidationError(
                        code="illegal_move",
                        message=(
                            f"Slot {idx + 1}: Move ID {move} no es legal "
                            f"en {reg.regulation_id}."
                        ),
                        slot_idx=idx,
                    )
                )

    return errors


# ---------------------------------------------------------------------------
# Helper de formato para UI
# ---------------------------------------------------------------------------


def format_errors_for_ui(errors: list[ValidationError]) -> list[str]:
    """
    Convierte lista de ValidationError a strings formateados para Streamlit.

    Args:
        errors: Lista de ValidationError.

    Returns:
        Lista de strings con prefijo ❌ para st.error() o st.warning().
    """
    return [f"❌ {err.message}" for err in errors]


__all__ = [
    "ValidationError",
    "validate_team",
    "validate_team_legality",
    "format_errors_for_ui",
]
