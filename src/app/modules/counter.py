"""
Heurístico v1 de counters para VGC Champions.

Calcula un score de "efectividad como counter" para cada Pokémon del roster
contra un equipo rival, combinando tres componentes:

  TYPE_ADVANTAGE (peso 0.5):
    Cuántos Pokémon rivales son débiles a los tipos del counter.
    +1.0 por rival débil, -0.5 por rival resistente, -1.0 por inmune.

  SPEED_TIER (peso 0.3):
    Proporción del equipo rival al que el counter supera en velocidad base.
    score = (faster - slower) / total_rival

  ITEM_SYNERGY (peso 0.2):
    Bonus estático por ítems que señalan roles counter reconocibles.

El score total se normaliza al rango [0, 1] dividiendo por el máximo absoluto.

Este módulo es el heurístico v1 — sin ML, basado solo en datos estáticos.
Será reemplazado por WP predict (Counter v2, Tarea 87) cuando haya
suficientes datos de replays. TYPE_CHART contiene la tabla de tipos
completa para los 18 tipos de Pokémon.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_POKEMON_MASTER_PATH = _PROJECT_ROOT / "data" / "pokemon_master.json"

# ---------------------------------------------------------------------------
# Tablas estáticas
# ---------------------------------------------------------------------------

TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0,
                 "Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water":    {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,
                 "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
                 "Flying": 2.0, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,
                 "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,
                 "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5,
                 "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,
                 "Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,
                 "Dark": 2.0, "Steel": 2.0, "Fairy": 0.5},
    "Poison":   {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
                 "Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
    "Ground":   {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,
                 "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying":   {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0,
                 "Bug": 2.0, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5,
                 "Dark": 0.0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Flying": 0.5,
                 "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5,
                 "Fairy": 0.5},
    "Rock":     {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,
                 "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost":    {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon":   {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0,
                 "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,
                 "Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
    "Fairy":    {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5,
                 "Dragon": 2.0, "Dark": 2.0, "Steel": 0.5},
}

ITEM_SYNERGY_SCORES: dict[str, float] = {
    "choice scarf":  0.3,
    "choice band":   0.2,
    "choice specs":  0.2,
    "focus sash":    0.1,
    "life orb":      0.2,
    "assault vest":  0.1,
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CounterResult:
    """
    Resultado de un Pokémon candidato como counter.

    Attributes:
        species: Nombre de la especie counter.
        score: Score normalizado total (0.0-1.0).
        type_advantage_score: Componente de ventaja de tipo (sin normalizar).
        speed_tier_score: Componente de speed tier.
        item_synergy_score: Componente de ítem.
        types: Tipos del counter.
        counters_directly: Especies rivales que este counter amenaza.
    """

    species: str
    score: float
    type_advantage_score: float
    speed_tier_score: float
    item_synergy_score: float
    types: list[str] = field(default_factory=list)
    counters_directly: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _load_pokemon_data() -> dict[str, dict[str, Any]]:
    """
    Carga datos de Pokémon desde pokemon_master.json.

    Returns:
        Dict {nombre_lower: {types, base_stats, ...}}.
        Dict vacío si el archivo no existe o hay error.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            log.warning(
                "pokemon_master.json no encontrado en %s", _POKEMON_MASTER_PATH
            )
            return {}
        raw = json.loads(_POKEMON_MASTER_PATH.read_text(encoding="utf-8"))
        pokemon_data = raw.get("pokemon", {})
        result: dict[str, dict[str, Any]] = {}
        for entry in pokemon_data.values():
            if isinstance(entry, dict):
                name = str(entry.get("name", "")).lower()
                if name:
                    result[name] = entry
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando pokemon_master.json: %s", exc)
        return {}


def _type_advantage_score(
    attacker_types: list[str],
    rival_slots: list[Any],
    pokemon_data: dict[str, dict[str, Any]],
) -> tuple[float, list[str]]:
    """
    Calcula el score de ventaja de tipo del counter contra el equipo rival.

    Args:
        attacker_types: Tipos del counter.
        rival_slots: Slots del equipo rival (objetos con .species).
        pokemon_data: Datos de Pokémon master.

    Returns:
        Tupla (score_raw, lista_rivales_amenazados).
    """
    score = 0.0
    threatened: list[str] = []

    for slot in rival_slots:
        species = str(getattr(slot, "species", "")).lower()
        rival_data = pokemon_data.get(species, {})
        rival_types = [str(t) for t in rival_data.get("types", [])]

        if not rival_types:
            continue

        best_multiplier = 1.0
        for atk_type in attacker_types:
            multiplier = 1.0
            type_row = TYPE_CHART.get(atk_type, {})
            for def_type in rival_types:
                multiplier *= type_row.get(def_type, 1.0)
            best_multiplier = max(best_multiplier, multiplier)

        if best_multiplier >= 2.0:
            score += 1.0
            threatened.append(str(getattr(slot, "species", "")))
        elif best_multiplier == 0.0:
            score -= 1.0
        elif best_multiplier <= 0.5:
            score -= 0.5

    return score, threatened


def _speed_tier_score(
    counter_spe: int,
    rival_slots: list[Any],
    pokemon_data: dict[str, dict[str, Any]],
) -> float:
    """
    Calcula el score de speed tier comparando el counter contra el rival.

    Args:
        counter_spe: Base speed del counter.
        rival_slots: Slots del equipo rival.
        pokemon_data: Datos de Pokémon master.

    Returns:
        Score entre -1.0 y 1.0.
    """
    if counter_spe == 0 or not rival_slots:
        return 0.0

    faster = 0
    slower = 0

    for slot in rival_slots:
        species = str(getattr(slot, "species", "")).lower()
        rival_data = pokemon_data.get(species, {})
        rival_spe = int(rival_data.get("base_stats", {}).get("speed", 0))
        if rival_spe == 0:
            continue
        if counter_spe > rival_spe:
            faster += 1
        elif counter_spe < rival_spe:
            slower += 1

    total = faster + slower
    if total == 0:
        return 0.0
    return (faster - slower) / total


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def heuristic_counter(
    rival_team: Any,
    roster: list[str],
    pokemon_data: dict[str, dict[str, Any]] | None = None,
    top_n: int = 20,
) -> list[CounterResult]:
    """
    Calcula los mejores counters heurísticos para un equipo rival.

    Para cada Pokémon del roster calcula un score combinado de:
      0.5 · type_advantage + 0.3 · speed_tier + 0.2 · item_synergy

    Los scores se normalizan al rango [0, 1] dividiendo por el máximo
    absoluto. Si pokemon_master.json no existe, retorna lista vacía sin crash.

    Args:
        rival_team: ParsedTeam del equipo rival. Debe tener atributo .slots.
        roster: Lista de nombres de Pokémon candidatos a counter.
        pokemon_data: Dict de datos maestros. Si None, lo carga desde disco.
        top_n: Número máximo de counters a retornar.

    Returns:
        Lista de CounterResult ordenada por score descendente.
        Lista vacía si el roster está vacío, el rival no tiene slots,
        o pokemon_master.json no está disponible.
    """
    if not roster:
        log.debug("Roster vacío — sin counters")
        return []

    rival_slots = getattr(rival_team, "slots", [])
    if not rival_slots:
        log.debug("Equipo rival sin slots — sin counters")
        return []

    if pokemon_data is None:
        pokemon_data = _load_pokemon_data()

    if not pokemon_data:
        log.warning(
            "pokemon_master.json vacío o no cargado — no se pueden calcular counters"
        )
        return []

    results: list[CounterResult] = []

    for species_name in roster:
        species_lower = species_name.lower()
        data = pokemon_data.get(species_lower, {})

        types = [str(t) for t in data.get("types", [])]
        base_spe = int(data.get("base_stats", {}).get("speed", 0))

        type_score, threatened = _type_advantage_score(
            types, rival_slots, pokemon_data
        )
        speed_score = _speed_tier_score(base_spe, rival_slots, pokemon_data)
        item_score = 0.0  # v1: sin ítem del roster individual

        raw_score = 0.5 * type_score + 0.3 * speed_score + 0.2 * item_score

        results.append(
            CounterResult(
                species=species_name,
                score=raw_score,
                type_advantage_score=type_score,
                speed_tier_score=speed_score,
                item_synergy_score=item_score,
                types=types,
                counters_directly=threatened,
            )
        )

    if not results:
        return []

    # Normalizar scores al rango [0, 1]
    max_score = max(abs(r.score) for r in results)
    if max_score > 0:
        for r in results:
            r.score = round(r.score / max_score, 4)

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


__all__ = [
    "CounterResult",
    "heuristic_counter",
    "TYPE_CHART",
    "ITEM_SYNERGY_SCORES",
]
