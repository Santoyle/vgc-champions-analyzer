"""
Funciones de fitness multi-objetivo para el GA NSGA-II de VGC Champions Analyzer.

El algoritmo NSGA-II maximiza los 4 objetivos simultáneamente. Todos los valores
están normalizados a [0, 1] donde 1.0 es el mejor posible:

  f1 — Cobertura defensiva: cuántos tipos del meta puede el equipo resistir o
       absorber con Tera. Premia equipos difíciles de cubrir ofensivamente.

  f2 — Sinergia ofensiva: cuántos tipos defensivos puede el equipo cubrir con
       ataques STAB (×2 o superior). Premia diversidad ofensiva.

  f3 — Anti-meta: qué tan bien el equipo contrarresta los top-N Pokémon más
       usados del meta actual. Premia ventajas de tipo específicas.

  f4 — Control de velocidad: distribución del speed tier del equipo (p75 + std).
       Premia equipos con mix de Pokémon rápidos y lentos (versatilidad táctica).

META_FALLBACK_TOP20 es el top-20 del meta Champions (Pikalytics abril 2026) y se
usa cuando no hay datos de uso disponibles en data/raw/. Cuando hay datos, la
función fitness_anti_meta recibe meta_top como parámetro explícito.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.app.modules.counter import TYPE_CHART
from src.app.modules.ga import (
    Chromosome,
    _load_pokemon_master,
)

log = logging.getLogger(__name__)

ALL_TYPES: list[str] = [
    "Normal", "Fire", "Water", "Electric", "Grass",
    "Ice", "Fighting", "Poison", "Ground", "Flying",
    "Psychic", "Bug", "Rock", "Ghost", "Dragon",
    "Dark", "Steel", "Fairy",
]

# Top-20 Pokémon del meta Champions como fallback cuando no hay datos de uso.
# Basado en datos reales de Pikalytics abril 2026.
META_FALLBACK_TOP20: list[str] = [
    "Incineroar", "Sneasler", "Garchomp",
    "Sinistcha", "Kingambit", "Basculegion",
    "Flutter Mane", "Urshifu", "Rillaboom",
    "Amoonguss", "Ogerpon", "Calyrex-Ice",
    "Miraidon", "Koraidon", "Tornadus",
    "Whimsicott", "Grimmsnarl", "Farigiraf",
    "Iron Hands", "Chi-Yu",
]


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _get_pokemon_types(
    species_id: int,
    pokemon_master: dict[int, dict[str, Any]],
) -> list[str]:
    """
    Obtiene los tipos de un Pokémon por dex_id.

    Args:
        species_id: Dex ID del Pokémon.
        pokemon_master: Datos maestros cargados.

    Returns:
        Lista de strings de tipos (ej: ["Fire", "Dark"]).
        Lista vacía si no se encuentra.
    """
    data = pokemon_master.get(species_id, {})
    return [str(t) for t in data.get("types", [])]


def _get_base_speed(
    species_id: int,
    pokemon_master: dict[int, dict[str, Any]],
) -> int:
    """
    Obtiene la velocidad base de un Pokémon.

    Args:
        species_id: Dex ID del Pokémon.
        pokemon_master: Datos maestros cargados.

    Returns:
        Velocidad base como int. 0 si no se encuentra.
    """
    data = pokemon_master.get(species_id, {})
    return int(data.get("base_stats", {}).get("speed", 0))


def _type_multiplier(
    attack_type: str,
    defender_types: list[str],
) -> float:
    """
    Calcula el multiplicador de daño de un tipo de ataque contra un defensor.

    Para defensores con dos tipos, multiplica ambos modificadores (e.g. Garchomp
    Dragon/Ground recibe 0.5 × 1.0 = 0.5x de Fire).

    Args:
        attack_type: Tipo del ataque.
        defender_types: Lista de tipos del defensor (1 o 2 elementos).

    Returns:
        Multiplicador total (producto de cada tipo defensivo).
    """
    multiplier = 1.0
    type_row = TYPE_CHART.get(attack_type, {})
    for def_type in defender_types:
        multiplier *= type_row.get(def_type, 1.0)
    return multiplier


# ---------------------------------------------------------------------------
# Objetivo 1: Cobertura defensiva
# ---------------------------------------------------------------------------


def fitness_defensive_coverage(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
) -> float:
    """
    Calcula el fitness de cobertura defensiva (f1).

    Un equipo con buena cobertura defensiva puede resistir ataques de muchos
    tipos distintos. Se añade un bonus por tipos Tera disponibles.

    Normalización: (tipos_cubiertos + tera_bonus) / (18 + 9)
      donde 18 = tipos totales y 9 = bonus Tera máximo asumido.

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros de Pokémon.

    Returns:
        Score en [0, 1]. Mayor = mejor cobertura defensiva.
    """
    if not chrom.slots:
        return 0.0

    team_type_sets = [
        _get_pokemon_types(s.species_id, pokemon_master)
        for s in chrom.slots
    ]
    tera_types = {s.tera_type for s in chrom.slots if s.tera_type}

    covered_types = 0
    tera_bonus = 0.0

    for attack_type in ALL_TYPES:
        team_resists = False
        for defender_types in team_type_sets:
            if not defender_types:
                continue
            mult = _type_multiplier(attack_type, defender_types)
            if mult <= 0.5:
                team_resists = True
                break

        if team_resists:
            covered_types += 1

        for tera_type in tera_types:
            tera_mult = _type_multiplier(attack_type, [tera_type])
            if tera_mult <= 0.5:
                tera_bonus += 0.5
                break

    max_score = 18.0 + 9.0
    score = (covered_types + tera_bonus) / max_score
    return float(min(1.0, score))


# ---------------------------------------------------------------------------
# Objetivo 2: Sinergia ofensiva
# ---------------------------------------------------------------------------


def fitness_offensive_synergy(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
) -> float:
    """
    Calcula el fitness de sinergia ofensiva (f2).

    Mide cuántos tipos defensivos puede cubrir el equipo con ataques STAB
    (tipos de los Pokémon del equipo). Premia la diversidad de tipos de ataque.

    Normalización: tipos_defensivos_cubiertos / 18

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros de Pokémon.

    Returns:
        Score en [0, 1]. Mayor = mejor sinergia ofensiva.
    """
    if not chrom.slots:
        return 0.0

    team_attack_types: set[str] = set()
    for slot in chrom.slots:
        types = _get_pokemon_types(slot.species_id, pokemon_master)
        team_attack_types.update(types)

    if not team_attack_types:
        return 0.0

    covered_defenders = 0
    for def_type in ALL_TYPES:
        for atk_type in team_attack_types:
            mult = _type_multiplier(atk_type, [def_type])
            if mult >= 2.0:
                covered_defenders += 1
                break

    return float(covered_defenders / len(ALL_TYPES))


# ---------------------------------------------------------------------------
# Objetivo 3: Anti-meta
# ---------------------------------------------------------------------------


def fitness_anti_meta(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
    meta_top: list[str] | None = None,
) -> float:
    """
    Calcula el fitness anti-meta (f3).

    Mide qué tan bien el equipo contrarresta a los Pokémon más usados del meta
    actual usando ventajas de tipo. Scoring por slot vs meta:
      ×4 ventaja → 1.0 pts | ×2 ventaja → 0.7 pts | ×1 neutro → 0.3 pts | <1 → 0 pts

    Normalización: suma_ventajas / (len(meta_top) × TEAM_SIZE)

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros de Pokémon.
        meta_top: Lista de nombres del meta. Si None, usa META_FALLBACK_TOP20.

    Returns:
        Score en [0, 1]. Mayor = mejor anti-meta.
    """
    if not chrom.slots:
        return 0.0

    if meta_top is None:
        meta_top = META_FALLBACK_TOP20

    name_to_types: dict[str, list[str]] = {}
    for data in pokemon_master.values():
        name = str(data.get("name", "")).lower()
        types = [str(t) for t in data.get("types", [])]
        if name and types:
            name_to_types[name] = types

    team_attack_types: list[list[str]] = [
        _get_pokemon_types(s.species_id, pokemon_master)
        for s in chrom.slots
    ]

    total_advantage = 0.0
    max_advantage = float(len(meta_top) * len(chrom.slots))

    for meta_pkm in meta_top:
        meta_types = name_to_types.get(meta_pkm.lower(), [])
        if not meta_types:
            continue

        for slot_types in team_attack_types:
            best_mult = 1.0
            for atk_type in slot_types:
                mult = _type_multiplier(atk_type, meta_types)
                best_mult = max(best_mult, mult)

            if best_mult >= 4.0:
                total_advantage += 1.0
            elif best_mult >= 2.0:
                total_advantage += 0.7
            elif best_mult == 1.0:
                total_advantage += 0.3

    if max_advantage == 0.0:
        return 0.0

    return float(min(1.0, total_advantage / max_advantage))


# ---------------------------------------------------------------------------
# Objetivo 4: Control de velocidad
# ---------------------------------------------------------------------------


def fitness_speed_control(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
) -> float:
    """
    Calcula el fitness de control de velocidad (f4).

    Premia equipos con alta varianza de velocidades (mix rápidos/lentos = más
    opciones tácticas) y también alta velocidad p75 (buen speed tier general).

    Normalización: (p75 / 180 + std / 50) / 2
      donde 180 = max base speed aproximado y 50 = std típica de referencia.

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros de Pokémon.

    Returns:
        Score en [0, 1]. Mayor = mejor control de velocidad.
    """
    if not chrom.slots:
        return 0.0

    speeds = [
        _get_base_speed(s.species_id, pokemon_master)
        for s in chrom.slots
    ]
    speeds = [s for s in speeds if s > 0]

    if not speeds:
        return 0.0

    speeds_arr = np.array(speeds, dtype=float)
    p75 = float(np.percentile(speeds_arr, 75))
    p75_norm = min(p75 / 180.0, 1.0)

    std = float(np.std(speeds_arr))
    variance_norm = min(std / 50.0, 1.0)

    return float((p75_norm + variance_norm) / 2.0)


# ---------------------------------------------------------------------------
# Función principal: evaluar los 4 objetivos
# ---------------------------------------------------------------------------


def evaluate_fitness(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
    meta_top: list[str] | None = None,
) -> tuple[float, float, float, float]:
    """
    Evalúa los 4 objetivos de fitness para un Chromosome.

    DEAP los maximiza todos simultáneamente con NSGA-II.
    Siempre retorna una tupla válida de 4 floats — nunca levanta excepción.

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros. Carga desde disco si None.
        meta_top: Lista del meta para f3. None = usa META_FALLBACK_TOP20.

    Returns:
        Tupla (f1, f2, f3, f4) con los 4 fitness en [0, 1].
        (0.0, 0.0, 0.0, 0.0) si hay error inesperado.
    """
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    try:
        f1 = fitness_defensive_coverage(chrom, pokemon_master)
        f2 = fitness_offensive_synergy(chrom, pokemon_master)
        f3 = fitness_anti_meta(chrom, pokemon_master, meta_top)
        f4 = fitness_speed_control(chrom, pokemon_master)
        log.debug("Fitness: f1=%.3f f2=%.3f f3=%.3f f4=%.3f", f1, f2, f3, f4)
        return f1, f2, f3, f4
    except Exception as exc:  # noqa: BLE001
        log.warning("Error evaluando fitness: %s", exc)
        return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "evaluate_fitness",
    "fitness_defensive_coverage",
    "fitness_offensive_synergy",
    "fitness_anti_meta",
    "fitness_speed_control",
    "META_FALLBACK_TOP20",
    "ALL_TYPES",
]
