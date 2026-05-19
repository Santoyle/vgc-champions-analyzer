"""
Warm-start para el GA NSGA-II de VGC.

El warm-start acelera la convergencia del algoritmo genético inyectando
conocimiento de regulaciones previas en el 30-40% inicial de la población.
En lugar de partir de individuos completamente aleatorios, se usan equipos
reales de alta performance del ladder de Showdown como semilla.

extract_agnostic_features mide la similitud entre equipos de distintas
regulaciones a través de características transferibles: roles estratégicos
(Fake Out, Trick Room, Intimidate, Redirección, Clima), velocidad media y
diversidad de tipos. Estos features son independientes del conjunto legal
específico de cada regulación.

map_illegal_to_similar preserva el "arquetipo" del equipo histórico: si un
equipo de Trick Room llevaba Hatterene (ilegal en la nueva regulación), se
busca otro setter de Trick Room del mismo tipo primario. Así el equipo
mantiene su identidad táctica aunque cambie la regulación.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.modules.ga import (
    NATURES,
    TEAM_SIZE,
    Chromosome,
    SlotGene,
    _load_pokemon_master,
)
from src.app.modules.ga_repair import repair

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Constantes de roles estratégicos
# ---------------------------------------------------------------------------

WEATHER_SETTERS: set[str] = {
    "politoed",
    "pelipper",
    "ninetales",
    "alolan-ninetales",
    "torkoal",
    "hippowdon",
    "tyranitar",
    "abomasnow",
}

REDIRECTION_POKEMON: set[str] = {
    "amoonguss",
    "togekiss",
    "clefairy",
    "indeedee",
    "jigglypuff",
}

TRICK_ROOM_SETTERS: set[str] = {
    "porygon2",
    "dusclops",
    "hatterene",
    "indeedee",
    "aromatisse",
    "musharna",
    "bronzong",
    "slowking",
}

_FAKE_OUT_USERS: set[str] = {
    "incineroar",
    "hariyama",
    "lopunny",
    "ambipom",
    "persian",
}

_TERA_TYPES: list[str] = [
    "Normal", "Fire", "Water", "Electric", "Grass",
    "Ice", "Fighting", "Poison", "Ground", "Flying",
    "Psychic", "Bug", "Rock", "Ghost", "Dragon",
    "Dark", "Steel", "Fairy",
]


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass
class WarmStartConfig:
    """
    Configuración del warm-start para el GA.

    Attributes:
        source_regulation_ids: Regulaciones de las que cargar equipos.
                                Lista vacía = todas las disponibles en data/raw.
        min_rating: Rating mínimo de los replays a incluir. Default 1600.
        max_teams: Máximo de equipos a cargar. Default 200.
        warm_fraction: Fracción de la población a llenar con warm-start [0, 1].
                        Default 0.35 (35%).
        map_illegal_to_similar: Si True, mapea Pokémon ilegales al legal más
                                  similar por tipo primario.
    """

    source_regulation_ids: list[str] = field(default_factory=list)
    min_rating: int = 1600
    max_teams: int = 200
    warm_fraction: float = 0.35
    map_illegal_to_similar: bool = True


# ---------------------------------------------------------------------------
# Carga de equipos históricos
# ---------------------------------------------------------------------------


def _load_teams_from_parquets(
    regulation_ids: list[str],
    min_rating: int = 1600,
    max_teams: int = 200,
) -> list[dict[str, Any]]:
    """
    Carga equipos desde los Parquets de Showdown de las regulaciones dadas.

    Aplica deduplicación por composición de Pokémon (nombres lowercase
    ordenados), mismo patrón que _load_recent_teams en counter.py.

    Args:
        regulation_ids: IDs de regulaciones a buscar.
                        Lista vacía = todas las carpetas reg=* disponibles.
        min_rating: Rating mínimo de los replays.
        max_teams: Máximo de equipos a retornar.

    Returns:
        Lista de dicts con keys: team (list[str]), regulation_id (str),
        rating (int). Ordenada por rating descendente. Lista vacía si no
        hay datos.
    """
    raw_dir = _DATA_DIR / "raw"
    if not raw_dir.exists():
        return []

    if not regulation_ids:
        reg_dirs = [
            d for d in raw_dir.iterdir()
            if d.is_dir() and d.name.startswith("reg=")
        ]
    else:
        reg_dirs = [raw_dir / f"reg={rid}" for rid in regulation_ids]

    teams: list[dict[str, Any]] = []
    seen: set[str] = set()

    for reg_dir in reg_dirs:
        if not reg_dir.exists():
            continue

        reg_id = reg_dir.name.replace("reg=", "")
        showdown_dir = reg_dir / "source=showdown"
        if not showdown_dir.exists():
            continue

        for pq_file in sorted(showdown_dir.glob("*.parquet"), reverse=True):
            if len(teams) >= max_teams:
                break
            try:
                df = pd.read_parquet(pq_file)
                for _, row in df.iterrows():
                    if len(teams) >= max_teams:
                        break
                    rating = int(row.get("rating", 0) or 0)
                    if rating < min_rating:
                        continue

                    for team_col in ("team_p1_json", "team_p2_json"):
                        try:
                            raw_val = row.get(team_col, "[]")
                            team: list[str] = json.loads(str(raw_val))
                            if not team or len(team) < 3:
                                continue
                            comp_key = "|".join(
                                sorted(str(p).lower() for p in team)
                            )
                            if comp_key in seen:
                                continue
                            seen.add(comp_key)
                            teams.append({
                                "team": [str(p) for p in team],
                                "regulation_id": reg_id,
                                "rating": rating,
                            })
                        except Exception:  # noqa: BLE001
                            continue
            except Exception as exc:  # noqa: BLE001
                log.debug("Error leyendo %s: %s", pq_file, exc)
                continue

    teams.sort(key=lambda x: x.get("rating", 0), reverse=True)
    log.info(
        "Warm-start: cargados %d equipos únicos de %d regulaciones",
        len(teams),
        len(reg_dirs),
    )
    return teams


# ---------------------------------------------------------------------------
# Mapeo de especies
# ---------------------------------------------------------------------------


def _map_to_legal_species(
    species_name: str,
    legal_species_ids: list[int],
    pokemon_master: dict[int, dict[str, Any]],
    rng: random.Random,
) -> int:
    """
    Mapea un nombre de Pokémon a un species_id legal en la nueva regulación.

    Estrategia en orden:
      1. Si el Pokémon ya es legal, retorna su dex_id.
      2. Busca un Pokémon legal con el mismo tipo primario.
      3. Fallback: retorna un legal aleatorio.
      4. Si legal_species_ids está vacío, retorna 0.

    Args:
        species_name: Nombre del Pokémon histórico (puede ser ilegal).
        legal_species_ids: IDs legales en la regulación objetivo.
        pokemon_master: Datos maestros.
        rng: RNG.

    Returns:
        dex_id de un Pokémon legal (o 0 si la lista legal está vacía).
    """
    if not legal_species_ids:
        return 0

    name_to_id: dict[str, int] = {
        str(data.get("name", "")).lower(): dex_id
        for dex_id, data in pokemon_master.items()
        if isinstance(data, dict)
    }

    legal_set = set(legal_species_ids)
    species_lower = species_name.lower()

    # Paso 1: ya es legal
    dex_id = name_to_id.get(species_lower)
    if dex_id is not None and dex_id in legal_set:
        return dex_id

    # Paso 2: mismo tipo primario
    source_types: list[str] = []
    if dex_id is not None:
        source_data = pokemon_master.get(dex_id, {})
        source_types = [str(t) for t in source_data.get("types", [])]

    if source_types:
        primary_type = source_types[0]
        candidates = [
            lid
            for lid in legal_species_ids
            if primary_type in [
                str(t) for t in pokemon_master.get(lid, {}).get("types", [])
            ]
        ]
        if candidates:
            return rng.choice(candidates)

    # Paso 3: fallback aleatorio
    return rng.choice(legal_species_ids)


# ---------------------------------------------------------------------------
# Conversión equipo → Chromosome
# ---------------------------------------------------------------------------


def _team_to_chromosome(
    team_dict: dict[str, Any],
    target_reg: Any,
    pokemon_master: dict[int, dict[str, Any]],
    rng: random.Random,
    map_illegal: bool = True,
) -> Chromosome | None:
    """
    Convierte un dict de equipo histórico a Chromosome para la regulación objetivo.

    Mapea los Pokémon ilegales si map_illegal=True, asigna ítems/moves/nature
    aleatorios válidos, y llama siempre a repair() para garantizar legalidad.

    Args:
        team_dict: Dict con key "team" (list[str]) y opcionalmente "rating".
        target_reg: RegulationConfig de la regulación objetivo.
        pokemon_master: Datos maestros.
        rng: RNG.
        map_illegal: Si True, mapea Pokémon ilegales al similar legal.

    Returns:
        Chromosome reparado y legal, o None si la conversión falla.
    """
    try:
        team_names: list[str] = team_dict.get("team", [])
        legal_ids: list[int] = list(target_reg.pokemon_legales)

        name_to_id: dict[str, int] = {
            str(data.get("name", "")).lower(): dex_id
            for dex_id, data in pokemon_master.items()
            if isinstance(data, dict)
        }
        legal_set = set(legal_ids)

        legal_items: list[Any] = list(target_reg.items_legales)
        legal_moves: list[Any] = list(target_reg.moves_legales)
        tera_enabled: bool = getattr(
            target_reg.mechanics, "tera_enabled", False
        )
        reg_id_str = str(getattr(target_reg, "regulation_id", ""))

        slots: list[SlotGene] = []
        for pokemon_name in team_names[:TEAM_SIZE]:
            species_lower = pokemon_name.lower()
            dex_id = name_to_id.get(species_lower, 0)

            if dex_id not in legal_set and map_illegal:
                dex_id = _map_to_legal_species(
                    pokemon_name, legal_ids, pokemon_master, rng
                )

            item = rng.choice(legal_items) if legal_items else ""
            moves = [
                rng.choice(legal_moves) if legal_moves else 0
                for _ in range(4)
            ]
            nature = rng.choice(NATURES)
            tera_type = rng.choice(_TERA_TYPES) if tera_enabled else ""

            slots.append(
                SlotGene(
                    species_id=dex_id,
                    item=item,
                    nature=nature,
                    moves=moves,
                    tera_type=tera_type,
                )
            )

        if not slots:
            return None

        chrom = Chromosome(slots=slots, regulation_id=reg_id_str)

        # repair() siempre: puede haber species_clause o item_clause violados
        repaired, repair_log = repair(chrom, target_reg, rng, pokemon_master)
        if not repair_log.is_clean:
            log.debug(
                "Warm-start chromosome reparado: %d correcciones",
                repair_log.total_fixes,
            )
        return repaired

    except Exception as exc:  # noqa: BLE001
        log.debug("Error convirtiendo equipo a chromosome: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Función principal de warm-start
# ---------------------------------------------------------------------------


def build_warm_start_population(
    target_reg: Any,
    config: WarmStartConfig | None = None,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
    rng: random.Random | None = None,
) -> list[Chromosome]:
    """
    Construye la lista de Chromosome para warm-start del GA NSGA-II.

    Carga equipos históricos de alta performance desde los Parquets de
    Showdown, los convierte a Chromosomes válidos para target_reg y los
    repara para garantizar legalidad.

    Args:
        target_reg: RegulationConfig de la nueva regulación.
        config: Configuración del warm-start. None = defaults.
        pokemon_master: Datos maestros. Carga desde disco si None.
        rng: RNG. None = random.Random(42).

    Returns:
        Lista de Chromosome legales para inyectar en la población inicial.
        Lista vacía (sin excepción) si no hay datos históricos disponibles.
    """
    if config is None:
        config = WarmStartConfig()
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()
    if rng is None:
        rng = random.Random(42)

    target_reg_id = str(getattr(target_reg, "regulation_id", ""))
    source_regs = [
        r for r in config.source_regulation_ids if r != target_reg_id
    ]

    teams = _load_teams_from_parquets(
        regulation_ids=source_regs,
        min_rating=config.min_rating,
        max_teams=config.max_teams,
    )

    if not teams:
        log.info(
            "Warm-start: sin equipos históricos disponibles para %s",
            target_reg_id,
        )
        return []

    warm_chromosomes: list[Chromosome] = []
    for team_dict in teams:
        chrom = _team_to_chromosome(
            team_dict,
            target_reg,
            pokemon_master,
            rng,
            map_illegal=config.map_illegal_to_similar,
        )
        if chrom is not None:
            warm_chromosomes.append(chrom)

    log.info(
        "Warm-start: %d/%d equipos convertidos a Chromosome legales para %s",
        len(warm_chromosomes),
        len(teams),
        target_reg_id,
    )
    return warm_chromosomes


# ---------------------------------------------------------------------------
# Features agnósticas de regulación
# ---------------------------------------------------------------------------


def extract_agnostic_features(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
) -> dict[str, float]:
    """
    Extrae features agnósticas de regulación de un Chromosome.

    Estas features son comparables entre distintas regulaciones y miden
    la identidad táctica del equipo: roles estratégicos presentes, perfil
    de velocidad y diversidad de tipos.

    Todas las features numéricas están normalizadas a [0, 1].

    Args:
        chrom: Chromosome a analizar.
        pokemon_master: Datos maestros.

    Returns:
        Dict con 8 features:
          has_fake_out (0/1), has_trick_room (0/1), has_intimidate (0/1),
          has_redirection (0/1), has_weather (0/1),
          mean_speed [0, 1], type_diversity [0, 1], speed_variance [0, 1].
    """
    features: dict[str, float] = {
        "has_fake_out": 0.0,
        "has_trick_room": 0.0,
        "has_intimidate": 0.0,
        "has_redirection": 0.0,
        "has_weather": 0.0,
        "mean_speed": 0.0,
        "type_diversity": 0.0,
        "speed_variance": 0.0,
    }

    if not chrom.slots:
        return features

    speeds: list[int] = []
    all_types: set[str] = set()

    for slot in chrom.slots:
        data = pokemon_master.get(slot.species_id, {})
        name = str(data.get("name", "")).lower()
        types = [str(t) for t in data.get("types", [])]
        base_speed = int(data.get("base_stats", {}).get("speed", 0))
        abilities = [str(a).lower() for a in data.get("abilities", [])]

        all_types.update(types)
        if base_speed > 0:
            speeds.append(base_speed)

        if name in REDIRECTION_POKEMON:
            features["has_redirection"] = 1.0
        if name in WEATHER_SETTERS:
            features["has_weather"] = 1.0
        if name in TRICK_ROOM_SETTERS:
            features["has_trick_room"] = 1.0
        if "intimidate" in abilities:
            features["has_intimidate"] = 1.0
        if name in _FAKE_OUT_USERS:
            features["has_fake_out"] = 1.0

    if speeds:
        import numpy as np

        # Normalizar: velocidad máxima esperada ~180 (Deoxys-S, Regieleki)
        features["mean_speed"] = float(sum(speeds) / len(speeds) / 180.0)
        # Normalizar: std máximo esperado ~50
        features["speed_variance"] = float(
            min(float(np.std(speeds)) / 50.0, 1.0)
        )

    # 18 tipos posibles
    features["type_diversity"] = float(len(all_types) / 18.0)

    return features


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "WarmStartConfig",
    "build_warm_start_population",
    "extract_agnostic_features",
]
