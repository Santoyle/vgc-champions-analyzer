"""
Cromosoma canónico del algoritmo genético NSGA-II para VGC Champions Analyzer.

Define la estructura de datos que representa un equipo VGC completo (Chromosome)
formado por 6 SlotGene, cada uno con su configuración competitiva completa.

Responsabilidades del módulo:
  - Cromosoma canónico (Chromosome + SlotGene): estructura de datos del GA.
  - encode/decode: serialización plana para DEAP. Estos son inversos aproximados —
    NO preservan tera_type ni stat_points (demasiada complejidad para el vector DEAP
    v1; ambos campos se manejan en la repair function fuera del ciclo evolutivo).
  - random_chromosome: función de inicialización de población. Usa random.Random
    con seed explícito para reproducibilidad determinista en tests.
  - chromosome_to_dict: conversión a dict con nombres legibles para la UI.

Los 4 objetivos de fitness (cobertura defensiva, sinergia ofensiva, anti-meta,
distribución de velocidades) se implementan en la Tarea 92 en un módulo separado.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_POKEMON_MASTER_PATH = _PROJECT_ROOT / "data" / "pokemon_master.json"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TEAM_SIZE: int = 6
MOVES_PER_SLOT: int = 4

NATURES: list[str] = [
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky",
]

# Genes por slot: species_id + item + nature_idx + 4 moves + mega_flag = 8
_GENES_PER_SLOT: int = 1 + 1 + 1 + MOVES_PER_SLOT + 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SlotGene:
    """
    Gen de un slot del equipo (un Pokémon).

    Representa la configuración completa de un Pokémon en el equipo del GA.

    Attributes:
        species_id: dex_id del Pokémon (int).
        item: Nombre del ítem equipado. String vacío si no tiene ítem.
        ability: Nombre de la habilidad. String vacío si no especificada.
        nature: Naturaleza del Pokémon.
        moves: Lista de 4 move IDs (int). 0 = slot vacío.
        tera_type: Tipo Tera. String vacío si no aplica.
        mega_flag: True si tiene Mega Stone.
        stat_points: Dict {stat: value} si la regulación usa SP.
                     Dict vacío si usa EVs/IVs.
    """

    species_id: int
    item: str = ""
    ability: str = ""
    nature: str = "Hardy"
    moves: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    tera_type: str = ""
    mega_flag: bool = False
    stat_points: dict[str, int] = field(default_factory=dict)


@dataclass
class Chromosome:
    """
    Cromosoma completo del GA — un equipo VGC de 6 Pokémon.

    Attributes:
        slots: Lista de 6 SlotGene.
        regulation_id: ID de la regulación que define la legalidad.
        fitness_values: Tupla de 4 valores de fitness (inicialmente vacía).
                        Se setea por DEAP en la Tarea 93.
    """

    slots: list[SlotGene] = field(default_factory=list)
    regulation_id: str = ""
    fitness_values: tuple[float, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.slots)

    def species_list(self) -> list[int]:
        """Retorna lista de species_id del equipo."""
        return [s.species_id for s in self.slots]

    def items_list(self) -> list[str]:
        """Retorna lista de ítems del equipo."""
        return [s.item for s in self.slots]

    def has_mega(self) -> bool:
        """True si algún slot tiene mega_flag."""
        return any(s.mega_flag for s in self.slots)

    def mega_count(self) -> int:
        """Número de slots con mega_flag."""
        return sum(1 for s in self.slots if s.mega_flag)


# ---------------------------------------------------------------------------
# I/O de datos maestros
# ---------------------------------------------------------------------------


def _load_pokemon_master() -> dict[int, dict[str, Any]]:
    """
    Carga pokemon_master.json mapeado por dex_id.

    Returns:
        Dict {dex_id (int): {name, types, base_stats, abilities, ...}}.
        Dict vacío si el archivo no existe o hay error de parseo.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            return {}
        raw: dict[str, Any] = json.loads(
            _POKEMON_MASTER_PATH.read_text(encoding="utf-8")
        )
        pokemon_data = raw.get("pokemon", {})
        return {
            int(dex_id): entry
            for dex_id, entry in pokemon_data.items()
            if isinstance(entry, dict)
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando pokemon_master.json: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Encoding / Decoding para DEAP
# ---------------------------------------------------------------------------


def encode(chromosome: Chromosome) -> list[int | str | float]:
    """
    Codifica un Chromosome a una lista plana de genes para DEAP.

    La representación plana tiene exactamente TEAM_SIZE * _GENES_PER_SLOT = 48
    elementos. Cada slot ocupa 8 genes:
      [species_id, item, nature_idx, move0, move1, move2, move3, mega_flag]

    No codifica tera_type ni stat_points — estos se manejan por separado en la
    repair function porque su representación no es trivialmente plana para DEAP.

    Args:
        chromosome: Chromosome a encodar.

    Returns:
        Lista plana con 48 elementos (6 slots × 8 genes).
    """
    genes: list[int | str | float] = []
    for slot in chromosome.slots[:TEAM_SIZE]:
        genes.append(slot.species_id)
        genes.append(slot.item)
        nature_idx = NATURES.index(slot.nature) if slot.nature in NATURES else 0
        genes.append(nature_idx)
        padded_moves = (slot.moves + [0, 0, 0, 0])[:MOVES_PER_SLOT]
        for move_id in padded_moves:
            genes.append(move_id)
        genes.append(int(slot.mega_flag))
    return genes


def decode(
    genes: list[int | str | float],
    regulation_id: str,
) -> Chromosome:
    """
    Decodifica una lista plana de genes a Chromosome.

    Inverso aproximado de encode(). Tolera listas más largas o más cortas —
    usa solo los primeros 48 genes y rellena con defaults si faltan.
    tera_type y stat_points se pierden en el round-trip (se recuperan en la
    repair function).

    Args:
        genes: Lista plana de genes (output de encode).
        regulation_id: Para etiquetar el cromosoma resultante.

    Returns:
        Chromosome reconstruido con los datos codificados.
    """
    slots: list[SlotGene] = []

    for i in range(TEAM_SIZE):
        base = i * _GENES_PER_SLOT
        if base >= len(genes):
            slots.append(SlotGene(species_id=0))
            continue

        def _get(offset: int, default: Any, _base: int = base) -> Any:
            idx = _base + offset
            return genes[idx] if idx < len(genes) else default

        species_id = int(_get(0, 0))
        item = str(_get(1, ""))
        nature_idx = int(_get(2, 0))
        nature = NATURES[nature_idx % len(NATURES)]
        moves = [int(_get(3 + j, 0)) for j in range(MOVES_PER_SLOT)]
        mega_flag = bool(int(_get(7, 0)))

        slots.append(
            SlotGene(
                species_id=species_id,
                item=item,
                nature=nature,
                moves=moves,
                mega_flag=mega_flag,
            )
        )

    return Chromosome(slots=slots, regulation_id=regulation_id)


# ---------------------------------------------------------------------------
# Generación aleatoria de cromosomas legales
# ---------------------------------------------------------------------------


def random_slot(
    reg: Any,
    rng: random.Random,
    pokemon_master: dict[int, dict[str, Any]],
    exclude_species: list[int] | None = None,
) -> SlotGene:
    """
    Genera un SlotGene aleatorio legal para la regulación dada.

    Selecciona aleatoriamente species, item, nature, moves y mega_flag
    dentro de los límites definidos por reg (RegulationConfig).

    Args:
        reg: RegulationConfig de la regulación.
        rng: Instancia de random.Random para reproducibilidad.
        pokemon_master: Datos maestros de Pokémon.
        exclude_species: Species IDs ya usados en el equipo (species_clause).

    Returns:
        SlotGene aleatorio y legal.
    """
    # Especie
    legal_species = list(reg.pokemon_legales)
    if exclude_species:
        legal_species = [s for s in legal_species if s not in exclude_species]
    if not legal_species:
        legal_species = list(reg.pokemon_legales)
    species_id = rng.choice(legal_species)

    # Ítem
    legal_items = list(reg.items_legales)
    item = rng.choice(legal_items) if legal_items else ""

    # Naturaleza
    nature = rng.choice(NATURES)

    # Moves
    legal_moves = list(reg.moves_legales)
    moves: list[int] = (
        [rng.choice(legal_moves) for _ in range(MOVES_PER_SLOT)]
        if legal_moves
        else [0] * MOVES_PER_SLOT
    )

    # Mega flag: True solo si mega_enabled y el ítem elegido es Mega Stone
    mega_flag = False
    if getattr(reg.mechanics, "mega_enabled", False):
        mega_stone_suffixes = ("ite", "onite", "ardite", "azite")
        if any(item.lower().endswith(s) for s in mega_stone_suffixes):
            mega_flag = True

    # Tera type
    tera_types: list[str] = [
        "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
        "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
        "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
    ]
    tera_type = (
        rng.choice(tera_types)
        if getattr(reg.mechanics, "tera_enabled", False)
        else ""
    )

    # Stat Points (si la regulación los usa)
    stat_points: dict[str, int] = {}
    if getattr(reg.mechanics, "stat_points_system", False):
        total = int(getattr(reg.mechanics, "stat_points_total", 66))
        cap = int(getattr(reg.mechanics, "stat_points_cap_per_stat", 32))
        stats = ["hp", "atk", "def", "spa", "spd", "spe"]
        remaining = total
        for stat in stats[:-1]:
            val = rng.randint(0, min(cap, remaining))
            stat_points[stat] = val
            remaining -= val
            if remaining <= 0:
                break
        stat_points[stats[-1]] = max(0, min(cap, remaining))

    return SlotGene(
        species_id=species_id,
        item=item,
        nature=nature,
        moves=moves,
        tera_type=tera_type,
        mega_flag=mega_flag,
        stat_points=stat_points,
    )


def random_chromosome(
    reg: Any,
    rng: random.Random | None = None,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> Chromosome:
    """
    Genera un Chromosome aleatorio completamente legal para la regulación dada.

    Respeta species_clause generando especies distintas para cada slot.

    Args:
        reg: RegulationConfig de la regulación.
        rng: Instancia de random.Random. Si None, crea una con seed=42.
        pokemon_master: Datos maestros de Pokémon. Si None, carga desde disco.

    Returns:
        Chromosome con TEAM_SIZE SlotGene legales.
    """
    if rng is None:
        rng = random.Random(42)

    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    species_clause: bool = getattr(reg.clauses, "species_clause", True)

    slots: list[SlotGene] = []
    used_species: list[int] = []

    for _ in range(TEAM_SIZE):
        slot = random_slot(
            reg,
            rng,
            pokemon_master,
            exclude_species=used_species if species_clause else None,
        )
        slots.append(slot)
        if species_clause:
            used_species.append(slot.species_id)

    return Chromosome(
        slots=slots,
        regulation_id=str(getattr(reg, "regulation_id", "")),
    )


# ---------------------------------------------------------------------------
# Utilidad de display
# ---------------------------------------------------------------------------


def chromosome_to_dict(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Convierte un Chromosome a dict con nombres legibles para la UI.

    Resuelve species_id a nombre usando pokemon_master.

    Args:
        chrom: Chromosome a convertir.
        pokemon_master: Datos maestros. Carga desde disco si None.

    Returns:
        Dict con regulation_id, fitness_values y lista de slots con nombres resueltos.
    """
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    slots_data: list[dict[str, Any]] = []
    for slot in chrom.slots:
        species_data = pokemon_master.get(slot.species_id, {})
        species_name = str(
            species_data.get("name", f"#{slot.species_id}")
        ).capitalize()
        slots_data.append({
            "species_id": slot.species_id,
            "species_name": species_name,
            "item": slot.item,
            "ability": slot.ability,
            "nature": slot.nature,
            "moves": slot.moves,
            "tera_type": slot.tera_type,
            "mega_flag": slot.mega_flag,
            "stat_points": slot.stat_points,
        })

    return {
        "regulation_id": chrom.regulation_id,
        "fitness_values": list(chrom.fitness_values),
        "slots": slots_data,
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SlotGene",
    "Chromosome",
    "NATURES",
    "TEAM_SIZE",
    "MOVES_PER_SLOT",
    "encode",
    "decode",
    "random_slot",
    "random_chromosome",
    "chromosome_to_dict",
]
