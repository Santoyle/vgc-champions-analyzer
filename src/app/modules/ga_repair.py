"""
Repair function determinista para cromosomas del GA NSGA-II de VGC.

Después de cada crossover y mutación en DEAP, el cromosoma puede quedar
ilegal. Este módulo aplica correcciones deterministas en orden fijo para
restaurar la legalidad sin penalizaciones en el fitness.

Orden de checks (fijo e invariante):
  1. team_size        — completar slots faltantes.
  2. illegal_species  — reemplazar especies fuera de reg.pokemon_legales.
  3. species_clause   — eliminar duplicados de especie.
  4. illegal_items    — reemplazar ítems fuera de reg.items_legales.
  5. item_clause      — eliminar ítems duplicados (si item_clause=True).
  6. mega_clause      — máximo 1 Mega Stone por equipo.
  7. stat_points      — normalizar SP al cap si la reg usa stat_points_system.

El orden garantiza que checks posteriores no invaliden los anteriores:
por ejemplo, species_clause se aplica después de illegal_species para que
los duplicados se resuelvan solo entre especies legales.

RepairLog registra cuántas correcciones se aplicaron por categoría, lo que
permite medir la "salud" del operador genético (tasa de cromosomas ilegales
generados por crossover/mutación) y ajustar los hiperparámetros del GA.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from src.app.modules.ga import (
    TEAM_SIZE,
    Chromosome,
    SlotGene,
    _load_pokemon_master,
    random_slot,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RepairLog
# ---------------------------------------------------------------------------


@dataclass
class RepairLog:
    """
    Registro de las correcciones aplicadas por la repair function.

    Útil para debugging y para medir qué tan "dañados" llegan los cromosomas
    del GA tras las operaciones de crossover y mutación.

    Attributes:
        species_fixes: Número de especies reemplazadas por species_clause.
        item_fixes: Número de ítems reemplazados por item_clause.
        mega_fixes: Número de mega flags desactivados.
        illegal_species_fixes: Especies ilegales corregidas.
        illegal_item_fixes: Ítems ilegales corregidos.
        stat_points_fixes: Slots con SP normalizados.
        size_fixes: Slots añadidos por team_size incompleto.
    """

    species_fixes: int = 0
    item_fixes: int = 0
    mega_fixes: int = 0
    illegal_species_fixes: int = 0
    illegal_item_fixes: int = 0
    stat_points_fixes: int = 0
    size_fixes: int = 0

    @property
    def total_fixes(self) -> int:
        """Total de correcciones aplicadas en todas las categorías."""
        return (
            self.species_fixes
            + self.item_fixes
            + self.mega_fixes
            + self.illegal_species_fixes
            + self.illegal_item_fixes
            + self.stat_points_fixes
            + self.size_fixes
        )

    @property
    def is_clean(self) -> bool:
        """True si no se aplicó ninguna corrección — el cromosoma ya era legal."""
        return self.total_fixes == 0


# ---------------------------------------------------------------------------
# Helpers privados (un check por función)
# ---------------------------------------------------------------------------


def _fix_team_size(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
    pokemon_master: dict[int, dict[str, Any]],
) -> int:
    """
    Completa el equipo a TEAM_SIZE slots si faltan.

    Puede ocurrir si el crossover produce un cromosoma con menos de 6 slots.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG para selección aleatoria.
        pokemon_master: Datos maestros de Pokémon.

    Returns:
        Número de slots añadidos.
    """
    fixes = 0
    used_species = [s.species_id for s in chrom.slots]
    species_clause: bool = getattr(reg.clauses, "species_clause", True)

    while len(chrom.slots) < TEAM_SIZE:
        slot = random_slot(
            reg,
            rng,
            pokemon_master,
            exclude_species=used_species if species_clause else None,
        )
        chrom.slots.append(slot)
        used_species.append(slot.species_id)
        fixes += 1

    return fixes


def _fix_illegal_species(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
    pokemon_master: dict[int, dict[str, Any]],
) -> int:
    """
    Reemplaza especies ilegales (no en reg.pokemon_legales) por legales.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG para selección aleatoria.
        pokemon_master: Datos maestros.

    Returns:
        Número de correcciones aplicadas.
    """
    legal_set = set(reg.pokemon_legales)
    legal_list = list(legal_set)
    fixes = 0
    for slot in chrom.slots:
        if slot.species_id not in legal_set:
            slot.species_id = rng.choice(legal_list)
            fixes += 1
    return fixes


def _fix_species_clause(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
    pokemon_master: dict[int, dict[str, Any]],
) -> int:
    """
    Corrige duplicados de especie (species_clause).

    Itera los slots en orden y reemplaza los duplicados por una especie legal
    que no esté ya en el equipo. Se llama después de _fix_illegal_species para
    garantizar que las especies de reemplazo sean siempre legales.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG para selección aleatoria.
        pokemon_master: Datos maestros.

    Returns:
        Número de correcciones aplicadas.
    """
    if not getattr(reg.clauses, "species_clause", True):
        return 0

    legal_set = set(reg.pokemon_legales)
    seen: set[int] = set()
    fixes = 0

    for slot in chrom.slots:
        if slot.species_id in seen:
            available = list(legal_set - seen)
            if not available:
                available = list(legal_set)
            slot.species_id = rng.choice(available)
            fixes += 1
        seen.add(slot.species_id)

    return fixes


def _fix_illegal_items(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
) -> int:
    """
    Reemplaza ítems ilegales (no en reg.items_legales) por ítems legales.

    Un ítem es ilegal si no está en reg.items_legales y no es string vacío.
    La comparación es case-insensitive para robustez.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG.

    Returns:
        Número de correcciones aplicadas.
    """
    legal_items = list(reg.items_legales)
    if not legal_items:
        return 0

    legal_set = {i.lower() for i in legal_items}
    fixes = 0

    for slot in chrom.slots:
        if slot.item and slot.item.lower() not in legal_set:
            slot.item = rng.choice(legal_items)
            fixes += 1

    return fixes


def _fix_item_clause(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
) -> int:
    """
    Corrige ítems duplicados (item_clause).

    Reemplaza el segundo y sucesivos ítems duplicados por ítems legales
    distintos. Solo se aplica si reg.clauses.item_clause=True.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG.

    Returns:
        Número de correcciones aplicadas.
    """
    if not getattr(reg.clauses, "item_clause", False):
        return 0

    legal_items = list(reg.items_legales)
    if not legal_items:
        return 0

    seen_items: set[str] = set()
    fixes = 0

    for slot in chrom.slots:
        if slot.item and slot.item in seen_items:
            available = [i for i in legal_items if i not in seen_items]
            if not available:
                available = legal_items
            slot.item = rng.choice(available)
            fixes += 1
        if slot.item:
            seen_items.add(slot.item)

    return fixes


def _fix_mega_clause(
    chrom: Chromosome,
    reg: Any,
) -> int:
    """
    Corrige exceso de Mega Stones.

    Si mega_enabled=True, solo puede haber 1 slot con mega_flag=True.
    Desactiva el flag en todos los slots excepto el primero.
    Si mega_enabled=False, limpia todos los flags.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.

    Returns:
        Número de correcciones aplicadas.
    """
    if not getattr(reg.mechanics, "mega_enabled", False):
        fixes = sum(1 for s in chrom.slots if s.mega_flag)
        for slot in chrom.slots:
            slot.mega_flag = False
        return fixes

    mega_max: int = int(getattr(reg.mechanics, "mega_max_per_battle", 1))
    mega_count = 0
    fixes = 0

    for slot in chrom.slots:
        if slot.mega_flag:
            if mega_count >= mega_max:
                slot.mega_flag = False
                fixes += 1
            else:
                mega_count += 1

    return fixes


def _fix_stat_points(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random,
) -> int:
    """
    Normaliza los Stat Points de cada slot.

    Si la regulación usa stat_points_system:
      1. Si el slot no tiene stat_points, asigna distribución aleatoria válida.
      2. Clampea cualquier stat que supere stat_points_cap_per_stat.
      3. Escala proporcionalmente si la suma supera stat_points_total.

    Args:
        chrom: Chromosome a reparar (in-place).
        reg: RegulationConfig.
        rng: RNG.

    Returns:
        Número de slots corregidos.
    """
    if not getattr(reg.mechanics, "stat_points_system", False):
        return 0

    total_cap: int = int(getattr(reg.mechanics, "stat_points_total", 66))
    per_stat_cap: int = int(getattr(reg.mechanics, "stat_points_cap_per_stat", 32))
    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    fixes = 0

    for slot in chrom.slots:
        if not slot.stat_points:
            # Asignar distribución aleatoria válida
            remaining = total_cap
            sp: dict[str, int] = {}
            for stat in stats[:-1]:
                val = rng.randint(0, min(per_stat_cap, remaining))
                sp[stat] = val
                remaining -= val
                if remaining <= 0:
                    break
            sp[stats[-1]] = max(0, min(per_stat_cap, remaining))
            slot.stat_points = sp
            fixes += 1
            continue

        modified = False

        # Clampear cada stat al cap
        for stat in stats:
            val = slot.stat_points.get(stat, 0)
            if val > per_stat_cap:
                slot.stat_points[stat] = per_stat_cap
                modified = True

        # Escalar si la suma supera el total
        current_total = sum(slot.stat_points.values())
        if current_total > total_cap and current_total > 0:
            scale = total_cap / current_total
            for stat in stats:
                slot.stat_points[stat] = int(
                    slot.stat_points.get(stat, 0) * scale
                )
            modified = True

        if modified:
            fixes += 1

    return fixes


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def repair(
    chrom: Chromosome,
    reg: Any,
    rng: random.Random | None = None,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> tuple[Chromosome, RepairLog]:
    """
    Repara un Chromosome post-crossover/mutación para que sea legal.

    Aplica los checks en orden estricto y determinista:
      1. team_size        — completar slots faltantes con random_slot().
      2. illegal_species  — reemplazar species_id fuera de pokemon_legales.
      3. species_clause   — eliminar duplicados de especie.
      4. illegal_items    — reemplazar ítems fuera de items_legales.
      5. item_clause      — eliminar ítems duplicados (si item_clause=True).
      6. mega_clause      — máximo 1 Mega Stone por equipo.
      7. stat_points      — normalizar SP si la reg usa stat_points_system.

    La repair function NO usa penalizaciones de fitness — siempre retorna
    un cromosoma completamente legal. El fitness se evalúa sobre el cromosoma
    ya reparado.

    Args:
        chrom: Chromosome a reparar. Se modifica IN-PLACE y también se retorna.
               DEAP permite usar el objeto original o el retorno; son el mismo.
        reg: RegulationConfig activa.
        rng: RNG para selecciones aleatorias. Si None, usa random.Random(42).
        pokemon_master: Datos maestros de Pokémon. Si None, carga desde disco.

    Returns:
        Tupla (chrom_reparado, repair_log).
        chrom_reparado es el mismo objeto modificado in-place.
    """
    if rng is None:
        rng = random.Random(42)
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    repair_log = RepairLog()

    # Aplicar checks en orden fijo. Cada check está aislado en try/except
    # para que un error interno no aborte las correcciones posteriores.

    try:
        repair_log.size_fixes = _fix_team_size(chrom, reg, rng, pokemon_master)
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_team_size falló: %s", exc)

    try:
        repair_log.illegal_species_fixes = _fix_illegal_species(
            chrom, reg, rng, pokemon_master
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_illegal_species falló: %s", exc)

    try:
        repair_log.species_fixes = _fix_species_clause(
            chrom, reg, rng, pokemon_master
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_species_clause falló: %s", exc)

    try:
        repair_log.illegal_item_fixes = _fix_illegal_items(chrom, reg, rng)
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_illegal_items falló: %s", exc)

    try:
        repair_log.item_fixes = _fix_item_clause(chrom, reg, rng)
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_item_clause falló: %s", exc)

    try:
        repair_log.mega_fixes = _fix_mega_clause(chrom, reg)
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_mega_clause falló: %s", exc)

    try:
        repair_log.stat_points_fixes = _fix_stat_points(chrom, reg, rng)
    except Exception as exc:  # noqa: BLE001
        log.warning("_fix_stat_points falló: %s", exc)

    if not repair_log.is_clean:
        log.debug(
            "Chromosome reparado: %d correcciones total "
            "(size=%d, ill_sp=%d, sp_clause=%d, ill_it=%d, "
            "it_clause=%d, mega=%d, stat_pts=%d)",
            repair_log.total_fixes,
            repair_log.size_fixes,
            repair_log.illegal_species_fixes,
            repair_log.species_fixes,
            repair_log.illegal_item_fixes,
            repair_log.item_fixes,
            repair_log.mega_fixes,
            repair_log.stat_points_fixes,
        )

    return chrom, repair_log


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "RepairLog",
    "repair",
]
