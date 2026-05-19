"""
Implementación de NSGA-II con DEAP para optimización multi-objetivo de equipos VGC.

El algoritmo evoluciona equipos de 6 Pokémon optimizando simultáneamente 4 objetivos:
cobertura defensiva (f1), sinergia ofensiva (f2), anti-meta (f3) y control de
velocidad (f4).

Parámetros de producción: pop=120, gen=60. Estos valores balancean calidad de
solución y tiempo de cómputo (~2-5 minutos en CPU estándar).

warm_start_chromosomes permite transferir conocimiento evolutivo entre regulaciones
(Tarea 94): equipos buenos de Regulación I se usan como semilla para Regulación J,
reduciendo el número de generaciones necesarias para converger.

La repair function se aplica después de cada operador genético para garantizar que
la población siempre es 100% legal — no se usan penalizaciones en el fitness.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import algorithms, base, creator, tools

from src.app.modules.ga import (
    TEAM_SIZE,
    Chromosome,
    SlotGene,
    _load_pokemon_master,
    decode,
    encode,
    random_chromosome,
)
from src.app.modules.ga_fitness import (
    META_FALLBACK_TOP20,
    evaluate_fitness,
)
from src.app.modules.ga_repair import repair

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del algoritmo
# ---------------------------------------------------------------------------

POP_SIZE: int = 120
N_GENERATIONS: int = 60
CXPB: float = 0.7   # probabilidad de crossover
MUTPB: float = 0.2  # probabilidad de mutación por individuo
TOURNAMENT_SIZE: int = 2


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------


@dataclass
class GAResult:
    """
    Resultado de una ejecución del GA NSGA-II.

    Attributes:
        pareto_front: Lista de Chromosome en el frente de Pareto final
                      (reducido a ~12 soluciones representativas).
        logbook: Estadísticas por generación (gen, avg, max por objetivo).
        n_generations: Generaciones ejecutadas.
        regulation_id: Regulación de la evolución.
        best_fitness_history: Lista de tuplas (gen, f1×1000, f2×1000,
                               f3×1000, f4×1000) por generación.
    """

    pareto_front: list[Chromosome]
    logbook: list[dict[str, Any]]
    n_generations: int
    regulation_id: str
    best_fitness_history: list[tuple[int, ...]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Setup DEAP
# ---------------------------------------------------------------------------


def _setup_deap(n_objectives: int = 4) -> tuple[base.Toolbox, Any]:
    """
    Configura las clases DEAP para NSGA-II.

    Crea FitnessMulti con n_objectives pesos positivos (maximización) y la
    clase Individual que extiende Chromosome. Usa delattr para limpiar clases
    previas y evitar el error "class already registered" de DEAP cuando se
    llama múltiples veces en el mismo proceso.

    Args:
        n_objectives: Número de objetivos de fitness. Default 4.

    Returns:
        Tupla (toolbox, creator_module) configurados.
    """
    for cls_name in ("FitnessMulti", "Individual"):
        if hasattr(creator, cls_name):
            delattr(creator, cls_name)

    creator.create(
        "FitnessMulti",
        base.Fitness,
        weights=tuple(1.0 for _ in range(n_objectives)),
    )
    creator.create(
        "Individual",
        Chromosome,
        fitness=creator.FitnessMulti,
    )

    toolbox = base.Toolbox()
    return toolbox, creator


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------


def _cx_chromosome(
    ind1: Chromosome,
    ind2: Chromosome,
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    """
    Crossover two-point a nivel de slots entre dos cromosomas.

    Intercambia el segmento [cx1:cx2] de SlotGene. Opera directamente sobre
    los slots (no sobre el encoding plano) para preservar tera_type y
    stat_points que encode/decode no preservan.

    Args:
        ind1: Primer padre (se modifica in-place).
        ind2: Segundo padre (se modifica in-place).
        rng: RNG para reproducibilidad.

    Returns:
        Tupla (hijo1, hijo2) modificados in-place.
    """
    if len(ind1.slots) < 2 or len(ind2.slots) < 2:
        return ind1, ind2

    size = min(len(ind1.slots), len(ind2.slots))
    cx1 = rng.randint(1, size - 1)
    cx2 = rng.randint(cx1, size)

    ind1.slots[cx1:cx2], ind2.slots[cx1:cx2] = (
        ind2.slots[cx1:cx2][:],
        ind1.slots[cx1:cx2][:],
    )
    return ind1, ind2


def _mut_chromosome(
    ind: Chromosome,
    reg: Any,
    rng: random.Random,
    pokemon_master: dict[int, dict[str, Any]],
    indpb: float = 0.1,
) -> tuple[Chromosome]:
    """
    Mutación por slot: reemplaza slots individuales con probabilidad indpb.

    Para cada slot, con probabilidad indpb genera un nuevo SlotGene aleatorio
    legal usando random_slot. Respeta species_clause acumulando los species_id
    ya asignados.

    Args:
        ind: Individuo a mutar (in-place).
        reg: RegulationConfig.
        rng: RNG.
        pokemon_master: Datos maestros.
        indpb: Probabilidad de mutación por slot. Default 0.1.

    Returns:
        Tupla de un elemento con el individuo mutado (convención DEAP).
    """
    from src.app.modules.ga import random_slot

    species_clause: bool = getattr(reg.clauses, "species_clause", True)
    used_species: list[int] = []

    for i, slot in enumerate(ind.slots):
        if rng.random() < indpb:
            new_slot = random_slot(
                reg,
                rng,
                pokemon_master,
                exclude_species=used_species if species_clause else None,
            )
            ind.slots[i] = new_slot
            if species_clause:
                used_species.append(new_slot.species_id)
        else:
            if species_clause:
                used_species.append(slot.species_id)

    return (ind,)


def _deep_copy_chromosome(chrom: Chromosome) -> Chromosome:
    """Crea una copia profunda de un Chromosome para usar como offspring."""
    return Chromosome(
        slots=[
            SlotGene(
                species_id=s.species_id,
                item=s.item,
                ability=s.ability,
                nature=s.nature,
                moves=s.moves[:],
                tera_type=s.tera_type,
                mega_flag=s.mega_flag,
                stat_points=dict(s.stat_points),
            )
            for s in chrom.slots
        ],
        regulation_id=chrom.regulation_id,
    )


# ---------------------------------------------------------------------------
# Clustering del frente de Pareto
# ---------------------------------------------------------------------------


def _cluster_pareto_front(
    front: list[Chromosome],
    target_size: int = 12,
) -> list[Chromosome]:
    """
    Reduce el frente de Pareto a target_size soluciones representativas.

    Usa KMeans sobre los vectores de fitness para seleccionar el individuo
    más cercano al centroide de cada cluster. Si sklearn no está disponible
    o falla, retorna los primeros target_size elementos del frente.

    Args:
        front: Lista de Chromosome en el frente de Pareto.
        target_size: Número deseado de soluciones representativas.

    Returns:
        Lista reducida de Chromosome (len <= target_size).
    """
    if len(front) <= target_size:
        return front

    try:
        from sklearn.cluster import KMeans

        fitness_matrix = np.array([
            list(ind.fitness.values)  # type: ignore[attr-defined]
            for ind in front
        ])

        kmeans = KMeans(n_clusters=target_size, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fitness_matrix)

        selected: list[Chromosome] = []
        for cluster_id in range(target_size):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_fits = fitness_matrix[cluster_mask]
            dists = np.linalg.norm(cluster_fits - centroid, axis=1)
            best_in_cluster = cluster_indices[np.argmin(dists)]
            selected.append(front[best_in_cluster])

        return selected

    except Exception as exc:  # noqa: BLE001
        log.debug(
            "Clustering del frente falló: %s — retornando primeros %d",
            exc,
            target_size,
        )
        return front[:target_size]


# ---------------------------------------------------------------------------
# Loop principal de evolución
# ---------------------------------------------------------------------------


def run_nsga2(
    reg: Any,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
    meta_top: list[str] | None = None,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GENERATIONS,
    seed: int = 42,
    warm_start_chromosomes: list[Chromosome] | None = None,
    progress_callback: Any = None,
) -> GAResult:
    """
    Ejecuta NSGA-II para optimizar equipos VGC multi-objetivo.

    Pipeline completo:
      1. Inicializar población aleatoria legal (+ warm-start si disponible).
      2. Evaluar fitness inicial de toda la población.
      3. Bucle generacional (selección → crossover → mutación → repair → fitness).
      4. Extraer y reducir el frente de Pareto final con clustering.

    Args:
        reg: RegulationConfig de la regulación activa.
        pokemon_master: Datos maestros de Pokémon. Carga desde disco si None.
        meta_top: Top Pokémon del meta para f3. None = META_FALLBACK_TOP20.
        pop_size: Tamaño de la población. Default POP_SIZE=120.
        n_gen: Número de generaciones. Default N_GENERATIONS=60.
        seed: Semilla para reproducibilidad determinista.
        warm_start_chromosomes: Cromosomas de regulaciones anteriores para
                                 warm-start (Tarea 94). Se inyectan en el
                                 35% inicial de la población tras repair.
        progress_callback: Callable(gen, total, pareto_preview) para UI.
                            None = sin callback.

    Returns:
        GAResult con frente de Pareto (≤12 equipos), logbook y estadísticas.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()
    if meta_top is None:
        meta_top = META_FALLBACK_TOP20

    toolbox, creator_mod = _setup_deap(n_objectives=4)
    reg_id = str(getattr(reg, "regulation_id", ""))

    def _eval_chrom(chrom: Chromosome) -> tuple[float, float, float, float]:
        return evaluate_fitness(chrom, pokemon_master, meta_top)

    toolbox.register("evaluate", _eval_chrom)
    toolbox.register("select", tools.selNSGA2)

    # ── Paso 1: Población inicial ───────────────────────────────────────────

    log.info("Inicializando población: %d individuos", pop_size)
    population: list[Chromosome] = []

    if warm_start_chromosomes:
        n_warm = min(int(pop_size * 0.35), len(warm_start_chromosomes))
        for i in range(n_warm):
            warm_chrom = warm_start_chromosomes[i % len(warm_start_chromosomes)]
            repaired, _ = repair(warm_chrom, reg, rng, pokemon_master)
            population.append(repaired)
        log.info("Warm-start: %d cromosomas históricos inyectados", n_warm)

    while len(population) < pop_size:
        chrom = random_chromosome(reg, rng, pokemon_master)
        population.append(chrom)

    # Adjuntar fitness DEAP a cada individuo
    for ind in population:
        if not hasattr(ind, "fitness"):
            ind.fitness = creator_mod.FitnessMulti()  # type: ignore[attr-defined]

    # ── Paso 2: Evaluación inicial ──────────────────────────────────────────

    for ind, fit in zip(population, map(_eval_chrom, population)):
        ind.fitness.values = fit  # type: ignore[attr-defined]

    population = toolbox.select(population, pop_size)

    # ── Estadísticas ────────────────────────────────────────────────────────

    stats = tools.Statistics(
        lambda ind: ind.fitness.values  # type: ignore[attr-defined]
    )
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    logbook_data: list[dict[str, Any]] = []
    best_history: list[tuple[int, ...]] = []

    log.info(
        "Iniciando evolución NSGA-II: %d gen × %d individuos",
        n_gen,
        pop_size,
    )

    # ── Bucle generacional ──────────────────────────────────────────────────

    for gen in range(n_gen):

        # Selección para offspring con crowding distance
        offspring = tools.selTournamentDCD(population, pop_size)
        offspring = [_deep_copy_chromosome(ind) for ind in offspring]

        # Adjuntar fitness a offspring
        for ind in offspring:
            if not hasattr(ind, "fitness"):
                ind.fitness = creator_mod.FitnessMulti()  # type: ignore[attr-defined]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if rng.random() < CXPB:
                offspring[i], offspring[i + 1] = _cx_chromosome(
                    offspring[i], offspring[i + 1], rng
                )

        # Mutación
        for ind in offspring:
            if rng.random() < MUTPB:
                _mut_chromosome(ind, reg, rng, pokemon_master)

        # Repair: garantizar legalidad después de cada operador
        for ind in offspring:
            repair(ind, reg, rng, pokemon_master)

        # Evaluar fitness de offspring
        for ind, fit in zip(offspring, map(_eval_chrom, offspring)):
            ind.fitness.values = fit  # type: ignore[attr-defined]

        # Selección NSGA-II: sobreviven los mejores de pop + offspring
        population = toolbox.select(population + offspring, pop_size)

        # Estadísticas de la generación
        record = stats.compile(population)
        max_fits: Any = record.get("max", np.zeros(4))
        logbook_data.append({
            "gen": gen,
            "avg": record.get("avg", np.zeros(4)).tolist(),
            "max": max_fits.tolist(),
        })
        best_history.append((
            gen,
            int(float(max_fits[0]) * 1000),
            int(float(max_fits[1]) * 1000),
            int(float(max_fits[2]) * 1000),
            int(float(max_fits[3]) * 1000),
        ))

        if gen % 10 == 0:
            log.info(
                "Gen %d/%d — max fitness: f1=%.3f f2=%.3f f3=%.3f f4=%.3f",
                gen,
                n_gen,
                float(max_fits[0]),
                float(max_fits[1]),
                float(max_fits[2]),
                float(max_fits[3]),
            )

        # Callback de progreso para la UI (sin acoplar a Streamlit)
        if progress_callback is not None:
            try:
                pareto_preview = tools.sortNondominated(
                    population, pop_size, first_front_only=True
                )[0]
                progress_callback(gen + 1, n_gen, pareto_preview)
            except Exception:  # noqa: BLE001
                pass

    # ── Paso 5: Frente de Pareto final ─────────────────────────────────────

    pareto_front = tools.sortNondominated(
        population, pop_size, first_front_only=True
    )[0]
    pareto_final = _cluster_pareto_front(pareto_front, target_size=12)

    log.info(
        "NSGA-II completado: %d equipos en Pareto (antes clustering: %d)",
        len(pareto_final),
        len(pareto_front),
    )

    return GAResult(
        pareto_front=pareto_final,
        logbook=logbook_data,
        n_generations=n_gen,
        regulation_id=reg_id,
        best_fitness_history=best_history,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "GAResult",
    "run_nsga2",
    "POP_SIZE",
    "N_GENERATIONS",
]
