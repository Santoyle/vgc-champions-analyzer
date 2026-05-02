"""
SPDO (SP Distribution Optimizer) — repartos de Stat Points en Champions.

El optimizador tiene cuatro modos previstos; este módulo implementa el
**Modo 2**: optimización defensiva frente a un ataque concreto (KO / supervivencia
con probabilidad mínima). El **Modo 1** (meta-óptimo vía NSGA-II) vivirá en
``spdo_meta.py`` (T-124).

Modo 2 expone ``optimize_defensive_sp()``: dado un atacante, un move y datos
del defensor, busca el mínimo coste en HP/Def (o HP/SpD) que garantiza vivir el
ataque con probabilidad mayor o igual al objetivo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.app.core.champions_calc import (
    calc_all_stats,
    champions_damage_calc,
)
from src.app.core.schema import SPSpread

log = logging.getLogger(__name__)


@dataclass
class DefensiveOptResult:
    """
    Resultado de la optimización defensiva SPDO
    Modo 2.

    Attributes:
        spread: SPSpread óptimo encontrado.
        survival_prob: Probabilidad de supervivencia
                       lograda (0.0-1.0).
        target_prob: Probabilidad objetivo solicitada.
        hp_sp_used: SP gastados en HP.
        def_sp_used: SP gastados en Def o SpD.
        total_sp_cost: HP_sp + def_sp_used.
        stat_used: "def" o "spd" según categoría
                    del ataque.
        def_stat_value: Valor final del stat
                         defensivo.
        hp_value: Valor final de HP.
        pokemon: Nombre del defensor.
        attacker: Nombre del atacante.
        move_name: Nombre del move.
        found: True si se encontró solución.
    """

    spread: SPSpread
    survival_prob: float
    target_prob: float
    hp_sp_used: int
    def_sp_used: int
    total_sp_cost: int
    stat_used: str
    def_stat_value: int
    hp_value: int
    pokemon: str
    attacker: str
    move_name: str
    found: bool


def optimize_defensive_sp(
    reg_id: str,
    pokemon_slug: str,
    attacker_slug: str,
    attacker_move: dict[str, Any],
    attacker_spread: SPSpread,
    attacker_nature: str,
    attacker_item: str | None = None,
    def_nature: str = "Hardy",
    def_item: str | None = None,
    field: dict[str, Any] | None = None,
    survival_target_pct: float = 0.95,
    fixed_offensive_sp: int = 32,
    with_stealth_rock: bool = False,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> DefensiveOptResult | None:
    """
    Encuentra el spread SP mínimo para que el
    defensor sobreviva el move del atacante con
    probabilidad >= survival_target_pct.

    Estrategia de búsqueda:
    - Determinar si el move es Physical o Special
    - Buscar exhaustivamente sobre HP_sp × Def_sp
      (o HP_sp × SpD_sp para Special)
    - Para cada combinación calcular survival_prob
      usando champions_damage_calc()
    - Retornar la combinación con menor costo total
      (HP_sp + def_sp) que cumple el target
    - Si hay empate en costo, preferir más HP_sp
      (más versátil que Def/SpD)

    Stealth Rock: si with_stealth_rock=True,
    sumar 1/8 del HP máximo (como fracción respecto a
    ese HP) al daño de cada roll antes de evaluar
    supervivencia.

    Args:
        reg_id: Regulación (para futuro uso con DB).
        pokemon_slug: Pokémon a optimizar.
        attacker_slug: Atacante de referencia.
        attacker_move: Dict con power, type,
                       category, name.
        attacker_spread: SPSpread del atacante.
        attacker_nature: Naturaleza del atacante.
        attacker_item: Ítem del atacante.
        def_nature: Naturaleza del defensor.
        def_item: Ítem del defensor.
        field: Condiciones de campo (weather,
               reflect_def, etc.).
               None = campo neutral.
        survival_target_pct: Probabilidad mínima
                              de supervivencia
                              requerida [0, 1].
        fixed_offensive_sp: SP fijos en el stat
                             ofensivo del defensor
                             (Atk o SpA). Default 32.
        with_stealth_rock: Si True, simular con
                           Stealth Rock activos.
        pokemon_master: Datos maestros. Carga si None.

    Returns:
        DefensiveOptResult con el spread óptimo
        y found=True si se encontró solución.
        DefensiveOptResult con found=False si
        ningún spread en el espacio cumple el target.
        None si hay error crítico (Pokémon no
        encontrado en pokemon_master).
    """
    _ = reg_id

    if field is None:
        field = {}

    if pokemon_master is None:
        from src.app.core.champions_calc import (
            _load_pokemon_master,
        )

        pokemon_master = _load_pokemon_master()

    # Determinar stat defensivo según categoría
    category = str(
        attacker_move.get("category", "Physical")
    )
    is_physical = category == "Physical"
    stat_used = "def" if is_physical else "spd"

    move_name = str(attacker_move.get("name", ""))

    # Verificar que el Pokémon existe
    test_spread = SPSpread(
        hp=0,
        atk=0,
        **{"def": 0},
        spa=0,
        spd=0,
        spe=0,
    )
    test_stats = calc_all_stats(
        pokemon_slug,
        test_spread,
        def_nature,
        pokemon_master,
    )
    if test_stats is None:
        log.warning(
            "Pokémon '%s' no encontrado en "
            "pokemon_master",
            pokemon_slug,
        )
        return None

    # Búsqueda exhaustiva sobre HP × stat_def
    best: DefensiveOptResult | None = None
    best_cost = 999

    for hp_sp in range(33):
        for def_sp in range(33):
            # Restricción: no exceder cap total
            total_used = (
                hp_sp + def_sp + fixed_offensive_sp
            )
            if total_used > 66:
                continue

            # Construir spread del defensor
            remainder = (
                66 - hp_sp - def_sp - fixed_offensive_sp
            )
            if is_physical:
                spread = SPSpread(
                    hp=hp_sp,
                    atk=0,
                    **{"def": def_sp},
                    spa=0,
                    spd=0,
                    spe=min(32, max(0, remainder)),
                )
            else:
                spread = SPSpread(
                    hp=hp_sp,
                    atk=0,
                    **{"def": 0},
                    spa=0,
                    spd=def_sp,
                    spe=min(32, max(0, remainder)),
                )

            # Calcular rolls de daño
            rolls = champions_damage_calc(
                attacker_slug,
                attacker_spread,
                attacker_nature,
                attacker_item,
                pokemon_slug,
                spread,
                def_nature,
                def_item,
                attacker_move,
                field,
                pokemon_master,
            )

            # Ajustar por Stealth Rock (chip = 1/8 del HP max)
            if with_stealth_rock:
                stats = calc_all_stats(
                    pokemon_slug,
                    spread,
                    def_nature,
                    pokemon_master,
                )
                if stats:
                    sr_damage = 1 / 8
                    # Rolls efectivos = daño + SR
                    effective_rolls = (
                        rolls + sr_damage
                    )
                else:
                    effective_rolls = rolls
            else:
                effective_rolls = rolls

            # Probabilidad de supervivencia
            survival_prob = float(
                (effective_rolls < 1.0).mean()
            )

            if survival_prob < survival_target_pct:
                continue

            # Verificar si es mejor solución
            cost = hp_sp + def_sp
            if cost < best_cost or (
                cost == best_cost
                and best is not None
                and hp_sp > best.hp_sp_used
            ):
                best_cost = cost
                def_stats = calc_all_stats(
                    pokemon_slug,
                    spread,
                    def_nature,
                    pokemon_master,
                )
                hp_val = (
                    def_stats["hp"]
                    if def_stats
                    else 0
                )
                def_val = (
                    def_stats[stat_used]
                    if def_stats
                    else 0
                )
                best = DefensiveOptResult(
                    spread=spread,
                    survival_prob=round(
                        survival_prob, 4
                    ),
                    target_prob=survival_target_pct,
                    hp_sp_used=hp_sp,
                    def_sp_used=def_sp,
                    total_sp_cost=cost,
                    stat_used=stat_used,
                    def_stat_value=def_val,
                    hp_value=hp_val,
                    pokemon=pokemon_slug,
                    attacker=attacker_slug,
                    move_name=move_name,
                    found=True,
                )

    if best is None:
        log.info(
            "Sin spread que garantice %.0f%% "
            "supervivencia para %s vs %s %s",
            survival_target_pct * 100,
            pokemon_slug,
            attacker_slug,
            move_name,
        )
        # Retornar resultado con found=False
        spread_zero = SPSpread(
            hp=0,
            atk=0,
            **{"def": 0},
            spa=0,
            spd=0,
            spe=0,
        )
        return DefensiveOptResult(
            spread=spread_zero,
            survival_prob=0.0,
            target_prob=survival_target_pct,
            hp_sp_used=0,
            def_sp_used=0,
            total_sp_cost=0,
            stat_used=stat_used,
            def_stat_value=0,
            hp_value=0,
            pokemon=pokemon_slug,
            attacker=attacker_slug,
            move_name=move_name,
            found=False,
        )

    return best


__all__ = [
    "DefensiveOptResult",
    "optimize_defensive_sp",
]
