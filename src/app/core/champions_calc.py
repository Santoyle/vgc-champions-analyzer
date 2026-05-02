"""
Champions Damage Calculator — módulo core.

Implementa la fórmula de stats con el sistema SP
de Pokémon Champions (1 SP = 8 EVs, cap 32/stat,
cap 66 total, IV fijo 31, nivel 50).

Funciones principales:
  stat_from_sp()       — calcula un stat individual
  calc_all_stats()     — calcula los 6 stats
  champions_damage_calc() — 16 damage rolls (T-115)
  type_effectiveness() — multiplicador de tipo

champions_damage_calc() retorna array[16] de rolls como fracción del
HP defensor. 0.0 = sin daño, 1.0 = OHKO exacto.

Validación: Garchomp 2/32/0/0/0/32 Jolly →
  HP=185, Atk=182, Def=130, SpA=90, SpD=95, Spe=169
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.app.core.schema import SPSpread
from src.app.modules.counter import TYPE_CHART

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_POKEMON_MASTER_PATH = (
    _PROJECT_ROOT / "data" / "pokemon_master.json"
)


NATURE_MULT: dict[str, dict[str, float]] = {
    # Neutral
    "Hardy": {},
    "Docile": {},
    "Serious": {},
    "Bashful": {},
    "Quirky": {},
    # +Atk
    "Lonely": {"atk": 1.1, "def": 0.9},
    "Brave": {"atk": 1.1, "spe": 0.9},
    "Adamant": {"atk": 1.1, "spa": 0.9},
    "Naughty": {"atk": 1.1, "spd": 0.9},
    # +Def
    "Bold": {"def": 1.1, "atk": 0.9},
    "Relaxed": {"def": 1.1, "spe": 0.9},
    "Impish": {"def": 1.1, "spa": 0.9},
    "Lax": {"def": 1.1, "spd": 0.9},
    # +Spe
    "Timid": {"spe": 1.1, "atk": 0.9},
    "Hasty": {"spe": 1.1, "def": 0.9},
    "Jolly": {"spe": 1.1, "spa": 0.9},
    "Naive": {"spe": 1.1, "spd": 0.9},
    # +SpA
    "Modest": {"spa": 1.1, "atk": 0.9},
    "Mild": {"spa": 1.1, "def": 0.9},
    "Quiet": {"spa": 1.1, "spe": 0.9},
    "Rash": {"spa": 1.1, "spd": 0.9},
    # +SpD
    "Calm": {"spd": 1.1, "atk": 0.9},
    "Gentle": {"spd": 1.1, "def": 0.9},
    "Sassy": {"spd": 1.1, "spe": 0.9},
    "Careful": {"spd": 1.1, "spa": 0.9},
}


def _get_nature_mult(
    nature: str,
    stat: str,
) -> float:
    """
    Retorna el multiplicador de naturaleza para
    un stat específico.

    Args:
        nature: Nombre de la naturaleza (ej "Jolly").
        stat: Stat a consultar: "atk", "def", "spa",
              "spd", "spe" (HP nunca tiene mult).

    Returns:
        1.1 si la naturaleza sube ese stat,
        0.9 si lo baja,
        1.0 si es neutral.
    """
    mults = NATURE_MULT.get(nature, {})
    return float(mults.get(stat, 1.0))


def stat_from_sp(
    base_stat: int,
    sp: int,
    nature_mult: float = 1.0,
    *,
    is_hp: bool = False,
    level: int = 50,
) -> int:
    """
    Calcula el valor final de un stat dado su
    base stat y SP en el sistema de Champions.

    Fórmula (1 SP = 8 EVs, IV = 31, cap = 256 EVs):
        ev = min(sp * 8, 252)  (cap Pydantic = 32 SP)
        HP:     floor((2*BS + 31 + ev//4) * L/100)
                + L + 10
        Non-HP: floor(floor((2*BS + 31 + ev//4)
                * L/100) + 5) * nature_mult

    Args:
        base_stat: Base stat del Pokémon.
        sp: SP asignados (0-32).
        nature_mult: Multiplicador de naturaleza
                     (0.9, 1.0 o 1.1). Default 1.0.
        is_hp: True si se está calculando HP.
        level: Nivel (siempre 50 en Champions).

    Returns:
        Valor final del stat como int.
    """
    sp = max(0, min(32, sp))
    ev = min(sp * 8, 252)
    raw = (2 * base_stat + 31 + ev // 4) * level // 100

    if is_hp:
        return raw + level + 10
    return math.floor((raw + 5) * nature_mult)


def _load_pokemon_master() -> dict[int, dict[str, Any]]:
    """
    Carga pokemon_master.json mapeado por dex_id.

    Returns:
        Dict {dex_id: {name, types, base_stats, ...}}.
        Dict vacío si el archivo no existe.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            log.warning(
                "pokemon_master.json no encontrado "
                "en %s",
                _POKEMON_MASTER_PATH,
            )
            return {}
        raw = json.loads(
            _POKEMON_MASTER_PATH.read_text(encoding="utf-8")
        )
        pokemon = raw.get("pokemon", raw)
        return {
            int(k): v
            for k, v in pokemon.items()
            if isinstance(v, dict)
        }
    except Exception as exc:
        log.warning(
            "Error cargando pokemon_master.json: %s",
            exc,
        )
        return {}


def _get_base_stats(
    slug: str,
    pokemon_master: dict[int, dict[str, Any]],
) -> dict[str, int] | None:
    """
    Obtiene los base stats de un Pokémon por nombre.

    Busca por nombre lowercase en pokemon_master.
    El pokemon_master usa keys de PokéAPI:
    hp, attack, defense, special-attack,
    special-defense, speed.

    Args:
        slug: Nombre del Pokémon en lowercase
              (ej "garchomp", "incineroar").
        pokemon_master: Datos maestros cargados.

    Returns:
        Dict normalizado con keys:
        hp, atk, def, spa, spd, spe.
        None si no se encuentra el Pokémon.
    """
    slug_lower = slug.lower()
    for entry in pokemon_master.values():
        name = str(entry.get("name", "")).lower()
        if name == slug_lower:
            bs = entry.get("base_stats", {})
            return {
                "hp": int(bs.get("hp", 0)),
                "atk": int(
                    bs.get("attack", bs.get("atk", 0))
                ),
                "def": int(
                    bs.get("defense", bs.get("def", 0))
                ),
                "spa": int(
                    bs.get(
                        "special-attack",
                        bs.get("spa", 0),
                    )
                ),
                "spd": int(
                    bs.get(
                        "special-defense",
                        bs.get("spd", 0),
                    )
                ),
                "spe": int(
                    bs.get("speed", bs.get("spe", 0))
                ),
            }
    return None


def calc_all_stats(
    slug: str,
    spread: SPSpread,
    nature: str,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> dict[str, int] | None:
    """
    Calcula los 6 stats finales de un Pokémon con
    un spread SP y naturaleza dados.

    Args:
        slug: Nombre del Pokémon (ej "garchomp").
        spread: SPSpread con la distribución de SP.
        nature: Nombre de la naturaleza (ej "Jolly").
        pokemon_master: Datos maestros. Si None,
                         carga desde disco.

    Returns:
        Dict {hp, atk, def, spa, spd, spe} con
        los stats finales calculados.
        None si el Pokémon no se encuentra.
    """
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    bs = _get_base_stats(slug, pokemon_master)
    if bs is None:
        log.warning(
            "Pokémon '%s' no encontrado en "
            "pokemon_master",
            slug,
        )
        return None

    return {
        "hp": stat_from_sp(
            bs["hp"], spread.hp, is_hp=True
        ),
        "atk": stat_from_sp(
            bs["atk"],
            spread.atk,
            _get_nature_mult(nature, "atk"),
        ),
        "def": stat_from_sp(
            bs["def"],
            spread.def_,
            _get_nature_mult(nature, "def"),
        ),
        "spa": stat_from_sp(
            bs["spa"],
            spread.spa,
            _get_nature_mult(nature, "spa"),
        ),
        "spd": stat_from_sp(
            bs["spd"],
            spread.spd,
            _get_nature_mult(nature, "spd"),
        ),
        "spe": stat_from_sp(
            bs["spe"],
            spread.spe,
            _get_nature_mult(nature, "spe"),
        ),
    }


def type_effectiveness(
    move_type: str,
    defender_types: list[str],
) -> float:
    """
    Calcula el multiplicador de efectividad de tipo
    de un ataque contra un defensor.

    Multiplica la efectividad de cada tipo del
    defensor usando TYPE_CHART de counter.py.

    Args:
        move_type: Tipo del ataque (ej "Water").
        defender_types: Lista de tipos del defensor
                        (1 o 2 elementos).

    Returns:
        Multiplicador total como float.
        0.25, 0.5, 1.0, 2.0 o 4.0 típicamente.
        0.0 para inmunidades.
    """
    mult = 1.0
    row_key = move_type.strip().title()
    type_row = TYPE_CHART.get(row_key, {})
    for def_type in defender_types:
        col_key = def_type.strip().title()
        mult *= float(
            type_row.get(col_key, 1.0)
        )
    return mult


def champions_damage_calc(
    att_slug: str,
    att_spread: SPSpread,
    att_nature: str,
    att_item: str | None,
    def_slug: str,
    def_spread: SPSpread,
    def_nature: str,
    def_item: str | None,
    move: dict[str, Any],
    field: dict[str, Any],
    pokemon_master: dict[int, dict[str, Any]] | None = None,
) -> NDArray[np.float64]:
    """
    Calcula los 16 damage rolls de un ataque en
    Pokémon Champions como fracción del HP máximo
    del defensor.

    Modificadores aplicados en orden gen 9:
    1. Spread damage ×0.75 si targets > 1 (doubles)
    2. Weather boost (rain/sun para Water/Fire)
    3. Screens (Reflect ×0.5/0.667, Light Screen)
    4. STAB ×1.5 si move_type en att_types
    5. Type effectiveness desde TYPE_CHART
    6. Burn ×0.5 si físico y atacante quemado
    7. Items: Life Orb ×1.3, CB/CS ×1.5, AV ×1.5 SpD
    8. Rolls (85..100)/100 — 16 valores discretos

    Args:
        att_slug: Nombre del atacante.
        att_spread: SPSpread del atacante.
        att_nature: Naturaleza del atacante.
        att_item: Ítem del atacante o None.
        def_slug: Nombre del defensor.
        def_spread: SPSpread del defensor.
        def_nature: Naturaleza del defensor.
        def_item: Ítem del defensor o None.
        move: Dict con keys:
              "power" (int), "type" (str),
              "category" (str: Physical/Special),
              "name" (str).
        field: Dict con keys opcionales:
               "targets" (int, default 1),
               "weather" (str: rain/sun/sand/snow),
               "reflect_def" (bool),
               "light_screen_def" (bool),
               "attacker_burned" (bool).
        pokemon_master: Datos maestros. Carga si None.

    Returns:
        np.ndarray de shape (16,) con los 16 rolls
        como fracción del HP máximo del defensor.
        Cada elemento en [0.0, 1.0+].
        Array de ceros si hay error de cálculo.
    """
    if pokemon_master is None:
        pokemon_master = _load_pokemon_master()

    try:
        # Calcular stats del atacante y defensor
        att_stats = calc_all_stats(
            att_slug,
            att_spread,
            att_nature,
            pokemon_master,
        )
        def_stats = calc_all_stats(
            def_slug,
            def_spread,
            def_nature,
            pokemon_master,
        )

        if att_stats is None or def_stats is None:
            log.warning(
                "No se encontraron stats para "
                "%s o %s",
                att_slug,
                def_slug,
            )
            return np.zeros(16, dtype=np.float64)

        # Stat de ataque y defensa según categoría
        category = move.get("category", "Physical")
        is_physical = category == "Physical"

        # Assault Vest sube SpD ×1.5 efectivo
        def_spd_mult = 1.0
        if (
            not is_physical
            and def_item == "Assault Vest"
        ):
            def_spd_mult = 1.5

        A = att_stats["atk"] if is_physical else att_stats["spa"]
        D = int(
            (
                def_stats["def"]
                if is_physical
                else def_stats["spd"]
            )
            * def_spd_mult
        )
        HP_def = def_stats["hp"]

        if D == 0 or HP_def == 0:
            return np.zeros(16, dtype=np.float64)

        power = int(move.get("power", 0))
        if power == 0:
            return np.zeros(16, dtype=np.float64)

        # Base damage (fórmula gen 9)
        base = (
            (2 * 50 / 5 + 2)
            * power
            * A
            / D
        ) / 50 + 2

        # Modificadores
        mod = 1.0

        # 1. Spread damage (doubles con >1 target)
        targets = int(field.get("targets", 1))
        if targets > 1:
            mod *= 0.75

        # Title case como TYPE_CHART ("Water", "Fire", …).
        # Debe ejecutarse antes de clima, STAB y efectividad.
        move_type = str(move.get("type", "")).strip().title()

        # 2. Weather
        weather = field.get("weather", "")
        if weather == "rain":
            if move_type == "Water":
                mod *= 1.5
            elif move_type == "Fire":
                mod *= 0.5
        elif weather == "sun":
            if move_type == "Fire":
                mod *= 1.5
            elif move_type == "Water":
                mod *= 0.5

        # 3. Screens
        if field.get("reflect_def") and is_physical:
            mod *= (
                2 / 3 if targets > 1 else 0.5
            )
        if (
            field.get("light_screen_def")
            and not is_physical
        ):
            mod *= (
                2 / 3 if targets > 1 else 0.5
            )

        # 4. STAB
        att_entry = None
        for entry in pokemon_master.values():
            if (
                str(entry.get("name", "")).lower()
                == att_slug.lower()
            ):
                att_entry = entry
                break

        if att_entry:
            att_types = [
                str(t).strip().title()
                for t in att_entry.get("types", [])
            ]
            if move_type in att_types:
                mod *= 1.5

        # 5. Type effectiveness
        def_entry = None
        for entry in pokemon_master.values():
            if (
                str(entry.get("name", "")).lower()
                == def_slug.lower()
            ):
                def_entry = entry
                break

        if def_entry:
            def_types = [
                str(t) for t in def_entry.get("types", [])
            ]
            eff = type_effectiveness(
                move_type, def_types
            )
            mod *= eff

        # 6. Burn
        if (
            field.get("attacker_burned")
            and is_physical
        ):
            mod *= 0.5

        # 7. Items del atacante
        if att_item == "Life Orb":
            mod *= 1.3
        elif att_item == "Choice Band" and is_physical:
            mod *= 1.5
        elif (
            att_item == "Choice Specs"
            and not is_physical
        ):
            mod *= 1.5

        # 8. Rolls: 16 valores (85..100)/100
        rolls = np.array(
            [(85 + i) / 100.0 for i in range(16)],
            dtype=np.float64,
        )
        damage = np.floor(base * mod * rolls)
        return np.divide(
            damage,
            np.float64(HP_def),
            dtype=np.float64,
        )

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Error en damage calc %s vs %s: %s",
            att_slug,
            def_slug,
            exc,
        )
        return np.zeros(16, dtype=np.float64)


__all__ = [
    "NATURE_MULT",
    "SPSpread",
    "stat_from_sp",
    "calc_all_stats",
    "type_effectiveness",
    "champions_damage_calc",
]
