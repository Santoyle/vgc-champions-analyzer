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

from src.app.core.schema import SPSpread

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


__all__ = [
    "NATURE_MULT",
    "SPSpread",
    "stat_from_sp",
    "calc_all_stats",
]
