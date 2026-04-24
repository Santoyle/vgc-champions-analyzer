"""
Parser de paste Showdown al cromosoma canónico del proyecto.

Convierte texto paste de Pokémon Showdown (formato estándar de equipos
competitivos) a estructuras de datos tipadas (ParsedSlot, ParsedTeam)
listas para persistencia en Parquet o display en la UI.

El parser es PERMISIVO: prefiere retornar datos parciales con
advertencias antes que fallar completamente. Un paste con 5 Pokémon
válidos y 1 bloque inválido retorna 5 slots + 1 warning en
parse_warnings — nunca levanta excepción.

mega_capable se detecta por el sufijo del ítem usando
_MEGA_STONE_PATTERN (ej: "Incineroite", "Charizardite X") sin
necesitar la lista de Mega Stones de la regulación activa.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_STAT_ALIASES: dict[str, str] = {
    "hp": "hp",
    "atk": "atk",
    "attack": "atk",
    "def": "def",
    "defense": "def",
    "defence": "def",
    "spa": "spa",
    "spatk": "spa",
    "special attack": "spa",
    "spd": "spd",
    "spdef": "spd",
    "special defense": "spd",
    "special defence": "spd",
    "spe": "spe",
    "speed": "spe",
}

_MEGA_STONE_PATTERN = re.compile(
    r"(?:ite|onite|ardite|azite|ite\s*X|ite\s*Y)$",
    re.IGNORECASE,
)


@dataclass
class ParsedSlot:
    """
    Un slot de equipo parseado desde paste Showdown.

    Representa un Pokémon con toda su configuración competitiva.
    Es el bloque de construcción del cromosoma canónico.

    Attributes:
        species: Nombre de la especie (sin nickname).
        nickname: Apodo si tiene, None si no.
        item: Ítem equipado. None si no especificado.
        ability: Habilidad. None si no especificada.
        level: Nivel (default 50 en VGC).
        tera_type: Tipo Tera. None si no especificado.
        nature: Naturaleza. None si no especificada.
        evs: Dict {stat: value} con los EVs asignados.
             Stats: "hp", "atk", "def", "spa", "spd", "spe".
             Valor 0 si no especificado.
        ivs: Dict {stat: value} con los IVs. Default 31 para todos.
        moves: Lista de hasta 4 moves.
        is_shiny: True si es Shiny.
        gender: "M", "F", o None.
        mega_capable: True si el item es una Mega Stone.
    """

    species: str
    nickname: str | None = None
    item: str | None = None
    ability: str | None = None
    level: int = 50
    tera_type: str | None = None
    nature: str | None = None
    evs: dict[str, int] = field(
        default_factory=lambda: {
            "hp": 0,
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
        }
    )
    ivs: dict[str, int] = field(
        default_factory=lambda: {
            "hp": 31,
            "atk": 31,
            "def": 31,
            "spa": 31,
            "spd": 31,
            "spe": 31,
        }
    )
    moves: list[str] = field(default_factory=list)
    is_shiny: bool = False
    gender: str | None = None
    mega_capable: bool = False


@dataclass
class ParsedTeam:
    """
    Un equipo completo parseado desde paste Showdown.

    Attributes:
        slots: Lista de hasta 6 ParsedSlot.
        raw_paste: Texto original del paste.
        player_name: Nombre del jugador si estaba en el paste.
        parse_warnings: Lista de advertencias durante el parsing.
    """

    slots: list[ParsedSlot] = field(default_factory=list)
    raw_paste: str = ""
    player_name: str | None = None
    parse_warnings: list[str] = field(default_factory=list)


def _parse_evs_ivs_line(line: str) -> dict[str, int]:
    """
    Parsea una línea de EVs o IVs de Showdown.

    Formato: "252 HP / 4 Atk / 252 Def"
    o: "0 Atk / 0 SpA"

    Args:
        line: Contenido después de "EVs:" o "IVs:".

    Returns:
        Dict {stat_key: value} con los valores encontrados.
        Stats no mencionadas no se incluyen.
    """
    result: dict[str, int] = {}
    for part in line.split("/"):
        part = part.strip()
        match = re.match(r"^(\d+)\s+(.+)$", part)
        if not match:
            continue
        try:
            value = int(match.group(1))
            stat_key = _STAT_ALIASES.get(match.group(2).strip().lower())
            if stat_key:
                result[stat_key] = value
        except (ValueError, KeyError):
            continue
    return result


def _parse_first_line(
    line: str,
) -> tuple[str, str | None, str | None, str | None]:
    """
    Parsea la primera línea de un bloque Pokémon.

    Formato soportado:
      "Nickname (Species) (Gender) @ Item"
      "Species (Gender) @ Item"
      "Species @ Item"
      "Species"

    Args:
        line: Primera línea del bloque Pokémon.

    Returns:
        Tupla (species, nickname, gender, item).
        nickname es None si no hay apodo.
        gender es "M", "F", o None.
        item es None si no hay @ en la línea.
    """
    nickname: str | None = None
    gender: str | None = None
    item: str | None = None

    # Separar ítem
    if " @ " in line:
        name_part, item_raw = line.split(" @ ", 1)
        item = item_raw.strip() or None
    else:
        name_part = line

    name_part = name_part.strip()

    # Extraer género si está entre paréntesis al final: "(M)" o "(F)"
    gender_match = re.search(r"\(([MF])\)\s*$", name_part)
    if gender_match:
        gender = gender_match.group(1)
        name_part = name_part[: gender_match.start()].strip()

    # Extraer especie si hay nickname: "Nickname (Species)"
    species_match = re.search(r"\(([^)]+)\)\s*$", name_part)
    if species_match:
        nickname = name_part[: species_match.start()].strip() or None
        species = species_match.group(1).strip()
    else:
        species = name_part.strip()

    return species, nickname, gender, item


def parse_slot(block: str) -> ParsedSlot | None:
    """
    Parsea un bloque de texto Showdown en un ParsedSlot.

    Un bloque es el texto de un solo Pokémon, sin líneas en blanco
    intermedias.

    Args:
        block: Texto del bloque Pokémon.

    Returns:
        ParsedSlot si el parsing fue exitoso.
        None si el bloque está vacío o es inválido.
    """
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    species, nickname, gender, item = _parse_first_line(lines[0])
    if not species:
        return None

    mega_capable = bool(item and _MEGA_STONE_PATTERN.search(item))

    slot = ParsedSlot(
        species=species,
        nickname=nickname,
        item=item,
        gender=gender,
        mega_capable=mega_capable,
    )

    for line in lines[1:]:
        line_lower = line.lower()

        if line_lower.startswith("ability:"):
            slot.ability = line[8:].strip() or None

        elif line_lower.startswith("level:"):
            try:
                slot.level = int(line[6:].strip())
            except ValueError:
                pass

        elif line_lower.startswith("tera type:"):
            slot.tera_type = line[10:].strip() or None

        elif line_lower.startswith("shiny:"):
            slot.is_shiny = "yes" in line_lower

        elif line_lower.startswith("evs:"):
            slot.evs.update(_parse_evs_ivs_line(line[4:].strip()))

        elif line_lower.startswith("ivs:"):
            slot.ivs.update(_parse_evs_ivs_line(line[4:].strip()))

        elif line_lower.endswith("nature"):
            nature_str = line.lower().replace("nature", "").strip()
            if nature_str:
                slot.nature = nature_str.capitalize()

        elif line.startswith("- "):
            move = line[2:].strip()
            if move and len(slot.moves) < 4:
                slot.moves.append(move)

    return slot


def parse_paste(paste_text: str) -> ParsedTeam:
    """
    Parsea un paste completo de Pokémon Showdown.

    Divide el paste en bloques por líneas en blanco y parsea cada
    bloque como un ParsedSlot. Siempre retorna un ParsedTeam — nunca
    levanta excepción. Errores de parsing van a parse_warnings.

    Maneja:
    - Pastes con nombre de jugador al inicio
    - Pastes con 1 a 6 Pokémon
    - Bloques vacíos o inválidos (se ignoran con warning)
    - Más de 6 Pokémon (se trunca con warning)

    Args:
        paste_text: Texto completo del paste.

    Returns:
        ParsedTeam con los slots parseados. Si el paste es inválido,
        retorna team vacío con advertencia en parse_warnings.
    """
    team = ParsedTeam(raw_paste=paste_text)

    if not paste_text or not paste_text.strip():
        team.parse_warnings.append("Paste vacío o solo espacios")
        return team

    raw_blocks = re.split(r"\n\s*\n", paste_text.strip())
    player_name_detected = False

    for raw_block in raw_blocks:
        block = raw_block.strip()
        if not block:
            continue

        first_line = block.splitlines()[0].strip()

        # Bloque de una sola línea sin @ ni move → posible nombre de jugador
        if (
            not player_name_detected
            and "@" not in first_line
            and not first_line.startswith("-")
            and "\n" not in block
        ):
            team.player_name = first_line
            player_name_detected = True
            continue

        slot = parse_slot(block)
        if slot is None:
            team.parse_warnings.append(
                f"Bloque no parseado: {block[:50]}..."
            )
            continue

        if len(team.slots) >= 6:
            team.parse_warnings.append(
                f"Paste con más de 6 Pokémon — ignorando: {slot.species}"
            )
            continue

        team.slots.append(slot)

    if not team.slots:
        team.parse_warnings.append(
            "No se pudo parsear ningún Pokémon del paste"
        )

    log.debug(
        "Paste parseado: %d slots, %d warnings",
        len(team.slots),
        len(team.parse_warnings),
    )
    return team


def team_to_records(
    team: ParsedTeam,
    regulation_id: str,
    source: str = "manual",
) -> list[dict[str, Any]]:
    """
    Convierte un ParsedTeam a lista de dicts planos para escritura en
    Parquet o display en UI.

    Serializa moves, evs e ivs como JSON strings para compatibilidad
    con esquemas tabulares.

    Columnas: regulation_id, source, player_name, slot_idx, species,
    nickname, item, ability, level, tera_type, nature, evs_json,
    ivs_json, moves_json, is_shiny, gender, mega_capable.

    Args:
        team: ParsedTeam a convertir.
        regulation_id: ID de la regulación activa.
        source: Origen del paste ("manual", "limitless", "vgcpastes").

    Returns:
        Lista de dicts con un registro por slot.
    """
    return [
        {
            "regulation_id": regulation_id,
            "source": source,
            "player_name": team.player_name,
            "slot_idx": idx,
            "species": slot.species,
            "nickname": slot.nickname,
            "item": slot.item,
            "ability": slot.ability,
            "level": slot.level,
            "tera_type": slot.tera_type,
            "nature": slot.nature,
            "evs_json": json.dumps(slot.evs),
            "ivs_json": json.dumps(slot.ivs),
            "moves_json": json.dumps(slot.moves),
            "is_shiny": slot.is_shiny,
            "gender": slot.gender,
            "mega_capable": slot.mega_capable,
        }
        for idx, slot in enumerate(team.slots)
    ]


__all__ = [
    "ParsedSlot",
    "ParsedTeam",
    "parse_slot",
    "parse_paste",
    "team_to_records",
]
