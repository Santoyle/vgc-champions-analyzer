"""
Parser de logs de batallas Pokémon Showdown a filas normalizadas (``replay_turns``).

Este módulo recorre el protocolo línea a línea: ``GameState`` se actualiza evento a
evento como un **autómata de estado finito**, y ante acciones observables (`move`,
`switch`, etc.) emite registros compatibles con el esquema de ``replay_turns``. Esas
filas son la base de las métricas **MLWR**, **LRE** y **METI** del Bloque 15.

No ejecuta DuckDB ni I/O en ``parse_replay_log`` — solo texto de entrada → lista de dicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Eventos del log que nos interesan
EVENTS_OF_INTEREST = {
    "move",
    "switch",
    "drag",
    "faint",
    "-damage",
    "-heal",
    "-mega",
    "-weather",
    "-fieldstart",
    "-fieldend",
    "-sidestart",
    "-sideend",
    "turn",
    "win",
}

# Mapeo de weather strings del log a valores
# normalizados
WEATHER_MAP: dict[str, str] = {
    "raindance": "rain",
    "primordialsea": "rain",
    "sunnyday": "sun",
    "desolateland": "sun",
    "sandstorm": "sand",
    "snowscape": "snow",
    "hail": "snow",
    "none": "",
}


@dataclass
class GameState:
    """
    Estado del juego en un momento dado.
    Se actualiza evento a evento durante el parsing.
    """

    turn: int = 0
    weather: str = ""
    terrain: str = ""
    trick_room_active: bool = False
    reflect_p1: bool = False
    reflect_p2: bool = False
    light_screen_p1: bool = False
    light_screen_p2: bool = False
    tailwind_p1_turns: int = 0
    tailwind_p2_turns: int = 0
    mega_used_p1: bool = False
    mega_used_p2: bool = False
    ko_diff: int = 0  # KOs p1 - KOs p2
    # HP actual por slot: {"p1a": 1.0, "p1b": 1.0, ...}
    hp: dict[str, float] = field(
        default_factory=lambda: {
            "p1a": 1.0,
            "p1b": 1.0,
            "p2a": 1.0,
            "p2b": 1.0,
        }
    )
    # Pokémon activo por slot
    active: dict[str, str] = field(default_factory=dict)


def _parse_hp_pct(hp_str: str) -> float:
    """
    Parsea un string de HP del log a fracción [0,1].

    Formatos soportados:
    - "45/100"  → 0.45
    - "45/95"   → 0.4736...
    - "0 fnt"   → 0.0
    - "100"     → 1.0

    Args:
        hp_str: String de HP del log de Showdown.

    Returns:
        Fracción de HP en [0.0, 1.0].
    """
    hp_str = hp_str.strip().split()[0]
    if "/" in hp_str:
        parts = hp_str.split("/")
        try:
            current = float(parts[0])
            max_hp = float(parts[1])
            if max_hp == 0:
                return 0.0
            return min(1.0, current / max_hp)
        except (ValueError, IndexError):
            return 0.0
    try:
        val = float(hp_str)
        return min(1.0, val / 100.0)
    except ValueError:
        return 0.0


def _parse_slot(slot_str: str) -> tuple[str, str]:
    """
    Parsea un slot string del log a (player, slot).

    Ejemplos:
    - "p1a" → ("p1", "a")
    - "p2b" → ("p2", "b")
    - "p1a: Garchomp" → ("p1", "a")

    Args:
        slot_str: String de slot del log.

    Returns:
        Tupla (player, slot).
    """
    # Tomar solo la parte antes de ":"
    slot_part = slot_str.split(":")[0].strip()
    if len(slot_part) >= 3:
        return slot_part[:2], slot_part[2]
    return "p1", "a"


def _normalize_slug(name: str) -> str:
    """
    Normaliza un nombre de Pokémon a slug
    kebab-case lowercase.

    Ejemplos:
    - "Incineroar" → "incineroar"
    - "Rotom-W" → "rotom-w"
    - "Urshifu-Rapid-Strike" → "urshifu-rapid-strike"

    Args:
        name: Nombre del Pokémon del log.

    Returns:
        Slug normalizado.
    """
    # Tomar solo la parte antes de "," (formas)
    name = name.split(",")[0].strip()
    return name.lower().replace(" ", "-")


def _winner_key_from_showdown_log(lines: list[str]) -> str:
    """
    Deriva ``p1`` / ``p2`` / ``""`` desde ``|player|`` y ``|win|``.

    Compara el username de ``|win|...`` con los de ``|player|p1|`` y
    ``|player|p2|``. Si no coincide (empate u otro caso), retorna ``""``.
    """
    p1_user: str | None = None
    p2_user: str | None = None
    win_user: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        event = parts[1].lower()
        if event == "player" and len(parts) >= 4:
            side = parts[2].lower()
            name = parts[3].strip()
            if not name:
                continue
            if side == "p1":
                p1_user = name
            elif side == "p2":
                p2_user = name
        elif event == "win" and len(parts) >= 3:
            w = parts[2].strip()
            if w:
                win_user = w

    if not win_user:
        return ""

    wcf = win_user.casefold()
    if p1_user is not None and wcf == p1_user.casefold():
        return "p1"
    if p2_user is not None and wcf == p2_user.casefold():
        return "p2"
    return ""


def parse_replay_log(
    replay_id: str,
    regulation_id: str,
    raw_log: str,
    winner: str | None,
) -> list[dict[str, Any]]:
    """
    Parsea el log completo de un replay de Showdown
    y retorna una lista de filas para replay_turns.

    El parser es un autómata de estado finito que
    procesa el log línea a línea, actualizando el
    GameState y emitiendo filas cuando detecta
    eventos de acción (move, switch).

    Args:
        replay_id: ID único del replay.
        regulation_id: Regulación del replay.
        raw_log: Log completo del replay como string.
        winner: "p1" o "p2" o None si no hay ganador.

    Returns:
        Lista de dicts con el schema de replay_turns.
        Lista vacía si el log está vacío o es inválido.
    """
    if not raw_log or not raw_log.strip():
        return []

    lines = raw_log.strip().split("\n")
    winner_key = _winner_key_from_showdown_log(lines)
    state = GameState()
    rows: list[dict[str, Any]] = []
    action_idx = 0

    def _make_row(
        player: str,
        slot: str,
        move_slug: str | None = None,
        target_player: str | None = None,
        target_slot: str | None = None,
        hp_before: float | None = None,
        hp_after: float | None = None,
        ko_dealt: bool = False,
        ko_received: bool = False,
    ) -> dict[str, Any]:
        nonlocal action_idx
        slot_key = f"{player}{slot}"
        row = {
            "regulation_id": regulation_id,
            "replay_id": replay_id,
            "turn": state.turn,
            "action_idx": action_idx,
            "player": player,
            "slot": slot,
            "pokemon_slug": state.active.get(slot_key),
            "move_slug": move_slug,
            "target_player": target_player,
            "target_slot": target_slot,
            "hp_before_pct": hp_before,
            "hp_after_pct": hp_after,
            "ko_dealt": ko_dealt,
            "ko_received": ko_received,
            "weather": state.weather or None,
            "terrain": state.terrain or None,
            "trick_room_active": state.trick_room_active,
            "reflect_p1": state.reflect_p1,
            "reflect_p2": state.reflect_p2,
            "light_screen_p1": state.light_screen_p1,
            "light_screen_p2": state.light_screen_p2,
            "tailwind_p1_turns": state.tailwind_p1_turns,
            "tailwind_p2_turns": state.tailwind_p2_turns,
            "mega_used_p1": state.mega_used_p1,
            "mega_used_p2": state.mega_used_p2,
            "ko_diff": state.ko_diff,
            "winner": winner_key,
        }
        action_idx += 1
        return row

    # Buffer para hp_before del siguiente damage
    pending_move: dict[str, dict[str, Any]] = {}

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("|"):
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        event = parts[1].lower()

        # Nuevo turno
        if event == "turn":
            action_idx = 0
            try:
                state.turn = int(parts[2])
            except (IndexError, ValueError):
                pass
            # Decrementar tailwind turns
            if state.tailwind_p1_turns > 0:
                state.tailwind_p1_turns -= 1
            if state.tailwind_p2_turns > 0:
                state.tailwind_p2_turns -= 1

        # Switch / drag
        elif event in ("switch", "drag"):
            if len(parts) < 4:
                continue
            player, slot = _parse_slot(parts[2])
            pkm_name = parts[3].split(",")[0].strip()
            slug = _normalize_slug(pkm_name)
            slot_key = f"{player}{slot}"
            state.active[slot_key] = slug

            hp_after = 1.0
            if len(parts) >= 5:
                hp_after = _parse_hp_pct(parts[4])
            state.hp[slot_key] = hp_after

            rows.append(
                _make_row(
                    player=player,
                    slot=slot,
                    hp_before=hp_after,
                    hp_after=hp_after,
                )
            )

        # Move
        elif event == "move":
            if len(parts) < 4:
                continue
            player, slot = _parse_slot(parts[2])
            move_name = parts[3].strip()
            move_slug = _normalize_slug(move_name)

            target_player = None
            target_slot = None
            if len(parts) >= 5 and parts[4].strip():
                tp, ts = _parse_slot(parts[4])
                target_player = tp
                target_slot = ts

            slot_key = f"{player}{slot}"
            hp_before = state.hp.get(slot_key, 1.0)

            row = _make_row(
                player=player,
                slot=slot,
                move_slug=move_slug,
                target_player=target_player,
                target_slot=target_slot,
                hp_before=hp_before,
                hp_after=hp_before,
            )
            rows.append(row)
            # Guardar referencia para actualizar
            # hp_after cuando llegue el -damage
            pending_move[f"{player}{slot}"] = row

        # Damage
        elif event == "-damage":
            if len(parts) < 4:
                continue
            player, slot = _parse_slot(parts[2])
            slot_key = f"{player}{slot}"
            hp_before = state.hp.get(slot_key, 1.0)
            hp_after = _parse_hp_pct(parts[3])
            state.hp[slot_key] = hp_after

            # Actualizar hp_after en la fila del
            # move pendiente si existe
            if slot_key in pending_move:
                pending_move[slot_key]["hp_after_pct"] = hp_after

        # Heal
        elif event == "-heal":
            if len(parts) < 4:
                continue
            player, slot = _parse_slot(parts[2])
            slot_key = f"{player}{slot}"
            hp_after = _parse_hp_pct(parts[3])
            state.hp[slot_key] = hp_after

        # Faint (KO)
        elif event == "faint":
            if len(parts) < 3:
                continue
            player, slot = _parse_slot(parts[2])
            slot_key = f"{player}{slot}"
            state.hp[slot_key] = 0.0

            # Actualizar ko_diff
            if player == "p1":
                state.ko_diff -= 1
                # Buscar el move que causó el faint
                # en el jugador contrario
                for key in list(pending_move.keys()):
                    if key.startswith("p2"):
                        pending_move[key]["ko_dealt"] = True
                # Marcar ko_received en fila p1
                if slot_key in pending_move:
                    pending_move[slot_key]["ko_received"] = True
            else:
                state.ko_diff += 1
                for key in list(pending_move.keys()):
                    if key.startswith("p1"):
                        pending_move[key]["ko_dealt"] = True
                if slot_key in pending_move:
                    pending_move[slot_key]["ko_received"] = True

        # Weather
        elif event == "-weather":
            if len(parts) >= 3:
                weather_raw = parts[2].strip().lower()
                state.weather = WEATHER_MAP.get(
                    weather_raw, ""
                )

        # Field start/end (Trick Room, Terrain)
        elif event == "-fieldstart":
            if len(parts) >= 3:
                field_str = parts[2].lower()
                if "trick room" in field_str:
                    state.trick_room_active = True
                elif "grassy" in field_str:
                    state.terrain = "grassy"
                elif "electric" in field_str:
                    state.terrain = "electric"
                elif "psychic" in field_str:
                    state.terrain = "psychic"
                elif "misty" in field_str:
                    state.terrain = "misty"

        elif event == "-fieldend":
            if len(parts) >= 3:
                field_str = parts[2].lower()
                if "trick room" in field_str:
                    state.trick_room_active = False
                else:
                    state.terrain = ""

        # Side conditions (Reflect, Light Screen,
        # Tailwind)
        elif event == "-sidestart":
            if len(parts) >= 4:
                side = parts[2].strip().lower()
                move_str = parts[3].lower()
                is_p1 = "p1" in side
                if "reflect" in move_str:
                    if is_p1:
                        state.reflect_p1 = True
                    else:
                        state.reflect_p2 = True
                elif "light screen" in move_str:
                    if is_p1:
                        state.light_screen_p1 = True
                    else:
                        state.light_screen_p2 = True
                elif "tailwind" in move_str:
                    if is_p1:
                        state.tailwind_p1_turns = 4
                    else:
                        state.tailwind_p2_turns = 4

        elif event == "-sideend":
            if len(parts) >= 4:
                side = parts[2].strip().lower()
                move_str = parts[3].lower()
                is_p1 = "p1" in side
                if "reflect" in move_str:
                    if is_p1:
                        state.reflect_p1 = False
                    else:
                        state.reflect_p2 = False
                elif "light screen" in move_str:
                    if is_p1:
                        state.light_screen_p1 = False
                    else:
                        state.light_screen_p2 = False

        # Mega Evolution
        elif event == "-mega":
            if len(parts) >= 3:
                player, _ = _parse_slot(parts[2])
                if player == "p1":
                    state.mega_used_p1 = True
                else:
                    state.mega_used_p2 = True

        # Limpiar pending_move al final del turno
        # (el turno nuevo resetea el buffer)
        if event == "turn":
            pending_move.clear()

    return rows


def parse_replays_from_parquet(
    parquet_path: str,
    regulation_id: str,
) -> pd.DataFrame:
    """
    Lee un Parquet de replays y parsea todos los
    logs, retornando un DataFrame de replay_turns.

    Args:
        parquet_path: Path al archivo Parquet.
        regulation_id: ID de la regulación.

    Returns:
        DataFrame con el schema de replay_turns.
        DataFrame vacío si no hay logs válidos.
    """
    try:
        df_raw = pd.read_parquet(parquet_path)
    except Exception as exc:
        log.error(
            "Error leyendo Parquet %s: %s",
            parquet_path,
            exc,
        )
        return pd.DataFrame()

    if df_raw.empty:
        return pd.DataFrame()

    required_cols = {"replay_id", "raw_log", "winner"}
    if not required_cols.issubset(set(df_raw.columns)):
        log.warning(
            "Parquet %s no tiene columnas "
            "requeridas: %s",
            parquet_path,
            required_cols - set(df_raw.columns),
        )
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []
    n_parsed = 0
    n_failed = 0

    for _, row in df_raw.iterrows():
        replay_id = str(row.get("replay_id", ""))
        raw_log = str(row.get("raw_log", ""))
        winner_raw = str(row.get("winner", ""))

        # Normalizar winner a "p1" o "p2"
        if str(row.get("p1", "")) == winner_raw:
            winner = "p1"
        elif str(row.get("p2", "")) == winner_raw:
            winner = "p2"
        else:
            winner = (
                winner_raw[:2].lower() if winner_raw else ""
            )

        try:
            rows = parse_replay_log(
                replay_id=replay_id,
                regulation_id=regulation_id,
                raw_log=raw_log,
                winner=winner,
            )
            all_rows.extend(rows)
            n_parsed += 1
        except Exception as exc:
            log.debug(
                "Error parseando replay %s: %s",
                replay_id,
                exc,
            )
            n_failed += 1

    log.info(
        "Parseados %d/%d replays de %s "
        "(%d filas generadas, %d fallos)",
        n_parsed,
        n_parsed + n_failed,
        parquet_path,
        len(all_rows),
        n_failed,
    )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


__all__ = [
    "GameState",
    "parse_replay_log",
    "parse_replays_from_parquet",
    "WEATHER_MAP",
    "EVENTS_OF_INTEREST",
]
