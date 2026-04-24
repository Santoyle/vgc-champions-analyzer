"""
Scraper de replays de Pokémon Showdown para la regulación activa.

Descarga y parsea replays competitivos de Pokémon Showdown usando
la API pública de replay.pokemonshowdown.com. El format_slug se
obtiene siempre de REGULATION_TO_FORMAT (smogon_chaos.py) para
mantener el mapeo centralizado — nunca se hardcodea aquí.

Flujo de uso:
    1. fetch_recent_replays(regulation_id, max_replays, min_rating)
       orquesta búsqueda paginada + descarga individual.
    2. replays_to_records(replays) convierte a dicts planos para
       escritura en Parquet.

El campo raw_log está disponible en ParsedReplay en memoria para
cálculos avanzados (IIT, TPI, correlaciones) pero no se persiste
en Parquet mediante replays_to_records para mantener el archivo
compacto. Los pipelines pueden accederlo antes de la serialización.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.app.data.scrapers.smogon_chaos import REGULATION_TO_FORMAT

log = logging.getLogger(__name__)

SHOWDOWN_REPLAY_BASE = "https://replay.pokemonshowdown.com"
SHOWDOWN_SEARCH_URL = f"{SHOWDOWN_REPLAY_BASE}/search.json"
MIN_RATING = 1500
REQUEST_DELAY_SEC = 0.5
MAX_PAGES = 10

USER_AGENT = (
    "vgc-champions-analyzer/0.1 "
    "showdown-replay-scraper "
    "(github.com/Santoyle/vgc-champions-analyzer)"
)


@dataclass
class ReplayMetadata:
    """
    Metadatos de un replay de Showdown sin el log.

    Attributes:
        replay_id: ID único del replay.
        format_slug: Formato de batalla.
        p1: Nombre del jugador 1.
        p2: Nombre del jugador 2.
        rating: Rating promedio de la batalla.
        upload_time: Timestamp Unix de subida.
    """

    replay_id: str
    format_slug: str
    p1: str
    p2: str
    rating: int
    upload_time: int


@dataclass
class ParsedReplay:
    """
    Replay parseado con equipos y resultado.

    Attributes:
        replay_id: ID único del replay.
        regulation_id: ID de regulación del proyecto.
        format_slug: Formato de batalla.
        p1: Nombre jugador 1.
        p2: Nombre jugador 2.
        rating: Rating de la batalla.
        upload_time: Timestamp Unix.
        team_p1: Lista de Pokémon del equipo p1.
        team_p2: Lista de Pokémon del equipo p2.
        winner: "p1", "p2", o None si no se determinó.
        raw_log: Log completo del replay (texto).
    """

    replay_id: str
    regulation_id: str
    format_slug: str
    p1: str
    p2: str
    rating: int
    upload_time: int
    team_p1: list[str] = field(default_factory=list)
    team_p2: list[str] = field(default_factory=list)
    winner: str | None = None
    raw_log: str = ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_json(
    client: httpx.Client,
    url: str,
    params: dict[str, str | int] | None = None,
) -> dict[str, Any]:
    """
    Descarga JSON desde una URL con retry automático.

    Args:
        client: Cliente httpx configurado.
        url: URL a descargar.
        params: Query parameters opcionales.

    Returns:
        Dict parseado del JSON.

    Raises:
        httpx.HTTPStatusError: En errores HTTP.
    """
    response = client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return response.json()  # type: ignore[no-any-return]


def _parse_battle_log(
    log_text: str,
    regulation_id: str,
    replay_id: str,
) -> tuple[list[str], list[str], str | None]:
    """
    Parsea el battle log de un replay para extraer
    equipos y resultado.

    Extrae:
    - Pokémon del equipo p1 (líneas |poke|p1|...)
    - Pokémon del equipo p2 (líneas |poke|p2|...)
    - Ganador (línea |win|...)

    Limpia los nombres de Pokémon eliminando
    suffixes de forma (ej: "Incineroar, L50" →
    "Incineroar").

    Args:
        log_text: Texto completo del battle log.
        regulation_id: Para logging.
        replay_id: Para logging.

    Returns:
        Tupla (team_p1, team_p2, winner).
        winner es el nombre del jugador ganador
        tal como aparece en el log, o None.
    """
    team_p1: list[str] = []
    team_p2: list[str] = []
    winner: str | None = None

    for line in log_text.splitlines():
        parts = line.split("|")

        if len(parts) >= 4 and parts[1] == "poke":
            player = parts[2]
            # "Incineroar, L50, M" → "Incineroar"
            pokemon_name = parts[3].split(",")[0].strip()
            if player == "p1" and pokemon_name:
                team_p1.append(pokemon_name)
            elif player == "p2" and pokemon_name:
                team_p2.append(pokemon_name)

        elif len(parts) >= 3 and parts[1] == "win":
            winner = parts[2].strip() or None

    log.debug(
        "Replay %s (%s): p1=%d pkm, p2=%d pkm, winner=%s",
        replay_id,
        regulation_id,
        len(team_p1),
        len(team_p2),
        winner,
    )
    return team_p1, team_p2, winner


def fetch_replay_search(
    regulation_id: str,
    page: int = 1,
    min_rating: int = MIN_RATING,
) -> list[ReplayMetadata]:
    """
    Busca replays disponibles para una regulación
    en Pokémon Showdown.

    Filtra por rating mínimo para quedarse con
    batallas competitivas relevantes.

    Args:
        regulation_id: ID de la regulación del proyecto.
                       Debe estar en REGULATION_TO_FORMAT.
        page: Número de página (1-indexed).
        min_rating: Rating mínimo para incluir replay.

    Returns:
        Lista de ReplayMetadata.
        Lista vacía si la regulación no está mapeada
        o si hay error de red.
    """
    format_slug = REGULATION_TO_FORMAT.get(regulation_id)
    if format_slug is None:
        log.warning(
            "regulation_id '%s' no tiene format_slug. "
            "IDs conocidos: %s",
            regulation_id,
            list(REGULATION_TO_FORMAT.keys()),
        )
        return []

    headers = {"User-Agent": USER_AGENT}

    try:
        with httpx.Client(headers=headers) as client:
            data = _fetch_json(
                client,
                SHOWDOWN_SEARCH_URL,
                params={"format": format_slug, "page": page},
            )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Error buscando replays para %s page %d: %s",
            regulation_id,
            page,
            exc,
        )
        return []

    battles = data if isinstance(data, list) else data.get("battles", [])
    if not isinstance(battles, list):
        battles = []

    results: list[ReplayMetadata] = []
    for battle in battles:
        if not isinstance(battle, dict):
            continue
        rating = int(battle.get("rating", 0))
        if rating < min_rating:
            continue
        results.append(
            ReplayMetadata(
                replay_id=str(battle.get("id", "")),
                format_slug=format_slug,
                p1=str(battle.get("p1", "")),
                p2=str(battle.get("p2", "")),
                rating=rating,
                upload_time=int(battle.get("uploadtime", 0)),
            )
        )

    log.info(
        "Showdown search: %d replays (rating>=%d) para %s page %d",
        len(results),
        min_rating,
        regulation_id,
        page,
    )
    return results


def fetch_single_replay(
    replay_id: str,
    regulation_id: str,
) -> ParsedReplay | None:
    """
    Descarga y parsea un replay individual de Showdown.

    Args:
        replay_id: ID del replay (ej: "gen9champs-123").
        regulation_id: ID de la regulación para etiquetar.

    Returns:
        ParsedReplay con equipos y resultado.
        None si el replay no existe o hubo error.
    """
    url = f"{SHOWDOWN_REPLAY_BASE}/{replay_id}.json"
    headers = {"User-Agent": USER_AGENT}

    try:
        with httpx.Client(headers=headers) as client:
            data = _fetch_json(client, url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            log.debug("Replay %s no encontrado (404)", replay_id)
        else:
            log.warning(
                "Error HTTP %d descargando replay %s",
                exc.response.status_code,
                replay_id,
            )
        return None
    except Exception as exc:  # noqa: BLE001
        log.error("Error descargando replay %s: %s", replay_id, exc)
        return None

    log_text = str(data.get("log", ""))
    format_slug = str(data.get("format", ""))
    p1 = str(data.get("p1", ""))
    p2 = str(data.get("p2", ""))
    rating = int(data.get("rating", 0))
    upload_time = int(data.get("uploadtime", 0))

    team_p1, team_p2, winner = _parse_battle_log(
        log_text, regulation_id, replay_id
    )

    return ParsedReplay(
        replay_id=replay_id,
        regulation_id=regulation_id,
        format_slug=format_slug,
        p1=p1,
        p2=p2,
        rating=rating,
        upload_time=upload_time,
        team_p1=team_p1,
        team_p2=team_p2,
        winner=winner,
        raw_log=log_text,
    )


def fetch_recent_replays(
    regulation_id: str,
    max_replays: int = 100,
    min_rating: int = MIN_RATING,
) -> list[ParsedReplay]:
    """
    Descarga y parsea los replays más recientes de
    Showdown para una regulación.

    Pagina automáticamente hasta obtener max_replays
    o agotar páginas disponibles (MAX_PAGES).

    Args:
        regulation_id: ID de la regulación.
        max_replays: Máximo de replays a descargar.
        min_rating: Rating mínimo para incluir.

    Returns:
        Lista de ParsedReplay parseados.
        Lista vacía si no hay datos o hay error.
    """
    all_metadata: list[ReplayMetadata] = []
    page = 1

    while len(all_metadata) < max_replays and page <= MAX_PAGES:
        batch = fetch_replay_search(regulation_id, page, min_rating)
        if not batch:
            break
        all_metadata.extend(batch)
        page += 1

    all_metadata = all_metadata[:max_replays]

    log.info(
        "Descargando %d replays para %s...",
        len(all_metadata),
        regulation_id,
    )

    parsed: list[ParsedReplay] = []
    for meta in all_metadata:
        replay = fetch_single_replay(meta.replay_id, regulation_id)
        if replay is not None:
            parsed.append(replay)

    log.info(
        "Replays parseados: %d/%d para %s",
        len(parsed),
        len(all_metadata),
        regulation_id,
    )
    return parsed


def replays_to_records(
    replays: list[ParsedReplay],
) -> list[dict[str, Any]]:
    """
    Convierte lista de ParsedReplay a dicts planos
    para escritura en Parquet.

    Los equipos se serializan como JSON strings
    para compatibilidad con esquemas tabulares.
    raw_log no se incluye para mantener el Parquet
    compacto.

    Columnas: regulation_id, replay_id, format_slug,
    p1, p2, rating, upload_time, team_p1_json,
    team_p2_json, winner.

    Args:
        replays: Lista de ParsedReplay a convertir.

    Returns:
        Lista de dicts con un registro por replay.
    """
    return [
        {
            "regulation_id": r.regulation_id,
            "replay_id": r.replay_id,
            "format_slug": r.format_slug,
            "p1": r.p1,
            "p2": r.p2,
            "rating": r.rating,
            "upload_time": r.upload_time,
            "team_p1_json": json.dumps(r.team_p1),
            "team_p2_json": json.dumps(r.team_p2),
            "winner": r.winner,
        }
        for r in replays
    ]


__all__ = [
    "ReplayMetadata",
    "ParsedReplay",
    "fetch_replay_search",
    "fetch_single_replay",
    "fetch_recent_replays",
    "replays_to_records",
    "SHOWDOWN_SEARCH_URL",
]
