"""
Scraper de resultados de torneos oficiales TPCi desde pokedata.ovh.

RK9.gg es la plataforma oficial de Play!Pokémon para resultados de
torneos presenciales (Regionals, Internationals, Worlds), pero su TOS
es restrictivo. pokedata.ovh es un mirror público de RK9 con TOS menos
restrictivo — se usa como fuente primaria en este scraper.

Torneos oficiales confirmados con Reg M-A (pendientes de resultados):
  - Indianapolis Regionals (29-31 mayo 2026)
  - NAIC — North America Internationals (12-14 junio 2026)

fetch_event_standings retorna lista vacía (no levanta excepción) para
torneos futuros sin resultados — un 404 o 403 es comportamiento normal
para un evento que aún no ha ocurrido o cuyos resultados no están
publicados.

El scraper usa solo stdlib + httpx (sin BeautifulSoup ni lxml) para
mantener el árbol de dependencias mínimo.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

POKEDATA_BASE = "https://www.pokedata.ovh"
POKEDATA_EVENTS_URL = f"{POKEDATA_BASE}/events/"
RK9_BASE = "https://rk9.gg"
REQUEST_DELAY_SEC = 1.0
USER_AGENT = (
    "vgc-champions-analyzer/0.1 "
    "rk9-pokedata-scraper "
    "(github.com/Santoyle/vgc-champions-analyzer)"
)


@dataclass
class OfficialEvent:
    """
    Evento oficial TPCi desde pokedata.ovh.

    Attributes:
        event_id: ID único del evento.
        name: Nombre del evento.
        event_type: Tipo ("regional", "international",
                    "worlds", "special", "unknown").
        regulation_id: ID de regulación del proyecto.
        event_date: Fecha del evento. None si futura
                    sin fecha confirmada.
        location: Ciudad/país del evento.
        num_players: Jugadores inscritos. 0 si desconocido.
        has_results: True si los resultados ya están disponibles.
        url: URL del evento en pokedata.ovh.
    """

    event_id: str
    name: str
    event_type: str
    regulation_id: str
    event_date: date | None
    location: str
    num_players: int
    has_results: bool
    url: str


@dataclass
class OfficialStanding:
    """
    Resultado de un jugador en un torneo oficial.

    Attributes:
        event_id: ID del evento.
        player_name: Nombre del jugador.
        player_id: ID del jugador en Play!Pokémon si disponible.
        final_rank: Posición final.
        record: Record W-L-T.
        resistance: Porcentaje de resistencia.
        country: País del jugador si disponible.
    """

    event_id: str
    player_name: str
    player_id: str | None
    final_rank: int
    record: str | None = None
    resistance: float | None = None
    country: str | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_html(
    client: httpx.Client,
    url: str,
) -> str:
    """
    Descarga HTML con retry automático.

    Args:
        client: Cliente httpx configurado.
        url: URL a descargar.

    Returns:
        Texto HTML.

    Raises:
        httpx.HTTPStatusError: En errores HTTP.
    """
    response = client.get(url, timeout=30.0)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return response.text


def _classify_event_type(name: str) -> str:
    """
    Clasifica el tipo de evento por nombre.

    Args:
        name: Nombre del evento (cualquier capitalización).

    Returns:
        Tipo: "regional", "international", "worlds",
        "special", o "unknown".
    """
    name_lower = name.lower()
    if any(w in name_lower for w in ("regional", "regionals")):
        return "regional"
    if any(w in name_lower for w in ("international", "naic", "euic", "laic", "ocic")):
        return "international"
    if any(w in name_lower for w in ("worlds", "world championship")):
        return "worlds"
    if any(w in name_lower for w in ("special", "challenge", "premier")):
        return "special"
    return "unknown"


def _extract_events_from_html(
    html: str,
    regulation_id: str,
) -> list[OfficialEvent]:
    """
    Extrae listado de eventos TPCi desde HTML de pokedata.ovh
    usando regex.

    Busca links a eventos y metadatos asociados. Tolerante a
    variaciones de layout — si no puede parsear un campo usa
    valores por defecto.

    Args:
        html: HTML de la página de eventos.
        regulation_id: Para etiquetar los eventos.

    Returns:
        Lista de OfficialEvent extraídos. Lista vacía si el
        layout no es reconocible.
    """
    events: list[OfficialEvent] = []
    seen_ids: set[str] = set()

    _NAV_SLUGS = frozenset(
        {"standings", "results", "info", "players", "teams", "bracket"}
    )

    event_pattern = re.compile(
        r'href=["\'][^"\']*?/events/'
        r'([a-zA-Z0-9\-_]+)/?["\'][^>]*>'
        r'([^<]+)</a>',
        re.IGNORECASE,
    )

    for match in event_pattern.finditer(html):
        eid = match.group(1).strip()
        name = match.group(2).strip()

        if not eid or not name:
            continue
        if eid in seen_ids:
            continue
        if eid in _NAV_SLUGS:
            continue

        seen_ids.add(eid)
        events.append(
            OfficialEvent(
                event_id=eid,
                name=name,
                event_type=_classify_event_type(name),
                regulation_id=regulation_id,
                event_date=None,
                location="",
                num_players=0,
                has_results=False,
                url=f"{POKEDATA_BASE}/events/{eid}/",
            )
        )

    log.debug(
        "Extraídos %d eventos de HTML para %s",
        len(events),
        regulation_id,
    )
    return events


def _extract_standings_from_html(
    html: str,
    event_id: str,
) -> list[OfficialStanding]:
    """
    Extrae standings de un evento desde HTML de pokedata.ovh.

    Args:
        html: HTML de la página de standings.
        event_id: ID del evento.

    Returns:
        Lista de OfficialStanding. Lista vacía si no hay
        resultados o el formato no es reconocido.
    """
    standings: list[OfficialStanding] = []

    row_pattern = re.compile(
        r"<tr[^>]*>\s*"
        r"(?:<td[^>]*>(\d+)</td>)?\s*"
        r"<td[^>]*>([^<]{2,50})</td>",
        re.DOTALL | re.IGNORECASE,
    )

    for match in row_pattern.finditer(html):
        try:
            rank_str = match.group(1)
            player = match.group(2).strip()

            if not player or len(player) > 50:
                continue

            rank = int(rank_str) if rank_str else 0
            if rank == 0:
                continue

            standings.append(
                OfficialStanding(
                    event_id=event_id,
                    player_name=player,
                    player_id=None,
                    final_rank=rank,
                )
            )
        except (ValueError, AttributeError):
            continue

    log.info(
        "Standings extraídos: %d jugadores del evento %s",
        len(standings),
        event_id,
    )
    return standings


def fetch_official_events(
    regulation_id: str,
) -> list[OfficialEvent]:
    """
    Obtiene listado de eventos oficiales TPCi desde pokedata.ovh
    para una regulación.

    Args:
        regulation_id: ID de la regulación del proyecto.

    Returns:
        Lista de OfficialEvent.
        Lista vacía si hay error o no hay eventos.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,*/*",
    }

    try:
        with httpx.Client(headers=headers) as client:
            html = _fetch_html(client, POKEDATA_EVENTS_URL)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Error scrapeando pokedata.ovh para %s: %s",
            regulation_id,
            exc,
        )
        return []

    events = _extract_events_from_html(html, regulation_id)

    log.info(
        "pokedata.ovh: %d eventos encontrados para %s",
        len(events),
        regulation_id,
    )
    return events


def fetch_event_standings(
    event_id: str,
    regulation_id: str,
) -> list[OfficialStanding]:
    """
    Descarga standings de un evento oficial desde pokedata.ovh.

    Retorna lista vacía (no levanta excepción) si el evento no tiene
    resultados todavía — esto es normal para torneos futuros como
    Indianapolis Regionals (29-31 mayo 2026) y NAIC (12-14 jun 2026).

    Args:
        event_id: ID del evento en pokedata.ovh.
        regulation_id: Para logging.

    Returns:
        Lista de OfficialStanding.
        Lista vacía si no hay resultados aún o hay error.
    """
    url = f"{POKEDATA_BASE}/events/{event_id}/standings/"
    headers = {"User-Agent": USER_AGENT}

    try:
        with httpx.Client(headers=headers) as client:
            html = _fetch_html(client, url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (403, 404):
            log.debug(
                "Evento %s sin standings aún (%d) — torneo futuro o sin publicar.",
                event_id,
                exc.response.status_code,
            )
        else:
            log.warning(
                "Error HTTP %d en standings de evento %s (%s)",
                exc.response.status_code,
                event_id,
                regulation_id,
            )
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Error descargando standings de evento %s: %s",
            event_id,
            exc,
        )
        return []

    return _extract_standings_from_html(html, event_id)


def events_to_records(
    events: list[OfficialEvent],
) -> list[dict[str, Any]]:
    """
    Convierte lista de OfficialEvent a dicts planos para escritura
    en Parquet.

    event_date se serializa como string ISO (YYYY-MM-DD) o None
    para compatibilidad con Parquet sin tipos fecha complejos.

    Columnas: event_id, name, event_type, regulation_id,
    event_date, location, num_players, has_results, url.

    Args:
        events: Lista de OfficialEvent.

    Returns:
        Lista de dicts con un registro por evento.
    """
    return [
        {
            "event_id": e.event_id,
            "name": e.name,
            "event_type": e.event_type,
            "regulation_id": e.regulation_id,
            "event_date": e.event_date.isoformat() if e.event_date else None,
            "location": e.location,
            "num_players": e.num_players,
            "has_results": e.has_results,
            "url": e.url,
        }
        for e in events
    ]


def standings_to_records(
    standings: list[OfficialStanding],
) -> list[dict[str, Any]]:
    """
    Convierte lista de OfficialStanding a dicts planos para
    escritura en Parquet.

    Columnas: event_id, player_name, player_id, final_rank,
    record, resistance, country.

    Args:
        standings: Lista de OfficialStanding.

    Returns:
        Lista de dicts con un registro por jugador.
    """
    return [
        {
            "event_id": s.event_id,
            "player_name": s.player_name,
            "player_id": s.player_id,
            "final_rank": s.final_rank,
            "record": s.record,
            "resistance": s.resistance,
            "country": s.country,
        }
        for s in standings
    ]


__all__ = [
    "OfficialEvent",
    "OfficialStanding",
    "fetch_official_events",
    "fetch_event_standings",
    "events_to_records",
    "standings_to_records",
    "POKEDATA_BASE",
]
