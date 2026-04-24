"""
Scraper best-effort para Limitless VGC (limitlesstcg.com).

Limitless aloja torneos oficiales y comunitarios de Pokémon VGC y
Champions. No tiene API pública oficial — los datos se obtienen
scrapeando HTML usando solo stdlib + httpx (sin BeautifulSoup ni
lxml para evitar dependencias extra).

Función crítica: count_recent_tournaments(reg_id, days) es usada
por get_active_regulation() en regulation_active.py para el
Escenario B (ventana de transición entre regulaciones). Esta función
SIEMPRE retorna int y nunca levanta excepción — cualquier error de
red o parsing devuelve 0.

Las demás funciones son best-effort: retornan listas vacías ante
cualquier error de red o cambio de layout en el HTML de Limitless.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
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

LIMITLESS_BASE = "https://limitlesstcg.com"
LIMITLESS_TOURNAMENTS_URL = f"{LIMITLESS_BASE}/tournaments"
REQUEST_DELAY_SEC = 1.0
USER_AGENT = (
    "vgc-champions-analyzer/0.1 "
    "limitless-scraper "
    "(github.com/Santoyle/vgc-champions-analyzer)"
)


@dataclass
class TournamentSummary:
    """
    Resumen de un torneo de Limitless VGC.

    Attributes:
        tournament_id: ID único del torneo en Limitless.
        name: Nombre del torneo.
        format_name: Nombre del formato (ej: "VGC 2026").
        regulation_id: ID de regulación del proyecto.
        date: Fecha del torneo.
        num_players: Número de jugadores.
        url: URL del torneo en Limitless.
    """

    tournament_id: str
    name: str
    format_name: str
    regulation_id: str
    date: date
    num_players: int
    url: str


@dataclass
class PlayerResult:
    """
    Resultado de un jugador en un torneo.

    Attributes:
        tournament_id: ID del torneo.
        player_name: Nombre del jugador.
        final_rank: Posición final.
        record: Record W-L-T si disponible.
        team_paste: Paste Showdown del equipo si disponible.
    """

    tournament_id: str
    player_name: str
    final_rank: int
    record: str | None = None
    team_paste: str | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_html(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None = None,
) -> str:
    """
    Descarga HTML de una URL con retry automático.

    Args:
        client: Cliente httpx configurado.
        url: URL a descargar.
        params: Query parameters opcionales.

    Returns:
        Texto HTML de la respuesta.

    Raises:
        httpx.HTTPStatusError: En errores HTTP.
    """
    response = client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return response.text


def _extract_tournaments_from_html(
    html: str,
    regulation_id: str,
) -> list[TournamentSummary]:
    """
    Extrae listado de torneos desde el HTML de Limitless usando regex.

    Busca patrones de torneos en el HTML:
    - Links a torneos con ID numérico o alfanumérico
    - Nombres de torneos
    - Fechas (si están disponibles)
    - Número de jugadores

    Es tolerante a cambios de layout — si no puede parsear un campo,
    usa valores por defecto.

    Args:
        html: HTML de la página de torneos.
        regulation_id: Para etiquetar los torneos.

    Returns:
        Lista de TournamentSummary extraídos.
        Lista vacía si el layout no es reconocible.
    """
    tournaments: list[TournamentSummary] = []

    tournament_block_pattern = re.compile(
        r'href=["\'][^"\']*?/tournaments/'
        r'([a-zA-Z0-9\-_]+)["\'][^>]*>'
        r'([^<]+)</a>',
        re.IGNORECASE,
    )

    # Slugs de navegación que no son IDs reales de torneos
    _NAV_SLUGS = frozenset(
        {"standings", "results", "decklists", "info", "teams", "bracket"}
    )

    seen_ids: set[str] = set()
    for match in tournament_block_pattern.finditer(html):
        tid = match.group(1).strip()
        name = match.group(2).strip()

        if not tid or not name:
            continue
        if tid in seen_ids:
            continue
        if tid in _NAV_SLUGS:
            continue

        seen_ids.add(tid)
        tournaments.append(
            TournamentSummary(
                tournament_id=tid,
                name=name,
                format_name="VGC Champions",
                regulation_id=regulation_id,
                date=date.today(),
                num_players=0,
                url=f"{LIMITLESS_BASE}/tournaments/{tid}/standings",
            )
        )

    log.debug(
        "Extraídos %d torneos de HTML para %s",
        len(tournaments),
        regulation_id,
    )
    return tournaments


def fetch_recent_tournaments(
    regulation_id: str,
    days: int = 30,
) -> list[TournamentSummary]:
    """
    Obtiene torneos recientes de Limitless VGC para
    una regulación específica.

    Dado que Limitless no tiene API, scrapea el HTML del listado de
    torneos filtrando por formato Champions. El parámetro days no se
    usa en el filtrado de URL (Limitless no soporta filtro temporal)
    — el caller puede usar la fecha aproximada de cada torneo para
    filtrar posteriormente.

    Args:
        regulation_id: ID de la regulación del proyecto.
        days: Ventana de días hacia atrás (referencia para el caller).

    Returns:
        Lista de TournamentSummary.
        Lista vacía si hay error o no hay torneos.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,*/*",
    }
    params: dict[str, str] = {"game": "POKEMON"}

    try:
        with httpx.Client(headers=headers) as client:
            html = _fetch_html(
                client,
                LIMITLESS_TOURNAMENTS_URL,
                params=params,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Error scrapeando Limitless para %s: %s",
            regulation_id,
            exc,
        )
        return []

    tournaments = _extract_tournaments_from_html(html, regulation_id)

    log.info(
        "Limitless: %d torneos encontrados para %s",
        len(tournaments),
        regulation_id,
    )
    return tournaments


def count_recent_tournaments(
    reg_id: str,
    days: int = 7,
) -> int:
    """
    Cuenta el número de torneos en los últimos N días
    para una regulación.

    Esta función es usada por get_active_regulation() en
    regulation_active.py para el Escenario B (ventana de transición
    entre regulaciones). Es la función más crítica del módulo.

    GARANTÍA: siempre retorna int, nunca levanta excepción. Cualquier
    error de red, parsing, o timeout devuelve 0 silenciosamente (con
    log.debug para diagnóstico).

    Args:
        reg_id: ID de la regulación a consultar.
        days: Ventana de días hacia atrás (default 7).

    Returns:
        Número de torneos encontrados (int >= 0).
        0 si hay cualquier error.
    """
    try:
        tournaments = fetch_recent_tournaments(reg_id, days=days)
        return len(tournaments)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "count_recent_tournaments falló para %s: %s — retornando 0",
            reg_id,
            exc,
        )
        return 0


def _extract_standings_from_html(
    html: str,
    tournament_id: str,
) -> list[PlayerResult]:
    """
    Extrae standings de un torneo desde HTML.

    Busca patrones de rank y nombre de jugador en la tabla de
    standings de Limitless usando regex.

    Args:
        html: HTML de la página de standings.
        tournament_id: Para etiquetar resultados.

    Returns:
        Lista de PlayerResult. Lista vacía si el layout no es
        reconocible.
    """
    results: list[PlayerResult] = []

    row_pattern = re.compile(
        r"<tr[^>]*>.*?"
        r"(\d+).*?"
        r"<td[^>]*>([^<]+)</td>",
        re.DOTALL | re.IGNORECASE,
    )

    for match in row_pattern.finditer(html):
        try:
            rank = int(match.group(1))
            player = match.group(2).strip()
            if player and len(player) < 50:
                results.append(
                    PlayerResult(
                        tournament_id=tournament_id,
                        player_name=player,
                        final_rank=rank,
                    )
                )
        except (ValueError, IndexError):
            continue

    log.info(
        "Standings extraídos: %d jugadores del torneo %s",
        len(results),
        tournament_id,
    )
    return results


def fetch_tournament_standings(
    tournament_id: str,
    regulation_id: str,
) -> list[PlayerResult]:
    """
    Descarga los standings de un torneo específico de Limitless.

    Args:
        tournament_id: ID del torneo en Limitless.
        regulation_id: Para etiquetar los resultados en logs.

    Returns:
        Lista de PlayerResult con posiciones.
        Lista vacía si hay error o no hay datos.
    """
    url = f"{LIMITLESS_BASE}/tournaments/{tournament_id}/standings"
    headers = {"User-Agent": USER_AGENT}

    try:
        with httpx.Client(headers=headers) as client:
            html = _fetch_html(client, url)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Error descargando standings de torneo %s (%s): %s",
            tournament_id,
            regulation_id,
            exc,
        )
        return []

    return _extract_standings_from_html(html, tournament_id)


def summaries_to_records(
    tournaments: list[TournamentSummary],
) -> list[dict[str, Any]]:
    """
    Convierte lista de TournamentSummary a dicts planos para
    escritura en Parquet.

    date se serializa como string ISO (YYYY-MM-DD) para
    compatibilidad con Parquet sin tipos fecha complejos.

    Columnas: tournament_id, name, format_name,
    regulation_id, date, num_players, url.

    Args:
        tournaments: Lista de TournamentSummary.

    Returns:
        Lista de dicts con un registro por torneo.
    """
    return [
        {
            "tournament_id": t.tournament_id,
            "name": t.name,
            "format_name": t.format_name,
            "regulation_id": t.regulation_id,
            "date": t.date.isoformat(),
            "num_players": t.num_players,
            "url": t.url,
        }
        for t in tournaments
    ]


__all__ = [
    "TournamentSummary",
    "PlayerResult",
    "fetch_recent_tournaments",
    "count_recent_tournaments",
    "fetch_tournament_standings",
    "summaries_to_records",
    "LIMITLESS_BASE",
]
