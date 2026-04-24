"""
Scraper best-effort para Pikalytics Champions.

Pikalytics no tiene API pública oficial, por lo que este scraper
utiliza dos estrategias de parsing en cascada:

  1. Endpoint markdown AI (PIKALYTICS_AI_URL): devuelve una tabla
     en texto plano que es más fácil de parsear y más estable.
  2. Página HTML principal (PIKALYTICS_CHAMPIONS_URL): fallback si
     el endpoint AI no está disponible o devuelve formato inesperado.

En caso de fallo total de ambas estrategias, retorna un
PikalyticsSnapshot vacío con parse_method="fallback" — nunca lanza
excepción. El pipeline que invoca al scraper es responsable de
decidir si el fallo es crítico.

Los datos capturados son LIVE: representan el estado actual del
meta de Pokémon Champions en Pikalytics en el momento de la
descarga, no datos históricos. Se etiquetan con la regulación
activa en ese momento para su almacenamiento en Parquet.
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

PIKALYTICS_BASE = "https://www.pikalytics.com"
PIKALYTICS_CHAMPIONS_URL = (
    f"{PIKALYTICS_BASE}/pokedex/championstournaments"
)
PIKALYTICS_AI_URL = (
    f"{PIKALYTICS_BASE}/ai/pokedex/championstournaments"
)
REQUEST_DELAY_SEC = 1.0
USER_AGENT = (
    "vgc-champions-analyzer/0.1 "
    "pikalytics-scraper "
    "(github.com/Santoyle/vgc-champions-analyzer)"
)


@dataclass
class PikalyticsEntry:
    """
    Datos de uso de un Pokémon desde Pikalytics.

    Attributes:
        pokemon: Nombre del Pokémon.
        usage_pct: Porcentaje de uso (0-100).
        win_pct: Win rate si está disponible. None si no.
        count: Número de apariciones si disponible.
        rank: Posición en el ranking (1 = más usado).
    """

    pokemon: str
    usage_pct: float
    win_pct: float | None = None
    count: int | None = None
    rank: int = 0


@dataclass
class PikalyticsSnapshot:
    """
    Snapshot completo de Pikalytics para una reg.

    Attributes:
        regulation_id: ID de la regulación.
        snapshot_date: Fecha de captura.
        source_url: URL de donde se obtuvieron datos.
        entries: Lista de PikalyticsEntry por Pokémon.
        parse_method: Método usado ("markdown_ai",
                      "html", o "fallback").
    """

    regulation_id: str
    snapshot_date: date
    source_url: str
    entries: list[PikalyticsEntry] = field(default_factory=list)
    parse_method: str = "unknown"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_url(
    client: httpx.Client,
    url: str,
) -> httpx.Response:
    """
    Descarga una URL con retry automático.

    Args:
        client: Cliente httpx configurado.
        url: URL a descargar.

    Returns:
        Response HTTP.

    Raises:
        httpx.HTTPStatusError: En errores 4xx/5xx.
        httpx.TimeoutException: Si tarda >30s (retry).
    """
    response = client.get(url, timeout=30.0)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return response


def _parse_markdown_table(
    text: str,
    regulation_id: str,
) -> list[PikalyticsEntry]:
    """
    Parsea una tabla markdown de Pikalytics.

    Intenta extraer columnas Pokémon, Usage %, Win %
    y Count de cualquier tabla markdown en el texto.

    Es robusto a variaciones de formato:
    - Columnas en distinto orden
    - Columnas faltantes (Win %, Count opcionales)
    - Filas con datos incompletos

    Args:
        text: Texto con tabla(s) markdown.
        regulation_id: Para logging.

    Returns:
        Lista de PikalyticsEntry parseadas.
        Lista vacía si no se pudo parsear nada.
    """
    entries: list[PikalyticsEntry] = []

    table_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("|")
    ]

    if not table_lines:
        log.debug("No se encontraron tablas markdown para %s", regulation_id)
        return entries

    header_line = table_lines[0]
    headers = [h.strip().lower() for h in header_line.split("|") if h.strip()]

    col_pokemon = next(
        (i for i, h in enumerate(headers) if "pokémon" in h or "pokemon" in h),
        None,
    )
    col_usage = next(
        (i for i, h in enumerate(headers) if "usage" in h),
        None,
    )
    col_win = next(
        (i for i, h in enumerate(headers) if "win" in h),
        None,
    )
    col_count = next(
        (i for i, h in enumerate(headers) if "count" in h or "total" in h),
        None,
    )

    if col_pokemon is None or col_usage is None:
        log.debug(
            "Columnas Pokémon/Usage no encontradas en headers: %s", headers
        )
        return entries

    rank = 0
    for line in table_lines[2:]:  # Skip header + separator row
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells or len(cells) <= max(col_pokemon, col_usage):
            continue

        try:
            pokemon = cells[col_pokemon].strip()
            usage_pct = float(cells[col_usage].replace("%", "").strip())

            win_pct: float | None = None
            if col_win is not None and col_win < len(cells):
                try:
                    win_pct = float(cells[col_win].replace("%", "").strip())
                except (ValueError, IndexError):
                    pass

            count: int | None = None
            if col_count is not None and col_count < len(cells):
                try:
                    count = int(float(cells[col_count].replace(",", "").strip()))
                except (ValueError, IndexError):
                    pass

            rank += 1
            entries.append(
                PikalyticsEntry(
                    pokemon=pokemon,
                    usage_pct=usage_pct,
                    win_pct=win_pct,
                    count=count,
                    rank=rank,
                )
            )

        except (ValueError, IndexError) as exc:
            log.debug(
                "Fila no parseable en %s: %s — %s", regulation_id, line, exc
            )
            continue

    return entries


def _parse_html_table(
    html: str,
    regulation_id: str,
) -> list[PikalyticsEntry]:
    """
    Parser HTML de fallback para Pikalytics.

    Usa regex para extraer datos de uso sin depender
    de BeautifulSoup. Busca patrones de nombre +
    porcentaje en el HTML.

    Args:
        html: HTML de la página de Pikalytics.
        regulation_id: Para logging.

    Returns:
        Lista de PikalyticsEntry. Lista vacía si falla.
    """
    entries: list[PikalyticsEntry] = []

    pattern = re.compile(
        r'data-name=["\']([^"\']+)["\'][^>]*>.*?(\d+\.?\d*)\s*%',
        re.IGNORECASE | re.DOTALL,
    )
    matches: list[tuple[str, str]] = pattern.findall(html)

    if not matches:
        alt_pattern = re.compile(
            r'([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z]+)?)\s*[|:]?\s*(\d+\.?\d+)\s*%'
        )
        matches = alt_pattern.findall(html[:50000])

    for rank, (pokemon, usage_str) in enumerate(matches[:200], start=1):
        try:
            entries.append(
                PikalyticsEntry(
                    pokemon=pokemon.strip(),
                    usage_pct=float(usage_str),
                    rank=rank,
                )
            )
        except ValueError:
            continue

    if entries:
        log.info(
            "HTML fallback: %d Pokémon parseados para %s",
            len(entries),
            regulation_id,
        )
    else:
        log.warning(
            "HTML fallback no encontró datos para %s", regulation_id
        )

    return entries


def fetch_pikalytics_snapshot(
    regulation_id: str,
) -> PikalyticsSnapshot:
    """
    Descarga y parsea el snapshot actual de Pikalytics
    para la regulación especificada.

    Intenta en orden:
    1. Endpoint markdown AI (PIKALYTICS_AI_URL)
    2. Página HTML principal (PIKALYTICS_CHAMPIONS_URL)
    3. Retorna snapshot vacío con warning

    La regulación actual de Champions se determina
    externamente — este scraper siempre descarga
    el estado ACTUAL de Pikalytics Champions.

    Args:
        regulation_id: ID de la regulación activa.
                       Solo para etiquetar los datos.

    Returns:
        PikalyticsSnapshot con los datos descargados.
        Nunca levanta excepción — retorna snapshot
        vacío en caso de error total.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,text/plain,*/*",
    }

    snapshot = PikalyticsSnapshot(
        regulation_id=regulation_id,
        snapshot_date=date.today(),
        source_url=PIKALYTICS_AI_URL,
    )

    with httpx.Client(headers=headers) as client:

        # Intento 1: endpoint markdown AI
        try:
            response = _fetch_url(client, PIKALYTICS_AI_URL)
            entries = _parse_markdown_table(response.text, regulation_id)
            if entries:
                snapshot.entries = entries
                snapshot.parse_method = "markdown_ai"
                log.info(
                    "Pikalytics markdown AI: %d Pokémon para %s",
                    len(entries),
                    regulation_id,
                )
                return snapshot
            log.debug(
                "Markdown AI no retornó datos útiles para %s — intentando HTML",
                regulation_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Pikalytics AI endpoint falló para %s: %s — intentando HTML",
                regulation_id,
                exc,
            )

        # Intento 2: HTML principal
        try:
            response = _fetch_url(client, PIKALYTICS_CHAMPIONS_URL)
            snapshot.source_url = PIKALYTICS_CHAMPIONS_URL
            entries = _parse_html_table(response.text, regulation_id)
            if entries:
                snapshot.entries = entries
                snapshot.parse_method = "html"
                log.info(
                    "Pikalytics HTML: %d Pokémon para %s",
                    len(entries),
                    regulation_id,
                )
                return snapshot
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Pikalytics HTML también falló para %s: %s",
                regulation_id,
                exc,
            )

    # Fallback: snapshot vacío
    log.warning(
        "Pikalytics no disponible para %s — retornando snapshot vacío",
        regulation_id,
    )
    snapshot.parse_method = "fallback"
    return snapshot


def snapshot_to_records(
    snapshot: PikalyticsSnapshot,
) -> list[dict[str, Any]]:
    """
    Convierte un PikalyticsSnapshot a lista de dicts
    planos para escritura en Parquet.

    Columnas: regulation_id, snapshot_date,
    source_url, parse_method, pokemon, usage_pct,
    win_pct, count, rank.

    Args:
        snapshot: PikalyticsSnapshot a convertir.

    Returns:
        Lista de dicts. Lista vacía si no hay entries.
    """
    return [
        {
            "regulation_id": snapshot.regulation_id,
            "snapshot_date": snapshot.snapshot_date.isoformat(),
            "source_url": snapshot.source_url,
            "parse_method": snapshot.parse_method,
            "pokemon": entry.pokemon,
            "usage_pct": entry.usage_pct,
            "win_pct": entry.win_pct,
            "count": entry.count,
            "rank": entry.rank,
        }
        for entry in snapshot.entries
    ]


__all__ = [
    "PikalyticsEntry",
    "PikalyticsSnapshot",
    "fetch_pikalytics_snapshot",
    "snapshot_to_records",
    "PIKALYTICS_CHAMPIONS_URL",
    "PIKALYTICS_AI_URL",
]
