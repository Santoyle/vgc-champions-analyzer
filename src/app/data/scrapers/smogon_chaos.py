"""
Scraper para descargar chaos JSON de Smogon Stats.

Los chaos JSON contienen datos de uso mensual por formato:
porcentaje de uso por Pokémon, habilidades, ítems, movimientos,
compañeros de equipo y spreads de EVs.

Punto de extensión: REGULATION_TO_FORMAT es la única constante
que mapea regulation_ids del proyecto a format_ids de Smogon.
Para agregar una nueva regulación basta con añadir una entrada
aquí — sin cambiar ninguna otra lógica.

Los datos descargados se usan como PRIOR de entrenamiento para
los modelos ML del pipeline: distribuciones de uso reales del
meta competitivo que informan los priors bayesianos del modelo
de optimización de equipos.
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

log = logging.getLogger(__name__)

SMOGON_STATS_BASE = "https://www.smogon.com/stats"
DEFAULT_CUTOFF = 1760
REQUEST_DELAY_SEC = 0.5

REGULATION_TO_FORMAT: dict[str, str] = {
    "I": "gen9vgc2025regi",
    "H": "gen9vgc2025regh",
    "F": "gen9vgc2025regf",
    "M-A": "gen9championsbssregma",
}


@dataclass
class SmogonUsageEntry:
    """
    Datos de uso de un Pokémon extraídos del chaos JSON.

    Attributes:
        pokemon: Nombre canónico del Pokémon.
        raw_count: Número absoluto de usos.
        usage_pct: Porcentaje de uso (raw_count / total).
        abilities: Dict {ability_name: pct_uso}.
        items: Dict {item_name: pct_uso}.
        moves: Dict {move_name: pct_uso}.
        teammates: Dict {pokemon_name: pct_correlation}.
        spreads: Dict {spread_str: pct_uso}.
    """

    pokemon: str
    raw_count: int
    usage_pct: float
    abilities: dict[str, float] = field(default_factory=dict)
    items: dict[str, float] = field(default_factory=dict)
    moves: dict[str, float] = field(default_factory=dict)
    teammates: dict[str, float] = field(default_factory=dict)
    spreads: dict[str, float] = field(default_factory=dict)


@dataclass
class SmogonMonthlyStats:
    """
    Resultado completo de un mes de stats de Smogon.

    Attributes:
        format_id: Identificador del formato Smogon.
        regulation_id: ID de la regulación del proyecto.
        year_month: String YYYY-MM del período.
        cutoff: Rating mínimo usado.
        total_battles: Total de batallas del período.
        entries: Lista de SmogonUsageEntry por Pokémon.
    """

    format_id: str
    regulation_id: str
    year_month: str
    cutoff: int
    total_battles: int
    entries: list[SmogonUsageEntry] = field(default_factory=list)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_chaos_json(
    client: httpx.Client,
    url: str,
) -> dict[str, Any]:
    """
    Descarga un chaos JSON desde Smogon con retry.

    Args:
        client: Cliente httpx con headers configurados.
        url: URL completa del chaos JSON.

    Returns:
        Dict parseado del JSON descargado.

    Raises:
        httpx.HTTPStatusError: Si el servidor retorna
            un código de error HTTP.
        httpx.TimeoutException: Si la request tarda
            más de 30 segundos (con retry automático).
    """
    response = client.get(url, timeout=30.0)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return response.json()  # type: ignore[no-any-return]


def _parse_chaos_json(
    raw: dict[str, Any],
    regulation_id: str,
    year_month: str,
    cutoff: int,
) -> SmogonMonthlyStats:
    """
    Parsea el chaos JSON de Smogon a SmogonMonthlyStats.

    Calcula usage_pct como raw_count / total_raw_count
    para normalizar los porcentajes correctamente.

    Args:
        raw: Dict crudo del chaos JSON.
        regulation_id: ID de regulación del proyecto.
        year_month: String YYYY-MM del período.
        cutoff: Rating mínimo del dump.

    Returns:
        SmogonMonthlyStats con todos los Pokémon parseados.
    """
    info = raw.get("info", {})
    data = raw.get("data", {})

    format_id = str(info.get("metagame", ""))
    total_battles = int(info.get("number of battles", 0))

    total_raw = sum(
        int(v.get("Raw count", 0)) for v in data.values() if isinstance(v, dict)
    )
    if total_raw == 0:
        total_raw = 1

    entries: list[SmogonUsageEntry] = []
    for pokemon_name, pdata in data.items():
        if not isinstance(pdata, dict):
            continue

        raw_count = int(pdata.get("Raw count", 0))
        usage_pct = raw_count / total_raw * 100.0

        entries.append(
            SmogonUsageEntry(
                pokemon=str(pokemon_name),
                raw_count=raw_count,
                usage_pct=round(usage_pct, 4),
                abilities={
                    str(k): float(v) for k, v in pdata.get("Abilities", {}).items()
                },
                items={
                    str(k): float(v) for k, v in pdata.get("Items", {}).items()
                },
                moves={
                    str(k): float(v) for k, v in pdata.get("Moves", {}).items()
                },
                teammates={
                    str(k): float(v) for k, v in pdata.get("Teammates", {}).items()
                },
                spreads={
                    str(k): float(v) for k, v in pdata.get("Spreads", {}).items()
                },
            )
        )

    entries.sort(key=lambda e: e.raw_count, reverse=True)

    return SmogonMonthlyStats(
        format_id=format_id,
        regulation_id=regulation_id,
        year_month=year_month,
        cutoff=cutoff,
        total_battles=total_battles,
        entries=entries,
    )


def fetch_smogon_stats(
    regulation_id: str,
    year_month: str,
    cutoff: int = DEFAULT_CUTOFF,
) -> SmogonMonthlyStats | None:
    """
    Descarga y parsea el chaos JSON de Smogon para
    una regulación y mes específicos.

    Construye la URL automáticamente desde el mapeo
    REGULATION_TO_FORMAT. Si el regulation_id no está
    en el mapeo, loggea un warning y retorna None.

    Args:
        regulation_id: ID de la regulación del proyecto.
                       Debe estar en REGULATION_TO_FORMAT.
        year_month: String YYYY-MM del período a descargar.
                    Ej: "2025-07", "2025-10"
        cutoff: Rating mínimo. Default 1760 (competitivo).

    Returns:
        SmogonMonthlyStats si la descarga fue exitosa.
        None si el regulation_id no está mapeado o
        si la URL retorna 404 (datos no disponibles
        para ese mes).

    Logs:
        INFO al iniciar descarga con URL.
        INFO al completar con conteo de Pokémon.
        WARNING si regulation_id no está en el mapeo.
        WARNING si la URL retorna 404.
        ERROR si falla después de reintentos.
    """
    format_id = REGULATION_TO_FORMAT.get(regulation_id)
    if format_id is None:
        log.warning(
            "regulation_id '%s' no está en "
            "REGULATION_TO_FORMAT. IDs conocidos: %s",
            regulation_id,
            list(REGULATION_TO_FORMAT.keys()),
        )
        return None

    url = f"{SMOGON_STATS_BASE}/{year_month}/chaos/{format_id}-{cutoff}.json"
    log.info("Descargando Smogon stats: %s", url)

    headers = {
        "User-Agent": (
            "vgc-champions-analyzer/0.1 "
            "smogon-chaos-fetcher "
            "(github.com/Santoyle/vgc-champions-analyzer)"
        )
    }

    try:
        with httpx.Client(headers=headers) as client:
            raw = _fetch_chaos_json(client, url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            log.warning(
                "Smogon stats no disponibles para "
                "%s/%s (404). El mes puede no tener "
                "datos o el formato es incorrecto.",
                regulation_id,
                year_month,
            )
            return None
        log.error(
            "Error HTTP %d descargando %s: %s",
            exc.response.status_code,
            url,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Error descargando Smogon stats %s/%s: %s",
            regulation_id,
            year_month,
            exc,
        )
        return None

    stats = _parse_chaos_json(raw, regulation_id, year_month, cutoff)
    log.info(
        "Smogon stats descargadas: %s/%s — %d Pokémon, %d batallas",
        regulation_id,
        year_month,
        len(stats.entries),
        stats.total_battles,
    )
    return stats


def stats_to_records(
    stats: SmogonMonthlyStats,
) -> list[dict[str, Any]]:
    """
    Convierte SmogonMonthlyStats a lista de dicts
    planos para ingestión con dlt o escritura directa
    a Parquet.

    Cada dict representa un Pokémon con sus stats
    serializadas como JSON strings para compatibilidad
    con esquemas tabulares.

    Args:
        stats: SmogonMonthlyStats a convertir.

    Returns:
        Lista de dicts con un registro por Pokémon.
        Campos: regulation_id, format_id, year_month,
        cutoff, total_battles, pokemon, raw_count,
        usage_pct, abilities_json, items_json,
        moves_json, teammates_json, spreads_json.
    """
    records = []
    for entry in stats.entries:
        records.append(
            {
                "regulation_id": stats.regulation_id,
                "format_id": stats.format_id,
                "year_month": stats.year_month,
                "cutoff": stats.cutoff,
                "total_battles": stats.total_battles,
                "pokemon": entry.pokemon,
                "raw_count": entry.raw_count,
                "usage_pct": entry.usage_pct,
                "abilities_json": json.dumps(entry.abilities),
                "items_json": json.dumps(entry.items),
                "moves_json": json.dumps(entry.moves),
                "teammates_json": json.dumps(entry.teammates),
                "spreads_json": json.dumps(entry.spreads),
            }
        )
    return records


__all__ = [
    "REGULATION_TO_FORMAT",
    "SmogonUsageEntry",
    "SmogonMonthlyStats",
    "fetch_smogon_stats",
    "stats_to_records",
]
