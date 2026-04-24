"""
Pipeline de ingesta LIVE diaria para la regulación activa de Champions.

Este pipeline es el corazón del sistema LIVE: descarga el estado actual
del meta competitivo desde 4 fuentes (Pikalytics, Pokémon Showdown,
Limitless VGC, pokedata.ovh) y persiste los datos en Parquet
hive-particionado en data/raw/.

El reg_id de la regulación activa se detecta automáticamente mediante
get_active_regulation() — nunca está hardcodeado. Esto garantiza que el
pipeline siga funcionando correctamente cuando la regulación cambie de
M-A a la siguiente sin modificar código.

Se ejecuta diariamente via GitHub Actions cron. A diferencia de
prior_ingest.py (one-shot histórico), este pipeline es idempotente:
si se ejecuta dos veces el mismo día, el segundo run sobreescribe el
Parquet del primero con datos más actualizados.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.app.core.regulation_active import get_active_regulation
from src.app.data.scrapers.limitless import (
    fetch_recent_tournaments,
    summaries_to_records,
)
from src.app.data.scrapers.pikalytics_champions import (
    fetch_pikalytics_snapshot,
    snapshot_to_records,
)
from src.app.data.scrapers.rk9_pokedata import (
    events_to_records,
    fetch_official_events,
)
from src.app.data.scrapers.showdown_replays import (
    fetch_recent_replays,
    replays_to_records,
)
from src.app.utils.db import get_parquet_path

log = logging.getLogger(__name__)

LIVE_SOURCES: list[str] = [
    "pikalytics",
    "showdown",
    "limitless",
    "rk9",
]


def _write_parquet(
    records: list[dict[str, Any]],
    reg_id: str,
    source: str,
    fecha: str,
) -> bool:
    """
    Escribe una lista de records como Parquet en la ruta hive
    particionada.

    Ruta: data/raw/reg={reg_id}/source={source}/{fecha}.parquet

    Añade la columna ingested_at con la fecha de hoy antes de
    escribir. Si records está vacío, loggea warning y retorna False.

    Args:
        records: Lista de dicts a escribir.
        reg_id: ID de la regulación activa.
        source: Nombre de la fuente.
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        True si se escribió correctamente.
        False si records vacío o hubo error.
    """
    if not records:
        log.warning(
            "Sin records para %s/%s — no se escribe Parquet",
            source,
            reg_id,
        )
        return False

    parquet_path = get_parquet_path(reg_id, source, fecha)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.DataFrame(records)
        df["ingested_at"] = date.today().isoformat()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(parquet_path), compression="snappy")
        log.info("Escrito: %s (%d filas)", parquet_path, len(df))
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Error escribiendo Parquet %s: %s", parquet_path, exc)
        return False


def ingest_pikalytics(
    reg_id: str,
    fecha: str,
) -> bool:
    """
    Descarga y persiste snapshot de Pikalytics Champions para la
    regulación activa.

    Args:
        reg_id: ID de la regulación activa.
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        True si exitoso, False si falló o sin datos.
    """
    log.info("Ingesta Pikalytics para %s...", reg_id)
    snapshot = fetch_pikalytics_snapshot(reg_id)

    if not snapshot.entries:
        log.warning(
            "Pikalytics: sin datos para %s (método: %s)",
            reg_id,
            snapshot.parse_method,
        )
        return False

    records = snapshot_to_records(snapshot)
    log.info(
        "Pikalytics: %d Pokémon descargados para %s (método: %s)",
        len(records),
        reg_id,
        snapshot.parse_method,
    )
    return _write_parquet(records, reg_id, "pikalytics", fecha)


def ingest_showdown(
    reg_id: str,
    fecha: str,
    max_replays: int = 50,
) -> bool:
    """
    Descarga y persiste replays recientes de Showdown para la
    regulación activa.

    Args:
        reg_id: ID de la regulación activa.
        fecha: Fecha en formato YYYY-MM-DD.
        max_replays: Máximo de replays a descargar.

    Returns:
        True si exitoso, False si falló o sin datos.
    """
    log.info(
        "Ingesta Showdown replays para %s (max: %d)...",
        reg_id,
        max_replays,
    )
    replays = fetch_recent_replays(reg_id, max_replays=max_replays)

    if not replays:
        log.warning("Showdown: sin replays para %s", reg_id)
        return False

    records = replays_to_records(replays)
    log.info("Showdown: %d replays descargados para %s", len(records), reg_id)
    return _write_parquet(records, reg_id, "showdown", fecha)


def ingest_limitless(
    reg_id: str,
    fecha: str,
) -> bool:
    """
    Descarga y persiste torneos recientes de Limitless para la
    regulación activa.

    Args:
        reg_id: ID de la regulación activa.
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        True si exitoso, False si falló o sin datos.
    """
    log.info("Ingesta Limitless torneos para %s...", reg_id)
    tournaments = fetch_recent_tournaments(reg_id, days=30)

    if not tournaments:
        log.warning("Limitless: sin torneos para %s", reg_id)
        return False

    records = summaries_to_records(tournaments)
    log.info("Limitless: %d torneos descargados para %s", len(records), reg_id)
    return _write_parquet(records, reg_id, "limitless", fecha)


def ingest_rk9(
    reg_id: str,
    fecha: str,
) -> bool:
    """
    Descarga y persiste eventos oficiales de pokedata.ovh para la
    regulación activa.

    Args:
        reg_id: ID de la regulación activa.
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        True si exitoso, False si falló o sin datos.
    """
    log.info("Ingesta pokedata.ovh eventos para %s...", reg_id)
    events = fetch_official_events(reg_id)

    if not events:
        log.warning("pokedata.ovh: sin eventos para %s", reg_id)
        return False

    records = events_to_records(events)
    log.info("pokedata.ovh: %d eventos para %s", len(records), reg_id)
    return _write_parquet(records, reg_id, "rk9", fecha)


def run_live_ingest(
    reg_id: str | None = None,
    fecha: str | None = None,
    sources: list[str] | None = None,
    max_showdown_replays: int = 50,
) -> dict[str, bool]:
    """
    Ejecuta la ingesta LIVE completa para la regulación activa.

    Cuando reg_id es None, detecta automáticamente la regulación
    activa con get_active_regulation(). Cada fuente se ejecuta de
    forma independiente — un fallo en una fuente no cancela las
    demás.

    Args:
        reg_id: ID de regulación a usar. Si None, detecta
                automáticamente.
        fecha: Fecha de ingesta (YYYY-MM-DD). None = hoy.
        sources: Lista de fuentes a ejecutar. None = todas.
        max_showdown_replays: Máximo de replays a descargar.

    Returns:
        Dict {source: bool} con resultado por fuente.
        Si no se puede detectar la regulación activa, retorna
        {source: False} para todas las fuentes.
    """
    if reg_id is None:
        try:
            active = get_active_regulation()
            reg_id = active.regulation_id
            log.info(
                "Regulación activa detectada: %s (estado: %s)",
                reg_id,
                active.state,
            )
        except FileNotFoundError as exc:
            log.error("No se pudo detectar regulación activa: %s", exc)
            return {s: False for s in LIVE_SOURCES}

    if fecha is None:
        fecha = date.today().isoformat()

    if sources is None:
        sources = LIVE_SOURCES

    log.info(
        "Iniciando ingesta LIVE: reg=%s fecha=%s fuentes=%s",
        reg_id,
        fecha,
        sources,
    )

    results: dict[str, bool] = {}

    if "pikalytics" in sources:
        results["pikalytics"] = ingest_pikalytics(reg_id, fecha)

    if "showdown" in sources:
        results["showdown"] = ingest_showdown(
            reg_id, fecha, max_replays=max_showdown_replays
        )

    if "limitless" in sources:
        results["limitless"] = ingest_limitless(reg_id, fecha)

    if "rk9" in sources:
        results["rk9"] = ingest_rk9(reg_id, fecha)

    exitos = sum(1 for v in results.values() if v)
    total = len(results)
    log.info(
        "Ingesta LIVE completada: %s | fecha=%s | %d/%d fuentes exitosas | %s",
        reg_id,
        fecha,
        exitos,
        total,
        results,
    )

    return results


def main() -> int:
    """
    Entry point del pipeline como CLI.

    Modos de uso:
        # Ingesta completa regulación activa:
        python -m src.app.data.pipelines.live_ingest

        # Solo fuentes específicas:
        python -m src.app.data.pipelines.live_ingest --sources pikalytics showdown

        # Regulación específica (testing):
        python -m src.app.data.pipelines.live_ingest --reg M-A

        # Más replays de Showdown:
        python -m src.app.data.pipelines.live_ingest --max-replays 200
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingesta LIVE diaria de datos de Champions"
    )
    parser.add_argument(
        "--reg",
        default=None,
        help="Regulación específica. Default: detectar automáticamente.",
    )
    parser.add_argument(
        "--fecha",
        default=None,
        help="Fecha YYYY-MM-DD. Default: hoy.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=LIVE_SOURCES,
        default=None,
        help="Fuentes a ejecutar. Default: todas.",
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=50,
        dest="max_replays",
        help="Máximo de replays Showdown. Default: 50.",
    )
    args = parser.parse_args()

    results = run_live_ingest(
        reg_id=args.reg,
        fecha=args.fecha,
        sources=args.sources,
        max_showdown_replays=args.max_replays,
    )

    fallos = sum(1 for v in results.values() if not v)
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "LIVE_SOURCES",
    "run_live_ingest",
    "ingest_pikalytics",
    "ingest_showdown",
    "ingest_limitless",
    "ingest_rk9",
]
