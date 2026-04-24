"""
Pipeline de ingesta de datos históricos de uso de Smogon (PRIOR).

Descarga los chaos JSON mensuales de Smogon Stats para las
regulaciones PRIOR (I, H, F) y los persiste como Parquet
hive-particionado en data/raw/reg={reg_id}/source=smogon_chaos/.

Los datos descargados alimentan los priors bayesianos del modelo
de optimización de equipos. Se ejecuta desde CLI o GitHub Actions
— sin dependencia de Streamlit ni de ningún componente de UI.

Punto de extensión: agregar nuevas regulaciones al mapeo
REGULATION_TO_FORMAT en smogon_chaos.py y añadir el reg_id
a la lista DEFAULT_PRIOR_REGS de este módulo.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.app.data.scrapers.smogon_chaos import (
    REGULATION_TO_FORMAT,
    SmogonMonthlyStats,
    fetch_smogon_stats,
    stats_to_records,
)
from src.app.utils.db import get_parquet_path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

DEFAULT_PRIOR_REGS: list[str] = ["I", "H", "F"]
_INTER_REG_DELAY_SEC = 0.5


def get_months_for_regulation(reg_id: str) -> list[tuple[int, int]]:
    """
    Retorna todos los (year, month) dentro del rango
    [date_start, date_end] de la regulación indicada.

    Lee el JSON de la regulación directamente desde disco
    para evitar dependencias de Streamlit.

    Args:
        reg_id: ID de la regulación. Debe existir como
                regulations/{reg_id}.json en la raíz del proyecto.

    Returns:
        Lista de tuplas (year, month) ordenadas ascendentemente.
        Ejemplo: Reg H (2025-09-01 → 2025-11-30) →
                 [(2025, 9), (2025, 10), (2025, 11)]

    Raises:
        FileNotFoundError: Si el JSON de la regulación no existe.
        KeyError: Si el JSON no contiene date_start o date_end.
    """
    reg_path = _PROJECT_ROOT / "regulations" / f"{reg_id}.json"
    with reg_path.open(encoding="utf-8") as f:
        reg_data: dict[str, Any] = json.load(f)

    start = date.fromisoformat(str(reg_data["date_start"]))
    end = date.fromisoformat(str(reg_data["date_end"]))

    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    return months


def run_prior_ingest(
    reg_ids: list[str] | None = None,
    months: list[tuple[int, int]] | None = None,
    cutoff: int = 1500,
) -> dict[str, int]:
    """
    Descarga datos históricos de uso de Smogon para las
    regulaciones PRIOR y los guarda como Parquet.

    Para cada combinación (reg_id, year, month):
      1. Construye la URL desde REGULATION_TO_FORMAT.
      2. Llama a fetch_smogon_stats — si retorna None,
         loggea y continua (no aborta el pipeline).
      3. Convierte a registros planos con stats_to_records.
      4. Escribe el Parquet en la ruta hive-particionada:
         data/raw/reg={reg_id}/source=smogon_chaos/YYYY-MM.parquet

    Args:
        reg_ids: IDs de regulaciones a procesar.
                 Si None, usa ["I", "H", "F"].
        months: Lista de (year, month) a descargar para cada
                regulación. Si None, usa todos los meses del
                rango [date_start, date_end] de cada regulación.
        cutoff: Rating mínimo del dump de Smogon. Default 1500.

    Returns:
        Dict {reg_id: total_records_saved} por regulación.
        Una regulación con 0 indica que ningún mes tuvo datos.
    """
    target_regs = reg_ids if reg_ids is not None else DEFAULT_PRIOR_REGS

    # Filtrar regulaciones sin mapeo antes de empezar
    known_regs = [r for r in target_regs if r in REGULATION_TO_FORMAT]
    unknown = [r for r in target_regs if r not in REGULATION_TO_FORMAT]
    if unknown:
        log.warning(
            "Las siguientes regulaciones no están en "
            "REGULATION_TO_FORMAT y se omitirán: %s",
            unknown,
        )

    results: dict[str, int] = {reg_id: 0 for reg_id in known_regs}

    for i, reg_id in enumerate(known_regs):
        if i > 0:
            time.sleep(_INTER_REG_DELAY_SEC)

        target_months: list[tuple[int, int]]
        if months is not None:
            target_months = months
        else:
            try:
                target_months = get_months_for_regulation(reg_id)
            except (FileNotFoundError, KeyError) as exc:
                log.error(
                    "No se pudo cargar la regulación '%s': %s. "
                    "Omitiendo.",
                    reg_id,
                    exc,
                )
                continue

        for year, month in target_months:
            year_month = f"{year}-{month:02d}"
            stats = fetch_smogon_stats(
                regulation_id=reg_id,
                year_month=year_month,
                cutoff=cutoff,
            )
            if stats is None:
                log.warning(
                    "Sin datos para %s/%s — omitiendo.",
                    reg_id,
                    year_month,
                )
                continue

            records = stats_to_records(stats)
            if not records:
                log.warning(
                    "stats_to_records retornó lista vacía para "
                    "%s/%s — omitiendo escritura.",
                    reg_id,
                    year_month,
                )
                continue

            _write_parquet(reg_id, year_month, records)
            results[reg_id] += len(records)
            log.info(
                "Guardados %d registros para %s/%s.",
                len(records),
                reg_id,
                year_month,
            )

    return results


def _write_parquet(
    reg_id: str,
    year_month: str,
    records: list[dict[str, Any]],
) -> None:
    """
    Escribe una lista de dicts como Parquet hive-particionado.

    Crea los directorios padre si no existen. La ruta sigue
    la convención:
        data/raw/reg={reg_id}/source=smogon_chaos/{year_month}.parquet

    Args:
        reg_id: ID de la regulación.
        year_month: String YYYY-MM del período.
        records: Lista de dicts planos (output de stats_to_records).
    """
    out_path = get_parquet_path(
        reg_id=reg_id,
        source="smogon_chaos",
        fecha=year_month,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)
    log.debug("Parquet escrito en %s (%d filas).", out_path, len(records))


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Descarga datos históricos de uso de Smogon (PRIOR)."
    )
    parser.add_argument(
        "--regs",
        nargs="+",
        default=None,
        metavar="REG_ID",
        help=(
            "IDs de regulaciones a procesar. "
            f"Default: {DEFAULT_PRIOR_REGS}"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1500,
        help="Rating mínimo del dump de Smogon. Default: 1500.",
    )
    args = parser.parse_args()

    totals = run_prior_ingest(reg_ids=args.regs, cutoff=args.cutoff)

    print("\nResumen de ingesta:")
    for reg, n in totals.items():
        print(f"  {reg}: {n} registros guardados")

    total_all = sum(totals.values())
    print(f"\nTotal: {total_all} registros en {len(totals)} regulaciones.")
    sys.exit(0 if total_all > 0 else 1)


__all__ = [
    "DEFAULT_PRIOR_REGS",
    "get_months_for_regulation",
    "run_prior_ingest",
]
