"""
Congela el estado del meta (uso, ítems y compañeros) en un momento dado para
referencia histórica y como posible entrada del pipeline de retrain del modelo WP.

Los snapshots se escriben bajo ``data/snapshots/{label}/`` con ``meta.json`` y CSVs.
Se recomienda nombrar ``label`` con la convención ``post_{evento}_{año}`` —
por ejemplo ``post_GC1_2026`` ó ``snapshot_2026-05-04``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_SNAPSHOTS_DIR = _DATA_DIR / "snapshots"


def _count_replays(regulation_id: str) -> int:
    """
    Cuenta el número total de replays disponibles
    en los Parquets de Showdown para la regulación.

    Args:
        regulation_id: ID de la regulación.

    Returns:
        Número total de filas en todos los Parquets
        de source=showdown para esa regulación.
        0 si no hay datos.
    """
    showdown_dir = (
        _DATA_DIR
        / "raw"
        / f"reg={regulation_id}"
        / "source=showdown"
    )
    if not showdown_dir.exists():
        return 0

    total = 0
    for pq_file in showdown_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            total += len(df)
        except Exception as exc:
            log.debug(
                "Error leyendo %s: %s",
                pq_file,
                exc,
            )
    return total


def _count_pikalytics_entries(
    regulation_id: str,
) -> int:
    """
    Cuenta el número de entradas en los Parquets
    de Pikalytics para la regulación.

    Args:
        regulation_id: ID de la regulación.

    Returns:
        Número total de filas en todos los Parquets
        de source=pikalytics.
        0 si no hay datos.
    """
    pikalytics_dir = (
        _DATA_DIR
        / "raw"
        / f"reg={regulation_id}"
        / "source=pikalytics"
    )
    if not pikalytics_dir.exists():
        return 0

    total = 0
    for pq_file in pikalytics_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            total += len(df)
        except Exception as exc:
            log.debug(
                "Error leyendo %s: %s",
                pq_file,
                exc,
            )
    return total


def _build_usage_top50(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str,
) -> pd.DataFrame:
    """
    Construye el top-50 de Pokémon por uso medio.

    Combina datos de Pikalytics y Smogon chaos JSON
    si están disponibles. Prioriza Pikalytics para
    Champions (dato más específico del formato).

    Args:
        con: Conexión DuckDB activa.
        regulation_id: ID de la regulación.

    Returns:
        DataFrame con columnas:
        pokemon, avg_usage_pct, n_months,
        regulation_id.
        Máximo 50 filas, ordenado por
        avg_usage_pct DESC.
    """
    try:
        from src.app.data.sql.views import (
            create_usage_by_reg,
            register_raw_view,
        )

        register_raw_view(con)
        df = create_usage_by_reg(con, regulation_id)
        if df.empty:
            log.warning(
                "Sin datos de uso para %s",
                regulation_id,
            )
            return pd.DataFrame(
                columns=[
                    "pokemon",
                    "avg_usage_pct",
                    "n_months",
                    "regulation_id",
                ]
            )

        df["regulation_id"] = regulation_id
        return (
            df.sort_values("avg_usage_pct", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )
    except Exception as exc:
        log.warning(
            "Error construyendo usage top-50 "
            "para %s: %s",
            regulation_id,
            exc,
        )
        return pd.DataFrame(
            columns=[
                "pokemon",
                "avg_usage_pct",
                "n_months",
                "regulation_id",
            ]
        )


def _build_items_by_pkm(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str,
    top_n_items: int = 5,
) -> pd.DataFrame:
    """
    Construye el top-N ítems por Pokémon.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: ID de la regulación.
        top_n_items: Número de ítems por Pokémon.

    Returns:
        DataFrame con columnas:
        pokemon, item, avg_pct, n_months_seen,
        regulation_id.
    """
    try:
        from src.app.data.sql.views import (
            create_items_by_pkm,
            register_raw_view,
        )

        register_raw_view(con)
        df = create_items_by_pkm(con, regulation_id)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "pokemon",
                    "item",
                    "avg_pct",
                    "n_months_seen",
                    "regulation_id",
                ]
            )

        # Top N por Pokémon
        df["regulation_id"] = regulation_id
        df = (
            df.sort_values(
                ["pokemon", "avg_pct"],
                ascending=[True, False],
            )
            .groupby("pokemon")
            .head(top_n_items)
            .reset_index(drop=True)
        )
        return df
    except Exception as exc:
        log.warning(
            "Error construyendo items para %s: %s",
            regulation_id,
            exc,
        )
        return pd.DataFrame(
            columns=[
                "pokemon",
                "item",
                "avg_pct",
                "n_months_seen",
                "regulation_id",
            ]
        )


def _build_teammates_by_pkm(
    con: duckdb.DuckDBPyConnection,
    regulation_id: str,
    top_n_teammates: int = 5,
) -> pd.DataFrame:
    """
    Construye el top-N compañeros por Pokémon.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: ID de la regulación.
        top_n_teammates: Número de compañeros
                          por Pokémon.

    Returns:
        DataFrame con columnas:
        pokemon, teammate, avg_correlation,
        n_months_seen, regulation_id.
    """
    try:
        from src.app.data.sql.views import (
            create_teammates_by_pkm,
            register_raw_view,
        )

        register_raw_view(con)
        df = create_teammates_by_pkm(con, regulation_id)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "pokemon",
                    "teammate",
                    "avg_correlation",
                    "n_months_seen",
                    "regulation_id",
                ]
            )

        df["regulation_id"] = regulation_id
        df = (
            df.sort_values(
                ["pokemon", "avg_correlation"],
                ascending=[True, False],
            )
            .groupby("pokemon")
            .head(top_n_teammates)
            .reset_index(drop=True)
        )
        return df
    except Exception as exc:
        log.warning(
            "Error construyendo teammates "
            "para %s: %s",
            regulation_id,
            exc,
        )
        return pd.DataFrame(
            columns=[
                "pokemon",
                "teammate",
                "avg_correlation",
                "n_months_seen",
                "regulation_id",
            ]
        )


def create_snapshot(
    regulation_id: str,
    label: str,
    top_n_items: int = 5,
    top_n_teammates: int = 5,
    overwrite: bool = False,
) -> Path:
    """
    Crea un snapshot completo del meta para
    una regulación y lo guarda en disco.

    Args:
        regulation_id: ID de la regulación.
        label: Etiqueta del snapshot
               (ej: "post_GC1_2026").
        top_n_items: Top N ítems por Pokémon.
        top_n_teammates: Top N compañeros.
        overwrite: Si True, sobreescribe snapshot
                   existente.

    Returns:
        Path al directorio del snapshot creado.

    Raises:
        FileExistsError: Si el snapshot ya existe
                          y overwrite=False.
    """
    snapshot_dir = _SNAPSHOTS_DIR / label

    if snapshot_dir.exists() and not overwrite:
        raise FileExistsError(
            f"El snapshot '{label}' ya existe en "
            f"{snapshot_dir}. "
            f"Usa --overwrite para sobreescribir."
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Conectar DuckDB en memoria
    con = duckdb.connect(":memory:")

    try:
        # Construir DataFrames
        log.info(
            "Construyendo snapshot '%s' "
            "para regulación %s...",
            label,
            regulation_id,
        )

        df_usage = _build_usage_top50(con, regulation_id)
        df_items = _build_items_by_pkm(
            con, regulation_id, top_n_items
        )
        df_teammates = _build_teammates_by_pkm(
            con, regulation_id, top_n_teammates
        )

        # Contar datos disponibles
        n_replays = _count_replays(regulation_id)
        n_pikalytics = _count_pikalytics_entries(
            regulation_id
        )

        # Guardar CSVs
        usage_path = snapshot_dir / "usage_top50.csv"
        items_path = snapshot_dir / "items_by_pkm.csv"
        teammates_path = (
            snapshot_dir / "teammates_by_pkm.csv"
        )

        df_usage.to_csv(usage_path, index=False)
        df_items.to_csv(items_path, index=False)
        df_teammates.to_csv(teammates_path, index=False)

        # Guardar metadata
        meta = {
            "label": label,
            "regulation_id": regulation_id,
            "snapshot_date": datetime.now(
                timezone.utc
            ).isoformat(),
            "n_replays": n_replays,
            "n_pikalytics_entries": n_pikalytics,
            "n_pokemon_in_usage": len(df_usage),
            "n_item_rows": len(df_items),
            "n_teammate_rows": len(df_teammates),
            "top_n_items": top_n_items,
            "top_n_teammates": top_n_teammates,
            "files": {
                "usage": "usage_top50.csv",
                "items": "items_by_pkm.csv",
                "teammates": "teammates_by_pkm.csv",
            },
        }
        meta_path = snapshot_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        log.info(
            "Snapshot '%s' creado en %s\n"
            "  Regulación:   %s\n"
            "  Replays:      %d\n"
            "  Pikalytics:   %d entradas\n"
            "  Pokémon top:  %d\n"
            "  Filas ítems:  %d\n"
            "  Filas compañ: %d",
            label,
            snapshot_dir,
            regulation_id,
            n_replays,
            n_pikalytics,
            len(df_usage),
            len(df_items),
            len(df_teammates),
        )

        return snapshot_dir

    finally:
        con.close()


def main() -> int:
    """
    Entry point del script.

    Uso:
        # Snapshot de la regulación activa:
        python scripts/snapshot_meta.py
            --label post_GC1_2026

        # Regulación específica:
        python scripts/snapshot_meta.py
            --reg M-A
            --label post_GC1_2026

        # Sobreescribir existente:
        python scripts/snapshot_meta.py
            --reg M-A
            --label post_GC1_2026
            --overwrite

        # Verificar snapshot existente:
        python scripts/snapshot_meta.py
            --label post_GC1_2026
            --inspect
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Congela el estado del meta en un "
            "snapshot para referencia histórica "
            "y retrain del modelo WP."
        )
    )
    parser.add_argument(
        "--reg",
        default=None,
        help=(
            "Regulación a snapshottear. "
            "Default: auto-detect activa."
        ),
    )
    parser.add_argument(
        "--label",
        required=True,
        help=(
            "Etiqueta del snapshot. "
            "Ej: post_GC1_2026, post_Indianapolis_2026"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Sobreescribir snapshot si ya existe."
        ),
    )
    parser.add_argument(
        "--top-items",
        type=int,
        default=5,
        dest="top_items",
        help="Top N ítems por Pokémon (default: 5).",
    )
    parser.add_argument(
        "--top-teammates",
        type=int,
        default=5,
        dest="top_teammates",
        help=(
            "Top N compañeros por Pokémon "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Mostrar contenido de un snapshot "
            "existente sin crear uno nuevo."
        ),
    )
    args = parser.parse_args()

    # Modo inspect
    if args.inspect:
        snapshot_dir = _SNAPSHOTS_DIR / args.label
        meta_path = snapshot_dir / "meta.json"
        if not meta_path.exists():
            log.error(
                "Snapshot '%s' no encontrado en %s",
                args.label,
                snapshot_dir,
            )
            return 1
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        print(
            f"\n{'=' * 50}\n"
            f"SNAPSHOT: {args.label}\n"
            f"{'=' * 50}"
        )
        for key, value in meta.items():
            if key != "files":
                print(f"  {key}: {value}")
        print(f"{'=' * 50}\n")
        return 0

    # Resolver regulación
    reg_id = args.reg
    if not reg_id:
        try:
            from src.app.core.regulation_active import (
                get_active_regulation,
            )

            active = get_active_regulation()
            reg_id = active.regulation_id
            log.info(
                "Regulación activa detectada: %s",
                reg_id,
            )
        except Exception as exc:
            log.error(
                "No se pudo detectar regulación "
                "activa: %s",
                exc,
            )
            return 1

    # Crear snapshot
    try:
        snapshot_dir = create_snapshot(
            regulation_id=reg_id,
            label=args.label,
            top_n_items=args.top_items,
            top_n_teammates=args.top_teammates,
            overwrite=args.overwrite,
        )
        print(
            f"\n✅ Snapshot '{args.label}' creado "
            f"en {snapshot_dir}\n"
            f"Inspeccionar con:\n"
            f"  python scripts/snapshot_meta.py "
            f"--label {args.label} --inspect\n"
        )
        return 0

    except FileExistsError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.error(
            "Error creando snapshot: %s",
            exc,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
