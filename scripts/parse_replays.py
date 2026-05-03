from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DB_PATH = _DATA_DIR / "champions.duckdb"

REPLAY_TURNS_DDL = """
CREATE TABLE IF NOT EXISTS replay_turns (
    regulation_id     VARCHAR   NOT NULL,
    replay_id         VARCHAR   NOT NULL,
    turn              SMALLINT  NOT NULL,
    action_idx        SMALLINT  NOT NULL,
    player            VARCHAR   NOT NULL,
    slot              VARCHAR   NOT NULL,
    pokemon_slug      VARCHAR,
    move_slug         VARCHAR,
    target_player     VARCHAR,
    target_slot       VARCHAR,
    hp_before_pct     FLOAT,
    hp_after_pct      FLOAT,
    ko_dealt          BOOLEAN   DEFAULT FALSE,
    ko_received       BOOLEAN   DEFAULT FALSE,
    weather           VARCHAR,
    terrain           VARCHAR,
    trick_room_active BOOLEAN   DEFAULT FALSE,
    reflect_p1        BOOLEAN   DEFAULT FALSE,
    reflect_p2        BOOLEAN   DEFAULT FALSE,
    light_screen_p1   BOOLEAN   DEFAULT FALSE,
    light_screen_p2   BOOLEAN   DEFAULT FALSE,
    tailwind_p1_turns SMALLINT  DEFAULT 0,
    tailwind_p2_turns SMALLINT  DEFAULT 0,
    mega_used_p1      BOOLEAN   DEFAULT FALSE,
    mega_used_p2      BOOLEAN   DEFAULT FALSE,
    mega_pokemon_slug VARCHAR,
    ko_diff           SMALLINT  DEFAULT 0,
    winner            VARCHAR   NOT NULL,
    PRIMARY KEY (
        regulation_id, replay_id,
        turn, action_idx
    )
)
"""


def main() -> int:
    """
    CLI para parsear replays de Showdown y
    cargar en replay_turns de DuckDB.

    Uso:
        # Parsear todos los Parquets de M-A:
        python scripts/parse_replays.py --reg M-A

        # Dry-run (parsear sin guardar en DB):
        python scripts/parse_replays.py
            --reg M-A --dry-run

        # Regulación activa auto-detectada:
        python scripts/parse_replays.py
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
            "Parsea logs de replays Showdown "
            "y los carga en la tabla replay_turns "
            "de DuckDB."
        )
    )
    parser.add_argument(
        "--reg",
        default=None,
        help=(
            "Regulación a parsear. "
            "Default: auto-detect activa."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Parsear sin guardar en DuckDB. "
            "Útil para testing."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=str(_DB_PATH),
        dest="db_path",
        help=(
            f"Path al archivo DuckDB. "
            f"Default: {_DB_PATH}"
        ),
    )
    args = parser.parse_args()

    # Resolver regulación
    reg_id = args.reg
    if not reg_id:
        try:
            from src.app.core.regulation_active import (
                get_active_regulation,
            )

            active = get_active_regulation()
            reg_id = active.regulation_id
            log.info("Regulación activa: %s", reg_id)
        except Exception as exc:
            log.error(
                "No se pudo detectar regulación "
                "activa: %s",
                exc,
            )
            return 1

    # Buscar Parquets de Showdown
    showdown_dir = (
        _DATA_DIR
        / "raw"
        / f"reg={reg_id}"
        / "source=showdown"
    )
    if not showdown_dir.exists():
        log.error(
            "No hay directorio de Showdown para "
            "%s: %s",
            reg_id,
            showdown_dir,
        )
        return 1

    parquet_files = list(showdown_dir.glob("*.parquet"))
    if not parquet_files:
        log.error(
            "Sin Parquets de Showdown para %s",
            reg_id,
        )
        return 1

    log.info(
        "Encontrados %d Parquets para %s",
        len(parquet_files),
        reg_id,
    )

    # Parsear todos los Parquets
    from src.app.data.parsers.replay_parser import (
        parse_replays_from_parquet,
    )

    all_dfs: list[pd.DataFrame] = []
    for pq_file in sorted(parquet_files):
        log.info("Parseando %s...", pq_file.name)
        df = parse_replays_from_parquet(str(pq_file), reg_id)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        log.warning(
            "Sin filas generadas para %s. "
            "Verifica que los Parquets tienen "
            "el campo raw_log poblado.",
            reg_id,
        )
        return 0

    df_total = pd.concat(all_dfs, ignore_index=True)

    # Deduplicar por PK
    pk_cols = [
        "regulation_id",
        "replay_id",
        "turn",
        "action_idx",
    ]
    n_before = len(df_total)
    df_total = df_total.drop_duplicates(subset=pk_cols)
    n_dedup = n_before - len(df_total)
    if n_dedup > 0:
        log.info(
            "Eliminadas %d filas duplicadas",
            n_dedup,
        )

    log.info(
        "Total filas a insertar: %d",
        len(df_total),
    )

    if args.dry_run:
        print(
            f"\n[DRY RUN] {len(df_total)} filas "
            f"generadas para replay_turns "
            f"(no se guardaron en DuckDB)\n"
        )
        print(df_total.head(10).to_string())
        return 0

    # Guardar en DuckDB
    con = duckdb.connect(args.db_path)
    try:
        # Crear tabla si no existe
        con.execute(REPLAY_TURNS_DDL)

        # Insertar con ON CONFLICT DO NOTHING
        # (deduplicación por PK)
        con.register("df_insert", df_total)
        con.execute("""
            INSERT OR IGNORE INTO replay_turns
            SELECT * FROM df_insert
        """)

        # Verificar resultado
        n_rows = con.execute(
            """
            SELECT COUNT(*) FROM replay_turns
            WHERE regulation_id = ?
        """,
            [reg_id],
        ).fetchone()[0]

        log.info(
            "replay_turns para %s: %d filas totales",
            reg_id,
            n_rows,
        )
        print(
            f"\n✅ Parseados {len(df_total)} "
            f"registros → replay_turns\n"
            f"   Regulación: {reg_id}\n"
            f"   Total en tabla: {n_rows}\n"
            f"   DB: {args.db_path}\n"
        )
        return 0

    except Exception as exc:
        log.error(
            "Error guardando en DuckDB: %s",
            exc,
        )
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
