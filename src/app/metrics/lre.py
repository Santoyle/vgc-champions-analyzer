"""
LRE (Lead Recommendation Engine) — Sub-bloque 15D del pipeline Champions.

Este módulo define el DDL de ``lead_pair_stats`` y rellena estadísticas
agregadas de pares de leads (doubles VGC) a partir de ``replay_turns``.
La función ``populate_lead_pair_stats()`` materializa la tabla en DuckDB;
``lead_recommendation()`` (consulta/recomendación sobre esos datos) se
implementa en T-127.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

LEAD_PAIR_STATS_DDL = """
CREATE TABLE IF NOT EXISTS lead_pair_stats (
    regulation_id     VARCHAR  NOT NULL,
    my_lead_a         VARCHAR  NOT NULL,
    my_lead_b         VARCHAR  NOT NULL,
    opp_lead_a        VARCHAR  NOT NULL,
    opp_lead_b        VARCHAR  NOT NULL,
    n_matches         INTEGER  DEFAULT 0,
    n_wins            INTEGER  DEFAULT 0,
    win_rate          FLOAT,
    win_rate_lower_95 FLOAT,
    win_rate_upper_95 FLOAT,
    last_updated      TIMESTAMP,
    PRIMARY KEY (
        regulation_id,
        my_lead_a, my_lead_b,
        opp_lead_a, opp_lead_b
    )
)
"""


def _extract_leads_from_replay_turns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extrae los leads (Pokémon del turno 1) de
    cada replay para cada jugador.

    En Champions VGC Doubles, los leads son los
    2 Pokémon que aparecen en turn=1 para cada
    jugador. Se detectan buscando filas con
    turn=1 y pokemon_slug no nulo.

    Args:
        df: DataFrame con columnas regulation_id,
            replay_id, turn, player, pokemon_slug,
            winner.

    Returns:
        DataFrame con columnas:
        regulation_id, replay_id, player,
        lead_a, lead_b, winner.
        lead_a y lead_b ordenados alfabéticamente.
        Una fila por (replay_id, player).
    """
    turn1 = df[
        (df["turn"] == 1)
        & df["pokemon_slug"].notna()
    ].copy()

    if turn1.empty:
        return pd.DataFrame(columns=[
            "regulation_id", "replay_id",
            "player", "lead_a", "lead_b", "winner",
        ])

    # Agrupar por replay + player, tomar los
    # primeros 2 Pokémon únicos
    leads_list = []
    grouped = turn1.groupby(
        ["regulation_id", "replay_id", "player",
         "winner"]
    )

    for (reg_id, replay_id, player, winner), grp in grouped:
        pkm_list = (
            grp["pokemon_slug"]
            .unique()
            .tolist()
        )
        if len(pkm_list) < 2:
            continue
        # Tomar los 2 primeros y ordenar
        lead_a, lead_b = sorted(pkm_list[:2])
        leads_list.append({
            "regulation_id": reg_id,
            "replay_id":     replay_id,
            "player":        player,
            "lead_a":        lead_a,
            "lead_b":        lead_b,
            "winner":        winner,
        })

    if not leads_list:
        return pd.DataFrame(columns=[
            "regulation_id", "replay_id",
            "player", "lead_a", "lead_b", "winner",
        ])

    return pd.DataFrame(leads_list)


def _compute_wilson_ci(
    n_wins: int,
    n_matches: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Calcula el intervalo de confianza Wilson
    para una proporción.

    Args:
        n_wins: Número de éxitos.
        n_matches: Número total de intentos.
        confidence: Nivel de confianza.

    Returns:
        Tupla (lower, upper) del IC.
        (0.0, 1.0) si n_matches == 0.
    """
    if n_matches == 0:
        return 0.0, 1.0
    try:
        from statsmodels.stats.proportion import (
            proportion_confint,
        )
        lo, hi = proportion_confint(
            count=n_wins,
            nobs=n_matches,
            alpha=1 - confidence,
            method="wilson",
        )
        return float(lo), float(hi)
    except Exception as exc:
        log.warning(
            "Error calculando Wilson CI: %s", exc
        )
        wr = n_wins / n_matches
        return max(0.0, wr - 0.1), min(1.0, wr + 0.1)


def populate_lead_pair_stats(
    reg_id: str,
    _con_path: str,
) -> None:
    """
    Parsea replay_turns y popula la tabla
    lead_pair_stats en DuckDB.

    Algoritmo:
    1. Cargar replay_turns con turn=1 para reg_id
    2. Extraer leads (2 Pokémon por jugador en T1)
    3. Para cada replay, cruzar leads de p1 vs p2
    4. Computar n_matches, n_wins, WR, Wilson CI
    5. UPSERT en lead_pair_stats (INSERT OR REPLACE)

    La perspectiva es siempre desde p1:
    - my_lead = leads del p1
    - opp_lead = leads del p2
    - win = (winner == "p1")

    Args:
        reg_id: Regulación a procesar.
        _con_path: Path al archivo DuckDB.
    """
    con = duckdb.connect(_con_path)

    try:
        # Crear tabla si no existe
        con.execute(LEAD_PAIR_STATS_DDL)

        # Cargar turn=1 desde replay_turns
        df = con.execute(
            """
            SELECT DISTINCT
                regulation_id,
                replay_id,
                turn,
                player,
                pokemon_slug,
                winner
            FROM replay_turns
            WHERE regulation_id = ?
              AND turn = 1
              AND pokemon_slug IS NOT NULL
            """,
            [reg_id],
        ).df()

        if df.empty:
            log.warning(
                "Sin datos de turn=1 para %s",
                reg_id,
            )
            return

        # Extraer leads por jugador
        leads_df = _extract_leads_from_replay_turns(
            df
        )

        if leads_df.empty:
            log.warning(
                "Sin leads extraídos para %s",
                reg_id,
            )
            return

        # Cruzar p1 vs p2 por replay_id
        p1_leads = leads_df[
            leads_df["player"] == "p1"
        ].rename(columns={
            "lead_a": "my_lead_a",
            "lead_b": "my_lead_b",
        })
        p2_leads = leads_df[
            leads_df["player"] == "p2"
        ].rename(columns={
            "lead_a": "opp_lead_a",
            "lead_b": "opp_lead_b",
        })

        matchups = p1_leads.merge(
            p2_leads[
                ["replay_id", "opp_lead_a",
                 "opp_lead_b"]
            ],
            on="replay_id",
            how="inner",
        )

        if matchups.empty:
            log.warning(
                "Sin matchups p1 vs p2 para %s",
                reg_id,
            )
            return

        matchups["won"] = (
            matchups["winner"] == "p1"
        )

        # Agrupar por par de leads
        grouped = (
            matchups
            .groupby([
                "my_lead_a", "my_lead_b",
                "opp_lead_a", "opp_lead_b",
            ])
            .agg(
                n_matches=("won", "count"),
                n_wins=("won", "sum"),
            )
            .reset_index()
        )

        # Calcular WR y Wilson CI
        now = datetime.now(timezone.utc)
        rows = []
        for _, row in grouped.iterrows():
            n_m = int(row["n_matches"])
            n_w = int(row["n_wins"])
            wr = n_w / n_m if n_m > 0 else 0.5
            lo, hi = _compute_wilson_ci(n_w, n_m)
            rows.append({
                "regulation_id":     reg_id,
                "my_lead_a":         row["my_lead_a"],
                "my_lead_b":         row["my_lead_b"],
                "opp_lead_a":        row["opp_lead_a"],
                "opp_lead_b":        row["opp_lead_b"],
                "n_matches":         n_m,
                "n_wins":            n_w,
                "win_rate":          round(wr, 4),
                "win_rate_lower_95": round(lo, 4),
                "win_rate_upper_95": round(hi, 4),
                "last_updated":      now,
            })

        if not rows:
            return

        df_insert = pd.DataFrame(rows)
        con.register("df_lps", df_insert)
        con.execute("""
            INSERT OR REPLACE INTO lead_pair_stats
            SELECT * FROM df_lps
        """)

        n_rows = con.execute("""
            SELECT COUNT(1) FROM lead_pair_stats
            WHERE regulation_id = ?
        """, [reg_id]).fetchone()[0]

        log.info(
            "lead_pair_stats para %s: %d filas",
            reg_id, n_rows,
        )

    finally:
        con.close()


__all__ = [
    "populate_lead_pair_stats",
    "LEAD_PAIR_STATS_DDL",
]
