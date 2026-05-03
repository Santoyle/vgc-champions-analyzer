"""
LRE (Lead Recommendation Engine) — Sub-bloque 15D del pipeline Champions.

Este módulo define el DDL de ``lead_pair_stats`` y rellena estadísticas
agregadas de pares de leads (doubles VGC) a partir de ``replay_turns``.
La función ``populate_lead_pair_stats()`` materializa la tabla en DuckDB;
``lead_recommendation()`` (consulta/recomendación sobre esos datos) se
implementa en T-127.
"""

from __future__ import annotations

import itertools
import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

from src.app.core.champions_calc import (
    _load_pokemon_master,
    type_effectiveness,
)

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


def _cold_start_score(
    my_lead_a: str,
    my_lead_b: str,
    opp_lead_a: str,
    opp_lead_b: str,
    pokemon_master: dict[int, dict[str, Any]],
) -> float:
    """
    Calcula un score de type matchup para el
    cold start (sin datos históricos suficientes).

    Suma la efectividad de tipo de cada Pokémon
    del equipo propio contra los del rival y
    normaliza a [0.35, 0.65].

    Args:
        my_lead_a: Primer Pokémon del equipo.
        my_lead_b: Segundo Pokémon del equipo.
        opp_lead_a: Primer Pokémon rival.
        opp_lead_b: Segundo Pokémon rival.
        pokemon_master: Datos maestros.

    Returns:
        Float en [0.35, 0.65].
    """

    def get_types(slug: str) -> list[str]:
        for entry in pokemon_master.values():
            if str(
                entry.get("name", "")
            ).lower() == slug.lower():
                return [
                    str(t).title()
                    for t in entry.get("types", [])
                ]
        return []

    my_types_a = get_types(my_lead_a)
    my_types_b = get_types(my_lead_b)
    opp_types_a = get_types(opp_lead_a)
    opp_types_b = get_types(opp_lead_b)

    score = 0.0
    comparisons = 0

    for my_types in [my_types_a, my_types_b]:
        for opp_types in [opp_types_a, opp_types_b]:
            if my_types and opp_types:
                for my_t in my_types:
                    eff = type_effectiveness(
                        my_t, opp_types
                    )
                    score += eff
                    comparisons += 1

    if comparisons == 0:
        return 0.5

    raw = score / comparisons
    # Normalizar a [0.35, 0.65]
    normalized = 0.35 + (
        min(raw, 4.0) / 4.0
    ) * 0.30
    return round(normalized, 4)


@st.cache_data(ttl=1800, show_spinner=False)  # type: ignore[untyped-decorator]
def lead_recommendation(
    reg_id: str,
    my_team_slugs: tuple[str, ...],
    opp_team_slugs: tuple[str, ...],
    _con: Any,
    min_n_per_cell: int = 5,
) -> pd.DataFrame:
    """
    Recomienda los mejores pares de leads contra
    un equipo rival usando expected value.

    Calcula EV para las C(6,2)=15 combinaciones
    de leads propios contra las C(6,2)=15
    combinaciones de leads del rival.

    Algoritmo (Opción B — expected value):
    1. Para cada opp_lead_pair (15 combinaciones
       del rival): obtener frecuencia histórica
       desde lead_pair_stats (smoothing Laplace).
    2. Normalizar para obtener P(opp_lead).
    3. Para cada my_lead_pair (15 combinaciones):
       EV = Σ P(opp_lead) × WR(my_lead | opp_lead)
       Cuando n < min_n_per_cell: usar cold start.
    4. Retornar top-15 ordenado por EV DESC.

    Args:
        reg_id: Regulación activa.
        my_team_slugs: Tuple con los 6 slugs del
                        equipo propio.
        opp_team_slugs: Tuple con los 6 slugs del
                         equipo rival.
        _con: Conexión DuckDB (prefijo _ para que
              st.cache_data no la hashee).
        min_n_per_cell: Mínimo de partidas para
                         usar datos reales vs
                         cold start fallback.

    Returns:
        DataFrame con columnas:
        my_lead_a, my_lead_b, expected_win_prob,
        n_matchups_with_data, regulation_id.
        15 filas ordenadas por expected_win_prob DESC.
        DataFrame vacío si hay error crítico.
    """
    if len(my_team_slugs) < 2:
        log.warning(
            "my_team_slugs necesita al menos 2 "
            "Pokémon, recibidos: %d",
            len(my_team_slugs),
        )
        return pd.DataFrame(columns=[
            "my_lead_a", "my_lead_b",
            "expected_win_prob",
            "n_matchups_with_data",
            "regulation_id",
        ])

    if len(opp_team_slugs) < 2:
        log.warning(
            "opp_team_slugs necesita al menos 2 "
            "Pokémon, recibidos: %d",
            len(opp_team_slugs),
        )
        return pd.DataFrame(columns=[
            "my_lead_a", "my_lead_b",
            "expected_win_prob",
            "n_matchups_with_data",
            "regulation_id",
        ])

    pokemon_master = _load_pokemon_master()

    # Cargar datos de lead_pair_stats
    try:
        df_lps = _con.execute(
            """
            SELECT
                my_lead_a, my_lead_b,
                opp_lead_a, opp_lead_b,
                n_matches, win_rate
            FROM lead_pair_stats
            WHERE regulation_id = ?
            """,
            [reg_id],
        ).df()
    except Exception as exc:
        log.warning(
            "Error cargando lead_pair_stats: %s",
            exc,
        )
        df_lps = pd.DataFrame()

    # Generar todas las combinaciones C(n,2)
    my_pairs = list(itertools.combinations(
        sorted(my_team_slugs), 2
    ))
    opp_pairs = list(itertools.combinations(
        sorted(opp_team_slugs), 2
    ))

    # Calcular frecuencia histórica de leads
    # del rival (smoothing Laplace)
    opp_freq: dict[tuple[str, str], float] = {}
    for opp_a, opp_b in opp_pairs:
        opp_a_s, opp_b_s = sorted([opp_a, opp_b])
        if not df_lps.empty:
            mask = (
                (df_lps["opp_lead_a"] == opp_a_s)
                & (df_lps["opp_lead_b"] == opp_b_s)
            )
            n_seen = int(
                df_lps[mask]["n_matches"].sum()
            )
        else:
            n_seen = 0
        # Laplace smoothing: max(n, 1)
        opp_freq[(opp_a_s, opp_b_s)] = max(
            n_seen, 1
        )

    total_opp_freq = sum(opp_freq.values())

    # Calcular EV para cada par propio
    rows = []
    for my_a, my_b in my_pairs:
        my_a_s, my_b_s = sorted([my_a, my_b])
        ev = 0.0
        n_with_data = 0

        for (opp_a_s, opp_b_s), freq in opp_freq.items():
            p_opp = freq / total_opp_freq

            # Buscar WR en lead_pair_stats
            wr: float | None = None
            if not df_lps.empty:
                mask = (
                    (df_lps["my_lead_a"] == my_a_s)
                    & (df_lps["my_lead_b"] == my_b_s)
                    & (df_lps["opp_lead_a"] == opp_a_s)
                    & (df_lps["opp_lead_b"] == opp_b_s)
                )
                matched = df_lps[mask]
                if not matched.empty:
                    n_cell = int(
                        matched["n_matches"].iloc[0]
                    )
                    if n_cell >= min_n_per_cell:
                        wr = float(
                            matched["win_rate"].iloc[0]
                        )
                        n_with_data += 1

            if wr is None:
                # Cold start fallback
                wr = _cold_start_score(
                    my_a_s, my_b_s,
                    opp_a_s, opp_b_s,
                    pokemon_master,
                )

            ev += p_opp * wr

        rows.append({
            "my_lead_a":            my_a_s,
            "my_lead_b":            my_b_s,
            "expected_win_prob":    round(ev, 4),
            "n_matchups_with_data": n_with_data,
            "regulation_id":        reg_id,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "expected_win_prob", ascending=False
        )
        .reset_index(drop=True)
    )


__all__ = [
    "populate_lead_pair_stats",
    "lead_recommendation",
    "LEAD_PAIR_STATS_DDL",
]
