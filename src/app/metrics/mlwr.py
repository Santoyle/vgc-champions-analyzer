"""
MLWR — Move-Level Win Rate (métricas de replays Champions).

Versión **L1**: correlación simple entre usar un movimiento en batalla y el
desenlace (**sin** ajuste por game state; eso será **MLWR L2** con XGBoost + SHAP
en T-119).

``replay_turns.winner`` a veces trae username del ganador (no ``p1``/``p2``);
``_resolve_winner_column`` lo detecta y evita estadísticas falsas hasta que el
pipeline normalice el campo.

``compute_meti()`` implementa **METI** (Mega Evolution Timing Index) del
Sub-bloque 15E; ``compute_meit()`` se implementa en T-131.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import duckdb
import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


def _resolve_winner_column(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Resuelve si el jugador activo (player) ganó
    la batalla.

    El campo winner en replay_turns contiene el
    username del ganador (ej "trainername"), no
    "p1"/"p2". Para determinar si el jugador
    activo ganó, comparamos player con p1_name
    y p2_name extraídos del replay.

    Como no tenemos p1_name/p2_name en la tabla
    actual, usamos una heurística:
    - Si winner == "p1" → el jugador "p1" ganó
    - Si winner == "p2" → el jugador "p2" ganó
    - Si winner es un username → no podemos
      resolver directamente, retornar NaN

    Esto es un workaround hasta que el parser
    normalice winner a "p1"/"p2".

    Args:
        df: DataFrame con columnas player y winner.

    Returns:
        Serie booleana: True si el player activo
        ganó, NaN si no se puede determinar.
    """
    # Caso ideal: winner ya es "p1" o "p2"
    is_p1_p2 = df["winner"].isin(["p1", "p2"])

    if is_p1_p2.mean() > 0.5:
        # La mayoría son p1/p2 — usar directamente
        return (df["player"] == df["winner"]).where(
            is_p1_p2
        )

    # Fallback: winner es username
    # No podemos determinar ganador sin join
    # con tabla de replays — retornar NaN
    log.warning(
        "winner column contains usernames, "
        "not p1/p2. MLWR results may be unreliable. "
        "Fix parse_replays.py to normalize winner."
    )
    return pd.Series(
        [float("nan")] * len(df),
        index=df.index,
        dtype=float,
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)  # type: ignore[untyped-decorator]
def compute_mlwr_level1(
    reg_id: str,
    _con: Any,
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Calcula el MLWR Nivel 1 para todos los pares
    (Pokémon, move) de una regulación.

    Nivel 1 = sin ajuste por game state.
    Solo mide la correlación cruda entre usar un
    move y ganar la batalla.

    Algoritmo:
    1. Cargar replay_turns desde DuckDB para reg_id
    2. Resolver columna winner → bool (ganó/no ganó)
    3. Calcular WR baseline por pokemon_slug
    4. Calcular WR por (pokemon_slug, move_slug)
    5. Filtrar pares con n_uses >= min_n
    6. MLWR = wr_with_move - wr_baseline
    7. Wilson 95% CI sobre wr_with_move

    Args:
        reg_id: Regulación a analizar.
        _con: Conexión DuckDB (prefijo _ para que
              st.cache_data no la hashee).
        min_n: Mínimo de usos para incluir un par.

    Returns:
        DataFrame con columnas:
        pokemon_slug, move_slug, n_uses,
        wr_with_move, wr_baseline, mlwr,
        wilson_low, wilson_high, regulation_id.
        Ordenado por mlwr DESC.
        DataFrame vacío si no hay datos.
    """
    try:
        # Cargar datos
        df = _con.execute(
            "SELECT regulation_id, replay_id, player, "
            "pokemon_slug, move_slug, winner "
            "FROM replay_turns "
            "WHERE regulation_id = ? "
            "AND pokemon_slug IS NOT NULL",
            [reg_id],
        ).df()
    except Exception as exc:
        log.warning(
            "Error cargando replay_turns para %s: %s",
            reg_id,
            exc,
        )
        return pd.DataFrame(
            columns=[
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        )

    if df.empty:
        log.info(
            "Sin datos en replay_turns para %s",
            reg_id,
        )
        return pd.DataFrame(
            columns=[
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        )

    # Resolver si ganó
    df["won"] = _resolve_winner_column(df)

    # Eliminar filas donde no se puede determinar
    df_valid = df.dropna(subset=["won"]).copy()
    df_valid["won"] = df_valid["won"].astype(bool)

    if df_valid.empty:
        log.warning(
            "No se pudo determinar ganador para "
            "ninguna fila de %s. "
            "Verificar que winner está normalizado "
            "a p1/p2 en el parser.",
            reg_id,
        )
        return pd.DataFrame(
            columns=[
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        )

    # Baseline WR por pokemon: una fila por
    # (replay_id, player, pokemon_slug) único
    df_per_battle = df_valid.drop_duplicates(
        subset=["replay_id", "player", "pokemon_slug"]
    )
    baseline = (
        df_per_battle.groupby("pokemon_slug")["won"]
        .agg(wr_baseline="mean", n_battles="count")
        .reset_index()
    )

    # WR por (pokemon_slug, move_slug)
    df_moves = df_valid[
        df_valid["move_slug"].notna()
    ].copy()

    if df_moves.empty:
        return pd.DataFrame(
            columns=[
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        )

    # Una fila por (replay_id, player, pokemon,
    # move) para no contar múltiples usos del
    # mismo move en la misma batalla
    df_move_per_battle = df_moves.drop_duplicates(
        subset=[
            "replay_id",
            "player",
            "pokemon_slug",
            "move_slug",
        ]
    )

    move_stats = (
        df_move_per_battle.groupby(
            ["pokemon_slug", "move_slug"]
        )["won"]
        .agg(wr_with_move="mean", n_uses="count")
        .reset_index()
    )

    # Filtrar por mínimo de usos
    move_stats = move_stats[
        move_stats["n_uses"] >= min_n
    ].copy()

    if move_stats.empty:
        log.info(
            "Sin pares (pokemon, move) con n >= %d "
            "para %s",
            min_n,
            reg_id,
        )
        return pd.DataFrame(
            columns=[
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        )

    # Unir con baseline
    result = move_stats.merge(
        baseline[["pokemon_slug", "wr_baseline"]],
        on="pokemon_slug",
        how="left",
    )
    result["wr_baseline"] = result[
        "wr_baseline"
    ].fillna(0.5)

    # MLWR
    result["mlwr"] = (
        result["wr_with_move"] - result["wr_baseline"]
    )

    # Wilson 95% CI
    try:
        from statsmodels.stats.proportion import (
            proportion_confint,
        )

        counts = (
            result["wr_with_move"] * result["n_uses"]
        ).round().astype(int)
        lo, hi = proportion_confint(
            count=counts,
            nobs=result["n_uses"],
            alpha=0.05,
            method="wilson",
        )
        result["wilson_low"] = (
            lo - result["wr_baseline"]
        )
        result["wilson_high"] = (
            hi - result["wr_baseline"]
        )
    except Exception as exc:
        log.warning(
            "Error calculando Wilson CI: %s",
            exc,
        )
        result["wilson_low"] = result["mlwr"] - 0.1
        result["wilson_high"] = result["mlwr"] + 0.1

    result["regulation_id"] = reg_id

    return (
        result[
            [
                "pokemon_slug",
                "move_slug",
                "n_uses",
                "wr_with_move",
                "wr_baseline",
                "mlwr",
                "wilson_low",
                "wilson_high",
                "regulation_id",
            ]
        ]
        .sort_values("mlwr", ascending=False)
        .reset_index(drop=True)
    )


def _get_mega_activation_turn(
    group: pd.DataFrame,
) -> int | None:
    """
    Dado un grupo de filas de un replay para
    un jugador específico, determina el turno
    en que activó la Mega Evolución.

    Detecta el primer turno donde mega_used_p1
    (o mega_used_p2 según el player) cambia de
    False a True.

    Args:
        group: DataFrame filtrado por
               (replay_id, player) con columnas
               turn, mega_used_p1, mega_used_p2,
               player.

    Returns:
        Número de turno de activación (1-based).
        None si nunca activó Mega en este replay.
    """
    if group.empty:
        return None

    player = str(group["player"].iloc[0])
    mega_col = (
        "mega_used_p1"
        if player == "p1"
        else "mega_used_p2"
    )

    if mega_col not in group.columns:
        return None

    sorted_g = group.sort_values("turn")
    mega_vals = sorted_g[mega_col].tolist()
    turns = sorted_g["turn"].tolist()

    prev = False
    for i, val in enumerate(mega_vals):
        if val and not prev:
            return int(turns[i])
        prev = bool(val)

    return None


def _turn_to_bucket(
    turn: int | None,
) -> Literal["T1", "T2", "T3", "T4+", "never"]:
    """
    Convierte un turno de activación a bucket.

    Args:
        turn: Turno de activación (1-based)
              o None si nunca activó.

    Returns:
        "T1", "T2", "T3", "T4+" o "never".
    """
    if turn is None:
        return "never"
    if turn == 1:
        return "T1"
    if turn == 2:
        return "T2"
    if turn == 3:
        return "T3"
    return "T4+"


@st.cache_data(ttl=3600, show_spinner=False)  # type: ignore[untyped-decorator]
def compute_meti(
    reg_id: str,
    _con: Any,
    mega_enabled: bool = True,
) -> pd.DataFrame:
    """
    Calcula el METI (Mega Evolution Timing Index)
    para todos los Pokémon que Megaevolucionan
    en una regulación.

    Mide si el turno de activación de la Mega
    está correlacionado con ganar la batalla.

    Buckets: T1, T2, T3, T4+, never.
    METI_delta = WR(bucket) - WR_baseline.

    Solo ejecutar si mega_enabled=True.
    Si mega_enabled=False, retorna DataFrame vacío
    con columnas correctas (not_applicable).

    Args:
        reg_id: Regulación a analizar.
        _con: Conexión DuckDB (prefijo _ para
              que st.cache_data no la hashee).
        mega_enabled: Si False, retorna vacío
                      inmediatamente.

    Returns:
        DataFrame con columnas:
        pokemon_slug, turn_bucket, n_matches,
        win_rate, baseline_wr, meti_delta,
        regulation_id.
        Ordenado por pokemon_slug, turn_bucket.
        DataFrame vacío si mega_enabled=False
        o sin datos.
    """
    empty_cols = [
        "pokemon_slug", "turn_bucket",
        "n_matches", "win_rate",
        "baseline_wr", "meti_delta",
        "regulation_id",
    ]

    if not mega_enabled:
        log.info(
            "METI no aplica para %s "
            "(mega_enabled=False)",
            reg_id,
        )
        return pd.DataFrame(columns=empty_cols)

    try:
        df = _con.execute(
            """
            SELECT
                replay_id,
                turn,
                player,
                pokemon_slug,
                mega_used_p1,
                mega_used_p2,
                winner
            FROM replay_turns
            WHERE regulation_id = ?
              AND pokemon_slug IS NOT NULL
            """,
            [reg_id],
        ).df()
    except Exception as exc:
        log.warning(
            "Error cargando replay_turns "
            "para METI %s: %s",
            reg_id, exc,
        )
        return pd.DataFrame(columns=empty_cols)

    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    # Resolver winner
    df["won"] = _resolve_winner_column(df)
    df_valid = df.dropna(subset=["won"]).copy()
    df_valid["won"] = df_valid["won"].astype(bool)

    if df_valid.empty:
        return pd.DataFrame(columns=empty_cols)

    # Calcular baseline WR por pokemon
    # (una fila por replay_id + player + pokemon)
    df_per_battle = df_valid.drop_duplicates(
        subset=["replay_id", "player", "pokemon_slug"]
    )
    baseline = (
        df_per_battle
        .groupby("pokemon_slug")["won"]
        .agg(baseline_wr="mean")
        .reset_index()
    )

    # Detectar turno de Mega por replay + player
    activation_rows = []
    grouped = df_valid.groupby(
        ["replay_id", "player"]
    )

    for (replay_id, player), grp in grouped:
        mega_turn = _get_mega_activation_turn(grp)
        bucket = _turn_to_bucket(mega_turn)

        # Solo procesar si hubo Mega o queremos
        # el "never" bucket
        pkm_list = (
            grp["pokemon_slug"].unique().tolist()
        )
        won_val = bool(grp["won"].iloc[0])

        # Tomar el Pokémon Mega del replay
        # (el que estaba activo al activar Mega)
        if mega_turn is not None:
            mega_row = grp[
                grp["turn"] == mega_turn
            ]
            if not mega_row.empty:
                pkm = str(
                    mega_row["pokemon_slug"].iloc[0]
                )
            else:
                pkm = pkm_list[0] if pkm_list else ""
        else:
            # Para "never", usar todos los Pokémon
            # del jugador en este replay
            for pkm in pkm_list:
                activation_rows.append({
                    "pokemon_slug": pkm,
                    "turn_bucket":  "never",
                    "won":          won_val,
                })
            continue

        if pkm:
            activation_rows.append({
                "pokemon_slug": pkm,
                "turn_bucket":  bucket,
                "won":          won_val,
            })

    if not activation_rows:
        return pd.DataFrame(columns=empty_cols)

    df_act = pd.DataFrame(activation_rows)

    # Agrupar por pokemon + bucket
    meti_stats = (
        df_act
        .groupby(["pokemon_slug", "turn_bucket"])
        ["won"]
        .agg(
            win_rate="mean",
            n_matches="count",
        )
        .reset_index()
    )

    # Unir con baseline
    result = meti_stats.merge(
        baseline, on="pokemon_slug", how="left"
    )
    result["baseline_wr"] = result[
        "baseline_wr"
    ].fillna(0.5)
    result["meti_delta"] = (
        result["win_rate"] - result["baseline_wr"]
    )
    result["regulation_id"] = reg_id

    return (
        result[[
            "pokemon_slug", "turn_bucket",
            "n_matches", "win_rate",
            "baseline_wr", "meti_delta",
            "regulation_id",
        ]]
        .sort_values(
            ["pokemon_slug", "turn_bucket"]
        )
        .reset_index(drop=True)
    )


__all__ = [
    "compute_mlwr_level1",
    "compute_meti",
]
