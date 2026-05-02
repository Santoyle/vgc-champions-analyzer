"""
MLWR — Move-Level Win Rate (métricas de replays Champions).

Versión **L1**: correlación simple entre usar un movimiento en batalla y el
desenlace (**sin** ajuste por game state; eso será **MLWR L2** con XGBoost + SHAP
en T-119).

``replay_turns.winner`` a veces trae username del ganador (no ``p1``/``p2``);
``_resolve_winner_column`` lo detecta y evita estadísticas falsas hasta que el
pipeline normalice el campo.
"""
from __future__ import annotations

import logging
from typing import Any

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


__all__ = [
    "compute_mlwr_level1",
]
