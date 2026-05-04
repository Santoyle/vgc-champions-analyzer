from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Resultado del drift check."""
    is_drift: bool
    p_val: float
    distance: float
    regulation_id: str
    top_drifted_pokemon: list[str] = field(
        default_factory=list
    )
    n_reference: int = 0
    n_current: int = 0


def check_drift(
    regulation_id: str,
    con: Any,
    threshold: float = 0.05,
    top_n: int = 50,
) -> DriftResult | None:
    """
    Detecta drift en la distribucion de uso del
    meta comparando los ultimos 7 dias vs los
    7 dias anteriores.

    Args:
        regulation_id: Regulacion a analizar.
        con: Conexion DuckDB (puede ser :memory:).
        threshold: Umbral de p-value para drift.
        top_n: Top N Pokemon a considerar.

    Returns:
        DriftResult con los resultados.
        None si no hay datos suficientes.
    """
    try:
        df = con.execute(
            """
            SELECT pokemon, usage_pct,
                   CAST(ingested_at AS DATE) AS date
            FROM read_parquet(
                'data/raw/reg=*/source=pikalytics/*.parquet',
                hive_partitioning=true
            )
            WHERE regulation_id = ?
              AND pokemon IS NOT NULL
            ORDER BY date DESC
            """,
            [regulation_id],
        ).df()
    except Exception as exc:
        log.warning(
            "Error leyendo datos para drift: %s",
            exc,
        )
        return None

    if df.empty or len(df["date"].unique()) < 2:
        log.info(
            "Sin datos suficientes para drift "
            "en %s",
            regulation_id,
        )
        return None

    dates = sorted(df["date"].unique())
    mid = len(dates) // 2
    recent_dates = dates[mid:]
    reference_dates = dates[:mid]

    df_recent = (
        df[df["date"].isin(recent_dates)]
        .groupby("pokemon")["usage_pct"]
        .mean()
        .reset_index()
    )
    df_ref = (
        df[df["date"].isin(reference_dates)]
        .groupby("pokemon")["usage_pct"]
        .mean()
        .reset_index()
    )

    if df_recent.empty or df_ref.empty:
        return None

    # Top N por uso
    top_pokemon = (
        df_recent
        .nlargest(top_n, "usage_pct")["pokemon"]
        .tolist()
    )

    df_recent_top = df_recent[
        df_recent["pokemon"].isin(top_pokemon)
    ].set_index("pokemon")["usage_pct"]

    df_ref_top = df_ref[
        df_ref["pokemon"].isin(top_pokemon)
    ].set_index("pokemon")["usage_pct"]

    # Alinear indices
    all_pkm = list(
        set(df_recent_top.index)
        | set(df_ref_top.index)
    )
    vec_recent = np.array([
        df_recent_top.get(p, 0.0)
        for p in all_pkm
    ])
    vec_ref = np.array([
        df_ref_top.get(p, 0.0)
        for p in all_pkm
    ])

    # Normalizar a distribuciones
    if vec_recent.sum() > 0:
        vec_recent = vec_recent / vec_recent.sum()
    if vec_ref.sum() > 0:
        vec_ref = vec_ref / vec_ref.sum()

    # Distancia L1 (Total Variation Distance)
    distance = float(
        np.abs(vec_recent - vec_ref).sum() / 2
    )

    # p-value aproximado via bootstrap
    try:
        from alibi_detect.cd import TabularDrift
        rng = np.random.default_rng(42)
        x_ref = vec_ref.reshape(1, -1)
        x_curr = vec_recent.reshape(1, -1)
        cd = TabularDrift(
            x_ref,
            p_val=threshold,
        )
        pred = cd.predict(x_curr)
        p_val = float(
            pred["data"]["p_val"].mean()
        )
        is_drift = bool(
            pred["data"]["is_drift"].any()
        )
    except Exception as exc:
        log.debug(
            "alibi-detect no disponible, "
            "usando threshold directo: %s", exc,
        )
        # Fallback: usar distancia L1 como proxy
        p_val = max(0.0, 1.0 - distance * 10)
        is_drift = distance > threshold

    # Top drifted Pokemon
    deltas = {
        p: abs(
            float(df_recent_top.get(p, 0.0))
            - float(df_ref_top.get(p, 0.0))
        )
        for p in all_pkm
    }
    top_drifted = sorted(
        deltas, key=deltas.get, reverse=True  # type: ignore[arg-type]
    )[:10]

    return DriftResult(
        is_drift=is_drift,
        p_val=p_val,
        distance=distance,
        regulation_id=regulation_id,
        top_drifted_pokemon=top_drifted,
        n_reference=len(reference_dates),
        n_current=len(recent_dates),
    )


def format_drift_for_discord(
    result: DriftResult,
) -> str:
    """
    Formatea el resultado de drift para Discord.

    Args:
        result: Output de check_drift().

    Returns:
        String formateado para Discord.
    """
    if not result.is_drift:
        return (
            f"Meta estable para "
            f"{result.regulation_id} — "
            f"p_val={result.p_val:.4f} "
            f"distance={result.distance:.4f}"
        )

    pkm_list = ", ".join(
        result.top_drifted_pokemon[:5]
    )
    return (
        f"Drift detectado en "
        f"{result.regulation_id}: "
        f"p_val={result.p_val:.4f} "
        f"distance={result.distance:.4f}. "
        f"Top: {pkm_list}"
    )


__all__ = [
    "DriftResult",
    "check_drift",
    "format_drift_for_discord",
]
