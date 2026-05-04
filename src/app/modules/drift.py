from __future__ import annotations
import logging
from typing import Any
import pandas as pd

log = logging.getLogger(__name__)


def check_drift(
    df_current: pd.DataFrame,
    df_reference: pd.DataFrame,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compara dos DataFrames de uso y detecta
    drift en la distribucion de Pokemon.

    Args:
        df_current: Datos actuales con columna
                    pokemon y usage_pct.
        df_reference: Datos de referencia.
        threshold: Umbral de cambio para alertar.

    Returns:
        Dict con keys:
        - drift_detected: bool
        - n_drifted: int
        - drifted_pokemon: list[str]
        - max_delta: float
        - threshold: float
    """
    if df_current.empty or df_reference.empty:
        return {
            "drift_detected": False,
            "n_drifted": 0,
            "drifted_pokemon": [],
            "max_delta": 0.0,
            "threshold": threshold,
        }

    merged = df_current.merge(
        df_reference,
        on="pokemon",
        suffixes=("_curr", "_ref"),
        how="outer",
    ).fillna(0.0)

    merged["delta"] = (
        merged["usage_pct_curr"]
        - merged["usage_pct_ref"]
    ).abs()

    drifted = merged[
        merged["delta"] >= threshold
    ]["pokemon"].tolist()

    return {
        "drift_detected": len(drifted) > 0,
        "n_drifted": len(drifted),
        "drifted_pokemon": drifted,
        "max_delta": float(merged["delta"].max()),
        "threshold": threshold,
    }


def format_drift_for_discord(
    result: dict[str, Any],
    reg_id: str,
) -> str:
    """
    Formatea el resultado de drift para Discord.

    Args:
        result: Output de check_drift().
        reg_id: Regulacion analizada.

    Returns:
        String formateado para Discord.
    """
    if not result["drift_detected"]:
        return (
            f"Meta estable para {reg_id} — "
            f"sin drift significativo "
            f"(threshold={result['threshold']})"
        )

    pkm_list = ", ".join(
        result["drifted_pokemon"][:5]
    )
    return (
        f"Drift detectado en {reg_id}: "
        f"{result['n_drifted']} Pokemon con "
        f"cambio >= {result['threshold']*100:.0f}% "
        f"(max={result['max_delta']:.3f}). "
        f"Top: {pkm_list}"
    )


__all__ = ["check_drift", "format_drift_for_discord"]
