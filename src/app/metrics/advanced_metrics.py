"""
Métricas avanzadas del **Bloque 15** (Champions).

Aquí viven **MEIT** (T-131), **TAI** (T-132) y **Shapley** (T-133) según el
roadmap. ``compute_meit`` reutiliza **METI** (``compute_meti`` en
``mlwr``) como sub-componente; el resto de funciones se añadirá en sus
tickets correspondientes.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from src.app.metrics.mlwr import compute_meti

log = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)  # type: ignore[untyped-decorator]
def compute_meit(
    reg_id: str,
    pokemon_slug: str,
    _con: Any,
    mega_enabled: bool = True,
    coefficients: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Calcula el MEIT (Mega Evolution Impact on
    Teams) para un Pokémon específico.

    MEIT = α·activation_rate
         + β·sustained_damage_ratio
         - γ·opportunity_cost

    Donde:
    - activation_rate: fracción de partidas donde
      el Pokémon Megaevoluciona (buckets T1-T4+)
      vs total donde está presente.
    - sustained_damage_ratio: WR promedio cuando
      está en bucket T1/T2/T3/T4+ (Mega activo)
      menos WR en bucket "never" (sin Mega).
      Si no hay datos de "never", usar 0.5.
    - opportunity_cost: fracción de activaciones
      que ocurren en T1 (turno donde no puede
      atacar por Megaevolucionar).

    Args:
        reg_id: Regulación a analizar.
        pokemon_slug: Pokémon a evaluar.
        _con: Conexión DuckDB (prefijo _ para
              que st.cache_data no la hashee).
        mega_enabled: Si False retorna dict con
                      found=False inmediatamente.
        coefficients: Dict con keys alpha, beta,
                      gamma. Default {0.4, 0.4, 0.2}.

    Returns:
        Dict con keys:
        - found: bool
        - pokemon: str
        - regulation_id: str
        - meit_score: float (0-1 aprox)
        - activation_rate: float
        - sustained_damage_ratio: float
        - opportunity_cost: float
        - alpha: float
        - beta: float
        - gamma: float
        - n_matches_total: int
        - not_applicable: bool (True si
          mega_enabled=False)
    """
    base_result: dict[str, Any] = {
        "found":                  False,
        "pokemon":                pokemon_slug,
        "regulation_id":          reg_id,
        "meit_score":             0.0,
        "activation_rate":        0.0,
        "sustained_damage_ratio": 0.0,
        "opportunity_cost":       0.0,
        "alpha":                  0.4,
        "beta":                   0.4,
        "gamma":                  0.2,
        "n_matches_total":        0,
        "not_applicable":         False,
    }

    if not mega_enabled:
        base_result["not_applicable"] = True
        log.info(
            "MEIT no aplica para %s "
            "(mega_enabled=False)",
            reg_id,
        )
        return base_result

    # Coeficientes
    alpha = float(
        (coefficients or {}).get("alpha", 0.4)
    )
    beta = float(
        (coefficients or {}).get("beta", 0.4)
    )
    gamma = float(
        (coefficients or {}).get("gamma", 0.2)
    )
    base_result["alpha"] = alpha
    base_result["beta"] = beta
    base_result["gamma"] = gamma

    # Obtener METI data para este Pokémon
    df_meti = compute_meti(
        reg_id, _con, mega_enabled=True
    )

    if df_meti.empty:
        log.warning(
            "Sin datos METI para %s en %s",
            pokemon_slug, reg_id,
        )
        return base_result

    # Filtrar por Pokémon
    df_pkm = df_meti[
        df_meti["pokemon_slug"] == pokemon_slug
    ].copy()

    if df_pkm.empty:
        log.info(
            "Pokémon '%s' no encontrado en "
            "datos METI para %s",
            pokemon_slug, reg_id,
        )
        return base_result

    # Total de partidas donde aparece el Pokémon
    n_total = int(df_pkm["n_matches"].sum())
    base_result["n_matches_total"] = n_total

    if n_total == 0:
        return base_result

    # activation_rate: % partidas con Mega activo
    mega_buckets = ["T1", "T2", "T3", "T4+"]
    df_mega = df_pkm[
        df_pkm["turn_bucket"].isin(mega_buckets)
    ]
    n_mega = int(df_mega["n_matches"].sum())
    activation_rate = n_mega / n_total

    # sustained_damage_ratio: WR con Mega - WR sin Mega
    wr_with_mega = (
        float(
            (df_mega["win_rate"] * df_mega["n_matches"])
            .sum() / n_mega
        )
        if n_mega > 0 else 0.5
    )

    df_never = df_pkm[
        df_pkm["turn_bucket"] == "never"
    ]
    wr_without_mega = (
        float(df_never["win_rate"].iloc[0])
        if not df_never.empty else 0.5
    )

    sustained_damage_ratio = (
        wr_with_mega - wr_without_mega
    )

    # opportunity_cost: % activaciones en T1
    df_t1 = df_pkm[df_pkm["turn_bucket"] == "T1"]
    n_t1 = int(
        df_t1["n_matches"].iloc[0]
        if not df_t1.empty else 0
    )
    opportunity_cost = (
        n_t1 / n_mega if n_mega > 0 else 0.0
    )

    # MEIT score
    meit_score = (
        alpha * activation_rate
        + beta * sustained_damage_ratio
        - gamma * opportunity_cost
    )

    return {
        "found":                  True,
        "pokemon":                pokemon_slug,
        "regulation_id":          reg_id,
        "meit_score":             round(meit_score, 4),
        "activation_rate":        round(
            activation_rate, 4
        ),
        "sustained_damage_ratio": round(
            sustained_damage_ratio, 4
        ),
        "opportunity_cost":       round(
            opportunity_cost, 4
        ),
        "alpha":                  alpha,
        "beta":                   beta,
        "gamma":                  gamma,
        "n_matches_total":        n_total,
        "not_applicable":         False,
    }


__all__ = [
    "compute_meit",
]
