"""
Métricas avanzadas del **Bloque 15** (Champions).

Aquí viven **MEIT** (T-131), **TAI** (T-132) y **Shapley** (T-133) según el
roadmap. ``compute_meit`` reutiliza **METI** (``compute_meti`` en
``mlwr``) como sub-componente; ``compute_tai`` cubre el índice de
adaptabilidad frente a arquetipos del meta (T-132).
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    import xgboost as xgb

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


def _estimate_wr_vs_archetype(
    team_slugs: tuple[str, ...],
    archetype_pokemon: list[str],
    wp_model: Any | None,
    reg_id: str,
) -> float:
    """
    Estima el WR del equipo contra un arquetipo
    del meta.

    Si wp_model disponible: usa predict_win_prob.
    Si no: usa heurístico de type coverage
    normalizado a [0.35, 0.65].

    Args:
        team_slugs: Pokémon del equipo a evaluar.
        archetype_pokemon: Pokémon representativos
                            del arquetipo rival.
        wp_model: Modelo XGBoost o None.
        reg_id: Regulación activa.

    Returns:
        WR estimada en [0.0, 1.0].
    """
    if wp_model is not None:
        try:
            from src.app.modules.wp import (
                predict_win_prob,
            )

            wr = predict_win_prob(
                list(team_slugs),
                archetype_pokemon,
                reg_id,
                wp_model,
            )
            return float(
                max(0.0, min(1.0, wr))
            )
        except Exception as exc:
            log.debug(
                "WP model falló para TAI: %s",
                exc,
            )

    # Heurístico: proporción de tipos cubiertos
    # Normalizado a [0.35, 0.65]
    try:
        from src.app.core.champions_calc import (
            type_effectiveness,
            _load_pokemon_master,
        )

        pm = _load_pokemon_master()

        def get_types(slug: str) -> list[str]:
            for entry in pm.values():
                if str(
                    entry.get("name", "")
                ).lower() == slug.lower():
                    return [
                        str(t).title()
                        for t in entry.get(
                            "types", []
                        )
                    ]
            return []

        score = 0.0
        comparisons = 0
        for my_pkm in team_slugs:
            my_types = get_types(my_pkm)
            for opp_pkm in archetype_pokemon:
                opp_types = get_types(opp_pkm)
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
        return 0.35 + (
            min(raw, 4.0) / 4.0
        ) * 0.30

    except Exception as exc:
        log.debug(
            "Heurístico TAI falló: %s", exc
        )
        return 0.5


@st.cache_data(ttl=3600, show_spinner=False)  # type: ignore[untyped-decorator]
def compute_tai(
    reg_id: str,
    team_slugs: tuple[str, ...],
    _con: Any,
    wp_model: Any | None = None,
    lambda_penalty: float = 0.5,
) -> dict[str, Any]:
    """
    Calcula el TAI (Teambuilder Adaptability Index)
    para un equipo dado.

    TAI = Σ_m usage(meta_m) · WR(team, meta_m)
          - λ · Var[WR]

    Args:
        reg_id: Regulación activa.
        team_slugs: Tuple con los slugs del equipo
                     (normalmente 6 Pokémon).
        _con: Conexión DuckDB (prefijo _ para
              que st.cache_data no la hashee).
        wp_model: Modelo XGBoost opcional.
                   Si None, usa heurístico.
        lambda_penalty: Coeficiente de penalización
                         de varianza. Default 0.5.

    Returns:
        Dict con keys:
        - tai_score: float
        - weighted_wr: float (sin penalización)
        - wr_variance: float
        - lambda_penalty: float
        - n_archetypes: int
        - archetype_breakdown: list[dict] con
          archetype_id, usage_pct, estimated_wr
          para cada arquetipo
        - regulation_id: str
        - team: list[str]
        - found: bool
    """
    empty_result: dict[str, Any] = {
        "tai_score":           0.0,
        "weighted_wr":         0.0,
        "wr_variance":         0.0,
        "lambda_penalty":      lambda_penalty,
        "n_archetypes":        0,
        "archetype_breakdown": [],
        "regulation_id":       reg_id,
        "team":                list(team_slugs),
        "found":               False,
    }

    if len(team_slugs) < 2:
        log.warning(
            "TAI requiere al menos 2 Pokémon "
            "en el equipo"
        )
        return empty_result

    # Obtener clusters del meta
    try:
        from src.app.modules.clustering import (
            cluster_kmeans,
        )

        df_usage = _con.execute(
            """
            SELECT pokemon, avg_usage_pct
            FROM main_curated.usage_by_reg
            WHERE regulation_id = ?
            ORDER BY avg_usage_pct DESC
            LIMIT 50
            """,
            [reg_id],
        ).df()

        if df_usage.empty:
            log.info(
                "Sin datos de uso para clusters "
                "de TAI %s",
                reg_id,
            )
            return empty_result

        cluster_result = cluster_kmeans(
            df_usage,
            n_clusters=min(8, len(df_usage)),
            feature_cols=["avg_usage_pct"],
        )
        df_clusters = cluster_result.df_clustered.rename(
            columns={"cluster": "cluster_id"}
        )
        df_clusters = df_clusters.rename(
            columns={"avg_usage_pct": "usage_pct"}
        )
        df_clusters = df_clusters.merge(
            df_usage, left_index=True,
            right_index=True, how="left",
        )
        df_clusters["pokemon"] = df_usage[
            "pokemon"
        ].values
    except Exception as exc:
        log.warning(
            "Error obteniendo clusters para "
            "TAI %s: %s",
            reg_id, exc,
        )
        return empty_result

    if df_clusters is None or df_clusters.empty:
        log.info(
            "Sin clusters disponibles para %s",
            reg_id,
        )
        return empty_result

    # Agrupar Pokémon por cluster
    cluster_groups: dict[
        int, list[str]
    ] = {}
    cluster_usage: dict[int, float] = {}

    for _, row in df_clusters.iterrows():
        cid = int(row.get("cluster_id", 0))
        pkm = str(row.get("pokemon", ""))
        usage = float(row.get("usage_pct", 0.0))

        if cid not in cluster_groups:
            cluster_groups[cid] = []
            cluster_usage[cid] = 0.0

        cluster_groups[cid].append(pkm)
        cluster_usage[cid] = max(
            cluster_usage[cid], usage
        )

    if not cluster_groups:
        return empty_result

    # Normalizar usage por cluster
    total_usage = sum(cluster_usage.values())
    if total_usage <= 0:
        total_usage = 1.0

    # Calcular WR estimada vs cada arquetipo
    wr_list: list[float] = []
    usage_list: list[float] = []
    breakdown: list[dict[str, Any]] = []

    for cid, pkm_list in cluster_groups.items():
        usage_norm = (
            cluster_usage[cid] / total_usage
        )
        # Tomar top-4 Pokémon del arquetipo
        archetype_pkm = pkm_list[:4]

        wr = _estimate_wr_vs_archetype(
            team_slugs,
            archetype_pkm,
            wp_model,
            reg_id,
        )

        wr_list.append(wr)
        usage_list.append(usage_norm)
        breakdown.append({
            "archetype_id": cid,
            "usage_pct": round(
                usage_norm * 100, 2
            ),
            "estimated_wr": round(wr, 4),
            "top_pokemon": archetype_pkm,
        })

    # TAI = Σ usage·WR - λ·Var[WR]
    wr_array = np.array(wr_list)
    usage_array = np.array(usage_list)

    weighted_wr = float(
        np.dot(usage_array, wr_array)
    )
    wr_variance = float(np.var(wr_array))
    tai_score = weighted_wr - (
        lambda_penalty * wr_variance
    )

    return {
        "tai_score":           round(tai_score, 4),
        "weighted_wr":         round(weighted_wr, 4),
        "wr_variance":         round(wr_variance, 6),
        "lambda_penalty":      lambda_penalty,
        "n_archetypes":        len(cluster_groups),
        "archetype_breakdown": sorted(
            breakdown,
            key=lambda x: x["usage_pct"],
            reverse=True,
        ),
        "regulation_id":       reg_id,
        "team":                list(team_slugs),
        "found":               True,
    }


def _estimate_team_wr(
    team_slugs: list[str],
    wp_model: Any | None,
    reg_id: str,
    fallback_wr: float = 0.5,
) -> float:
    """
    Estima el WR de un equipo usando el WP model
    o retorna fallback_wr si no hay modelo.

    Args:
        team_slugs: Lista de slugs del equipo.
        wp_model: Modelo XGBoost o None.
        reg_id: Regulación activa.
        fallback_wr: WR de fallback si no hay
                      modelo. Default 0.5.

    Returns:
        WR estimada en [0.0, 1.0].
    """
    if wp_model is None or len(team_slugs) < 2:
        return fallback_wr

    try:
        from src.app.modules.wp import (
            predict_win_prob,
        )

        wr = predict_win_prob(
            team_slugs,
            team_slugs,
            reg_id,
            wp_model,
        )
        return float(max(0.0, min(1.0, wr)))
    except Exception as exc:
        log.debug(
            "WP model falló en _estimate_team_wr:"
            " %s", exc,
        )
        return fallback_wr


@st.cache_data(ttl=3600, show_spinner=False)  # type: ignore[untyped-decorator]
def shapley_slot_value(
    reg_id: str,
    team_slugs: tuple[str, ...],
    _con: Any,
    wp_model: Any | None = None,
    n_permutations: int = 20,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    Calcula el Shapley Slot Value (LOO approx)
    para cada Pokémon del equipo.

    Aproximación LOO promediada sobre K
    permutaciones aleatorias:
      φᵢ ≈ mean_k[WR(S_k ∪ {i}) - WR(S_k)]

    Normaliza para que Σφᵢ = WR(team) - 0.5.

    Si wp_model es None: usa contribución
    uniforme φᵢ = (WR_team - 0.5) / n.

    Args:
        reg_id: Regulación activa.
        team_slugs: Tuple con slugs del equipo.
        _con: Conexión DuckDB (no usado directamente
              pero requerido para compatibilidad
              con cache).
        wp_model: Modelo XGBoost opcional.
        n_permutations: Número de permutaciones
                         aleatorias. Default 20.
        random_seed: Semilla para reproducibilidad.

    Returns:
        Dict {pokemon_slug: shapley_value}.
        Valores positivos = contribución por
        encima de la media.
        Suma total ≈ WR(team) - 0.5.
        Dict vacío si equipo < 2 Pokémon.
    """
    _ = _con

    if len(team_slugs) < 2:
        log.warning(
            "Shapley requiere al menos 2 "
            "Pokémon en el equipo"
        )
        return {}

    team_list = list(team_slugs)
    n = len(team_list)

    # WR del equipo completo
    wr_full = _estimate_team_wr(
        team_list, wp_model, reg_id,
        fallback_wr=0.5,
    )

    # Sin modelo: distribución uniforme
    if wp_model is None:
        uniform_val = (wr_full - 0.5) / n
        return {
            pkm: round(uniform_val, 4)
            for pkm in team_list
        }

    # LOO aproximado con K permutaciones
    rng = random.Random(random_seed)
    shapley_accum: dict[str, float] = {
        pkm: 0.0 for pkm in team_list
    }
    counts: dict[str, int] = {
        pkm: 0 for pkm in team_list
    }

    for _ in range(n_permutations):
        perm = team_list.copy()
        rng.shuffle(perm)

        for pkm in perm:
            # S = equipo sin pkm
            s_without = [
                p for p in perm if p != pkm
            ]
            # S ∪ {pkm}
            s_with = s_without + [pkm]

            wr_with = _estimate_team_wr(
                s_with, wp_model, reg_id,
                fallback_wr=0.5,
            )
            wr_without = _estimate_team_wr(
                s_without, wp_model, reg_id,
                fallback_wr=0.5,
            )

            marginal = wr_with - wr_without
            shapley_accum[pkm] += marginal
            counts[pkm] += 1

    # Promediar
    raw_shapley = {
        pkm: (
            shapley_accum[pkm] / counts[pkm]
            if counts[pkm] > 0 else 0.0
        )
        for pkm in team_list
    }

    # Normalizar: Σφᵢ = WR_full - 0.5
    raw_sum = sum(raw_shapley.values())
    target_sum = wr_full - 0.5

    if abs(raw_sum) > 1e-9:
        scale = target_sum / raw_sum
        normalized = {
            pkm: round(v * scale, 4)
            for pkm, v in raw_shapley.items()
        }
    else:
        uniform = target_sum / n
        normalized = {
            pkm: round(uniform, 4)
            for pkm in team_list
        }

    return normalized


__all__ = [
    "compute_meit",
    "compute_tai",
    "shapley_slot_value",
]
