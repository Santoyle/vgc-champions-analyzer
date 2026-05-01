"""
Métricas competitivas para VGC Champions.

Este módulo implementa métricas cuantitativas para analizar la efectividad
de Pokémon e ítems en el meta competitivo. Las métricas usan shrinkage
bayesiana para regularizar estimaciones con pocos meses de datos disponibles.

Métricas implementadas:
  - EI (Efficiency Item): mide si un ítem mejora el rendimiento de un Pokémon
    más allá del baseline (su ítem más usado). Usa prior beta(10,10) centrada
    en 0.5 (prior neutro, fuerte regularización con pocos datos).
  - STC (Speed Tier Control): mide qué fracción del meta un Pokémon supera en
    velocidad base. Score en [0, 1]; STC=1 significa más rápido que todos.
  - TPI (Threat Pressure Index): presión sobre el meta vía co-uso con líderes —
    correlación teammates ponderada por el uso del compañero.

La shrinkage beta(α=10, β=10) implica que necesitamos al menos ~20 meses de
datos para que el dato observado pese más que el prior. Con 1-2 meses, la
estimación queda fuertemente anclada a 50%, evitando EI extremos por muestra
pequeña.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_POKEMON_MASTER_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "pokemon_master.json"
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BETA_ALPHA: float = 10.0   # parámetro α de la prior beta(10,10)
BETA_BETA: float = 10.0    # parámetro β de la prior beta(10,10)
PRIOR_MEAN: float = 0.5    # media de beta(10,10): α / (α + β)


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------


@dataclass
class EIResult:
    """
    Resultado de la métrica EI para un par (Pokémon, ítem).

    Attributes:
        pokemon: Nombre del Pokémon.
        item: Nombre del ítem.
        regulation_id: Regulación analizada.
        raw_pct: Porcentaje de uso crudo del ítem en ese Pokémon
                 (avg_pct sin shrinkage, escala [0, 100]).
        shrunk_pct: Porcentaje ajustado con shrinkage beta(10,10),
                    escala [0, 100].
        baseline_pct: Porcentaje shrunk del ítem más popular del Pokémon
                      (baseline contra el que se compara), escala [0, 100].
        ei_score: EI final = shrunk_pct - baseline_pct.
                  Positivo → el ítem es más eficiente que el baseline.
                  Negativo → menos eficiente que el baseline.
                  Siempre 0.0 para is_baseline=True.
        n_months: Meses con datos para este par (pkm, item).
        is_baseline: True si este ítem es el baseline del Pokémon
                     (mayor avg_pct).
    """

    pokemon: str
    item: str
    regulation_id: str
    raw_pct: float
    shrunk_pct: float
    baseline_pct: float
    ei_score: float
    n_months: int
    is_baseline: bool


@dataclass
class STCResult:
    """
    Resultado de la métrica STC para un Pokémon individual del meta.

    Attributes:
        pokemon: Nombre del Pokémon (o "TEAM" si es el STC del equipo completo).
        regulation_id: Regulación analizada.
        base_speed: Velocidad base del Pokémon. 0 si es resultado de equipo.
        n_faster_than: Número de Pokémon del meta a los que supera en velocidad.
        n_meta_pokemon: Total de Pokémon del meta usados como referencia.
        stc_score: Score STC en [0, 1].
                   0.0 = más lento que todo el meta.
                   1.0 = más rápido que todos.
        usage_weight: Peso de uso del Pokémon en el meta (avg_usage_pct / 100).
                      1.0 para el resultado del equipo completo.
    """

    pokemon: str
    regulation_id: str
    base_speed: int
    n_faster_than: int
    n_meta_pokemon: int
    stc_score: float
    usage_weight: float


@dataclass
class TPIResult:
    """
    Resultado de la métrica TPI para un Pokémon.

    Attributes:
        pokemon: Nombre del Pokémon.
        regulation_id: Regulación analizada.
        tpi_score: Score TPI en [0, 1]. Mayor = más amenazante en el meta.
        n_teammates: Número de compañeros con datos de co-uso.
        top_teammate: Compañero con mayor correlación avg_correlation.
        avg_co_usage: Correlación de co-uso promedio con todos los compañeros.
        usage_weighted_score: Media de co_pct × uso_relativo / 100 (sin cap).
    """

    pokemon: str
    regulation_id: str
    tpi_score: float
    n_teammates: int
    top_teammate: str
    avg_co_usage: float
    usage_weighted_score: float


# ---------------------------------------------------------------------------
# Shrinkage bayesiana
# ---------------------------------------------------------------------------


def _apply_beta_shrinkage(
    observed_pct: float,
    n_months: int,
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
    prior_mean: float = PRIOR_MEAN,
) -> float:
    """
    Aplica shrinkage beta(α, β) a un porcentaje observado.

    Combina el prior con las observaciones ponderando por número de meses:

        shrunk = (α × prior_mean + n × observed_norm) / (α + β + n)

    Casos límite:
      - n_months = 0  → retorna prior_mean × 100 (sin datos, todo es prior).
      - n_months → ∞  → shrunk → observed_pct   (datos dominan al prior).

    Args:
        observed_pct: Porcentaje observado en escala [0, 100].
                      Se normaliza a [0, 1] internamente.
        n_months: Número de meses con observaciones. Clamp a 0 si negativo.
        alpha: Parámetro α de la prior beta. Default BETA_ALPHA = 10.
        beta: Parámetro β de la prior beta. Default BETA_BETA = 10.
        prior_mean: Media de la prior. Default PRIOR_MEAN = 0.5.

    Returns:
        Porcentaje ajustado en escala [0, 100] (misma escala que observed_pct).
    """
    if n_months <= 0:
        return float(prior_mean * 100.0)

    obs_norm = float(max(0.0, min(1.0, float(observed_pct) / 100.0)))
    shrunk_norm = (alpha * prior_mean + n_months * obs_norm) / (alpha + beta + n_months)
    return float(shrunk_norm * 100.0)


# ---------------------------------------------------------------------------
# Cálculo de EI
# ---------------------------------------------------------------------------


def compute_ei(
    df_items: pd.DataFrame,
    regulation_id: str,
    top_n_items: int = 5,
) -> pd.DataFrame:
    """
    Calcula la métrica EI para todos los pares (Pokémon, ítem) de una regulación.

    Para cada Pokémon:
      1. Identifica el ítem baseline (mayor avg_pct).
      2. Aplica shrinkage beta(10,10) a todos sus ítems.
      3. Calcula EI = shrunk_pct - baseline_shrunk_pct.

    El ítem baseline siempre tiene EI = 0.0 (se compara contra sí mismo).

    Args:
        df_items: DataFrame con columnas regulation_id, pokemon, item,
                  avg_pct, n_months_seen.
        regulation_id: Regulación a analizar (filtro sobre df_items).
        top_n_items: Máximo de ítems por Pokémon incluidos en el resultado.
                     Default 5.

    Returns:
        DataFrame con columnas pokemon, item, regulation_id, raw_pct,
        shrunk_pct, baseline_pct, ei_score, n_months, is_baseline.
        Ordenado por pokemon ASC, ei_score DESC.
        DataFrame vacío (mismas columnas) si no hay datos.
    """
    _empty_cols = [
        "pokemon", "item", "regulation_id",
        "raw_pct", "shrunk_pct", "baseline_pct",
        "ei_score", "n_months", "is_baseline",
    ]

    if df_items.empty:
        return pd.DataFrame(columns=_empty_cols)

    df = df_items[df_items["regulation_id"] == regulation_id].copy()
    if df.empty:
        return pd.DataFrame(columns=_empty_cols)

    results: list[EIResult] = []

    for pokemon, group in df.groupby("pokemon"):
        group_sorted = group.sort_values("avg_pct", ascending=False)
        top_items = group_sorted.head(top_n_items)

        if top_items.empty:
            continue

        baseline_row = group_sorted.iloc[0]
        baseline_raw = float(baseline_row["avg_pct"])
        baseline_n = int(baseline_row.get("n_months_seen", 1) or 1)
        baseline_shrunk = _apply_beta_shrinkage(baseline_raw, baseline_n)

        for _, row in top_items.iterrows():
            item = str(row["item"])
            raw_pct = float(row["avg_pct"])
            n_months = int(row.get("n_months_seen", 1) or 1)
            shrunk_pct = _apply_beta_shrinkage(raw_pct, n_months)
            is_baseline = item == str(baseline_row["item"])
            # Baseline siempre EI=0 por definición
            ei_score = 0.0 if is_baseline else shrunk_pct - baseline_shrunk

            results.append(
                EIResult(
                    pokemon=str(pokemon),
                    item=item,
                    regulation_id=regulation_id,
                    raw_pct=round(raw_pct, 4),
                    shrunk_pct=round(shrunk_pct, 4),
                    baseline_pct=round(baseline_shrunk, 4),
                    ei_score=round(ei_score, 4),
                    n_months=n_months,
                    is_baseline=is_baseline,
                )
            )

    if not results:
        return pd.DataFrame(columns=_empty_cols)

    df_result = pd.DataFrame([
        {
            "pokemon": r.pokemon,
            "item": r.item,
            "regulation_id": r.regulation_id,
            "raw_pct": r.raw_pct,
            "shrunk_pct": r.shrunk_pct,
            "baseline_pct": r.baseline_pct,
            "ei_score": r.ei_score,
            "n_months": r.n_months,
            "is_baseline": r.is_baseline,
        }
        for r in results
    ])

    return df_result.sort_values(
        ["pokemon", "ei_score"], ascending=[True, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def get_top_ei_items(
    df_ei: pd.DataFrame,
    top_n: int = 20,
    min_ei_score: float = 0.0,
) -> pd.DataFrame:
    """
    Retorna los pares (Pokémon, ítem) con mayor EI score, excluyendo baselines.

    Útil para el leaderboard de la página Analytics: muestra los ítems que
    superan significativamente al ítem más popular de su Pokémon.

    Args:
        df_ei: DataFrame output de compute_ei().
        top_n: Máximo de pares a retornar. Default 20.
        min_ei_score: EI mínimo para incluir un par. Default 0.0.

    Returns:
        DataFrame filtrado y ordenado por ei_score descendente.
        DataFrame vacío si df_ei está vacío.
    """
    if df_ei.empty:
        return df_ei

    return (
        df_ei[~df_ei["is_baseline"] & (df_ei["ei_score"] >= min_ei_score)]
        .sort_values("ei_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_ei_from_duckdb(
    con: Any,
    regulation_id: str,
    top_n_items: int = 5,
) -> pd.DataFrame:
    """
    Carga datos de ítems desde DuckDB y calcula EI directamente.

    Wrapper conveniente para uso desde la UI de Streamlit o scripts CLI.
    Importa las vistas de forma lazy para evitar dependencias circulares.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Regulación a analizar.
        top_n_items: Top N ítems por Pokémon. Default 5.

    Returns:
        DataFrame de EI (output de compute_ei). DataFrame vacío si hay error.
    """
    try:
        from src.app.data.sql.views import (  # noqa: PLC0415
            create_items_by_pkm,
            register_raw_view,
        )

        register_raw_view(con)
        df_items = create_items_by_pkm(con, regulation_id)
        return compute_ei(df_items, regulation_id, top_n_items)
    except Exception as exc:  # noqa: BLE001
        log.warning("Error calculando EI para %s: %s", regulation_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# STC — Speed Tier Control
# ---------------------------------------------------------------------------


def _load_speed_map() -> dict[str, int]:
    """
    Carga el mapeo nombre_pokemon → velocidad_base desde pokemon_master.json.

    Returns:
        Dict {nombre_lower: base_speed}.
        Dict vacío si el archivo no existe o hay error de lectura.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            return {}
        raw: dict[str, Any] = json.loads(
            _POKEMON_MASTER_PATH.read_text(encoding="utf-8")
        )
        result: dict[str, int] = {}
        for entry in raw.values():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).lower()
            speed = int(entry.get("base_stats", {}).get("speed", 0))
            if name:
                result[name] = speed
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando speed map: %s", exc)
        return {}


def compute_stc_pokemon(
    pokemon_name: str,
    regulation_id: str,
    df_usage: pd.DataFrame,
    speed_map: dict[str, int] | None = None,
) -> STCResult | None:
    """
    Calcula el STC para un Pokémon individual contra el meta de su regulación.

    El propio Pokémon está incluido en el meta (forma parte del meta actual),
    por lo que nunca se compara contra 0 competidores.

    Args:
        pokemon_name: Nombre del Pokémon a evaluar.
        regulation_id: Regulación del meta.
        df_usage: DataFrame con columnas pokemon, avg_usage_pct
                  (output de create_usage_by_reg).
        speed_map: Dict {nombre_lower: base_speed}. Carga desde disco si None.

    Returns:
        STCResult con el score del Pokémon.
        None si el Pokémon no tiene datos de velocidad en speed_map.
    """
    if speed_map is None:
        speed_map = _load_speed_map()

    pkm_lower = pokemon_name.lower()
    pkm_speed = speed_map.get(pkm_lower)

    if pkm_speed is None or pkm_speed == 0:
        log.debug("Sin datos de velocidad para %s", pokemon_name)
        return None

    # Velocidades del meta (incluyendo el propio Pokémon)
    meta_speeds: list[int] = []
    for _, row in df_usage.iterrows():
        meta_name = str(row["pokemon"]).lower()
        meta_speed = speed_map.get(meta_name, 0)
        if meta_speed > 0:
            meta_speeds.append(meta_speed)

    if not meta_speeds:
        return None

    n_faster = sum(1 for s in meta_speeds if pkm_speed > s)
    n_total = len(meta_speeds)
    stc_score = float(n_faster / n_total)

    usage_row = df_usage[df_usage["pokemon"].str.lower() == pkm_lower]
    usage_weight = float(
        usage_row.iloc[0]["avg_usage_pct"] / 100.0
        if not usage_row.empty
        else 0.0
    )

    return STCResult(
        pokemon=pokemon_name,
        regulation_id=regulation_id,
        base_speed=pkm_speed,
        n_faster_than=n_faster,
        n_meta_pokemon=n_total,
        stc_score=round(stc_score, 4),
        usage_weight=round(usage_weight, 4),
    )


def compute_stc_meta(
    df_usage: pd.DataFrame,
    regulation_id: str,
    speed_map: dict[str, int] | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Calcula el STC para todos los Pokémon del meta de una regulación.

    Ordena por stc_score descendente — los Pokémon que superan en velocidad
    a más rivales del meta quedan arriba.

    Args:
        df_usage: DataFrame con columnas pokemon, avg_usage_pct y opcionalmente
                  regulation_id (output de create_usage_by_reg).
        regulation_id: Regulación a analizar.
        speed_map: Mapeo velocidades. Carga desde disco si None.
        top_n: Top N Pokémon a incluir en el resultado. Default 30.

    Returns:
        DataFrame con columnas: pokemon, regulation_id, base_speed,
        n_faster_than, n_meta_pokemon, stc_score, usage_weight.
        Ordenado por stc_score DESC.
        DataFrame vacío si no hay datos.
    """
    if df_usage.empty:
        return pd.DataFrame()

    if speed_map is None:
        speed_map = _load_speed_map()

    df_reg = (
        df_usage[df_usage["regulation_id"] == regulation_id]
        if "regulation_id" in df_usage.columns
        else df_usage
    )
    if df_reg.empty:
        return pd.DataFrame()

    results: list[STCResult] = []
    candidate_rows = df_reg.head(min(top_n * 2, len(df_reg)))

    for _, row in candidate_rows.iterrows():
        result = compute_stc_pokemon(
            str(row["pokemon"]),
            regulation_id,
            df_reg,
            speed_map,
        )
        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame([
        {
            "pokemon": r.pokemon,
            "regulation_id": r.regulation_id,
            "base_speed": r.base_speed,
            "n_faster_than": r.n_faster_than,
            "n_meta_pokemon": r.n_meta_pokemon,
            "stc_score": r.stc_score,
            "usage_weight": r.usage_weight,
        }
        for r in results
    ])

    return (
        df_result.sort_values("stc_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_stc_from_duckdb(
    con: Any,
    regulation_id: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Carga datos de uso desde DuckDB y calcula STC.

    Importa las vistas de forma lazy para evitar dependencias circulares.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Regulación a analizar.
        top_n: Top N Pokémon en el resultado. Default 30.

    Returns:
        DataFrame de STC. DataFrame vacío si hay error.
    """
    try:
        from src.app.data.sql.views import (  # noqa: PLC0415
            create_usage_by_reg,
            register_raw_view,
        )

        register_raw_view(con)
        df_usage = create_usage_by_reg(con, regulation_id)
        speed_map = _load_speed_map()
        return compute_stc_meta(df_usage, regulation_id, speed_map, top_n)
    except Exception as exc:  # noqa: BLE001
        log.warning("Error calculando STC para %s: %s", regulation_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# TPI — Threat Pressure Index
# ---------------------------------------------------------------------------


def compute_tpi(
    df_teammates: pd.DataFrame,
    df_usage: pd.DataFrame,
    regulation_id: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Calcula el TPI para todos los Pokémon de una regulación.

    Combina co-uso (teammates_by_pkm) con uso total (usage_by_reg) para
    ponderar compañeros frecuentes en el meta.

    Args:
        df_teammates: pokemon, teammate, avg_correlation, n_months_seen,
                      regulation_id (opcional).
        df_usage: pokemon, avg_usage_pct, regulation_id (opcional).
        regulation_id: Regulación a analizar.
        top_n: Top N Pokémon en el resultado ordenado por tpi_score.

    Returns:
        DataFrame con columnas esperadas y orden DESC por tpi_score.
        DataFrame vacío con columnas fijas si faltan datos.
    """
    _empty_cols = [
        "pokemon", "regulation_id", "tpi_score",
        "n_teammates", "top_teammate",
        "avg_co_usage", "usage_weighted_score",
    ]

    if df_teammates.empty or df_usage.empty:
        return pd.DataFrame(columns=_empty_cols)

    df_tm = (
        df_teammates[
            df_teammates["regulation_id"] == regulation_id
        ].copy()
        if "regulation_id" in df_teammates.columns
        else df_teammates.copy()
    )

    df_use = (
        df_usage[df_usage["regulation_id"] == regulation_id].copy()
        if "regulation_id" in df_usage.columns
        else df_usage.copy()
    )

    if df_tm.empty or df_use.empty:
        return pd.DataFrame(columns=_empty_cols)

    usage_map: dict[str, float] = {
        str(row["pokemon"]).lower(): float(row["avg_usage_pct"])
        for _, row in df_use.iterrows()
    }
    max_usage = max(usage_map.values()) if usage_map else 1.0
    if max_usage == 0:
        max_usage = 1.0

    results: list[TPIResult] = []

    for pokemon, group in df_tm.groupby("pokemon"):
        pokemon_str = str(pokemon)
        n_teammates = len(group)

        if n_teammates == 0:
            continue

        weighted_scores: list[float] = []
        co_usages: list[float] = []
        top_tm = ""
        top_co = 0.0

        for _, row in group.iterrows():
            teammate = str(row["teammate"])
            co_pct = float(row.get("avg_correlation", 0.0))
            co_usages.append(co_pct)

            tm_usage = usage_map.get(teammate.lower(), 0.0)
            usage_weight = tm_usage / max_usage

            weighted_scores.append(co_pct * usage_weight / 100.0)

            if co_pct > top_co:
                top_co = co_pct
                top_tm = teammate

        avg_co = float(np.mean(co_usages)) if co_usages else 0.0
        weighted_avg = float(np.mean(weighted_scores)) if weighted_scores else 0.0
        tpi_score = min(1.0, weighted_avg)

        results.append(
            TPIResult(
                pokemon=pokemon_str,
                regulation_id=regulation_id,
                tpi_score=round(tpi_score, 4),
                n_teammates=n_teammates,
                top_teammate=top_tm,
                avg_co_usage=round(avg_co, 4),
                usage_weighted_score=round(weighted_avg, 4),
            )
        )

    if not results:
        return pd.DataFrame(columns=_empty_cols)

    df_result = pd.DataFrame([
        {
            "pokemon": r.pokemon,
            "regulation_id": r.regulation_id,
            "tpi_score": r.tpi_score,
            "n_teammates": r.n_teammates,
            "top_teammate": r.top_teammate,
            "avg_co_usage": r.avg_co_usage,
            "usage_weighted_score": r.usage_weighted_score,
        }
        for r in results
    ])

    return (
        df_result.sort_values("tpi_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_tpi_from_duckdb(
    con: Any,
    regulation_id: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Carga teammates y uso desde DuckDB y calcula TPI.

    Args:
        con: Conexión DuckDB activa.
        regulation_id: Regulación a analizar.
        top_n: Top N Pokémon en el resultado.

    Returns:
        DataFrame de TPI. Vacío si hay error.
    """
    try:
        from src.app.data.sql.views import (  # noqa: PLC0415
            create_teammates_by_pkm,
            create_usage_by_reg,
            register_raw_view,
        )

        register_raw_view(con)
        df_tm = create_teammates_by_pkm(con, regulation_id)
        df_use = create_usage_by_reg(con, regulation_id)
        return compute_tpi(df_tm, df_use, regulation_id, top_n)
    except Exception as exc:  # noqa: BLE001
        log.warning("Error calculando TPI para %s: %s", regulation_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "EIResult",
    "STCResult",
    "TPIResult",
    "compute_ei",
    "get_top_ei_items",
    "compute_ei_from_duckdb",
    "compute_stc_meta",
    "compute_stc_from_duckdb",
    "compute_tpi",
    "compute_tpi_from_duckdb",
    "BETA_ALPHA",
    "BETA_BETA",
    "_apply_beta_shrinkage",
    "_load_speed_map",
]
