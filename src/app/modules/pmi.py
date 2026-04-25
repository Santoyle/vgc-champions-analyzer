"""
Cálculo de PMI (Pointwise Mutual Information) sobre datos de co-uso de Pokémon.

PMI mide cuánto más frecuentemente aparecen dos Pokémon juntos en un equipo
de lo que se esperaría si fueran independientes. Un PMI positivo indica una
sinergia real; un PMI negativo indica que se evitan mutuamente.

  PMI(i, j) = log( P(i,j) / (P(i) · P(j)) )

PPMI (Positive PMI) = max(PMI, 0) elimina el ruido negativo y es la métrica
recomendada para recomendaciones — los pares con PMI negativo simplemente no
se muestran en lugar de penalizar activamente.

build_pmi_matrix() genera una matriz cuadrada N×N de PPMI útil para heatmaps
en la página de Analytics.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PMIPair:
    """
    Par de Pokémon con su PMI calculado.

    Attributes:
        pokemon: Pokémon de referencia.
        teammate: Compañero sugerido.
        pmi: Pointwise Mutual Information.
             Positivo = aparecen juntos más de lo esperado por azar.
             0 = independientes. Negativo = aparecen menos juntos.
        ppmi: PMI positivo (max(PMI, 0)) — versión sin ruido negativo,
              más útil para recomendaciones.
        co_usage_pct: Porcentaje de co-uso directo (avg_correlation).
        n_months: Meses con datos para este par.
    """

    pokemon: str
    teammate: str
    pmi: float
    ppmi: float
    co_usage_pct: float
    n_months: int


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------


def compute_pmi_from_teammates(
    df_teammates: pd.DataFrame,
    df_usage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula PMI para todos los pares de Pokémon a partir de los datos
    de teammates y usage.

    Fórmula:
        PMI(i, j) = log( P(i,j) / (P(i) · P(j)) )

    Donde las probabilidades se aproximan con:
        P(i,j) ≈ avg_correlation / 100  (co-uso del par)
        P(i)   ≈ avg_usage_pct / 100    (uso individual de i)
        P(j)   ≈ avg_usage_pct / 100    (uso individual de j)

    La normalización es aproximada pero suficiente para recomendaciones
    relativas — lo que importa es el orden, no el valor absoluto.

    PPMI = max(PMI, 0): elimina pares que aparecen menos de lo esperado.

    Args:
        df_teammates: DataFrame con columnas pokemon, teammate,
                      avg_correlation, n_months_seen, regulation_id.
        df_usage: DataFrame con columnas pokemon, avg_usage_pct,
                  regulation_id.

    Returns:
        DataFrame con columnas: pokemon, teammate, pmi, ppmi,
        co_usage_pct, n_months_seen. Ordenado por pokemon ASC, ppmi DESC.
        DataFrame vacío si los inputs están vacíos.
    """
    if df_teammates.empty or df_usage.empty:
        return pd.DataFrame(
            columns=["pokemon", "teammate", "pmi", "ppmi", "co_usage_pct", "n_months_seen"]
        )

    usage_map: dict[str, float] = dict(
        zip(df_usage["pokemon"], df_usage["avg_usage_pct"])
    )

    rows = []
    for _, row in df_teammates.iterrows():
        pokemon = str(row["pokemon"])
        teammate = str(row["teammate"])
        co_pct = float(row.get("avg_correlation", 0.0))
        n_months = int(row.get("n_months_seen", 0))

        p_i = usage_map.get(pokemon, 0.0) / 100.0
        p_j = usage_map.get(teammate, 0.0) / 100.0
        p_ij = co_pct / 100.0

        if p_i <= 0 or p_j <= 0 or p_ij <= 0:
            pmi = 0.0
        else:
            try:
                pmi = math.log(p_ij / (p_i * p_j))
            except (ValueError, ZeroDivisionError):
                pmi = 0.0

        ppmi = max(pmi, 0.0)
        rows.append(
            {
                "pokemon": pokemon,
                "teammate": teammate,
                "pmi": round(pmi, 4),
                "ppmi": round(ppmi, 4),
                "co_usage_pct": round(co_pct, 4),
                "n_months_seen": n_months,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["pokemon", "teammate", "pmi", "ppmi", "co_usage_pct", "n_months_seen"]
        )

    df_pmi = pd.DataFrame(rows)
    return df_pmi.sort_values(
        ["pokemon", "ppmi"], ascending=[True, False]
    ).reset_index(drop=True)


def get_top_teammates(
    df_pmi: pd.DataFrame,
    pokemon: str,
    top_n: int = 10,
    min_ppmi: float = 0.0,
    exclude: list[str] | None = None,
) -> list[PMIPair]:
    """
    Retorna los mejores compañeros para un Pokémon ordenados por PPMI desc.

    Excluye automáticamente el propio Pokémon y los que ya están en el
    equipo (parámetro exclude).

    Args:
        df_pmi: DataFrame con PMI calculado (output de
                compute_pmi_from_teammates).
        pokemon: Nombre del Pokémon de referencia.
        top_n: Número máximo de sugerencias.
        min_ppmi: Umbral mínimo de PPMI. Default 0.0 (todos positivos).
        exclude: Pokémon a excluir de las sugerencias (ya en el equipo).

    Returns:
        Lista de PMIPair ordenada por ppmi desc.
        Lista vacía si no hay datos para el Pokémon.
    """
    if df_pmi.empty:
        return []

    exclude_set = set(exclude or [])
    exclude_set.add(pokemon)

    mask = (
        (df_pmi["pokemon"] == pokemon)
        & (df_pmi["ppmi"] >= min_ppmi)
        & (~df_pmi["teammate"].isin(exclude_set))
    )
    filtered = df_pmi[mask].head(top_n)

    return [
        PMIPair(
            pokemon=pokemon,
            teammate=str(row["teammate"]),
            pmi=float(row["pmi"]),
            ppmi=float(row["ppmi"]),
            co_usage_pct=float(row["co_usage_pct"]),
            n_months=int(row["n_months_seen"]),
        )
        for _, row in filtered.iterrows()
    ]


def build_pmi_matrix(
    df_pmi: pd.DataFrame,
    top_pokemon: int = 20,
) -> pd.DataFrame:
    """
    Construye una matriz cuadrada de PPMI para los top_pokemon más frecuentes.

    Útil para visualizar como heatmap en la página de Analytics.

    Args:
        df_pmi: DataFrame con PMI calculado.
        top_pokemon: Número de Pokémon a incluir en la matriz.

    Returns:
        DataFrame cuadrado (N×N) con PPMI como valores, indexado y
        columneado por nombre de Pokémon. Cero donde no hay dato.
        DataFrame vacío si df_pmi está vacío.
    """
    if df_pmi.empty:
        return pd.DataFrame()

    top = (
        df_pmi.groupby("pokemon")["ppmi"]
        .sum()
        .nlargest(top_pokemon)
        .index.tolist()
    )

    df_filtered = df_pmi[
        df_pmi["pokemon"].isin(top) & df_pmi["teammate"].isin(top)
    ]

    if df_filtered.empty:
        return pd.DataFrame()

    matrix = df_filtered.pivot_table(
        index="pokemon",
        columns="teammate",
        values="ppmi",
        fill_value=0.0,
    )

    all_pokemon = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=all_pokemon, columns=all_pokemon, fill_value=0.0)

    return matrix


__all__ = [
    "PMIPair",
    "compute_pmi_from_teammates",
    "get_top_teammates",
    "build_pmi_matrix",
]
