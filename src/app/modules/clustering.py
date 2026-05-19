"""
Módulo de clustering para análisis de uso en VGC.

Implementa dos estrategias:

  KMeans (cluster_kmeans):
    Requiere especificar el número de clusters k explícitamente.
    Más rápido y determinista. Útil cuando se tiene una hipótesis
    sobre cuántos grupos existen en el meta (ej. staples, nicho,
    situacionales, dominantes).
    Usa find_optimal_kmeans_k para seleccionar k automáticamente
    mediante silhouette_score si no se quiere especificar k a mano.

  HDBSCAN (cluster_hdbscan):
    Detecta el número de clusters automáticamente a partir de la
    densidad de los datos. Clasifica como ruido (-1) los puntos que
    no pertenecen a ningún cluster denso.
    Requiere calibrar min_cluster_size según el tamaño del dataset.
    Para datasets VGC típicos (200-300 Pokémon) el valor recomendado
    es min_cluster_size=15. Valores más bajos dan más clusters pequeños;
    valores más altos agrupan más agresivamente.

Ninguna función importa streamlit — este módulo es lógica de dominio pura.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """
    Resultado de un clustering sobre Pokémon de uso.

    Attributes:
        labels: Array de etiquetas de cluster por Pokémon. -1 = ruido (HDBSCAN).
        n_clusters: Número de clusters encontrados (excluye ruido -1).
        silhouette: Score de silueta si calculable.
                    None si < 2 clusters o error numérico.
        method: "kmeans" o "hdbscan".
        df_clustered: DataFrame original con columnas "cluster" y
                      "cluster_label" añadidas.
        noise_count: Puntos clasificados como ruido (solo relevante en HDBSCAN).
    """

    labels: np.ndarray
    n_clusters: int
    silhouette: float | None
    method: str
    df_clustered: pd.DataFrame
    noise_count: int = 0


# ---------------------------------------------------------------------------
# Helper privado
# ---------------------------------------------------------------------------


def _prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Prepara y escala features para clustering.

    Filtra las columnas disponibles en df, elimina filas con NaN en ellas
    y escala con StandardScaler.

    Args:
        df: DataFrame con los datos (debe tener columna "pokemon").
        feature_cols: Lista de columnas candidatas a usar como features.

    Returns:
        Tupla (X_scaled, df_filtered) donde X_scaled es el array escalado
        y df_filtered solo contiene filas sin NaN en las features disponibles.

    Raises:
        ValueError: Si no hay columnas disponibles o el DataFrame queda vacío.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(
            f"Ninguna de las columnas requeridas está disponible: {feature_cols}"
        )

    df_clean = df[["pokemon"] + available].dropna(subset=available).copy()

    if df_clean.empty:
        raise ValueError("DataFrame vacío tras eliminar NaN en las features.")

    X = df_clean[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)

    return X_scaled, df_clean


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------


def cluster_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> ClusterResult:
    """
    Aplica KMeans sobre los datos de uso de Pokémon.

    Args:
        df: DataFrame con columnas de uso. Debe tener "pokemon" y las
            feature_cols.
        n_clusters: Número de clusters deseados.
        feature_cols: Columnas a usar como features.
                      Default: avg_usage_pct, max_usage_pct, min_usage_pct.
        random_state: Semilla para reproducibilidad.

    Returns:
        ClusterResult con method="kmeans" y noise_count=0.

    Raises:
        ValueError: Si el DataFrame tiene menos filas que n_clusters, o si
                    las features no están disponibles.
    """
    if feature_cols is None:
        feature_cols = ["avg_usage_pct", "max_usage_pct", "min_usage_pct"]

    if len(df) < n_clusters:
        raise ValueError(
            f"DataFrame tiene {len(df)} filas pero se pidieron {n_clusters} clusters."
        )

    X_scaled, df_clean = _prepare_features(df, feature_cols)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels: np.ndarray = kmeans.fit_predict(X_scaled)

    silhouette: float | None = None
    if n_clusters >= 2 and len(set(labels.tolist())) >= 2:
        try:
            silhouette = float(silhouette_score(X_scaled, labels))
        except Exception as exc:  # noqa: BLE001
            log.debug("Silhouette KMeans no calculable: %s", exc)

    df_result = df_clean.copy()
    df_result["cluster"] = labels
    df_result["cluster_label"] = (
        "Cluster " + pd.Series(labels, index=df_clean.index).astype(str)
    )

    log.info(
        "KMeans: %d clusters sobre %d Pokémon (silhouette: %s)",
        n_clusters,
        len(df_clean),
        f"{silhouette:.3f}" if silhouette is not None else "N/A",
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        silhouette=silhouette,
        method="kmeans",
        df_clustered=df_result,
        noise_count=0,
    )


def cluster_hdbscan(
    df: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    feature_cols: list[str] | None = None,
) -> ClusterResult:
    """
    Aplica HDBSCAN sobre los datos de uso de Pokémon.

    HDBSCAN detecta automáticamente el número de clusters y clasifica como
    ruido (-1) los puntos que no pertenecen a ningún cluster denso.
    min_cluster_size=15 es el valor recomendado para datasets VGC típicos
    (200-300 Pokémon).

    Args:
        df: DataFrame con columnas de uso.
        min_cluster_size: Tamaño mínimo de cluster. Default 15.
        min_samples: Parámetro de robustez al ruido. Si None, usa
                     min_cluster_size.
        feature_cols: Columnas a usar como features.

    Returns:
        ClusterResult con method="hdbscan". noise_count contiene los puntos
        con label=-1. cluster_label="🔘 Ruido" para puntos de ruido.

    Raises:
        ImportError: Si el paquete hdbscan no está instalado.
        ValueError: Si el DataFrame tiene menos filas que min_cluster_size.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError(
            "hdbscan no está instalado. Instala con: pip install hdbscan"
        )

    if feature_cols is None:
        feature_cols = ["avg_usage_pct", "max_usage_pct", "min_usage_pct"]

    if len(df) < min_cluster_size:
        raise ValueError(
            f"DataFrame tiene {len(df)} filas pero min_cluster_size={min_cluster_size}."
        )

    if min_samples is None:
        min_samples = min_cluster_size

    X_scaled, df_clean = _prepare_features(df, feature_cols)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels: np.ndarray = clusterer.fit_predict(X_scaled)

    unique_labels: set[int] = set(labels.tolist())
    noise_count = int(np.sum(labels == -1))
    n_clusters = len(unique_labels - {-1})

    silhouette: float | None = None
    mask = labels != -1
    if n_clusters >= 2 and int(np.sum(mask)) >= 2:
        try:
            silhouette = float(silhouette_score(X_scaled[mask], labels[mask]))
        except Exception as exc:  # noqa: BLE001
            log.debug("Silhouette HDBSCAN no calculable: %s", exc)

    df_result = df_clean.copy()
    df_result["cluster"] = labels
    df_result["cluster_label"] = [
        f"Cluster {lbl}" if lbl != -1 else "🔘 Ruido" for lbl in labels.tolist()
    ]

    log.info(
        "HDBSCAN: %d clusters, %d ruido sobre %d Pokémon (silhouette: %s)",
        n_clusters,
        noise_count,
        len(df_clean),
        f"{silhouette:.3f}" if silhouette is not None else "N/A",
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        silhouette=silhouette,
        method="hdbscan",
        df_clustered=df_result,
        noise_count=noise_count,
    )


def find_optimal_kmeans_k(
    df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 12,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> int:
    """
    Encuentra el k óptimo para KMeans usando silhouette_score.

    Evalúa k desde k_min hasta min(k_max, len(df)-1) y retorna el k con
    mayor silhouette_score. Prefiere silhouette sobre elbow method porque
    es más robusto en datasets pequeños (< 300 Pokémon típicos en VGC).

    Args:
        df: DataFrame con los datos.
        k_min: k mínimo a evaluar (inclusive).
        k_max: k máximo a evaluar (inclusive, acotado por len(df)-1).
        feature_cols: Columnas de features.
        random_state: Semilla.

    Returns:
        k óptimo según silhouette_score. Retorna k_min si el DataFrame
        es demasiado pequeño para evaluar o si ocurre cualquier error.
    """
    if feature_cols is None:
        feature_cols = ["avg_usage_pct", "max_usage_pct", "min_usage_pct"]

    try:
        X_scaled, _ = _prepare_features(df, feature_cols)
    except ValueError:
        return k_min

    best_k = k_min
    best_score = -1.0
    k_max_actual = min(k_max, len(df) - 1)

    for k in range(k_min, k_max_actual + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            if len(set(labels.tolist())) < 2:
                continue
            score = float(silhouette_score(X_scaled, labels))
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:  # noqa: BLE001
            continue

    log.info(
        "K óptimo: %d (silhouette=%.3f) en rango [%d, %d]",
        best_k,
        best_score,
        k_min,
        k_max_actual,
    )
    return best_k


__all__ = [
    "ClusterResult",
    "cluster_kmeans",
    "cluster_hdbscan",
    "find_optimal_kmeans_k",
]
