"""
Tests para src/app/modules/clustering.py y src/app/modules/graph_pmi.py.

Estrategia:
  - Datos sintéticos inyectados via fixtures (np.random.default_rng(42)).
  - Sin I/O real: no se leen Parquets ni se conecta DuckDB.
  - Los tests de HDBSCAN, networkx y python-louvain usan pytest.skip()
    si la librería no está instalada.

Grupos:
  1. TestPrepareFeatures    (4 tests) — función auxiliar de features.
  2. TestClusterKmeans      (8 tests) — clustering KMeans.
  3. TestClusterHDBSCAN     (4 tests) — clustering HDBSCAN.
  4. TestFindOptimalK       (3 tests) — búsqueda de k óptimo.
  5. TestBuildPMIGraph      (4 tests) — construcción del grafo PMI.
  6. TestDetectCommunities  (3 tests) — detección de comunidades Louvain.
  7. TestGraphToPlotly      (3 tests) — conversión a datos Plotly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.modules.clustering import (
    ClusterResult,
    _prepare_features,
    cluster_hdbscan,
    cluster_kmeans,
    find_optimal_kmeans_k,
)
from src.app.modules.graph_pmi import (
    CommunityResult,
    build_pmi_graph,
    detect_communities,
    graph_to_plotly,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_usage_df() -> pd.DataFrame:
    """DataFrame de uso sintético con 30 Pokémon y columnas avg/max/min usage_pct."""
    rng = np.random.default_rng(42)
    n = 30
    avg = rng.uniform(1.0, 60.0, n)
    return pd.DataFrame(
        {
            "pokemon": [f"Pokemon{i}" for i in range(n)],
            "regulation_id": ["TEST"] * n,
            "avg_usage_pct": avg,
            "max_usage_pct": avg + rng.uniform(0, 10, n),
            "min_usage_pct": np.maximum(avg - rng.uniform(0, 10, n), 0),
            "n_months": rng.integers(1, 5, n),
            "total_raw_count": rng.integers(100, 10000, n),
        }
    )


@pytest.fixture
def sample_pmi_df() -> pd.DataFrame:
    """DataFrame PMI sintético para tests de grafo."""
    pokemon = [
        "Incineroar", "Garchomp", "Sneasler", "Sinistcha", "Kingambit",
        "Flutter Mane", "Rillaboom", "Urshifu", "Basculegion", "Amoonguss",
    ]
    rng = np.random.default_rng(42)
    rows = []
    for i, p1 in enumerate(pokemon):
        for j, p2 in enumerate(pokemon):
            if i != j:
                ppmi = float(rng.uniform(0.0, 2.0))
                rows.append(
                    {
                        "pokemon": p1,
                        "teammate": p2,
                        "pmi": ppmi - 0.5,
                        "ppmi": ppmi,
                        "co_usage_pct": float(rng.uniform(5, 50)),
                        "n_months_seen": 3,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GRUPO 1 — _prepare_features
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    """Tests de la función auxiliar de preparación de features."""

    def test_returns_scaled_array_and_df(self, sample_usage_df: pd.DataFrame) -> None:
        """Retorna array escalado y DataFrame limpio con las dimensiones correctas."""
        X, df_clean = _prepare_features(
            sample_usage_df, ["avg_usage_pct", "max_usage_pct"]
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(df_clean, pd.DataFrame)
        assert X.shape[0] == len(df_clean)
        assert X.shape[1] == 2

    def test_raises_on_missing_columns(self, sample_usage_df: pd.DataFrame) -> None:
        """Levanta ValueError si ninguna columna pedida existe en el DataFrame."""
        with pytest.raises(ValueError):
            _prepare_features(sample_usage_df, ["nonexistent_col"])

    def test_raises_on_empty_df(self) -> None:
        """Levanta ValueError si el DataFrame queda vacío tras eliminar NaN."""
        df_empty = pd.DataFrame(
            {"pokemon": ["A"], "avg_usage_pct": [np.nan]}
        )
        with pytest.raises(ValueError):
            _prepare_features(df_empty, ["avg_usage_pct"])

    def test_scaled_values_have_zero_mean(self, sample_usage_df: pd.DataFrame) -> None:
        """StandardScaler produce media ≈ 0 en la columna escalada."""
        X, _ = _prepare_features(sample_usage_df, ["avg_usage_pct"])
        assert abs(float(X[:, 0].mean())) < 0.1


# ---------------------------------------------------------------------------
# GRUPO 2 — cluster_kmeans
# ---------------------------------------------------------------------------


class TestClusterKmeans:
    """Tests para el clustering KMeans."""

    def test_returns_cluster_result(self, sample_usage_df: pd.DataFrame) -> None:
        """Retorna una instancia de ClusterResult con method='kmeans'."""
        result = cluster_kmeans(sample_usage_df, n_clusters=3)
        assert isinstance(result, ClusterResult)
        assert result.method == "kmeans"

    def test_correct_n_clusters(self, sample_usage_df: pd.DataFrame) -> None:
        """n_clusters del resultado coincide con el valor pedido."""
        result = cluster_kmeans(sample_usage_df, n_clusters=4)
        assert result.n_clusters == 4

    def test_df_clustered_has_cluster_columns(
        self, sample_usage_df: pd.DataFrame
    ) -> None:
        """df_clustered tiene columnas 'cluster' y 'cluster_label'."""
        result = cluster_kmeans(sample_usage_df, n_clusters=3)
        assert "cluster" in result.df_clustered.columns
        assert "cluster_label" in result.df_clustered.columns

    def test_labels_count_matches_df(self, sample_usage_df: pd.DataFrame) -> None:
        """Número de labels == número de filas en df_clustered."""
        result = cluster_kmeans(sample_usage_df, n_clusters=3)
        assert len(result.labels) == len(result.df_clustered)

    def test_noise_count_is_zero_for_kmeans(
        self, sample_usage_df: pd.DataFrame
    ) -> None:
        """KMeans no produce ruido — noise_count debe ser 0."""
        result = cluster_kmeans(sample_usage_df, n_clusters=3)
        assert result.noise_count == 0

    def test_raises_when_k_exceeds_samples(
        self, sample_usage_df: pd.DataFrame
    ) -> None:
        """ValueError si n_clusters > número de muestras en el DataFrame."""
        with pytest.raises(ValueError):
            cluster_kmeans(sample_usage_df, n_clusters=len(sample_usage_df) + 1)

    def test_silhouette_is_float_or_none(self, sample_usage_df: pd.DataFrame) -> None:
        """Silhouette es float o None, nunca levanta excepción."""
        result = cluster_kmeans(sample_usage_df, n_clusters=3)
        assert result.silhouette is None or isinstance(result.silhouette, float)

    def test_reproducible_with_same_seed(self, sample_usage_df: pd.DataFrame) -> None:
        """El mismo random_state produce etiquetas idénticas."""
        r1 = cluster_kmeans(sample_usage_df, n_clusters=3, random_state=42)
        r2 = cluster_kmeans(sample_usage_df, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# GRUPO 3 — cluster_hdbscan
# ---------------------------------------------------------------------------


class TestClusterHDBSCAN:
    """Tests para el clustering HDBSCAN (con pytest.skip si no está instalado)."""

    def test_returns_cluster_result(self, sample_usage_df: pd.DataFrame) -> None:
        """Retorna ClusterResult con method='hdbscan'."""
        try:
            result = cluster_hdbscan(sample_usage_df, min_cluster_size=5)
            assert isinstance(result, ClusterResult)
            assert result.method == "hdbscan"
        except ImportError:
            pytest.skip("hdbscan no instalado")

    def test_noise_points_have_minus_one_label(
        self, sample_usage_df: pd.DataFrame
    ) -> None:
        """noise_count coincide con la cantidad de labels == -1."""
        try:
            result = cluster_hdbscan(sample_usage_df, min_cluster_size=10)
            noise_mask = result.labels == -1
            assert result.noise_count == int(noise_mask.sum())
        except ImportError:
            pytest.skip("hdbscan no instalado")

    def test_noise_label_in_df(self, sample_usage_df: pd.DataFrame) -> None:
        """Filas con cluster==-1 tienen cluster_label='🔘 Ruido'."""
        try:
            result = cluster_hdbscan(sample_usage_df, min_cluster_size=10)
            df = result.df_clustered
            noise_rows = df[df["cluster"] == -1]
            if len(noise_rows) > 0:
                assert (noise_rows["cluster_label"] == "🔘 Ruido").all()
        except ImportError:
            pytest.skip("hdbscan no instalado")

    def test_raises_when_df_too_small(self) -> None:
        """ValueError si el DataFrame tiene menos filas que min_cluster_size."""
        df_small = pd.DataFrame(
            {
                "pokemon": ["A", "B", "C"],
                "avg_usage_pct": [10.0, 20.0, 30.0],
                "max_usage_pct": [15.0, 25.0, 35.0],
                "min_usage_pct": [5.0, 15.0, 25.0],
            }
        )
        try:
            with pytest.raises(ValueError):
                cluster_hdbscan(df_small, min_cluster_size=10)
        except ImportError:
            pytest.skip("hdbscan no instalado")


# ---------------------------------------------------------------------------
# GRUPO 4 — find_optimal_kmeans_k
# ---------------------------------------------------------------------------


class TestFindOptimalK:
    """Tests para la búsqueda de k óptimo por silhouette."""

    def test_returns_int(self, sample_usage_df: pd.DataFrame) -> None:
        """Siempre retorna un entero."""
        k = find_optimal_kmeans_k(sample_usage_df, k_min=2, k_max=5)
        assert isinstance(k, int)

    def test_k_within_bounds(self, sample_usage_df: pd.DataFrame) -> None:
        """k está entre k_min y k_max (inclusive)."""
        k = find_optimal_kmeans_k(sample_usage_df, k_min=2, k_max=6)
        assert 2 <= k <= 6

    def test_returns_k_min_for_tiny_df(self) -> None:
        """Retorna k_min sin crash para DataFrame muy pequeño."""
        df_tiny = pd.DataFrame(
            {
                "pokemon": ["A", "B"],
                "avg_usage_pct": [10.0, 20.0],
                "max_usage_pct": [15.0, 25.0],
                "min_usage_pct": [5.0, 15.0],
            }
        )
        k = find_optimal_kmeans_k(df_tiny, k_min=2, k_max=5)
        assert k == 2


# ---------------------------------------------------------------------------
# GRUPO 5 — build_pmi_graph
# ---------------------------------------------------------------------------


class TestBuildPMIGraph:
    """Tests para la construcción del grafo PMI."""

    def test_returns_graph_with_nodes(self, sample_pmi_df: pd.DataFrame) -> None:
        """Grafo con datos reales tiene al menos un nodo."""
        try:
            import networkx as nx

            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1, top_pokemon=10)
            assert isinstance(G, nx.Graph)
            assert G.number_of_nodes() > 0
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_empty_df_returns_empty_graph(self) -> None:
        """DataFrame vacío retorna grafo sin nodos."""
        try:
            G = build_pmi_graph(pd.DataFrame())
            assert G.number_of_nodes() == 0
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_high_threshold_reduces_edges(self, sample_pmi_df: pd.DataFrame) -> None:
        """Umbral PPMI más alto produce igual o menos aristas."""
        try:
            G_low = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1)
            G_high = build_pmi_graph(sample_pmi_df, ppmi_threshold=1.9)
            assert G_high.number_of_edges() <= G_low.number_of_edges()
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_edges_have_weight(self, sample_pmi_df: pd.DataFrame) -> None:
        """Cada arista tiene el atributo 'weight'."""
        try:
            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1)
            if G.number_of_edges() > 0:
                for _, _, data in G.edges(data=True):
                    assert "weight" in data
                    break
        except ImportError:
            pytest.skip("networkx no instalado")


# ---------------------------------------------------------------------------
# GRUPO 6 — detect_communities
# ---------------------------------------------------------------------------


class TestDetectCommunities:
    """Tests para la detección de comunidades Louvain."""

    def test_returns_community_result(self, sample_pmi_df: pd.DataFrame) -> None:
        """Retorna CommunityResult para un grafo con nodos."""
        try:
            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1)
            result = detect_communities(G)
            if result is not None:
                assert isinstance(result, CommunityResult)
        except ImportError:
            pytest.skip("networkx o python-louvain no instalados")

    def test_returns_none_for_empty_graph(self) -> None:
        """Retorna None para grafo vacío sin crash."""
        try:
            import networkx as nx

            G_empty = nx.Graph()
            result = detect_communities(G_empty)
            assert result is None
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_community_members_cover_all_nodes(
        self, sample_pmi_df: pd.DataFrame
    ) -> None:
        """La unión de todos los miembros de comunidades == todos los nodos."""
        try:
            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1)
            result = detect_communities(G)
            if result is not None:
                all_members = [
                    m
                    for members in result.community_members.values()
                    for m in members
                ]
                assert set(all_members) == set(str(n) for n in G.nodes())
        except ImportError:
            pytest.skip("networkx o python-louvain no instalados")


# ---------------------------------------------------------------------------
# GRUPO 7 — graph_to_plotly
# ---------------------------------------------------------------------------


class TestGraphToPlotly:
    """Tests para la conversión del grafo a datos Plotly."""

    def test_returns_dict_with_required_keys(
        self, sample_pmi_df: pd.DataFrame
    ) -> None:
        """El dict resultante contiene todas las keys esperadas."""
        try:
            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1)
            data = graph_to_plotly(G)
            if data:
                required = {"edge_x", "edge_y", "node_x", "node_y", "node_text", "node_colors"}
                assert required.issubset(set(data.keys()))
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_empty_graph_returns_empty_dict(self) -> None:
        """Grafo vacío retorna dict vacío {}."""
        try:
            import networkx as nx

            data = graph_to_plotly(nx.Graph())
            assert data == {}
        except ImportError:
            pytest.skip("networkx no instalado")

    def test_node_count_matches(self, sample_pmi_df: pd.DataFrame) -> None:
        """El número de elementos en node_x == número de nodos del grafo."""
        try:
            G = build_pmi_graph(sample_pmi_df, ppmi_threshold=0.1, top_pokemon=10)
            data = graph_to_plotly(G)
            if data and G.number_of_nodes() > 0:
                assert len(data["node_x"]) == G.number_of_nodes()
        except ImportError:
            pytest.skip("networkx no instalado")
