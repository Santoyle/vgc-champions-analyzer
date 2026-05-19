"""
Grafo PMI y detección de comunidades Louvain para VGC.

El grafo PMI representa co-aparición *inesperada* de Pokémon en equipos: una
arista entre A y B con PPMI alto significa que ambos aparecen juntos más de lo
que se esperaría por sus frecuencias individuales. Esto permite identificar
"arquetipos de equipo" en lugar de meras listas de uso.

El algoritmo Louvain maximiza la modularidad del grafo para particionar los
nodos en comunidades densamente conectadas internamente. Cada comunidad Louvain
representa un arquetipo o sinergia de equipo: Pokémon que co-aparecen de forma
inusualmente frecuente.

La visualización se hace con Plotly (go.Scatter) en lugar de pyvis o d3.js para
garantizar compatibilidad con Streamlit Cloud sin dependencias extras de frontend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CommunityResult:
    """
    Resultado del análisis de comunidades Louvain sobre el grafo PMI.

    Attributes:
        communities: Dict {pokemon: community_id}.
        n_communities: Número de comunidades detectadas.
        modularity: Modularidad del particionado (mayor = comunidades más densas).
        node_count: Número de nodos en el grafo.
        edge_count: Número de aristas en el grafo.
        community_members: Dict {community_id: list[pokemon]} con los miembros
                           de cada comunidad, ordenados alfabéticamente.
    """

    communities: dict[str, int]
    n_communities: int
    modularity: float
    node_count: int
    edge_count: int
    community_members: dict[int, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------


def build_pmi_graph(
    df_pmi: pd.DataFrame,
    ppmi_threshold: float = 0.1,
    top_pokemon: int = 50,
) -> Any:
    """
    Construye un grafo networkx desde el DataFrame de PMI.

    Nodos = Pokémon. Aristas = pares con PPMI > ppmi_threshold.
    El peso de cada arista es el valor PPMI del par.
    Solo incluye los top_pokemon por suma de PPMI para mantener
    el grafo manejable en la UI.

    Args:
        df_pmi: DataFrame con columnas pokemon, teammate, ppmi, co_usage_pct.
        ppmi_threshold: Umbral mínimo de PPMI para crear una arista.
        top_pokemon: Máximo de nodos a incluir (filtrado por suma de PPMI).

    Returns:
        networkx.Graph con nodos y aristas ponderadas.
        Grafo vacío (sin nodos) si df_pmi está vacío o no hay aristas
        sobre el umbral.

    Raises:
        ImportError: Si networkx no está instalado.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx no está instalado. Instala con: pip install networkx"
        )

    G: Any = nx.Graph()

    if df_pmi.empty:
        return G

    top_nodes = (
        df_pmi.groupby("pokemon")["ppmi"]
        .sum()
        .nlargest(top_pokemon)
        .index.tolist()
    )

    df_filtered = df_pmi[
        df_pmi["pokemon"].isin(top_nodes)
        & df_pmi["teammate"].isin(top_nodes)
        & (df_pmi["ppmi"] > ppmi_threshold)
    ]

    if df_filtered.empty:
        log.debug("Sin aristas sobre umbral PPMI=%.2f", ppmi_threshold)
        return G

    for _, row in df_filtered.iterrows():
        pokemon = str(row["pokemon"])
        teammate = str(row["teammate"])
        ppmi = float(row["ppmi"])
        co_pct = float(row.get("co_usage_pct", 0.0))  # type: ignore[arg-type]
        G.add_edge(pokemon, teammate, weight=ppmi, co_usage=co_pct)

    log.info(
        "Grafo PMI: %d nodos, %d aristas (umbral PPMI=%.2f)",
        G.number_of_nodes(),
        G.number_of_edges(),
        ppmi_threshold,
    )
    return G


def detect_communities(
    graph: Any,
    resolution: float = 1.0,
    random_state: int = 42,
) -> CommunityResult | None:
    """
    Detecta comunidades en el grafo PMI usando el algoritmo Louvain.

    Louvain maximiza la modularidad del grafo: comunidades con muchas aristas
    internas y pocas externas. resolution > 1.0 produce más comunidades
    pequeñas; resolution < 1.0 produce menos comunidades grandes.

    Args:
        graph: networkx.Graph a analizar.
        resolution: Parámetro de resolución Louvain. Default 1.0.
        random_state: Semilla para reproducibilidad.

    Returns:
        CommunityResult con el particionado y métricas.
        None si el grafo está vacío o si ocurre cualquier error.

    Raises:
        ImportError: Si python-louvain o networkx no están instalados.
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError(
            "python-louvain no está instalado. "
            "Instala con: pip install python-louvain"
        )

    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        raise ImportError("networkx no está instalado.")

    if graph.number_of_nodes() == 0:
        log.debug("Grafo vacío — sin comunidades")
        return None

    partition: dict[str, int] = community_louvain.best_partition(
        graph,
        weight="weight",
        resolution=resolution,
        random_state=random_state,
    )

    modularity = float(
        community_louvain.modularity(partition, graph, weight="weight")
    )

    n_communities = len(set(partition.values()))

    community_members: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        if comm_id not in community_members:
            community_members[comm_id] = []
        community_members[comm_id].append(str(node))

    for comm_id in community_members:
        community_members[comm_id].sort()

    log.info(
        "Louvain: %d comunidades, modularidad=%.3f sobre %d nodos",
        n_communities,
        modularity,
        graph.number_of_nodes(),
    )

    return CommunityResult(
        communities=partition,
        n_communities=n_communities,
        modularity=modularity,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        community_members=community_members,
    )


def graph_to_plotly(
    graph: Any,
    community_result: CommunityResult | None = None,
    layout: str = "spring",
) -> dict[str, Any]:
    """
    Convierte el grafo networkx a datos Plotly para visualización de red.

    Calcula posiciones de nodos con el layout especificado. Los nodos se
    colorean por community_id si se provee community_result; de lo contrario
    usan un color uniforme (0).

    Args:
        graph: networkx.Graph a visualizar.
        community_result: Resultado Louvain para colorear nodos. None = color 0.
        layout: "spring" (default), "circular" o "kamada_kawai".

    Returns:
        Dict con keys edge_x, edge_y, node_x, node_y, node_text, node_colors.
        Dict vacío si el grafo no tiene nodos.
    """
    try:
        import networkx as nx
    except ImportError:
        return {}

    if graph.number_of_nodes() == 0:
        return {}

    seed = 42
    pos: dict[Any, Any]
    if layout == "spring":
        pos = nx.spring_layout(graph, weight="weight", seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        try:
            pos = nx.kamada_kawai_layout(graph, weight="weight")
        except Exception:  # noqa: BLE001
            pos = nx.spring_layout(graph, seed=seed)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in graph.edges():
        xy0 = pos[u]
        xy1 = pos[v]
        edge_x.extend([float(xy0[0]), float(xy1[0]), None])
        edge_y.extend([float(xy0[1]), float(xy1[1]), None])

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_colors: list[int] = []

    for node in graph.nodes():
        xy = pos[node]
        node_x.append(float(xy[0]))
        node_y.append(float(xy[1]))
        node_text.append(str(node))
        if community_result is not None:
            node_colors.append(community_result.communities.get(str(node), 0))
        else:
            node_colors.append(0)

    return {
        "edge_x": edge_x,
        "edge_y": edge_y,
        "node_x": node_x,
        "node_y": node_y,
        "node_text": node_text,
        "node_colors": node_colors,
    }


__all__ = [
    "CommunityResult",
    "build_pmi_graph",
    "detect_communities",
    "graph_to_plotly",
]
