# graphs.py
from enum import Enum
from typing import Iterable, List, Any
import numpy as np
from dataclasses import dataclass

from base_graph import (
    create_mrng,
    create_knn_graph_2d,
    create_minimum_spanning_tree_2d,
    create_relative_neighborhood_graph_2d,
    create_gabriel_graph_2d,
    create_delaunay_graph_2d,
)
from deg import get_deg_edges
import deglib

class GraphKind(Enum):
    KNN = "KNN"
    MST = "MST"
    MRNG = "MRNG"
    RNG = "RNG"
    GG = "GG"
    DG = "DG"
    DEG1 = "DEG1"
    DEG2 = "DEG2"
    DEG3 = "DEG3"
    DEG4 = "DEG4"
    DEG5 = "DEG5"

@dataclass
class GraphItem:
    kind: GraphKind
    title: str
    vertices: Any
    edges: Any
    graph: Any = None


# Friendly titles for plotting
_TITLES = {
    GraphKind.KNN: "kNN Graph",
    GraphKind.MST: "MST",
    GraphKind.MRNG: "MRNG",
    GraphKind.RNG: "RNG",
    GraphKind.GG: "GG",
    GraphKind.DG: "DG",
    GraphKind.DEG1: "DEG1",
    GraphKind.DEG2: "DEG2",
    GraphKind.DEG3: "DEG3",
    GraphKind.DEG4: "DEG4",
    GraphKind.DEG5: "DEG5",
}

# Fixed parameters for DEG variants
_DEG_EXTEND_K = 10
_DEG_EXTEND_EPS = 0.2
_DEG_IMPROVE_K = 10
_DEG_IMPROVE_EPS = 0.001
_DEG_SWAP_TRIES = 1000


def extract_features(gdf) -> np.ndarray:
    if "feature" not in gdf:
        raise ValueError("GeoDataFrame must contain a 'feature' column")
    col = gdf[("feature")]
    # Convert a column of arrays to a dense 2D float32 array
    vals = col.to_numpy() if hasattr(col, "to_numpy") else np.array(list(col))
    if vals.dtype == object:
        arr = np.vstack([np.asarray(row, dtype=np.float32) for row in vals])
    else:
        arr = np.asarray(vals, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("feature column must contain 2D vectors")
    return arr

def build_graphs(
    gdf,
    kinds: Iterable["GraphKind"],
    *,
    edges_per_vertex: int = 4,
) -> List[GraphItem]:
    """
    Returns a list[GraphItem] with known attributes:
    - kind: GraphKind
    - title: str
    - vertices: GeoDataFrame
    - edges: GeoDataFrame
    - ro_graph: optional deglib ReadOnlyGraph (only for DEG)

    Vertices are always the feature vectors from gdf['feature'].
    Base 2D graphs use geometry coordinates. DEG graphs use feature.
    Only edges_per_vertex is configurable; other DEG parameters are fixed.
    """
    selected = list(kinds)
    out: List[GraphItem] = []

    # Base graphs from geometry
    if GraphKind.KNN in selected:
        out.append(GraphItem(GraphKind.KNN, _TITLES[GraphKind.KNN], gdf, create_knn_graph_2d(gdf)))
    if GraphKind.MST in selected:
        out.append(GraphItem(GraphKind.MST, _TITLES[GraphKind.MST], gdf, create_minimum_spanning_tree_2d(gdf)))
    if GraphKind.MRNG in selected:
        out.append(GraphItem(GraphKind.MRNG, _TITLES[GraphKind.MRNG], gdf, create_mrng(gdf)))
    if GraphKind.RNG in selected:
        out.append(GraphItem(GraphKind.RNG, _TITLES[GraphKind.RNG], gdf, create_relative_neighborhood_graph_2d(gdf)))
    if GraphKind.GG in selected:
        out.append(GraphItem(GraphKind.GG, _TITLES[GraphKind.GG], gdf, create_gabriel_graph_2d(gdf)))
    if GraphKind.DG in selected:
        out.append(GraphItem(GraphKind.DG, _TITLES[GraphKind.DG], gdf, create_delaunay_graph_2d(gdf)))

        # DEG graphs (features -> edges)
    if any(k in selected for k in (GraphKind.DEG1, GraphKind.DEG2, GraphKind.DEG3, GraphKind.DEG4, GraphKind.DEG5)):
        if deglib is None:
            raise ImportError("deglib is required for DEG graphs but is not installed")
        X = extract_features(gdf)

        # DEG1
        if GraphKind.DEG1 in selected:
            g1 = deglib.builder.build_from_data(
                X,
                edges_per_vertex=edges_per_vertex,
                extend_k=_DEG_EXTEND_K,
                extend_eps=_DEG_EXTEND_EPS,
                optimization_target=deglib.builder.OptimizationTarget.StreamingData,
            )
            gro1 = deglib.graph.ReadOnlyGraph.from_graph(g1)
            out.append(GraphItem(GraphKind.DEG1, _TITLES[GraphKind.DEG1], gdf, get_deg_edges(gro1), gro1))

        # DEG2
        if GraphKind.DEG2 in selected:
            g2 = deglib.builder.build_from_data(
                X,
                edges_per_vertex=edges_per_vertex,
                extend_k=_DEG_EXTEND_K,
                extend_eps=_DEG_EXTEND_EPS,
                optimization_target=deglib.builder.OptimizationTarget.LowLID,
            )
            gro2 = deglib.graph.ReadOnlyGraph.from_graph(g2)
            out.append(GraphItem(GraphKind.DEG2, _TITLES[GraphKind.DEG2], gdf, get_deg_edges(gro2), gro2))

        # DEG3 base (+ DEG4/DEG5)
        if any(k in selected for k in (GraphKind.DEG3, GraphKind.DEG4, GraphKind.DEG5)):
            g3 = deglib.builder.build_from_data(
                X,
                edges_per_vertex=edges_per_vertex,
                extend_k=_DEG_EXTEND_K,
                extend_eps=_DEG_EXTEND_EPS,
                optimization_target=deglib.builder.OptimizationTarget.HighLID,
            )
            gro3 = deglib.graph.ReadOnlyGraph.from_graph(g3)
            if GraphKind.DEG3 in selected:
                out.append(GraphItem(GraphKind.DEG3, _TITLES[GraphKind.DEG3], gdf, get_deg_edges(gro3), gro3))

            if GraphKind.DEG4 in selected or GraphKind.DEG5 in selected:
                builder = deglib.builder.EvenRegularGraphBuilder(
                    g3, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
                )
                builder.build()
                gro4 = deglib.graph.ReadOnlyGraph.from_graph(g3)
                if GraphKind.DEG4 in selected:
                    out.append(
                        GraphItem(GraphKind.DEG4, _TITLES[GraphKind.DEG4], gdf, get_deg_edges(gro4), gro4))

            if GraphKind.DEG5 in selected:
                g3.remove_non_mrng_edges()
                gro5 = deglib.graph.ReadOnlyGraph.from_graph(g3)
                out.append(GraphItem(GraphKind.DEG5, _TITLES[GraphKind.DEG5], gdf, get_deg_edges(gro5), gro5))

    return out