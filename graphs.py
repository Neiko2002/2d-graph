# graphs.py
from enum import Enum
from typing import Iterable, List, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import geopandas as gpd

from base_graph import (
    create_mrng,
    create_rng,
    create_knn_graph_2d,
    create_minimum_spanning_tree_2d,
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
    DEGStream = "DEGStream"
    DEGHigh = "DEGHigh"
    DEGLow = "DEGLow"
    DEGStreamRNGChecked = "DEGStreamChecked"
    DEGHighRNGChecked = "DEGHighChecked"
    DEGLowRNGChecked = "DEGLowChecked"
    DEGStreamOpt = "DEGStreamOpt"
    DEGHighOpt = "DEGHighOpt"
    DEGLowOpt = "DEGLowOpt"
    DEGStreamOptRNGChecked = "DEGStreamOptChecked"
    DEGHighOptRNGChecked = "DEGHighOptChecked"
    DEGLowOptRNGChecked = "DEGLowOptChecked"

@dataclass
class GraphItem:
    kind: GraphKind
    title: str
    is_directed: bool
    vertices: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    graph: Any = None
    annotations: List[str] = field(default_factory=list)
    custom_plot_render: Callable[[Any], None] = None

# Friendly titles for plotting
_TITLES = {
    GraphKind.KNN: "kNN Graph",
    GraphKind.MST: "MST",
    GraphKind.MRNG: "MRNG",
    GraphKind.RNG: "RNG",
    GraphKind.GG: "GG",
    GraphKind.DG: "DG",
    GraphKind.DEGStream: "DEGStream",
    GraphKind.DEGHigh: "crEG scheme (D)",
    GraphKind.DEGLow: "crEG scheme (C)",
    GraphKind.DEGStreamRNGChecked: "DEGStreamChecked",
    GraphKind.DEGHighRNGChecked: "DEGHighChecked",
    GraphKind.DEGLowRNGChecked: "DEGLowChecked",
    GraphKind.DEGStreamOpt: "DEGStreamOpt",
    GraphKind.DEGHighOpt: "DEGHighOpt",
    GraphKind.DEGLowOpt: "DEGLowOpt",
    GraphKind.DEGStreamOptRNGChecked: "DEGStreamOptChecked",
    GraphKind.DEGHighOptRNGChecked: "DEGHighOptChecked",
    GraphKind.DEGLowOptRNGChecked: "DEGLowOptChecked"
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
    out: List[GraphItem] = []

    # Cache for DEG graphs to avoid rebuilding
    deg_cache = {}

    def get_deg_base(target):
        if target not in deg_cache:
            X = extract_features(gdf)
            g = deglib.builder.build_from_data(
                X,
                edges_per_vertex=edges_per_vertex,
                extend_k=_DEG_EXTEND_K,
                extend_eps=_DEG_EXTEND_EPS,
                optimization_target=target,
            )
            deg_cache[target] = g
        return deg_cache[target]

    for kind in kinds:
        if kind == GraphKind.KNN:
            out.append(GraphItem(kind, _TITLES[kind], True, gdf, create_knn_graph_2d(gdf, k=edges_per_vertex)))
        elif kind == GraphKind.MST:
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, create_minimum_spanning_tree_2d(gdf)))
        elif kind == GraphKind.MRNG:
            out.append(GraphItem(kind, _TITLES[kind], True, gdf, create_mrng(gdf)))
        elif kind == GraphKind.RNG:
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, create_rng(gdf)))
        elif kind == GraphKind.GG:
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, create_gabriel_graph_2d(gdf)))
        elif kind == GraphKind.DG:
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, create_delaunay_graph_2d(gdf)))
        elif kind == GraphKind.DEGStream:
            g = get_deg_base(deglib.builder.OptimizationTarget.StreamingData)
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGHigh:
            g = get_deg_base(deglib.builder.OptimizationTarget.HighLID)
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGLow:
            g = get_deg_base(deglib.builder.OptimizationTarget.LowLID)
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))

        elif kind == GraphKind.DEGStreamRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.StreamingData)
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGHighRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.HighLID)
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGLowRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.LowLID)
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))

        elif kind == GraphKind.DEGStreamOpt:
            g = get_deg_base(deglib.builder.OptimizationTarget.StreamingData)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGHighOpt:
            g = get_deg_base(deglib.builder.OptimizationTarget.HighLID)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGLowOpt:
            g = get_deg_base(deglib.builder.OptimizationTarget.LowLID)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))

        elif kind == GraphKind.DEGStreamOptRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.StreamingData)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGHighOptRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.HighLID)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))
        elif kind == GraphKind.DEGLowOptRNGChecked:
            g = get_deg_base(deglib.builder.OptimizationTarget.LowLID)
            builder = deglib.builder.EvenRegularGraphBuilder(
                g, improve_k=_DEG_IMPROVE_K, improve_eps=_DEG_IMPROVE_EPS, swap_tries=_DEG_SWAP_TRIES
            )
            builder.build()
            g.remove_non_mrng_edges()
            gro = deglib.graph.ReadOnlyGraph.from_graph(g)
            out.append(GraphItem(kind, _TITLES[kind], False, gdf, get_deg_edges(gro), gro))

    return out