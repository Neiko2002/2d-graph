import city2graph
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

def _pairwise_dist(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

def create_rng(coords):
    """
    Construct an undirected Relative Neighborhood Graph (RNG).
    An edge (i, j) exists if no other point k is closer to both i and j
    than i and j are to each other (lune condition).
    Returns GeoDataFrame with [source, target, geometry, weight, color]
    """
    if isinstance(coords, gpd.GeoDataFrame):
        coords = np.array(coords['feature'].tolist(), dtype=np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)

    dist = _pairwise_dist(coords)
    edges = set()

    for i in range(n):
        for j in range(i+1, n):
            dij = dist[i, j]
            keep = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if dist[i, k] < dij and dist[j, k] < dij:
                    keep = False
                    break
            if keep:
                edges.add((i, j))

    rows = []
    for i, j in sorted(edges):
        dij = float(dist[i, j])
        rows.append((i, j, LineString([coords[i,:2], coords[j,:2]]), dij, "black"))

    return gpd.GeoDataFrame(rows, columns=["source", "target", "geometry", "weight", "color"])

def create_mrng(coords):
    """
    Construct a directed Monotonic Relative Neighborhood Graph (MRNG)
    following Definition 5 in NSG paper.
    Each node p has directed edges p -> q determined by:
      1) Sort all q ≠ p by distance δ(p, q)
      2) Always include the nearest neighbor q₀
      3) For each remaining q, include p -> q if no already-selected r
         satisfies δ(p, r) < δ(p, q) and δ(q, r) < δ(p, q)
    Returns GeoDataFrame with [source, target, geometry, weight, color]
    """
    if isinstance(coords, gpd.GeoDataFrame):
        coords = np.array(coords['feature'].tolist(), dtype=np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    edges = set()

    for i in range(n):
        order = np.argsort(dist[i])
        L = []
        for j in order:
            if i == j:
                continue
            # Check lune condition against already selected neighbors
            skip = False
            for r in L:
                if dist[i, r] < dist[i, j] and dist[j, r] < dist[i, j]:
                    skip = True
                    break
            if not skip:
                edges.add((i, j))  # directed edge
                L.append(j)

    # Always include nearest neighbor edges to ensure connectivity
    for i in range(n):
        j = np.argmin(dist[i] + np.eye(n)[i]*1e9)
        edges.add((i, j))

    rows = []
    for i, j in sorted(edges):
        dij = float(dist[i, j])
        rows.append((i, j, LineString([coords[i], coords[j]]), dij, "black"))

    return gpd.GeoDataFrame(rows, columns=["source", "target", "geometry", "weight", "color"])


def create_knn_graph(coords, k=4):
    """
    Create a directed k-nearest neighbor graph.
    For each point, a directed edge is created to its k-nearest neighbors.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    if isinstance(coords, gpd.GeoDataFrame):
        coords = np.array(coords['feature'].tolist(), dtype=np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    dist = _pairwise_dist(coords)

    # For each point, find the k-nearest neighbors
    # np.argsort returns indices that would sort the array.
    # We skip the first one (index 0) because it's the point itself.
    neighbors = np.argsort(dist, axis=1)[:, 1:k+1]

    # build GeoDataFrame
    rows = []
    for i in range(n):
        for j in neighbors[i]:
            dij = float(dist[i, j])
            rows.append((i, j, LineString([coords[i], coords[j]]), dij, "black"))

    # Sort by source, then target for consistent output
    rows.sort()
    return gpd.GeoDataFrame(rows, columns=["source", "target", "geometry", "weight", "color"])

# ------------------------------------------------------------------------------------------------
# --------------------------------------- 2d base graphs -----------------------------------------
# ------------------------------------------------------------------------------------------------

def add_source_target(gdf, coords, tol=1e-6):
    """
    Add 'source' and 'target' columns to an edge GeoDataFrame
    by matching LINESTRING endpoints to vertex coordinates.
    gdf: GeoDataFrame with 'geometry' column (LINESTRINGs)
    coords: Nx2 numpy array of vertex coordinates
    tol: tolerance for coordinate comparison
    """
    if isinstance(coords, gpd.GeoDataFrame):
        coords = np.array(coords['feature'].tolist(), dtype=np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
    sources, targets = [], []

    for line in gdf.geometry:
        x1, y1, x2, y2 = *line.coords[0], *line.coords[-1]
        # find nearest vertex for each endpoint
        start_idx = np.argmin(np.linalg.norm(coords - np.array([x1, y1]), axis=1))
        end_idx = np.argmin(np.linalg.norm(coords - np.array([x2, y2]), axis=1))

        # sanity check (optional)
        if np.linalg.norm(coords[start_idx] - [x1, y1]) > tol:
            print(f"Warning: start point mismatch for {line}")
        if np.linalg.norm(coords[end_idx] - [x2, y2]) > tol:
            print(f"Warning: end point mismatch for {line}")

        sources.append(start_idx)
        targets.append(end_idx)

    gdf = gdf.copy()
    gdf["source"] = sources
    gdf["target"] = targets
    gdf["color"] = "black"
    return gdf

def create_knn_graph_2d(coords, k=4):
    """
    Create k-nearest neighbor graph from 2D coordinates.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    return create_knn_graph(coords, k=k)

def create_minimum_spanning_tree_2d(coords):
    """
    Create minimum spanning tree from 2D coordinates.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    mst_vertices, mst_edges = city2graph.euclidean_minimum_spanning_tree(coords, distance_metric="euclidean")
    return add_source_target(mst_edges, coords)

def create_relative_neighborhood_graph_2d(coords):
    """
    Create relative neighborhood graph from 2D coordinates.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    rng_vertices, rng_edges = city2graph.relative_neighborhood_graph(coords, distance_metric="euclidean")
    return add_source_target(rng_edges, coords)

def create_gabriel_graph_2d(coords):
    """
    Create gabriel graph from 2D coordinates.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    gg_vertices, gg_edges = city2graph.gabriel_graph(coords, distance_metric="euclidean")
    return add_source_target(gg_edges, coords)

def create_delaunay_graph_2d(coords):
    """
    Create delaunay graph from 2D coordinates.
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    dg_vertices, dg_edges = city2graph.delaunay_graph(coords, distance_metric="euclidean")
    return add_source_target(dg_edges, coords)