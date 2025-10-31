import city2graph
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

def _pairwise_dist(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

def _rng_edges(coords, dist):
    n = len(coords)
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
    return edges

def _to_adj(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj

def _add_edge(i, j, edges, adj):
    a, b = (i, j) if i < j else (j, i)
    if (a, b) not in edges:
        edges.add((a, b))
        adj[i].add(j)
        adj[j].add(i)

def create_mrng(coords):
    """
    Full monotonic enforcement:
    1) start from RNG
    2) for each ordered pair (u, v), enforce a path u->...->v where d(next,v) < d(curr,v)
       by adding repair edges when greedy descent gets stuck
    Returns GeoDataFrame with [source, target, geometry, weight]
    """
    if isinstance(coords, gpd.GeoDataFrame):
        coords = np.array(coords['feature'].tolist(), dtype=np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    dist = _pairwise_dist(coords)

    # 1) initialize with RNG
    edges = _rng_edges(coords, dist)
    adj = _to_adj(n, edges)

    # 2) enforce monotonicity for all targets v
    for v in range(n):
        order = np.argsort(dist[:, v])  # optional heuristic
        for u in order:
            if u == v:
                continue
            p = u
            seen = {p}
            while p != v:
                # try greedy descent along existing edges
                neigh = [q for q in adj[p] if dist[q, v] < dist[p, v]]
                if neigh:
                    # pick neighbor with minimal d(q, v)
                    q = neigh[int(np.argmin([dist[x, v] for x in neigh]))]
                    if q in seen:  # safety against cycles
                        break
                    p = q
                    seen.add(p)
                    continue

                # stuck: add a repair edge to some r closer to v
                closer = np.where(dist[:, v] < dist[p, v])[0]
                if closer.size == 0:
                    break  # cannot improve
                # pick r that is closest to p among points closer to v
                r = closer[int(np.argmin(dist[p, closer]))]
                _add_edge(p, r, edges, adj)
                p = r
                seen.add(p)

    # build GeoDataFrame
    rows = []
    for i, j in sorted(edges):
        dij = float(dist[i, j])
        rows.append((i, j, LineString([coords[i], coords[j]]), dij, "black"))
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
    knn_vertices, knn_edges = city2graph.knn_graph(coords, k=k, distance_metric="euclidean")
    return add_source_target(knn_edges, coords)

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