import deglib
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

def _get_coords(g, index):
    return np.array(g.get_feature_vector(index), dtype=np.float32)

def get_deg_coords(graph_file):
    g = deglib.graph.load_readonly_graph(graph_file)
    return np.array([_get_coords(g,i) for i in range(g.size())], dtype=np.float32)

def get_deg_edges(graph_file):
    g = deglib.graph.load_readonly_graph(graph_file)
    edges = []
    for i in range(g.size()):
        src_coords = _get_coords(g,i)
        for j in g.get_neighbor_indices(i):
            if i < j:
                dst_coords = _get_coords(g,j)
                line = LineString([src_coords, dst_coords])
                dist = line.length
                edges.append((i, j, line, dist))
    return gpd.GeoDataFrame(edges, columns=["source", "target", "geometry", "weight"])