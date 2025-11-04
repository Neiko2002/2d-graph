import deglib
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point


def _get_feature_at_index(g, index):
    return np.array(g.get_feature_vector(index), dtype=np.float32)

def read_deg_vertices(graph_file):
    return get_deg_vertices(deglib.graph.load_readonly_graph(graph_file))

def get_deg_vertices(g):
    features = []
    for i in range(g.size()):
        feat = _get_feature_at_index(g, i)
        features.append({
            'id': f"$x_{{{i + 1}}}$",
            'feature': feat,
            'color': '#FFE599',
            'geometry': Point(feat[0], feat[1])
        })
    return gpd.GeoDataFrame(features)

def read_deg_edges(graph_file):
    return get_deg_edges(deglib.graph.load_readonly_graph(graph_file))

def get_deg_edges(g):
    edges = []
    for i in range(g.size()):
        src_coords = _get_feature_at_index(g, i)
        for j in g.get_neighbor_indices(i):
            if i < j:
                dst_coords = _get_feature_at_index(g, j)
                line = LineString([src_coords[:2], dst_coords[:2]])
                dist = line.length
                edges.append((i, j, line, dist, 'black'))
    return gpd.GeoDataFrame(edges, columns=["source", "target", "geometry", "weight", "color"])