import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def extract_features(gdf):
    return np.array(gdf['feature'].tolist())

def create_random_coords(n=100, scale=100.0, seed=None, dims=2):
    """
    Create n random coordinates in range [0, scale) as float32 and return as GeoDataFrame.

    Args:
        n (int): Number of points.
        scale (float): Maximum value for coordinates.
        seed (int, optional): Random seed.
        dims (int): Number of dimensions (must be at least 2).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with 'id', 'feature', 'color', and 'geometry' columns.
    """
    if dims < 2:
        raise ValueError("dims must be at least 2")
    if seed is not None:
        np.random.seed(seed)
    coords = (np.random.rand(n, dims) * scale).astype(np.float32)

    gdf = gpd.GeoDataFrame({
        'id': range(n),
        'feature': [list(row) for row in coords],
        'color': "black",
        'geometry': [Point(row[0], row[1]) for row in coords]
    })

    return gdf