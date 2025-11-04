import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def extract_features(gdf):
    return np.array(gdf['feature'].tolist())

def create_coords(coords):
    """
    Create a GeoDataFrame from a list of coordinates.

    Args:
        coords (list): List of coordinate lists, e.g., [[1, 2], [2, 2]].

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with 'id', 'feature', 'color', and 'geometry' columns.
    """
    coords_array = np.array(coords, dtype=np.float32)

    if coords_array.ndim != 2 or coords_array.shape[1] < 2:
        raise ValueError("coords must be a 2D array with at least 2 columns")

    n = len(coords_array)

    gdf = gpd.GeoDataFrame({
        'id': [f"$x_{{{i + 1}}}$" for i in range(n)],
        'feature': [list(row) for row in coords_array],
        'color': '#FFE599',
        'geometry': [Point(row[0], row[1]) for row in coords_array]
    })

    return gdf

def create_random_coords(n=100, scale=100.0, seed=None, dims=2, distribution='uniform'):
    """
    Create n random coordinates in range [0, scale) as float32 and return as GeoDataFrame.

    Args:
        n (int): Number of points.
        scale (float): Maximum value for coordinates.
        seed (int, optional): Random seed.
        dims (int): Number of dimensions (must be at least 2).
        distribution (str): Distribution type - 'uniform', 'normal', 'exponential', 'gaussian_mixture' or 'clusters'.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with 'id', 'feature', 'color', and 'geometry' columns.
    """
    if dims < 2:
        raise ValueError("dims must be at least 2")
    if seed is not None:
        np.random.seed(seed)

    if distribution == 'uniform':
        coords = np.random.rand(n, dims) * scale
    elif distribution == 'normal':
        coords = np.random.randn(n, dims) * (scale / 4) + (scale / 2)
        coords = np.clip(coords, 0, scale)
    elif distribution == 'exponential':
        coords = np.random.exponential(scale / 4, (n, dims))
        coords = np.clip(coords, 0, scale)
    elif distribution == 'clusters':
        # Create k clusters with points normally distributed around cluster centers
        k = max(2, n // 20)  # Number of clusters
        cluster_centers = np.random.rand(k, dims) * scale
        cluster_ids = np.random.randint(0, k, n)
        coords = np.zeros((n, dims))
        for i in range(n):
            coords[i] = cluster_centers[cluster_ids[i]] + np.random.randn(dims) * (scale / 20)
        coords = np.clip(coords, 0, scale)
    elif distribution == 'gaussian_mixture':
        # Mixture of Gaussians with different variances
        k = max(2, n // 15)
        cluster_centers = np.random.rand(k, dims) * scale
        cluster_stds = np.random.rand(k) * (scale / 10) + (scale / 30)
        cluster_ids = np.random.choice(k, n)
        coords = np.zeros((n, dims))
        for i in range(n):
            coords[i] = cluster_centers[cluster_ids[i]] + np.random.randn(dims) * cluster_stds[cluster_ids[i]]
        coords = np.clip(coords, 0, scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Scale 2nd dimension to half
    coords[:, 1] *= 0.8

    coords = coords.astype(np.float32)

    gdf = gpd.GeoDataFrame({
        'id': [f"$x_{{{i + 1}}}$" for i in range(n)],
        'feature': [list(row) for row in coords],
        'color': '#FFE599',
        'geometry': [Point(row[0], row[1]) for row in coords]
    })

    return gdf