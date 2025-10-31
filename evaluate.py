import numpy as np

def calc_recall(indices_exact, indices_approx):
    return np.mean([
        len(set(indices_approx[i]).intersection(indices_exact[i])) / len(indices_exact[i])
        for i in range(len(indices_exact))
    ])

def exact_knn(coords, k=5):
    """
    Compute exact k-nearest neighbors using brute force distance matrix.
    coords: Nx2 numpy array
    returns: (indices, distances)
    """
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)  # ignore self-distance
    indices = np.argsort(distances, axis=1)[:, :k]
    dist_sorted = np.take_along_axis(distances, indices, axis=1)
    return indices, dist_sorted

def print_edge_overlap(name, gdf_a, gdf_b):
    """
    Compare undirected edge sets of two GeoDataFrames and print formatted results.
    Prints:
      <name>: overlap_a/total_a (ratio_a_in_b)   overlap_b/total_b (ratio_b_in_a)  avg=<avg_dist_b>
    Returns tuple of all computed values.
    """
    a = gdf_a.copy()
    b = gdf_b.copy()

    # enforce undirected order
    a[['u', 'v']] = np.sort(a[['source', 'target']].values, axis=1)
    b[['u', 'v']] = np.sort(b[['source', 'target']].values, axis=1)

    merged = a.merge(b, on=['u', 'v'])
    overlap = len(merged)
    total_a = len(a)
    total_b = len(b)

    ratio_a_in_b = overlap / total_a if total_a else 0
    ratio_b_in_a = overlap / total_b if total_b else 0
    avg_dist_b = b["weight"].mean() if total_b else 0.0

    # print formatted line
    print("{:<8} {:>3}/{:>3} ({:7.2%})   {:>3}/{:>3} ({:7.2%})  avg={:7.3f}".format(
        name, overlap, total_a, ratio_a_in_b, overlap, total_b, ratio_b_in_a, avg_dist_b))