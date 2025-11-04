import numpy as np

def color_overlapping_edges(
    edges_ref,
    edges_to_color,
    undirected_ref: bool = True,
    undirected_to_color: bool = True,
    source_col: str = "source",
    target_col: str = "target",
    color: str = "red",
    inplace: bool = False
):
    """
    Color edges in edges_to_color that overlap with gdf_a.

    Args:
        edges_ref: Reference GeoDataFrame with edges
        edges_to_color: GeoDataFrame with edges to potentially color
        undirected_ref: If True, treats gdf_a as undirected
        undirected_to_color: If True, treats edges_to_color as undirected
        source_col: Name of the source column
        target_col: Name of the target column
        color: Color to apply to overlapping edges
        inplace: If True, modifies edges_to_color in place

    Returns:
        Modified GeoDataFrame (or None if inplace=True)
    """
    result = edges_to_color if inplace else edges_to_color.copy()

    # Calculate overlap
    overlap_df = calc_edge_overlap(
        edges_ref,
        result,
        undirected_a=undirected_ref,
        undirected_b=undirected_to_color,
        source_col=source_col,
        target_col=target_col
    )

    # Create a temporary column in result for matching
    if undirected_to_color:
        result['_match_key'] = result.apply(
            lambda row: tuple(sorted([row[source_col], row[target_col]])),
            axis=1
        )
        overlap_df['_match_key'] = overlap_df.apply(
            lambda row: tuple(sorted([row['u'], row['v']])),
            axis=1
        )
    else:
        result['_match_key'] = result.apply(
            lambda row: (row[source_col], row[target_col]),
            axis=1
        )
        overlap_df['_match_key'] = overlap_df.apply(
            lambda row: (row['u'], row['v']),
            axis=1
        )

    # Update colors for matching edges
    result.loc[result['_match_key'].isin(overlap_df['_match_key']), 'color'] = color

    # Clean up temporary column
    result.drop(columns=['_match_key'], inplace=True)

    if not inplace:
        return result

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

def calc_edge_overlap(
    gdf_a,
    gdf_b,
    undirected_a: bool = True,
    undirected_b: bool = True,
    source_col: str = "source",
    target_col: str = "target"
):
    """
    Compare edge sets of two GeoDataFrames and return overlap statistics.

    Args:
        gdf_a: First GeoDataFrame with source and target columns
        gdf_b: Second GeoDataFrame with source and target columns
        undirected_a: If True, treats gdf_a as undirected (default: True)
        undirected_b: If True, treats gdf_b as undirected (default: True)
        source_col: Name of the source column (default: "source")
        target_col: Name of the target column (default: "target")

    Returns:
        DataFrame with overlapping edges
    """
    a = gdf_a.copy()
    b = gdf_b.copy()

    if undirected_a:
        a[['u', 'v']] = np.sort(a[[source_col, target_col]].values, axis=1)
    else:
        a[['u', 'v']] = a[[source_col, target_col]].values

    if undirected_b:
        b[['u', 'v']] = np.sort(b[[source_col, target_col]].values, axis=1)
    else:
        b[['u', 'v']] = b[[source_col, target_col]].values

    return a.merge(b, on=['u', 'v'])

def calc_edge_overlap_ratio(gdf_a, gdf_b,
    undirected_a: bool = True,
    undirected_b: bool = True,
    source_col: str = "source",
    target_col: str = "target"):
    """
    Compare undirected edge sets of two GeoDataFrames and return overlap ratio.
    Returns: overlap / total_a

    Args:
        gdf_a: First GeoDataFrame with source and target columns
        gdf_b: Second GeoDataFrame with source and target columns
        undirected_a: If True, treats gdf_a as undirected (default: True)
        undirected_b: If True, treats gdf_b as undirected (default: True)
        source_col: Name of the source column (default: "source")
        target_col: Name of the target column (default: "target")
    """
    overlap = len(calc_edge_overlap(gdf_a, gdf_b, undirected_a=undirected_a, undirected_b=undirected_b, source_col=source_col, target_col=target_col))
    total_a = len(gdf_a)
    return overlap / total_a if total_a else 0

def print_edge_overlap(name, gdf_a, gdf_b,
    undirected_a: bool = True,
    undirected_b: bool = True,
    source_col: str = "source",
    target_col: str = "target"):
    """
    Compare undirected edge sets of two GeoDataFrames and print formatted results.

    Args:
        name: Name label for the output
        gdf_a: First GeoDataFrame with source and target columns
        gdf_b: Second GeoDataFrame with source and target columns
        undirected_a: If True, treats gdf_a as undirected (default: True)
        undirected_b: If True, treats gdf_b as undirected (default: True)
        source_col: Name of the source column (default: "source")
        target_col: Name of the target column (default: "target")

    Prints:
      <name>: overlap_a/total_a (ratio_a_in_b)   overlap_b/total_b (ratio_b_in_a)  avg=<avg_dist_b>
    Returns tuple of all computed values.
    """
    overlap = len(calc_edge_overlap(gdf_a, gdf_b, undirected_a=undirected_a, undirected_b=undirected_b, source_col=source_col, target_col=target_col))
    total_a = len(gdf_a)
    total_b = len(gdf_b)

    ratio_a_in_b = overlap / total_a if total_a else 0
    ratio_b_in_a = overlap / total_b if total_b else 0
    avg_dist_b = gdf_b["weight"].mean() if total_b else 0.0

    # print formatted line
    print("{:<8} {:>3}/{:>3} ({:7.2%})   {:>3}/{:>3} ({:7.2%})  avg={:7.3f}".format(
        name, overlap, total_a, ratio_a_in_b, overlap, total_b, ratio_b_in_a, avg_dist_b))