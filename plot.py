import math
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.collections import LineCollection
from pathlib import Path

def plot_graphs(vertices=None, graphs=None, bg_color='white', text_color='black', marker_size=50, edge_color='black', edge_width=1, show_labels=True, save_file=None):
    """
    Plots vertices and/or edges from GeoDataFrames.

    Args:
        vertices (gpd.GeoDataFrame, optional): GeoDataFrame with 'id', 'color', and 'geometry' (Point) columns.
        graph (gpd.GeoDataFrame, optional): GeoDataFrame with 'geometry' (LineString) column.
        bg_color (str): Background color for the plot.
        text_color (str): Color for the annotation text.
        marker_size (float): Size of the scatter plot markers.
        edge_color (str): Color for the edges.
        edge_width (float): Width of the edge lines.
        show_labels (bool): Whether to show vertex ID labels.
        save_file (str|Path|None): output path, e.g., 'plot.svg'

    """
    if vertices is None or not isinstance(vertices, gpd.GeoDataFrame):
        raise ValueError("vertices must be a GeoDataFrame.")
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")

    size = len(graphs)
    rows = 1 if size < 4 else 2
    cols = math.ceil(size / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = np.atleast_1d(axes).ravel()

    # Precompute vertex arrays
    x = vertices.geometry.x.values
    y = vertices.geometry.y.values
    vcolors = vertices['color'] if 'color' in vertices.columns else 'black'

    for ax, g in zip(axes, graphs):
        title, edges = g.value
        ax.set_facecolor(bg_color)

        # Draw edges grouped by color
        if edges is not None:
            if not isinstance(edges, gpd.GeoDataFrame):
                raise ValueError("edges must be a GeoDataFrame.")
            # Build segments grouped by effective color
            segments_by_color = {}

            for row in edges.itertuples(index=False):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                coords = np.asarray(geom.coords)
                if coords.shape[0] < 2:
                    continue
                c = getattr(row, "color", None)
                if c is None or (isinstance(c, float) and np.isnan(c)):
                    c = edge_color
                segments_by_color.setdefault(c, []).append(coords)

            for c, segs in segments_by_color.items():
                lc = LineCollection(segs, colors=c, linewidths=edge_width, zorder=1)
                ax.add_collection(lc)

            ax.autoscale()

        # Plot vertices
        ax.scatter(x, y, c=vcolors, marker='o', s=marker_size, zorder=2)

        # Add labels from 'id'
        if show_labels:
            for _, row in vertices.iterrows():
                ax.annotate(row['id'], (row.geometry.x, row.geometry.y),
                            textcoords="offset points", xytext=(0, 10), ha='center',
                            color=text_color)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    # hide unused subplot if fewer than 8
    for ax in axes[size:]:
        ax.axis('off')

    plt.tight_layout()

    # Save if requested
    if save_file:
        save_path = Path(save_file)
        fmt = (save_path.suffix.lstrip('.').lower() or 'svg') if save_path.suffix else 'svg'
        fig.savefig(save_path, format=fmt, bbox_inches='tight')

    plt.show()