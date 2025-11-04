import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from graphs import GraphItem


def plot_graphs(graphs:List[GraphItem]=None, bg_color='white', text_color='black', annotation_size=16, title_size=24, marker_size=50, border_color='black', border_padding=8, edge_width=1, max_per_row=6, save_file=None):
    """
    Plots vertices and/or edges from GeoDataFrames.

    Args:
        graph (gpd.GeoDataFrame, optional): GeoDataFrame with 'geometry' (LineString) column.
        bg_color (str): Background color for the plot.
        text_color (str): Color for the annotation and title text.
        annotation_size (int): Font size for the annotation text.
        title_size (int): Font size for the title text.
        marker_size (float): Size of the scatter plot markers.
        border_color (str): Color of the subplot border.
        border_padding (int): Padding around the subplot border.
        edge_width (float): Width of the edge lines.
        max_per_row (int): Maximum number of subplots per row.
        save_file (str|Path|None): output path, e.g., 'plot.svg'

    """
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")

    # Calculate the total bounds across all graphs to unify axes limits
    global_min_x, global_min_y, global_max_x, global_max_y = np.inf, np.inf, -np.inf, -np.inf
    for g in graphs:
        if g.vertices is not None and not g.vertices.empty:
            bounds = g.vertices.total_bounds  # minx, miny, maxx, maxy
            global_min_x = min(global_min_x, bounds[0])
            global_min_y = min(global_min_y, bounds[1])
            global_max_x = max(global_max_x, bounds[2])
            global_max_y = max(global_max_y, bounds[3])

    # Determine the center and the maximum range to make the plot square
    center_x = (global_min_x + global_max_x) / 2
    center_y = (global_min_y + global_max_y) / 2
    range_x = global_max_x - global_min_x
    range_y = global_max_y - global_min_y
    max_range = max(range_x, range_y)

    # Add padding to the max_range
    padding = max_range * (border_padding / 100)
    padded_range = max_range + 2 * padding

    xlim = (center_x - padded_range / 2, center_x + padded_range / 2)
    ylim = (center_y - padded_range / 2, center_y + padded_range / 2)

    size = len(graphs)
    rows = max(1, math.ceil(size / max_per_row))
    cols = math.ceil(size / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 8))
    axes = np.atleast_1d(axes).ravel()

    # Marker radius in points (s is area in points^2 for scatter circles)
    node_radius_pts = math.sqrt(marker_size / math.pi)
    arrow_end_shrink_pts = node_radius_pts + 1.0  # small padding

    for ax, g in zip(axes, graphs):
        title, is_directed, vertices, edges = g.title, g.is_directed, g.vertices, g.edges
        ax.set_facecolor(bg_color)

        # Draw edges grouped by color
        if edges is not None:
            segments_by_style = {}
            for row in edges.itertuples(index=False):
                coords = np.asarray(row.geometry.coords)
                color = row.color if hasattr(row, 'color') else 'black'
                linestyle = row.linestyle if hasattr(row, 'linestyle') else 'solid'
                style_key = (color, linestyle)
                segments_by_style.setdefault(style_key, []).append(coords)

            for (color, linestyle), segs in segments_by_style.items():
                lc = LineCollection(segs, colors=color, linestyles=linestyle, linewidths=edge_width, zorder=1)
                ax.add_collection(lc)

            # Arrowheads for directed graphs
            if is_directed:
                for row in edges.itertuples(index=False):
                    coords = np.asarray(row.geometry.coords)
                    color = row.color
                    x1, y1 = coords[-2]
                    x2, y2 = coords[-1]
                    ax.annotate(
                        '',
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color=color,
                            lw=edge_width,
                            shrinkA=0,
                            shrinkB=arrow_end_shrink_pts,
                            mutation_scale=max(18, marker_size * 0.05)
                        ),
                        zorder=1.5
                    )

            ax.autoscale()

        # Plot vertices: yellow fill with colored border
        x = vertices.geometry.x.values
        y = vertices.geometry.y.values
        vcolors = vertices['color'] if 'color' in vertices.columns else 'black'
        ax.scatter(
            x, y,
            s=marker_size,
            marker='o',
            facecolors=vcolors,
            edgecolors=border_color,
            linewidths=edge_width,
            zorder=2
        )

        # Add vertex labels from 'id' column
        if annotation_size > 0:
            for _, row in vertices.iterrows():
                ax.annotate(
                    row['id'],
                    (row.geometry.x, row.geometry.y),
                    textcoords="offset points",
                    xytext=(-annotation_size/2, annotation_size/1.5),
                    ha='center',
                    color=text_color,
                    fontsize=annotation_size,
                    fontname='Calibri'
                )

        # Set title
        ax.set_title(title, fontsize=title_size, fontname='Calibri', y=1.08)

        # Add annotations below the graph
        if g.annotations:
            annotation_text = "\n".join(g.annotations)
            ax.text(0.95, 0.0, annotation_text,
                    transform=ax.transAxes,
                    fontsize=title_size * 0.8,
                    fontname='Calibri',
                    ha='right', va='bottom',
                    linespacing=1.1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.axis('off')

    # hide unused subplot if fewer than 8
    for ax in axes[size:]:
        ax.axis('off')

    # Apply tight_layout first to arrange subplots
    plt.tight_layout(h_pad=border_padding, w_pad=border_padding)

    # Now, shrink each axis and add the border
    for i, ax in enumerate(axes):
        if i >= size:
            continue

        # Add the rounded border
        for spine in ax.spines.values():
            spine.set_visible(False)

        border = FancyBboxPatch(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.05,rounding_size=0.05",
            fill=False,
            edgecolor=border_color,
            linewidth=edge_width * 1.5,
            joinstyle='round',
            zorder=3,
            clip_on=False
        )
        ax.add_artist(border)

    # Save if requested
    if save_file:
        save_path = Path(save_file)
        fmt = (save_path.suffix.lstrip('.').lower() or 'svg') if save_path.suffix else 'svg'
        fig.savefig(save_path, format=fmt, bbox_inches='tight')

    plt.show()