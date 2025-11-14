import math
from typing import List, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from graphs import GraphItem

import matplotlib as mpl
import matplotlib.font_manager as fm

# detect LM fonts in system locations
lm_paths = [
    p for p in fm.findSystemFonts()
    if "lmroman" in p.lower() or "lm" in p.lower()
]

# register fonts for this Python process
for p in lm_paths:
    fm.fontManager.addfont(p)

# configure Matplotlib to use Latin Modern fonts
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams["font.serif"]  = ["Latin Modern Roman"]

# math fonts: use custom fontset with LM
mpl.rcParams["mathtext.fontset"] = "cm"              # Computer Modern math (pdfLaTeX math)

# keep text as text in SVG
mpl.rcParams["svg.fonttype"] = "none"

# Define the acceptable title positions
TitlePosition = Literal[
    'top-left', 'top-center', 'top-right',
    'bottom-left', 'bottom-center', 'bottom-right'
]


def plot_graphs(graphs: List[GraphItem] = None, bg_color='white', text_color='black', vertex_label_size=16, annotation_size=16,
                title_size=24, title_position: TitlePosition = 'top-center',  # New parameter
                marker_size=50, border_color='black', border_padding_x=8, border_padding_y=8, edge_width=1, max_per_row=6, total_fig_width=24, save_file=None):
    """
    Plots vertices and/or edges from GeoDataFrames.

    Args:
        graphs (List[GraphItem], optional): List of GraphItem objects to plot.
        bg_color (str): Background color for the plot.
        text_color (str): Color for the annotation and title text.
        vertex_label_size (int): ont size for the vertex labels.
        annotation_size (int): Font size for the annotation text.
        title_size (int): Font size for the title text.
        title_position (TitlePosition): Position of the title in the subplot.
        marker_size (float): Size of the scatter plot markers.
        border_color (str): Color of the subplot border.
        border_padding_x (int): Horizontal padding around the subplot border as a percentage of range.
        border_padding_y (int): Vertical padding around the subplot border as a percentage of range.
        edge_width (float): Width of the edge lines.
        max_per_row (int): Maximum number of subplots per row.
        total_fig_width (float): The total width of the output figure in inches.
        save_file (str|Path|None): output path, e.g., 'plot.svg'

    """
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")

    # Calculate the total bounds across all graphs to unify axes limits
    global_min_x, global_min_y, global_max_x, global_max_y = np.inf, np.inf, -np.inf, -np.inf
    for g in graphs:
        if g.vertices is not None and not g.vertices.empty:
            bounds = g.vertices.total_bounds
            global_min_x = min(global_min_x, bounds[0])
            global_min_y = min(global_min_y, bounds[1])
            global_max_x = max(global_max_x, bounds[2])
            global_max_y = max(global_max_y, bounds[3])

    range_x = global_max_x - global_min_x
    range_y = global_max_y - global_min_y
    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1

    # Add padding to the max_range
    padding_x = range_x * (border_padding_x / 100)
    padding_y = range_y * (border_padding_y / 100)

    xlim = (global_min_x - padding_x, global_max_x + padding_x)
    ylim = (global_min_y - padding_y, global_max_y + padding_y)

    padded_range_x = xlim[1] - xlim[0]
    padded_range_y = ylim[1] - ylim[0]

    size = len(graphs)
    cols = min(size, max_per_row)
    rows = math.ceil(size / cols)

    fig_height = total_fig_width * (rows * padded_range_y) / (cols * padded_range_x)
    fig, axes = plt.subplots(rows, cols, figsize=(total_fig_width, fig_height))
    axes = np.atleast_1d(axes).ravel()

    # Marker radius in points (s is area in points^2 for scatter circles)
    node_radius_pts = math.sqrt(marker_size / math.pi)
    arrow_end_shrink_pts = node_radius_pts + 1.0

    for ax, g in zip(axes, graphs):
        title, is_directed, vertices, edges = g.title, g.is_directed, g.vertices, g.edges
        ax.set_facecolor(bg_color)

        # Draw edges grouped by color
        if edges is not None:
            segments_by_style = {}
            for row in edges.itertuples(index=False):
                coord = np.asarray(row.geometry.coords)
                color = getattr(row, 'color', 'black')
                linestyle = getattr(row, 'linestyle', 'solid')

                # Convert dash pattern tuple to LineCollection format: (offset, (on, off, ...))
                if isinstance(linestyle, tuple):
                    linestyle = (0, linestyle)

                style_key = (color, linestyle)
                segments_by_style.setdefault(style_key, []).append(coord)

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
        if vertex_label_size > 0:
            for _, row in vertices.iterrows():
                ax.annotate(
                    row['id'],
                    (row.geometry.x, row.geometry.y),
                    textcoords="offset points",
                    xytext=(-vertex_label_size / 2, vertex_label_size / 1.5),
                    ha='center',
                    color=text_color,
                    fontsize=vertex_label_size
                )

        # Set title
        if title:
            pos_map = {
                'top-left': ('left', 1.08),
                'top-center': ('center', 1.08),
                'top-right': ('right', 1.08),
                'bottom-left': ('left', 0.05),
                'bottom-center': ('center', 0.05),
                'bottom-right': ('right', 0.05)
            }

            if title_position not in pos_map:
                raise ValueError("title_position must be one of: "
                                 "'top-left', 'top-center', 'top-right', "
                                 "'bottom-left', 'bottom-center', 'bottom-right'")

            title_loc, title_y = pos_map[title_position]
            ax.set_title(
                title,
                fontsize=title_size,
                color=text_color,
                loc=title_loc, # Horizontal position based on 'ha' from original map
                y=title_y      # Vertical position based on 'ty' from original map
            )

        # Add annotations below the graph
        if g.annotations and annotation_size > 0:
            annotation_text = "\n".join(g.annotations)
            ax.text(0.95, 0.0, annotation_text,
                    transform=ax.transAxes,
                    fontsize=annotation_size,
                    ha='right', va='bottom',
                    linespacing=1.1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('off')

    # hide unused subplot if fewer than 8
    for ax in axes[size:]:
        ax.axis('off')

    # Apply tight_layout first to arrange subplots
    plt.tight_layout(h_pad=8, w_pad=8 if cols > 2 else 12)

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
        fmt = (save_path.suffix.lstrip('.').lower() or 'pdf') if save_path.suffix else 'pdf'
        if fmt == 'svg':
            # Warn the user if they select SVG, as it might lose subscript fidelity
            # in certain browsers unless fonts are installed.
            print(
                "Warning: Saving as SVG may result in non-selectable math characters or display errors if viewer lacks Latin Modern fonts.")
        fig.savefig(save_path, format=fmt, bbox_inches='tight')

    plt.show()