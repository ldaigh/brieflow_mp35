"""Shared smart-label placement for Plotly scatter-style figures.

Uses adjustText on an offscreen matplotlib axis to compute non-overlapping label
positions, which are then transferred into a Plotly figure as annotations with
leader lines. Falls back to a small fixed offset if adjustText is unavailable.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except Exception:
    HAS_ADJUST_TEXT = False


def smart_label_positions(label_x, label_y, all_x, all_y, labels, xlim, ylim, figsize=(7, 5)):
    """Compute non-overlapping label positions (data coords) via adjustText on an
    offscreen matplotlib axis that mirrors the Plotly axis ranges, avoiding every
    point. Returns a list of (text_x, text_y) aligned with the input order.

    Args:
        label_x, label_y: Coordinates of the points being labeled.
        all_x, all_y: Coordinates of every point (for collision avoidance).
        labels: Label strings, aligned with label_x / label_y.
        xlim, ylim: (min, max) axis ranges, shared with the Plotly figure.
        figsize: Offscreen figure size (only affects placement aspect ratio).
    """
    if not HAS_ADJUST_TEXT:
        dy = 0.02 * (ylim[1] - ylim[0])
        return [(x, y + dy) for x, y in zip(label_x, label_y)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    texts = [ax.text(x, y, str(lab), fontsize=8) for x, y, lab in zip(label_x, label_y, labels)]
    fig.canvas.draw()  # renderer needed so adjustText can measure text bboxes
    adjust_text(
        texts,
        x=list(all_x),
        y=list(all_y),
        ax=ax,
        force_text=(0.4, 0.8),
        force_static=(0.2, 0.4),
        expand=(1.3, 1.5),
        only_move={"text": "xy", "static": "xy", "pull": "xy", "explode": "xy"},
        time_lim=3,
    )
    positions = [t.get_position() for t in texts]
    plt.close(fig)
    return positions


def add_point_labels(fig, point_x, point_y, labels, all_x, all_y, xlim, ylim):
    """Add non-overlapping text labels for the given points to a Plotly figure as
    annotations, each with a leader line pointing back to its point."""
    positions = smart_label_positions(point_x, point_y, all_x, all_y, labels, xlim, ylim)
    for (lx, ly), px, py, text in zip(positions, point_x, point_y, labels):
        fig.add_annotation(
            x=px,
            y=py,
            ax=lx,
            ay=ly,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=str(text),
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.6,
            arrowcolor="#888888",
            font=dict(size=10, color="#222222"),
            bgcolor="rgba(255,255,255,0.7)",
        )
