""" Corner plot utils """

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from detection.chesscorner import ChESSCorner

AXIS0_COLOR = "#17a2b8"  # teal
AXIS1_COLOR = "#ff8c42"  # orange


def plot_chess_corners(
    ax,
    corner: list[ChESSCorner],
    show_orientation: bool = True,
    arrow_length: float = 20.0,
    show_labels: bool = True,
):
    """Plot chess corners on image with both grid-axis directions.

    Args:
        ax: matplotlib Axes to draw on.
        corner: list of `ChESSCorner` loaded from a detector output JSON.
        show_orientation: if True, overlay the two grid-axis arrows per corner.
        arrow_length: arrow length in image pixels.
        show_labels: toggle the axes title.

    Convention for axes: `axes[0].angle ∈ [0, π)`, and
    `axes[1].angle ∈ (axes[0], axes[0] + π)` with the sector in between
    being a **dark** sector of the corner (bright-to-dark polarity
    encoded in the ordering).
    """
    xs = [c.x for c in corner]
    ys = [c.y for c in corner]

    responses = np.array([c.response for c in corner], dtype=float)
    if responses.size:
        r_min = float(responses.min())
        r_max = float(responses.max())
        denom = r_max - r_min
        if denom <= 1e-12:
            t = np.zeros_like(responses)
        else:
            t = (responses - r_min) / denom
        point_colors = plt.get_cmap("viridis")(t)
    else:
        point_colors = "#1d3557"

    def _quiver(angles: np.ndarray, color: str, label: str | None) -> None:
        if angles.size == 0:
            return
        us = np.cos(angles) * arrow_length
        vs = np.sin(angles) * arrow_length
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.002,
            label=label,
        )

    if show_orientation and corner:
        axis0 = np.array(
            [c.axes[0].angle if len(c.axes) >= 1 else c.orientation for c in corner],
            dtype=float,
        )
        axis1 = np.array(
            [c.axes[1].angle if len(c.axes) >= 2 else np.nan for c in corner],
            dtype=float,
        )
        _quiver(axis0, AXIS0_COLOR, label="axis 0 (line ∈ [0,π))")
        if np.any(np.isfinite(axis1)):
            _quiver(axis1, AXIS1_COLOR, label="axis 1 (dark-CCW)")

    ax.scatter(
        xs,
        ys,
        s=18,
        facecolors="none",
        edgecolors=point_colors,
        linewidths=0.7,
        label="chess-corners",
    )

    if show_labels:
        ax.set_title(f"ChESS corners: {len(corner)} detected (two-axis)")
    ax.axis("off")

    return ax

def plot_harris_corners(ax, harris_pts: np.ndarray, show_labels: bool = True) -> None:
    """ Plot Harris corners on given axes

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        harris_pts (np.ndarray): Harris corner points
    """
    ax.scatter(
        harris_pts[:, 0],
        harris_pts[:, 1],
        s=18,
        c="red",
        marker="x",
        linewidths=0.7,
        label="Harris (OpenCV)",
    )

def plot_chessboard_corners(ax, chessboard_pts: np.ndarray, show_labels: bool = True) -> None:
    """Plot OpenCV chessboard corners."""
    ax.scatter(
        chessboard_pts[:, 0],
        chessboard_pts[:, 1],
        s=32,
        facecolors="none",
        edgecolors="#2b9348",
        marker="s",
        linewidths=0.8,
        label="findChessboardCornersSB",
    )

def plot_overlay(
    img: np.ndarray,
    chess_pts: list[ChESSCorner],
    harris_pts: np.ndarray | None = None,
    chessboard_pts: np.ndarray | None = None,
    split: bool = False,
) -> plt.Figure:
    plots = [("ChESS corners", chess_pts, plot_chess_corners)]
    if harris_pts is not None:
        plots.append(("Harris corners", harris_pts, plot_harris_corners))
    if chessboard_pts is not None:
        plots.append(("findChessboardCornersSB", chessboard_pts, plot_chessboard_corners))

    if split:
        fig, axes = plt.subplots(
            1,
            len(plots),
            figsize=(6 * len(plots), 6),
            sharex=True,
            sharey=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, (title, pts, fn) in zip(axes, plots):
            ax.imshow(img, cmap="gray")
            fn(ax, pts)
            ax.set_title(f"{title} ({len(pts)} pts)")
            ax.set_axis_off()
            ax.legend(loc="lower right", framealpha=0.6)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, cmap="gray")
        plot_chess_corners(ax, chess_pts)
        if harris_pts is not None:
            plot_harris_corners(ax, harris_pts)
        if chessboard_pts is not None:
            plot_chessboard_corners(ax, chessboard_pts)
        ax.set_title("Detected corners")
        ax.set_axis_off()
        ax.legend(loc="lower right", framealpha=0.6)

    fig.tight_layout()
    return fig

def plot_offset_hist(offsets: np.ndarray, title: str, path: Path) -> None:
    if offsets.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(offsets, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Nearest GT distance (px)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_scatter(errors: np.ndarray, title: str, path: Path) -> None:
    if errors.size == 0:
        return
    x = np.arange(len(errors))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, errors, s=10, alpha=0.7, color="darkorange")
    ax.set_xlabel("Detection index")
    ax.set_ylabel("Nearest GT distance (px)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
