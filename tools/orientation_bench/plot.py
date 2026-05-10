"""Plotting helpers (matplotlib optional).

If `matplotlib` is not importable, every plot function becomes a no-op
that logs a warning. Callers should treat plot generation as
best-effort metadata, not as a failure mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    _MPL = True
except Exception:  # pragma: no cover - exercised when matplotlib missing
    plt = None  # type: ignore[assignment]
    _MPL = False


__all__ = [
    "plot_error_cdf",
    "plot_error_vs_param",
    "plot_sigma_calibration",
    "plot_residual_heatmap",
    "plot_subpixel_aliasing",
    "have_matplotlib",
]


def have_matplotlib() -> bool:
    return _MPL


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _warn_skip(name: str) -> None:
    logger.warning("matplotlib unavailable; skipping plot %s", name)


def plot_error_cdf(per_method_deltas: dict[str, np.ndarray], out_path: str | Path) -> None:
    if not _MPL:
        _warn_skip("error_cdf")
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, arr in per_method_deltas.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        x = np.sort(np.abs(arr) * 180.0 / np.pi)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, label=f"{name} (n={x.size})")
    ax.set_xlabel("|delta theta| (deg)")
    ax.set_ylabel("Fraction <= x")
    ax.set_title("Orientation error CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    out = Path(out_path)
    _ensure_dir(out)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_vs_param(
    rows: list[dict[str, Any]], sweep_name: str, axis: str, out_path: str | Path
) -> None:
    """Plot bias and RMSE versus the swept parameter value."""
    if not _MPL:
        _warn_skip("error_vs_param")
        return
    if not rows:
        return
    xs = [r["param_value"] for r in rows]
    bias = [r.get(f"bias_{axis}_deg") for r in rows]
    rmse = [r.get(f"rmse_{axis}_deg") for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    try:
        xs_num = [float(x) for x in xs]
    except (TypeError, ValueError):
        xs_num = list(range(len(xs)))
        ax.set_xticks(xs_num)
        ax.set_xticklabels([str(x) for x in xs])
    ax.plot(xs_num, bias, label=f"bias {axis}", marker="o")
    ax.plot(xs_num, rmse, label=f"rmse {axis}", marker="s")
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(sweep_name)
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(f"{sweep_name}: bias and RMSE ({axis})")
    out = Path(out_path)
    _ensure_dir(out)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sigma_calibration(
    deltas: np.ndarray, sigmas: np.ndarray, out_path: str | Path
) -> None:
    if not _MPL:
        _warn_skip("sigma_calibration")
        return
    deltas = np.asarray(deltas, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    mask = np.isfinite(deltas) & np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(mask):
        return
    deltas = deltas[mask]
    sigmas = sigmas[mask]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(deltas / sigmas, bins=60, color="#457b9d", alpha=0.85)
    axes[0].axvline(-2.0, color="black", linestyle=":")
    axes[0].axvline(2.0, color="black", linestyle=":")
    axes[0].set_title("z = delta / sigma")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(sigmas, np.abs(deltas), s=6, alpha=0.4, color="#1d3557")
    upper = float(max(np.max(sigmas), np.max(np.abs(deltas))))
    axes[1].plot([0.0, upper], [0.0, upper], color="red", alpha=0.7, label="y=x")
    axes[1].set_xlabel("reported sigma (rad)")
    axes[1].set_ylabel("|delta| (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    out = Path(out_path)
    _ensure_dir(out)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_residual_heatmap(
    grid_results: np.ndarray, out_path: str | Path
) -> None:
    if not _MPL:
        _warn_skip("residual_heatmap")
        return
    grid = np.asarray(grid_results, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(grid, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_title("Residual heatmap")
    out = Path(out_path)
    _ensure_dir(out)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_subpixel_aliasing(grid: np.ndarray, out_path: str | Path) -> None:
    """Plot 11x11 (dx, dy) heatmap of RMSE."""
    if not _MPL:
        _warn_skip("subpixel_aliasing")
        return
    grid = np.asarray(grid, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        grid, cmap="magma", origin="lower", extent=(-0.5, 0.5, -0.5, 0.5)
    )
    fig.colorbar(im, ax=ax, label="RMSE (deg)")
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title("Subpixel aliasing (RMSE)")
    out = Path(out_path)
    _ensure_dir(out)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
