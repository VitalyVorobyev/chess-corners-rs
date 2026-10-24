#!/usr/bin/env python3
"""Render the figures of the book's uncertainty Part into ``book/src/img/``.

Run from the workspace root, with the Python bindings installed in the
active environment (``cd crates/chess-corners-py && maturin develop --release``):

    python tools/render_uncertainty_overlays.py

Every figure is computed from synthetic chessboards whose corner positions
are known exactly, formed as a camera forms them (Gaussian lens blur, then
integration over each pixel), and detected with the ``chess_corners`` Python
bindings:

- ``uncertainty_bias_precision.png`` — refined positions of one corner over
  repeated noise draws, per refiner: the cloud centre's offset from the truth
  is the refiner's bias, the cloud's spread its precision; the saddle-point
  refiner's predicted 1σ / 2σ ellipses are drawn on its cloud.
- ``uncertainty_board_ellipses.png`` — predicted covariance ellipses
  (exaggerated) on a board whose lens blur grows from left to right.
- ``uncertainty_calibration.png`` — predicted vs measured error per imaging
  condition, and the whitened errors against a unit normal.

Deterministic: fixed seeds. Takes about a minute.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

import chess_corners as cc  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[1]
BOOK_IMG = WORKSPACE / "book" / "src" / "img"

# Palette: categorical slots 1-3, text inks and surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 11,
    }
)


# ---------------------------------------------------------------------------
# Synthetic boards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    """An infinite chessboard: board coordinates ``(u, v)`` (cell units) map
    to detector pixel coordinates by ``p = origin + A @ (u, v)``. Corners sit
    at integer ``(u, v)``. Pixel ``i`` covers ``[i - 0.5, i + 0.5]``, so a
    detector coordinate of ``i`` is that pixel's centre."""

    a: np.ndarray  # 2×2, columns = the two lattice vectors in px
    origin: np.ndarray  # 2, a corner position in px
    dark: float = 40.0
    light: float = 200.0

    def render(self, w: int, h: int, psf: float, rows: int | None = None) -> np.ndarray:
        """The board as a camera forms it: the hard pattern blurred by a
        Gaussian lens PSF of σ ``psf`` px, then integrated over each pixel's
        unit square (the same formation as ``optical_chessboard`` in the
        Rust test fixtures).

        Each pixel is the pattern weighted by the PSF-blurred pixel aperture
        ``h(x)·h(y)``. Across an image row the pattern is piecewise constant,
        so the ``x`` integral is exact (differences of the antiderivative of
        ``h``); down the rows, split at horizontal edges, it uses ``rows``
        midpoint samples per pixel,
        by default ``max(16, ⌈16/σ⌉)``, within 0.02 gray levels of the exact
        value."""
        ap = _Aperture(psf)
        rows = rows or max(16, int(np.ceil(16.0 / psf)))
        inv = np.linalg.inv(self.a)
        ox, oy = self.origin
        # The row integral is continuous in y except across horizontal
        # edges, where the whole row changes sign: split the y range there
        # and place the midpoints within each piece.
        lo, hi = -ap.reach, h - 1 + ap.reach
        cuts = [lo, hi]
        for r in (0, 1):
            if abs(inv[r, 0]) < 1e-12:  # u_r depends on y only
                u0, u1 = inv[r, 1] * (lo - oy), inv[r, 1] * (hi - oy)
                k = np.arange(np.ceil(min(u0, u1)), np.floor(max(u0, u1)) + 1)
                cuts.extend(oy + k / inv[r, 1])
        cuts = np.unique(np.clip(cuts, lo, hi))
        ys, dys = [], []
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            n = max(1, int(np.ceil((y1 - y0) * rows)))
            dy = (y1 - y0) / n
            ys.append(y0 + (np.arange(n) + 0.5) * dy)
            dys.append(np.full(n, dy))
        ys, dys = np.concatenate(ys), np.concatenate(dys)
        xlo, xhi = -ap.reach, w - 1 + ap.reach
        cols = np.arange(w, dtype=float)
        across = np.empty((len(ys), w))
        for i, y in enumerate(ys):
            # Edge crossings of this row: u_r(x) = inv[r, 0]·(x − ox) + c_r.
            xs = [np.array([xlo, xhi])]
            for r in (0, 1):
                slope, c = inv[r, 0], inv[r, 1] * (y - oy)
                if abs(slope) < 1e-12:
                    continue
                u0, u1 = slope * (xlo - ox) + c, slope * (xhi - ox) + c
                k = np.arange(np.ceil(min(u0, u1)), np.floor(max(u0, u1)) + 1)
                xs.append(ox + (k - c) / slope)
            bp = np.sort(np.concatenate(xs))
            mid = 0.5 * (bp[1:] + bp[:-1]) - ox
            u = inv[0, 0] * mid + inv[0, 1] * (y - oy)
            v = inv[1, 0] * mid + inv[1, 1] * (y - oy)
            light = np.where((np.floor(u) + np.floor(v)) % 2 == 0, 1.0, 0.0)
            cum = ap.cumulative(bp[None, :] - cols[:, None])
            across[i] = (np.diff(cum, axis=1) * light).sum(1)
        down = ap.density(ys[None, :] - np.arange(h, dtype=float)[:, None]) * dys[None, :]
        cover = down @ across
        return self.dark + (self.light - self.dark) * cover

    def corners(self, w: int, h: int, margin: float) -> np.ndarray:
        inv = np.linalg.inv(self.a)
        box = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) - self.origin
        uv = box @ inv.T
        lo, hi = np.floor(uv.min(0)) - 1, np.ceil(uv.max(0)) + 1
        uu, vv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        pts = np.stack([uu.ravel(), vv.ravel()], 1) @ self.a.T + self.origin
        keep = (
            (pts[:, 0] >= margin)
            & (pts[:, 0] <= w - 1 - margin)
            & (pts[:, 1] >= margin)
            & (pts[:, 1] <= h - 1 - margin)
        )
        return pts[keep]


def lattice(cell: float, axis_angle_deg: float = 90.0, rotation_deg: float = 0.0) -> np.ndarray:
    """Lattice vectors of length ``cell`` px, ``axis_angle_deg`` apart."""
    r = np.deg2rad(rotation_deg)
    t = np.deg2rad(rotation_deg + axis_angle_deg)
    return cell * np.array([[np.cos(r), np.cos(t)], [np.sin(r), np.sin(t)]])


def _erf(x: np.ndarray) -> np.ndarray:
    """Error function, absolute error below 1.5e-7 (Abramowitz & Stegun
    7.1.26)."""
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.copysign(1.0 - poly * np.exp(-x * x), x)


def _cdf(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(t / np.sqrt(2.0)))


def _pdf(t: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


class _Aperture:
    """One axis of a unit pixel aperture blurred by a Gaussian PSF of σ px:
    ``h(t) = Φ((t + ½)/σ) − Φ((t − ½)/σ)``, truncated at ``½ + 5σ``."""

    def __init__(self, sigma: float):
        self.sigma = sigma
        self.reach = 0.5 + 5.0 * sigma

    def density(self, t: np.ndarray) -> np.ndarray:
        s = self.sigma
        return _cdf((t + 0.5) / s) - _cdf((t - 0.5) / s)

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        """``∫ h`` up to ``t``, from ``∫ Φ = τΦ(τ) + φ(τ)``."""
        s = self.sigma

        def psi(u):
            return u * _cdf(u) + _pdf(u)

        return s * (psi((t + 0.5) / s) - psi((t - 0.5) / s))


def quantize(img: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    noisy = img + rng.normal(0.0, noise, img.shape) if noise > 0 else img
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def detector(refiner: "cc.ChessRefiner", noise_sigma: float | None = None) -> "cc.Detector":
    cfg = cc.DetectorConfig.chess().without_orientation()
    cfg.strategy.chess.refiner = refiner
    if noise_sigma is not None:
        cfg = cfg.with_noise_sigma(noise_sigma)
    return cc.Detector(cfg)


def match(xy: np.ndarray, truth: np.ndarray, radius: float = 0.5):
    """Index of the detection nearest each truth corner, or -1."""
    if len(xy) == 0:
        return np.full(len(truth), -1)
    d = np.linalg.norm(truth[:, None, :] - xy[None, :, :], axis=2)
    j = d.argmin(1)
    return np.where(d[np.arange(len(truth)), j] <= radius, j, -1)


def note(ax, text: str, y: float = 0.03, va: str = "bottom") -> None:
    ax.text(
        0.03,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=INK2,
        va=va,
        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2),
    )


def frame(ax, lim: float) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # image convention: y down
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))


def ellipse(ax, centre, cov, k, scale=1.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    width, height = 2 * k * scale * np.sqrt(vals[1]), 2 * k * scale * np.sqrt(vals[0])
    ax.add_patch(Ellipse(centre, width, height, angle=angle, fill=False, **kw))


# ---------------------------------------------------------------------------
# Figure 1: bias vs precision for one corner
# ---------------------------------------------------------------------------


def fig_bias_precision(out: Path, draws: int) -> None:
    w = h = 48
    truth = np.array([24.30, 23.80])
    board = Board(lattice(8.0), truth)
    noise, blur = 2.0, 1.5
    clean = board.render(w, h, blur)
    refiners = [
        ("CenterOfMass", cc.ChessRefiner.center_of_mass(), AQUA),
        ("Förstner", cc.ChessRefiner.forstner(), ORANGE),
        ("SaddlePoint", cc.ChessRefiner.saddle_point(), BLUE),
    ]
    rng = np.random.default_rng(1)
    images = [quantize(clean, noise, rng) for _ in range(draws)]

    # Every refiner on the same axes, so cloud sizes and offsets compare
    # directly; the saddle-point detail is an inset with its region marked.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), sharex=True, sharey=True)
    lim = 0.42
    for ax, (name, refiner, color) in zip(axes, refiners):
        det = detector(refiner, noise_sigma=noise)
        errs, covs = [], []
        for img in images:
            d = det.detect(img)
            j = match(d.xy, truth[None, :])[0]
            if j < 0:
                continue
            errs.append(d.xy[j] - truth)
            if not np.isnan(d.position_covariance[j]).any():
                covs.append(d.position_covariance[j])
        errs = np.array(errs)
        mean = errs.mean(0)
        scatter_cov = np.cov(errs.T)
        sigma = np.sqrt(np.trace(scatter_cov) / 2)
        stats = f"bias {np.linalg.norm(mean):.3f} px\nσ per axis {sigma:.3f} px"
        ax.scatter(errs[:, 0], errs[:, 1], s=6, color=color, alpha=0.35, lw=0)
        ax.plot(0, 0, marker="+", color=INK, ms=14, mew=1.8, ls="none")
        ax.plot(*mean, marker="x", color=INK, ms=8, mew=1.8, ls="none")
        if covs:
            # Inset: the cloud with its predicted ellipses, magnified.
            pred = np.array(covs).mean(0)
            zoom = 5
            ins = ax.inset_axes([0.62, 0.1, 0.35, 0.35])
            ins.scatter(errs[:, 0], errs[:, 1], s=5, color=color, alpha=0.45, lw=0)
            ins.plot(0, 0, marker="+", color=INK, ms=10, mew=1.5, ls="none")
            ellipse(ins, (0, 0), pred, 1, color=INK, lw=1.2)
            ellipse(ins, (0, 0), pred, 2, color=INK, lw=1.2, ls="--")
            ins.set_xlim(-lim / zoom, lim / zoom)
            ins.set_ylim(lim / zoom, -lim / zoom)
            ins.set_aspect("equal")
            ins.tick_params(labelsize=7)
            ins.xaxis.set_major_locator(FixedLocator([-0.05, 0, 0.05]))
            ins.yaxis.set_major_locator(FixedLocator([-0.05, 0, 0.05]))
            ins.set_title(f"{zoom}× zoom", fontsize=8, color=INK2, pad=3)
            ax.indicate_inset_zoom(ins, edgecolor=INK2, alpha=0.8)
            r2 = np.sum(whiten(errs, np.array(covs)) ** 2, 1)
            stats += (
                f"\ninside predicted 1σ: {np.mean(r2 <= 1):.0%} (normal: 39%)"
                f"\ninside predicted 2σ: {np.mean(r2 <= 4):.0%} (normal: 86%)"
            )
        else:
            ellipse(ax, mean, scatter_cov, 2, color=INK2, lw=1.2, ls=":")
        ax.set_title(name)
        note(ax, stats, y=0.97, va="top")
        frame(ax, lim)
        ax.set_xlabel("x error (px)")
    axes[0].set_ylabel("y error (px)")

    handles = [
        plt.Line2D([], [], marker="+", color=INK, ms=12, mew=1.8, ls="none", label="true corner"),
        plt.Line2D([], [], marker="x", color=INK, ms=8, mew=1.8, ls="none", label="cloud centre"),
        plt.Line2D([], [], color=INK2, lw=1.2, ls=":", label="scatter 2σ (no prediction)"),
        plt.Line2D([], [], color=INK, lw=1.2, label="predicted 1σ"),
        plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="predicted 2σ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"One corner, {len(images)} noise draws (8 px cells, lens blur σ {blur} px, noise σ {noise:g} gray levels), "
        "same scale in every panel",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")


# ---------------------------------------------------------------------------
# Figure 2: covariance ellipses across a lens-blur gradient
# ---------------------------------------------------------------------------


def blur_gradient(board: Board, w: int, h: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Lens blur that grows linearly from ``lo`` (left edge) to ``hi``
    (right): each column blends the two nearest of a stack of boards formed
    with uniform blur."""
    levels = np.linspace(lo, hi, 9)
    stack = np.stack([board.render(w, h, s) for s in levels])
    target = lo + (hi - lo) * np.arange(w) / (w - 1)
    pos = np.interp(target, levels, np.arange(len(levels)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(levels) - 2)
    t = pos - i0
    cols = np.arange(w)
    out = (1 - t)[None, :] * stack[i0, :, cols].T + t[None, :] * stack[i0 + 1, :, cols].T
    return out, target


def fig_board_ellipses(out: Path) -> None:
    w, h = 560, 230
    board = Board(lattice(20.0, axis_angle_deg=55.0, rotation_deg=10.0), np.array([13.3, 11.6]))
    clean, blur_at = blur_gradient(board, w, h, 0.5, 2.5)
    rng = np.random.default_rng(7)
    noise = 2.0
    img = quantize(clean, noise, rng)
    det = detector(cc.ChessRefiner.saddle_point()).detect(img)  # σ estimated
    truth = board.corners(w, h, margin=8)
    idx = match(det.xy, truth)
    scale = 150.0

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    ax.grid(False)
    for j in idx[idx >= 0]:
        cov = det.position_covariance[j]
        if np.isnan(cov).any():
            continue
        ellipse(ax, det.xy[j], cov, 1, scale=scale, color=ORANGE, lw=1.6)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ticks = np.linspace(0, w - 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{blur_at[int(round(t))]:.1f}" for t in ticks])
    ax.set_yticks([])
    ax.set_xlabel("lens blur σ at this column (px)")
    ax.set_title(
        f"1σ position ellipses, drawn {scale:g}× enlarged; noise σ {noise:g} gray levels, estimated from the image"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")


# ---------------------------------------------------------------------------
# Figure 3: calibration over imaging conditions
# ---------------------------------------------------------------------------


def clean_boards(cell: float, axis_angle: float, rotation: float, psf: float, boards: int, seed: int):
    """``boards`` noise-free 200² renders with random subpixel origins."""
    w = h = 200
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(boards):
        origin = np.array([100.0, 100.0]) + rng.uniform(0, 1, 2)
        board = Board(lattice(cell, axis_angle, rotation), origin)
        out.append((board, board.render(w, h, psf)))
    return out


def run_condition(clean, noise: float, seed: int):
    """Errors and covariances of matched corners that carry a covariance,
    and the fraction of true corners matched within 0.5 px, on noisy copies
    of the ``clean`` boards."""
    rng = np.random.default_rng(seed)
    det = detector(cc.ChessRefiner.saddle_point())  # σ estimated per image
    errs, covs, n_truth, n_match = [], [], 0, 0
    for board, img in clean:
        h, w = img.shape
        d = det.detect(quantize(img, noise, rng))
        truth = board.corners(w, h, margin=10)
        idx = match(d.xy, truth)
        n_truth += len(truth)
        n_match += int((idx >= 0).sum())
        for t, j in zip(truth, idx):
            if j < 0 or np.isnan(d.position_covariance[j]).any():
                continue
            errs.append(d.xy[j] - t)
            covs.append(d.position_covariance[j])
    return np.array(errs), np.array(covs), n_match / max(n_truth, 1)


def whiten(errs: np.ndarray, covs: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covs)
    return np.linalg.solve(chol, errs[..., None])[..., 0]


def fig_calibration(out: Path, boards: int) -> None:
    blurs = [0.5, 0.8, 1.5, 2.5]
    noises = [1.0, 2.0, 4.0, 8.0]
    # (label, axis angle, rotation, colour, marker, grid-aligned)
    groups = [
        ("square, grid-aligned", 90.0, 0.0, BLUE, "o", True),
        ("60° shear, one axis grid-aligned", 60.0, 0.0, AQUA, "s", True),
        ("square, rotated 8°", 90.0, 8.0, ORANGE, "^", False),
        ("square, rotated 45°", 90.0, 45.0, RED, "D", False),
    ]
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    aligned, rotated = [], []
    seed = 100
    for label, angle, rot, color, marker, is_aligned in groups:
        pred, emp = [], []
        for b in blurs:
            seed += 1
            clean = clean_boards(8.0, angle, rot, b, boards, seed)
            for n in noises:
                seed += 1
                errs, covs, rate = run_condition(clean, n, seed)
                if len(errs) < 200 or rate < 0.99:
                    print(f"  skip {label} blur {b} noise {n}: {len(errs)} corners, match rate {rate:.2f}")
                    continue
                pred.append(np.sqrt(np.mean(covs[:, 0, 0] + covs[:, 1, 1]) / 2))
                emp.append(np.sqrt(np.mean(np.sum(errs**2, 1)) / 2))
                z = whiten(errs, covs)
                s0 = np.mean(np.sum(z**2, 1)) / 2
                print(f"  {label:34s} blur {b:3.1f} noise {n:3.0f}: σ̂₀² {s0:5.2f} ({len(errs)} corners)")
                (aligned if is_aligned else rotated).append(z.ravel())
        ax.scatter(emp, pred, s=36, color=color, marker=marker, label=label, edgecolor=SURFACE, lw=0.8, zorder=3)
    lo, hi = 0.005, 0.15
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK, lw=1.2, label="prediction = measurement")
    ax.fill_between(xs, xs / 1.25, xs * 1.25, color=GRID, alpha=0.6, lw=0, label="±25%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("measured RMS error per axis (px)")
    ax.set_ylabel("predicted σ per axis (px)")
    ax.set_title("One point per lens blur × noise condition (8 px cells)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    bins = np.linspace(-5, 5, 61)
    za, zr = np.clip(np.concatenate(aligned), -5, 5), np.clip(np.concatenate(rotated), -5, 5)
    axh.hist(za, bins=bins, density=True, color=BLUE, alpha=0.7, edgecolor=SURFACE, lw=0.5,
             label=f"grid-aligned boards (std {np.concatenate(aligned).std():.2f})")
    axh.hist(zr, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.8,
             label=f"rotated 8° and 45° (std {np.concatenate(rotated).std():.2f})")
    g = np.linspace(-5, 5, 400)
    axh.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), color=INK, lw=1.4, label="unit normal")
    axh.set_xlabel("whitened error component")
    axh.set_ylabel("density")
    axh.set_title("Whitened errors, all conditions (tails folded into ±5)")
    axh.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=600, help="noise draws for the single-corner figure")
    ap.add_argument("--boards", type=int, default=6, help="boards per condition for the calibration figure")
    ap.add_argument("--only", choices=["bias", "board", "calibration"], help="render one figure")
    args = ap.parse_args()
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "bias"):
        fig_bias_precision(BOOK_IMG / "uncertainty_bias_precision.png", args.draws)
    if args.only in (None, "board"):
        fig_board_ellipses(BOOK_IMG / "uncertainty_board_ellipses.png")
    if args.only in (None, "calibration"):
        fig_calibration(BOOK_IMG / "uncertainty_calibration.png", args.boards)


if __name__ == "__main__":
    main()
