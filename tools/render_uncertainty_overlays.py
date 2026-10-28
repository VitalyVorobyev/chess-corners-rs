#!/usr/bin/env python3
"""Render the figures of the book's uncertainty Part into ``book/src/img/``.

Run from the workspace root, with the Python bindings installed in the
active environment (``cd crates/chess-corners-py && maturin develop --release``)
plus ``numpy``, ``matplotlib``, ``scipy`` and ``opencv-python``:

    python tools/render_uncertainty_overlays.py

Every synthetic figure is computed from chessboards whose corner positions
are known exactly, detected with the ``chess_corners`` Python bindings under
the ``Uncertainty.PROPAGATED`` tier. Two independent image
generators form the boards: ``Board.render`` below (straight edges, Gaussian
lens blur, exact integration over each pixel) and
``tools/refiner_robustness/synth.py`` (supersampled perspective boards with
any PSF, illumination and lens distortion).

- ``uncertainty_bias_precision.png`` — refined positions of one corner over
  repeated noise draws, per localizer, on one shared scale: the cloud
  centre's offset from the truth is the localizer's bias, the cloud's spread
  its precision, and the predicted 1σ / 2σ ellipses are drawn around the
  truth.
- ``uncertainty_board_ellipses.png`` — predicted 1σ ellipses (enlarged) of
  two localizers on a board whose lens blur grows from left to right.
- ``uncertainty_calibration.png`` — predicted σ against the measured RMS
  error per imaging condition, per localizer, on both generators; prints
  σ̂₀² per condition.
- ``uncertainty_real_data.png`` — aggregate numbers from a real stereo
  calibration set (constants below; no image data): what the calibration
  residual is made of, and each localizer's corner-local error.

Deterministic: fixed seeds. Takes a few minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
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
sys.path.insert(0, str(WORKSPACE / "tools" / "refiner_robustness"))

# Palette: categorical slots, text inks and surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
VIOLET = "#7d5bd0"
GRAY = "#8a8984"
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
# Localizers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Localizer:
    name: str
    radon: bool
    refiner: str  # ChessRefiner / RadonRefiner constructor name
    color: str

    def detector(self, noise_sigma: float | None = None) -> "cc.Detector":
        """Detector with this localizer; ``noise_sigma=None`` estimates the
        noise from each image."""
        if self.radon:
            cfg = cc.DetectorConfig.radon().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
            cfg = cfg.with_radon(refiner=getattr(cc.RadonRefiner, self.refiner)())
        else:
            cfg = cc.DetectorConfig.chess().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
            cfg.strategy.chess.refiner = getattr(cc.ChessRefiner, self.refiner)()
        if noise_sigma is not None:
            cfg = cfg.with_noise_sigma(noise_sigma)
        return cc.Detector(cfg)


COM = Localizer("CenterOfMass", False, "center_of_mass", AQUA)
FORSTNER = Localizer("Förstner", False, "forstner", ORANGE)
SADDLE = Localizer("SaddlePoint", False, "saddle_point", BLUE)
RADON = Localizer("Radon peak fit", True, "peak_fit", RED)
RADON_SP = Localizer("Radon + SaddlePoint", True, "saddle_point", VIOLET)
LOCALIZERS = [COM, FORSTNER, SADDLE, RADON]


# ---------------------------------------------------------------------------
# Synthetic boards (first generator)
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


def match(xy: np.ndarray, truth: np.ndarray, radius: float = 0.5):
    """Index of the detection nearest each truth corner, or -1."""
    if len(xy) == 0:
        return np.full(len(truth), -1)
    d = np.linalg.norm(truth[:, None, :] - xy[None, :, :], axis=2)
    j = d.argmin(1)
    return np.where(d[np.arange(len(truth)), j] <= radius, j, -1)


def whiten(errs: np.ndarray, covs: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covs)
    return np.linalg.solve(chol, errs[..., None])[..., 0]


def sigma0_sq(errs: np.ndarray, covs: np.ndarray) -> float:
    """Variance factor: mean squared whitened error per axis."""
    return float(np.mean(np.sum(whiten(errs, covs) ** 2, 1)) / 2)


def collect(det, image: np.ndarray, truth: np.ndarray, errs: list, covs: list, radius: float = 0.5) -> tuple[int, int]:
    """Append the errors and covariances of corners matched within
    ``radius`` px that carry a covariance; return (true corners, matched)."""
    d = det.detect(image)
    idx = match(d.xy, truth, radius)
    for t, j in zip(truth, idx):
        if j < 0 or np.isnan(d.position_covariance[j]).any():
            continue
        errs.append(d.xy[j] - t)
        covs.append(d.position_covariance[j])
    return len(truth), int((idx >= 0).sum())


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
# Figure 1: bias vs precision for one corner, every localizer
# ---------------------------------------------------------------------------


def fig_bias_precision(out: Path, draws: int) -> None:
    w = h = 48
    truth = np.array([24.30, 23.80])
    rotation, noise, blur = 20.0, 2.0, 1.5
    board = Board(lattice(8.0, rotation_deg=rotation), truth)
    clean = board.render(w, h, blur)
    rng = np.random.default_rng(1)
    images = [quantize(clean, noise, rng) for _ in range(draws)]

    # Every localizer on the same axes, so cloud sizes, offsets and ellipses
    # compare directly; small clouds get a marked, magnified inset.
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 5.4), sharex=True, sharey=True)
    lim = 0.62
    for ax, loc in zip(axes, LOCALIZERS):
        det = loc.detector(noise_sigma=noise)
        errs, covs = [], []
        for img in images:
            collect(det, img, truth[None, :], errs, covs, radius=1.0)
        errs, covs = np.array(errs), np.array(covs)
        mean = errs.mean(0)
        scatter = np.sqrt(np.trace(np.cov(errs.T)) / 2)
        pred = covs.mean(0)
        r2 = np.sum(whiten(errs, covs) ** 2, 1)
        stats = (
            f"bias {np.linalg.norm(mean):.3f} px\n"
            f"scatter σ per axis {scatter:.3f} px\n"
            f"predicted σ per axis {np.sqrt(np.trace(pred) / 2):.3f} px\n"
            f"inside predicted 2σ: {np.mean(r2 <= 4):.0%}"
        )
        print(f"  {loc.name:16s} bias {np.linalg.norm(mean):.3f} scatter {scatter:.3f} "
              f"pred {np.sqrt(np.trace(pred) / 2):.3f} σ̂₀² {np.mean(r2) / 2:.2f}")
        ax.scatter(errs[:, 0], errs[:, 1], s=6, color=loc.color, alpha=0.35, lw=0)
        ax.plot(0, 0, marker="+", color=INK, ms=14, mew=1.8, ls="none")
        ax.plot(*mean, marker="x", color=INK, ms=8, mew=1.8, ls="none")
        ellipse(ax, (0, 0), pred, 1, color=INK, lw=1.2)
        ellipse(ax, (0, 0), pred, 2, color=INK, lw=1.2, ls="--")
        if scatter < 0.03 and np.linalg.norm(mean) < 0.03:
            zoom = 4
            ins = ax.inset_axes([0.6, 0.05, 0.37, 0.37])
            ins.scatter(errs[:, 0], errs[:, 1], s=5, color=loc.color, alpha=0.45, lw=0)
            ins.plot(0, 0, marker="+", color=INK, ms=10, mew=1.5, ls="none")
            ins.plot(*mean, marker="x", color=INK, ms=7, mew=1.5, ls="none")
            ellipse(ins, (0, 0), pred, 1, color=INK, lw=1.2)
            ellipse(ins, (0, 0), pred, 2, color=INK, lw=1.2, ls="--")
            ins.set_xlim(-lim / zoom, lim / zoom)
            ins.set_ylim(lim / zoom, -lim / zoom)
            ins.set_aspect("equal")
            ins.tick_params(labelsize=7)
            ins.xaxis.set_major_locator(FixedLocator([-0.1, 0, 0.1]))
            ins.yaxis.set_major_locator(FixedLocator([-0.1, 0, 0.1]))
            ins.set_title(f"{zoom}× zoom", fontsize=8, color=INK2, pad=3)
            ax.indicate_inset_zoom(ins, edgecolor=INK2, alpha=0.8)
        ax.set_title(loc.name)
        note(ax, stats, y=0.97, va="top")
        frame(ax, lim)
        ax.set_xlabel("x error (px)")
    axes[0].set_ylabel("y error (px)")

    handles = [
        plt.Line2D([], [], marker="+", color=INK, ms=12, mew=1.8, ls="none", label="true corner"),
        plt.Line2D([], [], marker="x", color=INK, ms=8, mew=1.8, ls="none", label="cloud centre"),
        plt.Line2D([], [], color=INK, lw=1.2, label="predicted 1σ"),
        plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="predicted 2σ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"One corner, {len(images)} noise draws (8 px cells rotated {rotation:g}°, lens blur σ {blur} px, "
        f"noise σ {noise:g} gray levels), same scale in every panel",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=130)
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
    noise = 4.0
    img = quantize(clean, noise, rng)
    truth = board.corners(w, h, margin=8)
    scale = 30.0
    rows = [SADDLE, FORSTNER]

    fig, axes = plt.subplots(len(rows), 1, figsize=(10.5, 4.3 * len(rows)))
    for ax, loc in zip(axes, rows):
        det = loc.detector().detect(img)  # σ estimated from the image
        idx = match(det.xy, truth, radius=1.5)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(-0.5, w - 0.5, h - 0.5, -0.5))
        ax.grid(False)
        for j in idx[idx >= 0]:
            cov = det.position_covariance[j]
            if np.isnan(cov).any():
                continue
            ellipse(ax, det.xy[j], cov, 1, scale=scale, color=loc.color, lw=1.6)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ticks = np.linspace(0, w - 1, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{blur_at[int(round(t))]:.1f}" for t in ticks])
        ax.set_yticks([])
        ax.set_title(loc.name)
    axes[-1].set_xlabel("lens blur σ at this column (px)")
    fig.suptitle(
        f"1σ position ellipses, drawn {scale:g}× enlarged; noise σ {noise:g} gray levels, estimated from the image",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")


# ---------------------------------------------------------------------------
# Figure 3: predicted vs measured over imaging conditions, both generators
# ---------------------------------------------------------------------------


def board_conditions(boards: int):
    """First generator: 200² boards, 8 px cells, random subpixel origins.
    Yields (group, label, noise, [(image, truth)])."""
    seed = 100
    shapes = [(90.0, 0.0, "0°"), (90.0, 8.0, "8°"), (90.0, 45.0, "45°"), (60.0, 0.0, "60° shear")]
    for angle, rot, tag in shapes:
        for blur in (0.5, 1.5, 2.5):
            seed += 1
            rng = np.random.default_rng(seed)
            clean = []
            for _ in range(boards):
                board = Board(lattice(8.0, angle, rot), np.array([100.0, 100.0]) + rng.uniform(0, 1, 2))
                clean.append((board.render(200, 200, blur), board.corners(200, 200, margin=10)))
            for noise in (1.0, 2.0, 8.0):
                yield "A", f"{tag} blur {blur} noise {noise:g}", noise, clean, seed * 10 + int(noise)


def synth_conditions(draws: int):
    """Second generator (``refiner_robustness/synth.py``): 128² perspective
    boards. A Gaussian-PSF grid at three noise levels, plus non-Gaussian
    PSFs and illumination at noise σ 2."""
    import synth  # noqa: PLC0415  (needs scipy and opencv)

    s, size, margin = 16, 128, 12

    def geometry(cell, roll, phase, persp=False):
        if persp:
            return synth.Geometry(size, size, cell=cell, roll=roll, tilt=(35.0, -25.0), focal=400.0, phase=phase, k1=0.15)
        return synth.Geometry(size, size, cell=cell, roll=roll, tilt=(8.0, -6.0), focal=2000.0, phase=phase)

    rng = np.random.default_rng(24)
    seed = 500
    for cell in (8.0, 12.0):
        for roll in (0.0, 8.0, 45.0):
            for blur in (0.5, 1.5, 2.5):
                geom = geometry(cell, roll, tuple(rng.uniform(0, 1, 2)))
                lin = synth.render(geom, synth.Optics(psf=synth.psf_gaussian(blur, s)), s=s)
                truth = geom.corners(margin)
                for noise in (1.0, 2.0, 8.0):
                    seed += 1
                    yield "B", f"cell {cell:g} roll {roll:g} blur {blur} noise {noise:g}", noise, [(lin, truth)] * draws, seed
    optics = {
        "defocus disk": dict(psf=lambda: synth.psf_disk(1.5, s)),
        "decentred defocus": dict(psf=lambda: synth.psf_disk(1.5, s, apod=0.6, apod_angle=-50.0)),
        "coma": dict(psf=lambda: synth.psf_coma(0.6, 2.0, 30.0, s)),
        "vignetting 40%": dict(vignette=0.4),
        "ramp ±20%": dict(illum=("ramp", 0.2)),
        "shading ±5%": dict(illum=("wave", 0.05)),
        "shading ±15%": dict(illum=("wave", 0.15)),
        "perspective + distortion": dict(persp=True),
    }
    for name, spec in optics.items():
        pairs = []
        for roll, ph in ((5.0, (0.37, 0.61)), (17.0, (0.12, 0.83)), (38.0, (0.71, 0.29))):
            geom = geometry(14.0 if spec.get("persp") else 10.0, roll, ph, spec.get("persp", False))
            o = synth.Optics(psf=spec["psf"]() if "psf" in spec else synth.psf_gaussian(1.0, s))
            if "illum" in spec:
                o.illum, o.illum_amp = spec["illum"]
            o.vignette = spec.get("vignette", 0.0)
            pairs.append((synth.render(geom, o, s=s), geom.corners(margin)))
        seed += 1
        yield "C", name, 2.0, pairs * max(1, draws // 3), seed


def run_conditions(conditions, results: dict) -> None:
    """Per condition and localizer: measured RMS error per axis, median
    predicted σ per axis, σ̂₀², and the share of corners predicted beyond
    0.5 px."""
    for group, label, noise, pairs, seed in conditions:
        dets = {loc.name: loc.detector(noise_sigma=noise) for loc in LOCALIZERS + [RADON_SP]}
        acc = {name: ([], [], 0, 0) for name in dets}
        rng = np.random.default_rng(seed)
        for clean, truth in pairs:
            img = quantize(clean, noise, rng)
            for name, det in dets.items():
                errs, covs, n_truth, n_match = acc[name]
                # A generous radius, so large errors are not cut off.
                nt, nm = collect(det, img, truth, errs, covs, radius=2.0)
                acc[name] = (errs, covs, n_truth + nt, n_match + nm)
        for name, (errs, covs, n_truth, n_match) in acc.items():
            if len(errs) < 100:
                print(f"  skip {name:20s} [{group}] {label}: {len(errs)} corners, match rate {n_match / max(n_truth, 1):.2f}")
                continue
            errs, covs = np.array(errs), np.array(covs)
            sig = np.sqrt((covs[:, 0, 0] + covs[:, 1, 1]) / 2)
            pred = float(np.median(sig))
            wide = float(np.mean(sig > 0.5))
            meas = float(np.sqrt(np.mean(np.sum(errs**2, 1)) / 2))
            s0 = sigma0_sq(errs, covs)
            results.setdefault(name, []).append((group, label, meas, pred, s0, wide))
            print(f"  {name:20s} [{group}] {label:38s} meas {meas:.3f} pred {pred:.3f} σ̂₀² {s0:5.2f} wide {wide:.1%}")


def fig_calibration(out: Path, boards: int, draws: int, replot: bool) -> None:
    cache = WORKSPACE / "out" / "uncertainty_calibration.json"
    if replot and cache.exists():
        results = json.loads(cache.read_text())
    else:
        results = {}
        run_conditions(board_conditions(boards), results)
        run_conditions(synth_conditions(draws), results)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(results))

    groups = {
        "A": ("lens-blur boards (first generator)", BLUE, "o"),
        "B": ("perspective boards, Gaussian PSF (second generator)", ORANGE, "^"),
        "C": ("other PSFs, illumination, distortion (second generator)", INK2, "D"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 11.0), sharex=True, sharey=True)
    lo, hi = 0.005, 1.0
    xs = np.array([lo, hi])
    for ax, loc in zip(axes.ravel(), LOCALIZERS):
        rows = results.get(loc.name, [])
        for g, (label, color, marker) in groups.items():
            pts = np.array([(r[2], r[3]) for r in rows if r[0] == g])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=34, color=color, marker=marker, label=label,
                           edgecolor=SURFACE, lw=0.7, zorder=3)
        ax.plot(xs, xs, color=INK, lw=1.2, label="prediction = measurement")
        ax.fill_between(xs, xs / 1.25, xs * 1.25, color=GRID, alpha=0.6, lw=0, label="±25%")
        s0 = np.array([r[4] for r in rows])
        ax.set_title(f"{loc.name}: σ̂₀² median {np.median(s0):.2f}, max {s0.max():.1f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
            axis.set_tick_params(labelbottom=True, labelleft=True)
    for ax in axes[1]:
        ax.set_xlabel("measured RMS error per axis (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted σ per axis, median (px)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Predicted against measured error, one point per imaging condition (noise σ supplied)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98), h_pad=2.0)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    for name, rows in results.items():
        s0 = np.array([r[4] for r in rows])
        print(f"  σ̂₀² {name:20s} n {len(s0):3d} median {np.median(s0):.2f} p90 {np.percentile(s0, 90):.2f} max {s0.max():.2f}")
    print(f"[ok] {out.relative_to(WORKSPACE)}")


# ---------------------------------------------------------------------------
# Figure 4: real calibration data, aggregate numbers only
# ---------------------------------------------------------------------------

# A real stereo calibration set on an accurate ChArUco target: rig
# calibration, residuals decomposed into a smooth term over the image (lens
# model), a per-corner 3-D target term, a smooth per-view term and the
# corner-local remainder. RMS px per axis over the corners every localizer
# finds.
REAL_SADDLE_PARTS = [
    ("lens model and target\n(same in every view)", 0.231),
    ("per view, common to\nall corners (defocus, depth)", 0.058),
    ("corner-local", 0.040),
]
REAL_LOCAL = [  # (localizer, corner-local RMS per axis, colour)
    ("Radon + SaddlePoint", 0.038, VIOLET),
    ("SaddlePoint", 0.040, BLUE),
    ("cv2.cornerSubPix 11×11", 0.045, GRAY),
    ("Radon peak fit", 0.056, RED),
    ("CenterOfMass", 0.103, AQUA),
    ("Förstner", 0.164, ORANGE),
]


def fig_real_data(out: Path) -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
    labels = [p[0] for p in REAL_SADDLE_PARTS]
    vals = [p[1] for p in REAL_SADDLE_PARTS]
    y = np.arange(len(vals))[::-1]
    axl.barh(y, vals, color=[GRAY, GRAY, BLUE], height=0.6)
    for yy, v in zip(y, vals):
        axl.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    axl.set_yticks(y)
    axl.set_yticklabels(labels, fontsize=9)
    axl.set_xlim(0, 0.27)
    axl.set_xlabel("RMS per axis (px)")
    axl.set_title("SaddlePoint: what the calibration residual holds")
    axl.grid(axis="y", visible=False)

    names = [r[0] for r in REAL_LOCAL]
    y = np.arange(len(names))[::-1]
    axr.barh(y, [r[1] for r in REAL_LOCAL], color=[r[2] for r in REAL_LOCAL], height=0.6)
    for yy, (_, v, _) in zip(y, REAL_LOCAL):
        axr.text(v + 0.003, yy, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    axr.set_yticks(y)
    axr.set_yticklabels(names, fontsize=9)
    axr.set_xlim(0, 0.19)
    axr.set_xlabel("corner-local error, RMS per axis (px)")
    axr.set_title("Corner-local error per localizer")
    axr.grid(axis="y", visible=False)
    fig.suptitle("Real stereo calibration on an accurate ChArUco target (aggregate over all views)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=600, help="noise draws for the single-corner figure")
    ap.add_argument("--boards", type=int, default=4, help="boards per condition, first generator")
    ap.add_argument("--synth-draws", type=int, default=12, help="noise draws per scene, second generator")
    ap.add_argument("--only", choices=["bias", "board", "calibration", "real"], help="render one figure")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the calibration figure from out/uncertainty_calibration.json")
    args = ap.parse_args()
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "bias"):
        fig_bias_precision(BOOK_IMG / "uncertainty_bias_precision.png", args.draws)
    if args.only in (None, "board"):
        fig_board_ellipses(BOOK_IMG / "uncertainty_board_ellipses.png")
    if args.only in (None, "calibration"):
        fig_calibration(BOOK_IMG / "uncertainty_calibration.png", args.boards, args.synth_draws, args.replot)
    if args.only in (None, "real"):
        fig_real_data(BOOK_IMG / "uncertainty_real_data.png")


if __name__ == "__main__":
    main()
