#!/usr/bin/env python3
"""Render the book figure for the Radon detector's saddle test into ``book/src/img/``.

Run from the workspace root, with the Python bindings installed in the
active environment (``cd crates/chess-corners-py && maturin develop --release``):

    python tools/render_radon_saddle_overlay.py

``radon_saddle_test.png`` has three panels, all on synthetic boards formed
as a camera forms them (the ``Board`` renderer of
``render_uncertainty_overlays.py``):

- a sharp board rotated 45° with every peak of the four-ray response that
  passes the threshold and non-maximum suppression: the corners, and the
  peaks beside the edges between them;
- the response map of the same crop, whose ridges along the edges carry
  those peaks;
- the saddle ratio (smaller over larger principal curvature of a quadratic
  fit to the image, negative when both have the same sign) of every such
  peak over a grid of boards, corners against edge peaks, with the
  acceptance bound.

The response, peak search and saddle fit are re-implemented here in NumPy
so the rejected peaks can be drawn; the peaks the figure keeps are checked
against the ``chess_corners`` Radon detector. Deterministic, about a
minute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import chess_corners as cc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_uncertainty_overlays import (  # noqa: E402
    BLUE,
    BOOK_IMG,
    INK2,
    ORANGE,
    SURFACE,
    WORKSPACE,
    Board,
    lattice,
    note,
)

RAY = 4  # ray radius in working (2× upsampled) pixels, the preset
NMS = 4
THRESHOLD = 0.28
MIN_RATIO = 0.2


def upsample2(img: np.ndarray) -> np.ndarray:
    """Bilinear 2× upsample sampling the input at ``i / 2``, rounded to u8
    (the detector's working image)."""
    h, w = img.shape
    ys = np.arange(2 * h) * 0.5
    xs = np.arange(2 * w) * 0.5
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    f = img.astype(float)
    a = f[y0][:, x0] + (f[y0][:, x1] - f[y0][:, x0]) * tx
    b = f[y1][:, x0] + (f[y1][:, x1] - f[y1][:, x0]) * tx
    return np.clip(np.rint(a + (b - a) * ty), 0, 255)


def ray_response(work: np.ndarray) -> np.ndarray:
    """``(max_α S_α − min_α S_α)²`` over the 0°, 90°, 45° and 135° rays of
    ``2·RAY + 1`` samples, then a 3×3 box blur; zero within ``RAY`` of the
    border."""
    h, w = work.shape
    sums = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        s = np.zeros_like(work)
        for t in range(-RAY, RAY + 1):
            s += np.roll(work, (-t * dy, -t * dx), axis=(0, 1))
        sums.append(s)
    sums = np.stack(sums)
    r = (sums.max(0) - sums.min(0)) ** 2
    r[:RAY] = r[-RAY:] = 0
    r[:, :RAY] = r[:, -RAY:] = 0
    pad = np.pad(r, 1, mode="edge")
    return sum(pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0


def peaks(resp: np.ndarray) -> np.ndarray:
    """Working-pixel ``(x, y)`` of strict local maxima within ``NMS`` above
    ``THRESHOLD`` of the maximum, at least ``RAY + NMS + 1`` from the border."""
    h, w = resp.shape
    b = RAY + NMS + 1
    thr = THRESHOLD * resp.max()
    out = []
    for y in range(b, h - b):
        for x in range(b, w - b):
            v = resp[y, x]
            if v <= thr:
                continue
            win = resp[y - NMS : y + NMS + 1, x - NMS : x + NMS + 1]
            if v >= win.max() and (win == v).sum() == 1:
                out.append((x, y))
    return np.array(out, dtype=int).reshape(-1, 2)


_OFF = np.array([(dx, dy) for dy in range(-RAY, RAY + 1) for dx in range(-RAY, RAY + 1) if dx * dx + dy * dy <= RAY * RAY])


def saddle_ratio(work: np.ndarray, x: int, y: int) -> float:
    """Smaller over larger principal curvature of the quadratic fit over the
    disk of radius ``RAY``; negative when both curvatures share a sign."""
    dx, dy = _OFF[:, 0].astype(float), _OFF[:, 1].astype(float)
    i = work[y + _OFF[:, 1], x + _OFF[:, 0]]
    r2 = dx * dx + dy * dy
    u = r2 - r2.mean()
    v = dx * dx - dy * dy
    p = (u * i).sum() / (u * u).sum()
    q = (v * i).sum() / (v * v).sum()
    b = (dx * dy * i).sum() / ((dx * dy) ** 2).sum()
    hess = np.array([[2 * (p + q), b], [b, 2 * (p - q)]])
    lam = np.linalg.eigvalsh(hess)
    small, large = np.sort(np.abs(lam))
    if large == 0:
        return 0.0
    return small / large if lam[0] * lam[1] < 0 else -small / large


def classify(board: Board, size: int, psf: float):
    """Peaks of the board in input pixels, their saddle ratios and whether
    each lies within 1 px of a corner."""
    img = np.clip(np.rint(board.render(size, size, psf)), 0, 255).astype(np.uint8)
    work = upsample2(img.astype(float))
    pk = peaks(ray_response(work))
    ratio = np.array([saddle_ratio(work, x, y) for x, y in pk])
    xy = pk / 2.0
    truth = board.corners(size, size, margin=0)
    d = np.linalg.norm(xy[:, None, :] - truth[None, :, :], axis=2).min(1)
    return img, work, xy, ratio, d < 1.0


def fig_saddle(out: Path) -> None:
    size = 72
    board = Board(lattice(12.0, 90.0, 45.0), np.array([36.3, 36.6]))
    img, work, xy, ratio, is_corner = classify(board, size, 0.5)
    keep = ratio >= MIN_RATIO

    # The kept peaks are the Radon detector's corners.
    det = cc.Detector(cc.DetectorConfig.radon().without_orientation())
    found = np.asarray(det.detect(img).xy).reshape(-1, 2)
    for p in xy[keep]:
        assert np.linalg.norm(found - p, axis=1).min() < 0.75, p
    assert len(found) == keep.sum(), (len(found), keep.sum())

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(-0.5, size - 0.5, size - 0.5, -0.5))
    ax.scatter(*xy[keep].T, s=60, facecolors="none", edgecolors=BLUE, linewidths=1.6, label="kept: saddle")
    ax.scatter(*xy[~keep].T, s=40, marker="x", color=ORANGE, linewidths=1.6, label="dropped: beside an edge")
    ax.set_title("Response peaks, sharp board rotated 45°")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    resp = ray_response(work)
    ax.imshow(np.sqrt(resp), cmap="magma", extent=(-0.5, size - 0.5, size - 0.5, -0.5))
    ax.scatter(*xy[~keep].T, s=40, marker="x", color=ORANGE, linewidths=1.4)
    ax.set_title("√ response: corner peaks and edge ridges")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Saddle ratios over a grid of boards.
    corner_r, edge_r = [], []
    for cell in (8.0, 12.0, 16.0):
        for rot in (0.0, 22.5, 45.0):
            for axis in (90.0, 70.0):
                for psf in (0.5, 1.0):
                    b = Board(lattice(cell, axis, rot), np.array([40.3, 40.7]))
                    _, _, _, r, c = classify(b, 80, psf)
                    corner_r.extend(r[c])
                    edge_r.extend(r[~c])
    ax = axes[2]
    bins = np.linspace(-1, 1, 41)
    ax.hist(corner_r, bins=bins, color=BLUE, alpha=0.8, label=f"corner peaks ({len(corner_r)})")
    ax.hist(edge_r, bins=bins, color=ORANGE, alpha=0.8, label=f"other peaks ({len(edge_r)})")
    ax.axvline(MIN_RATIO, color=INK2, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("saddle ratio (smaller / larger curvature; < 0: same sign)")
    ax.set_ylabel("peaks")
    ax.set_title("Saddle ratio over 36 synthetic boards")
    ax.legend(loc="upper left", fontsize=8)
    note(ax, f"kept at ≥ {MIN_RATIO}", y=0.55)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"[ok] {out.relative_to(WORKSPACE)}")
    print(f"     crop: {keep.sum()} kept, {(~keep).sum()} dropped; grid: corners min ratio {min(corner_r):.2f}, others max {max(edge_r):.2f}")


def main() -> None:
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    fig_saddle(BOOK_IMG / "radon_saddle_test.png")


if __name__ == "__main__":
    main()
