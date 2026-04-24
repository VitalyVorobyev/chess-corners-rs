#!/usr/bin/env python3
"""Run ``cv2.cornerSubPix`` over the same synthetic fixture the Rust
``bench_sweep`` example uses, and emit a JSON report in the same
schema so the plotter can treat it as a sixth refiner column.

The synthetic renderer is reimplemented here in NumPy to exactly
mirror ``crates/chess-corners/examples/bench_sweep.rs`` (8×
supersampled AA chessboard, separable Gaussian blur, box-Muller
additive noise). Keep the two in sync — any divergence shows up
as a visible gap in the book's plots.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np


SIZE = 45
CENTER = 22.0
N_GRID = 6
SUPER = 8


def render_aa_chessboard(cell: int, offset: tuple[float, float], dark: int, bright: int) -> np.ndarray:
    """8×-supersampled AA hard-cell chessboard identical to the Rust version."""
    out = np.empty((SIZE, SIZE), dtype=np.uint8)
    ox, oy = offset
    c = float(cell)
    inv = 1.0 / (SUPER * SUPER)
    # Vectorised supersample: compute all subsample cell parities at once.
    sx = (np.arange(SUPER, dtype=np.float32) + 0.5) / SUPER - 0.5  # (SUPER,)
    sy = sx  # same shape
    for y in range(SIZE):
        yf = y + sy  # (SUPER,)
        cy = np.floor((yf - oy) / c).astype(np.int32)  # (SUPER,)
        for x in range(SIZE):
            xf = x + sx  # (SUPER,)
            cx = np.floor((xf - ox) / c).astype(np.int32)
            # Broadcast: (SUPER, SUPER) parity map for this output pixel.
            parity = (cx[None, :] + cy[:, None]) & 1
            acc = (parity * bright + (1 - parity) * dark).astype(np.float32)
            out[y, x] = np.clip(round(acc.sum() * inv), 0, 255)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img
    radius = max(1, int(math.ceil(3.0 * sigma)))
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)


def add_noise_boxmuller(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    if sigma <= 0.0:
        return img
    # Match Rust's LCG + Box-Muller so plots against the Rust JSON are on
    # the same noise realisation.
    state = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    n = img.size
    out = np.empty(n, dtype=np.float32)
    idx = 0
    while idx < n:
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & 0xFFFFFFFFFFFFFFFF
        u1 = ((state >> 33) + 1) / (0xFFFFFFFF + 2)
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & 0xFFFFFFFFFFFFFFFF
        u2 = ((state >> 33) + 1) / (0xFFFFFFFF + 2)
        z = math.sqrt(max(-2.0 * math.log(u1), 0.0)) * math.cos(2.0 * math.pi * u2)
        out[idx] = z
        idx += 1
    result = img.astype(np.float32) + out.reshape(img.shape) * sigma
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def run_corner_subpix(img: np.ndarray, seed_xy: tuple[float, float], win: int = 5) -> tuple[float, float]:
    """Run ``cv2.cornerSubPix`` at one seed and return the refined (x, y)."""
    corners = np.array([[seed_xy]], dtype=np.float32)  # shape (1, 1, 2)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    refined = cv2.cornerSubPix(img, corners, (win, win), (-1, -1), term)
    return float(refined[0, 0, 0]), float(refined[0, 0, 1])


def rust_round(x: float) -> int:
    """Round half-away-from-zero, matching Rust's ``f32::round``.

    Python's built-in ``round`` uses banker's rounding (``round(0.5) == 0``),
    which differs from Rust at every ``.5`` offset and would change the
    seed pixel for cornerSubPix — see PR #42 review.
    """
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def condition_stats(cell: int, blur: float, noise: float, win: int, seed_xor: int) -> dict:
    errs: List[float] = []
    refine_elapsed = 0.0
    iters_total = 0
    for kx in range(N_GRID):
        for ky in range(N_GRID):
            ox = CENTER + kx / N_GRID
            oy = CENTER + ky / N_GRID
            img = render_aa_chessboard(cell, (ox, oy), 30, 230)
            img = gaussian_blur(img, blur)
            img = add_noise_boxmuller(img, noise, seed_xor ^ (kx * N_GRID + ky))
            seed = (rust_round(ox), rust_round(oy))
            # Time the refinement itself with multiple iters to get a
            # stable per-call number. Fixture construction stays outside
            # the timed block so ``time_us_per_call`` reflects
            # ``cornerSubPix`` only.
            reps = 50
            t_start = time.perf_counter()
            for _ in range(reps):
                rx, ry = run_corner_subpix(img, seed, win=win)
            refine_elapsed += time.perf_counter() - t_start
            iters_total += reps
            dx = rx - ox
            dy = ry - oy
            errs.append(math.hypot(dx, dy))
    errs.sort()
    n = len(errs)
    mean = sum(errs) / n
    p95 = errs[min(int(0.95 * n), n - 1)]
    worst = errs[-1]
    time_us = (refine_elapsed * 1e6) / iters_total if iters_total else 0.0
    return {
        "refiner": "OpenCV_subpix",
        "cell": cell,
        "blur_sigma": blur,
        "noise_sigma": noise,
        "mean": mean,
        "p95": p95,
        "worst": worst,
        "accept": n,
        "total": n,
        "time_us_per_call": time_us,
        "errors": errs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("book/src/img/bench/opencv_subpix_sweep.json"),
        help="Output JSON path",
    )
    parser.add_argument("--win", type=int, default=5, help="cornerSubPix winSize (half)")
    args = parser.parse_args()

    blur_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    noise_values = [0.0, 1.0, 2.0, 5.0, 8.0, 12.0]
    cell_values = [5, 6, 7, 8, 10]

    rows: List[dict] = []
    for b in blur_values:
        rows.append(condition_stats(8, b, 0.0, args.win, 0))
        print(f"  blur={b}: mean={rows[-1]['mean']:.4f}", flush=True)
    for n in noise_values:
        rows.append(condition_stats(8, 0.7, n, args.win, 0xC0FFEE))
        print(f"  noise={n}: mean={rows[-1]['mean']:.4f}", flush=True)
    for c in cell_values:
        rows.append(condition_stats(c, 0.0, 0.0, args.win, 0))
        print(f"  cell={c}: mean={rows[-1]['mean']:.4f}", flush=True)

    payload = {
        "schema": "opencv_subpix_sweep/v1",
        "tool": "cv2.cornerSubPix",
        "window_half": args.win,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
