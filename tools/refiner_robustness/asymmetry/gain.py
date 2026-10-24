"""Analytic gain G = db/dc versus the finite-difference gain, and numpy
per-corner timings (a Rust implementation would compute G analytically:
G = X^+[:2] dA/dc, with dA/dc from the bilinear sampler's derivative taps).

Run: .venv-uq26/bin/python tools/refiner_robustness/asymmetry/gain.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import check  # noqa: E402
import core  # noqa: E402
import estimators as est  # noqa: E402
import synth  # noqa: E402

RA = 4


def analytic_gain(img, c):
    k = RA + 1
    side = 2 * k + 1
    n = len(c)
    blk, t = core.block(img, c)
    idx, _, dx, dy = core._half(RA)
    Wx, Wy = core.dwmat(t, k)
    ddx = np.einsum("nkp,np->nk", Wx, blk).reshape(-1, side, side)
    ddy = np.einsum("nkp,np->nk", Wy, blk).reshape(-1, side, side)
    ax = (0.5 * (ddx - ddx[:, ::-1, ::-1]))[:, 1:-1, 1:-1].reshape(n, -1)[:, idx]
    ay = (0.5 * (ddy - ddy[:, ::-1, ::-1]))[:, 1:-1, 1:-1].reshape(n, -1)[:, idx]
    P = np.einsum("nkp,np->nk", core.wmat(t, k), blk).reshape(-1, side, side)
    S = 0.5 * (P + P[:, ::-1, ::-1])
    gx = (0.5 * (S[:, 1:-1, 2:] - S[:, 1:-1, :-2])).reshape(n, -1)[:, idx]
    gy = (0.5 * (S[:, 2:, 1:-1] - S[:, :-2, 1:-1])).reshape(n, -1)[:, idx]
    Si = S[:, 1:-1, 1:-1].reshape(n, -1)[:, idx]
    X = np.stack([gx, gy, dx * Si, dy * Si], 2)
    pinv = core.inv(np.einsum("nhi,nhj->nij", X, X)) @ np.swapaxes(X, 1, 2)
    return pinv[:, :2] @ np.stack([ax, ay], 2)


def main():
    for cond in ["baseline", "illum wave +-15%", "gauss 2.0", "decentred defocus"]:
        lin, sen, truth = check.setup(cond)
        u8 = synth.sensor(lin, sen, np.random.default_rng(3))
        img = u8.astype(float)
        xy, _ = est.chess(u8, "SaddlePoint")
        j = est.match(xy, truth)
        c, ok, Lc = core.saddle(img, xy[j[j >= 0]], 2)
        a = core.asym_full(img, c, Lc, RA, "mul")
        Ga = analytic_gain(img, c)
        d = np.abs(Ga - a["G"]).max(axis=(1, 2))
        bA = np.einsum("nij,nj->ni", core.inv(Ga), a["b"])
        bF = a["beta"]
        print(f"{cond:20s} G_FD diag median {np.median(a['G'][:, [0, 1], [0, 1]]):.3f}; |G_an - G_FD| median {np.median(d):.3f} "
              f"max {d.max():.3f}; |beta_an - beta_FD| RMS {np.sqrt(((bA - bF) ** 2).sum(1).mean()):.4f} "
              f"(beta RMS {np.sqrt((bF ** 2).sum(1).mean()):.4f})")
    n = len(c)
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        core.saddle(img, c, 2)
    t1 = time.perf_counter()
    for _ in range(reps):
        core.asym_fit(img, c, RA, "mul")
        analytic_gain(img, c)
    t2 = time.perf_counter()
    print(f"numpy, vectorised over {n} corners: SaddlePoint fixed point + map (converged start) "
          f"{1e6 * (t1 - t0) / reps / n:.1f} us/corner; asymmetry fit + analytic gain {1e6 * (t2 - t1) / reps / n:.1f} us/corner")


if __name__ == "__main__":
    main()
