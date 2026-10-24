"""Fit the per-localizer systematic-error tables of the position covariance.

Input: the noise-free per-corner records written by

    UQ24_SYNTH_DIR=tools/refiner_robustness/out/uq24 \
      cargo test -p chess-corners-core --release dump_systematic -- --ignored

(``bench_results/uq24/systematic.csv``: both generators, every localizer,
window centre, estimate, truth and edge width per corner). For every edge-width
knot it pools the corners of both generators within the knot's bin and fits

- ``gain``: the least-squares slope of the estimate's offset from its window
  centre on the true offset (single-shot localizers whose error is a pull
  toward the window centre); 1 for the re-centred saddle-point fit and for
  the center of mass, whose error does not scale with the offset;
- ``residual``: the RMS per axis that makes the whitened noise-free error
  ``e' Sigma_sys^-1 e / 2`` average 1 in the bin (the least-squares residual
  where no value reaches 1),

and prints them as the Rust ``Knot`` tables (``SYSTEMATIC`` next to each
estimator in ``chess-corners-core``).

Usage: python tools/refiner_robustness/fit_systematic.py [CSV]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
KNOTS = [1.09, 1.14, 1.18, 1.24, 1.30, 1.35, 1.40, 1.47, 1.58, 1.68]
# (localizer rows, grid spacing px, fit the gain?)
MODELS = {
    "center_of_mass.rs": ("CoM", 1.0, False),
    "forstner.rs": ("Forstner", 1.0, True),
    "radon/detect.rs": ("RadonPeak", 0.5, True),
    "saddle_point.rs": ("SaddlePoint", 1.0, False),
}


def sigma0_sq(e: np.ndarray, uhat: np.ndarray, gain: float, rho: float, h: float) -> float:
    """Mean e' Sigma_sys^-1 e / 2 for Sigma_sys = b b' + V I."""
    p = h * h / 12
    den = gain * gain * p + rho * rho
    b = uhat * (1 - gain * p / den)
    v = p * rho * rho / den
    bb, eb, ee = (b * b).sum(1), (e * b).sum(1), (e * e).sum(1)
    return float(((ee - eb * eb / (v + bb)) / v).mean() / 2)


def fit(d: pd.DataFrame, h: float, fit_gain: bool):
    u = np.stack([d.tx - d.cx, d.ty - d.cy], 1)
    uhat = np.stack([d.x - d.cx, d.y - d.cy], 1)
    e = uhat - u
    if fit_gain:
        gain = float((u * uhat).sum() / (u * u).sum())
        rho_ls = float(np.sqrt(((uhat - gain * u) ** 2).mean()))
        obs = uhat
    else:
        gain = 1.0
        rho_ls = float(np.sqrt((e**2).mean()))
        # Without a pull the window offset carries no information.
        obs = np.zeros_like(uhat)
    try:
        rho = brentq(lambda r: sigma0_sq(e, obs, gain, r, h) - 1.0, 1e-4, 1.0)
    except ValueError:
        rho = rho_ls
    return gain, rho, float(np.sqrt((e**2).mean())), len(d)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench_results" / "uq24" / "systematic.csv"
    df = pd.read_csv(path).dropna(subset=["width"])
    edges = [0.0] + [(a + b) / 2 for a, b in zip(KNOTS[:-1], KNOTS[1:])] + [np.inf]
    for file, (loc, h, fit_gain) in MODELS.items():
        d = df[df.localizer == loc]
        print(f"// {file}: {loc} (n = {len(d)}; width, gain, residual; noise-free RMS error/axis)")
        print("knots: &[")
        for k, w in enumerate(KNOTS):
            b = d[(d.width > edges[k]) & (d.width <= edges[k + 1])]
            gain, rho, rms, n = fit(b, h, fit_gain)
            print(f"    knot({w:.2f}, {gain:.3f}, {rho:.4f}), // rms {rms:.4f}, n {n}")
        print("],\n")


if __name__ == "__main__":
    main()
