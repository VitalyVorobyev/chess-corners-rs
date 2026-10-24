"""Sanity checks of the independent generator.

1. Bayer mosaic + demosaic of a flat gray scene stays flat (the RGGB mapping
   to OpenCV's code is right).
2. Pixel values against ``Board.render`` (tools/render_uncertainty_overlays.py)
   on the same board and Gaussian lens PSF.
3. True corners: zero-PSF render is point-symmetric about each truth corner.
4. SaddlePoint / Förstner noise-free RMS error on both renders.

Run: .venv-audit/bin/python tools/refiner_robustness/sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synth  # noqa: E402
import estimators as est  # noqa: E402
from render_uncertainty_overlays import Board  # noqa: E402

S = 16


def main() -> None:
    # 1. Bayer flatness.
    flat = np.full((32, 32), 150.0)
    out = synth.sensor(flat, synth.Sensor(noise=0, bayer=True), None)
    print(f"bayer flat: min {out.min()} max {out.max()} (expect 150 +-1)")

    # 2. Pixel agreement with Board.render: similarity board, rotation 7 deg.
    for roll, sig in [(0.0, 1.0), (7.0, 1.0), (30.0, 0.7), (7.0, 2.0)]:
        g = synth.Geometry(160, 160, cell=10.0, roll=roll, focal=1e7, phase=(0.37, 0.61))
        lin = synth.render(g, synth.Optics(psf=synth.psf_gaussian(sig, S)), s=S)
        hm = g.homography()
        hm = hm / hm[2, 2]
        a = hm[:2, :2]
        origin = hm[:2, 2]
        board = Board(a, origin)
        ref = board.render(160, 160, sig)
        # Board parity convention may be complemented; compare both.
        d1 = np.abs(lin - ref)[20:-20, 20:-20]
        d2 = np.abs(lin - (240 - ref))[20:-20, 20:-20]
        d = d1 if d1.mean() < d2.mean() else d2
        print(f"render vs Board.render roll {roll:4.1f} psf {sig}: max |diff| {d.max():.3f} GL, mean {d.mean():.4f} GL")

        truth = g.corners(12)
        # Noise-free, 8-bit, refiners on both renders.
        for name, img in [("synth", lin), ("Board", ref if d is d1 else 240 - ref)]:
            u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
            res = est.run_all(u8, names=["SaddlePoint", "Forstner", "RC-Forstner", "cSP-5"])
            line = []
            for k, (xy, _cov) in res.items():
                j = est.match(xy, truth, 1.0)
                e = xy[j[j >= 0]] - truth[j >= 0]
                line.append(f"{k} {np.sqrt((e**2).sum(1).mean()):.4f} ({(j >= 0).mean():.2f})")
            print(f"   {name:6s} noise-free RMS: " + ", ".join(line))

    # 3. Point symmetry of the PSF-free and symmetric-PSF renders about truth:
    # sample the image bilinearly at truth +- d.
    import scipy.ndimage as ndi

    g = synth.Geometry(160, 160, cell=10.0, roll=7.0, focal=1e7, phase=(0.37, 0.61))
    lin = synth.render(g, synth.Optics(psf=synth.psf_gaussian(1.0, S)), s=S)
    truth = g.corners(12)
    d = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 1.0], [1.5, -0.5]])
    asym = []
    for dd in d:
        p = ndi.map_coordinates(lin, [truth[:, 1] + dd[1], truth[:, 0] + dd[0]], order=3)
        q = ndi.map_coordinates(lin, [truth[:, 1] - dd[1], truth[:, 0] - dd[0]], order=3)
        asym.append(np.abs(p - q).max())
    print(f"point symmetry I(c+d)-I(c-d), Gaussian psf: max {max(asym):.3f} GL (cubic interp.)")

    # 4. Perspective + distortion truth: zero-PSF symmetry is no longer exact,
    # but the truth must sit on the edge crossings: check the corner is a
    # saddle at truth by sign pattern of a heavily supersampled render.
    g = synth.Geometry(200, 200, cell=10.0, roll=12.0, tilt=(35.0, -20.0), focal=500.0, phase=(0.3, 0.2), k1=0.12)
    lin = synth.render(g, synth.Optics(psf=synth.psf_gaussian(0.8, S)), s=S)
    truth = g.corners(12)
    u8 = np.clip(np.rint(lin), 0, 255).astype(np.uint8)
    res = est.run_all(u8, names=["SaddlePoint", "cSP-5"])
    for k, (xy, _c) in res.items():
        j = est.match(xy, truth, 1.0)
        e = xy[j[j >= 0]] - truth[j >= 0]
        print(f"perspective+distortion noise-free {k}: RMS {np.sqrt((e**2).sum(1).mean()):.4f}, mean {e.mean(0)}, matched {(j >= 0).mean():.2f} of {len(truth)}")


if __name__ == "__main__":
    main()
