"""Why a re-centred Förstner is not as good as the re-centred saddle fit.

1. The numpy single-shot Förstner reproduces the built-in refiner.
2. At the true corner, the Förstner offset u(truth) of the resampled window
   is exactly 0 at integer / half-integer phases (symmetric sampling) and
   non-zero at generic phases (bilinear resampling is not point-symmetric to
   third order).
3. The fixed-point gain -dF/do (eigenvalues of the window-offset Jacobian at
   the truth) shrinks with blur and crosses zero near lens sigma 2 px: the
   fixed point then barely constrains the corner and noise is amplified ~1/gain.
   Implied fixed-point shift = |J^-1 u(truth)|.
4. Coma: per-instance mean shift by board roll (is the asymmetric-PSF bias a
   uniform, absorbable shift?).

Run: .venv-audit/bin/python tools/refiner_robustness/diag_forstner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import estimators as est  # noqa: E402
import run_synth as rs  # noqa: E402
import synth  # noqa: E402


def main() -> None:
    # 1
    g = synth.Geometry(160, 160, cell=10.0, roll=7.0, focal=1e7, phase=(0.37, 0.61))
    for sig in (0.5, 1.0, 2.0):
        img = np.clip(np.rint(synth.render(g, synth.Optics(psf=synth.psf_gaussian(sig, 16)))), 0, 255).astype(np.uint8)
        r = est.run_all(img, ["Forstner", "Forstner-py"])
        a, b = r["Forstner"][0], r["Forstner-py"][0]
        j = est.match(b, a, 0.5)
        d = b[j[j >= 0]] - a[j >= 0]
        same = np.abs(d).max(1) < 1e-4
        print(
            f"1. blur {sig}: numpy vs built-in Förstner identical (<1e-4 px) on {same.mean():.1%} of {len(d)} corners; "
            f"the rest differ by the seed pixel (max {np.abs(d).max():.2f} px): the built-in is seeded at the NMS peak, "
            "the prototype at a rounded refined position"
        )
    # 2
    for sig in (0.5, 1.0, 1.5):
        line = []
        for ph in [(0.0, 0.0), (0.3, 0.0), (0.25, 0.4), (0.5, 0.5)]:
            g = synth.Geometry(161, 161, cell=10.0, roll=0.0, focal=1e9, phase=(ph[0] / 10, ph[1] / 10))
            lin = synth.render(g, synth.Optics(psf=synth.psf_gaussian(sig, 16)))
            t = g.corners(14)
            u, _ = est.forstner_u(lin, t)
            line.append(f"frac {tuple(np.round(t[0] % 1, 2))}: {np.sqrt((u**2).sum(1).mean()):.4f}")
        print(f"2. blur {sig}: |u(truth)| RMS  " + ";  ".join(line))
    # 3
    for sig in (0.5, 1.0, 1.5, 2.0):
        g = synth.Geometry(160, 160, cell=10.0, roll=7.0, focal=1e7, phase=(0.37, 0.61))
        im = np.rint(synth.render(g, synth.Optics(psf=synth.psf_gaussian(sig, 16))))
        t = g.corners(12)
        u, _ = est.forstner_u(im, t)
        h = 0.02
        jx = (est.forstner_u(im, t + [h, 0])[0] - u) / h
        jy = (est.forstner_u(im, t + [0, h])[0] - u) / h
        J = np.stack([jx, jy], 2)
        ev = -np.linalg.eigvals(J).real
        sh = -np.linalg.solve(J, u[..., None])[..., 0]
        print(
            f"3. blur {sig}: gain median {np.median(ev):.2f} [p5 {np.percentile(ev, 5):.2f}, p95 {np.percentile(ev, 95):.2f}], "
            f"|u(truth)| {np.sqrt((u**2).sum(1).mean()):.4f}, implied fixed-point shift {np.sqrt((sh**2).sum(1).mean()):.4f} px"
        )
    # 4
    psf = synth.psf_coma(0.6, 2.0, 30, 16)
    for geom in rs.instances(False):
        lin = synth.render(geom, synth.Optics(psf=psf))
        t = geom.corners(16)
        img = np.clip(np.rint(lin), 0, 255).astype(np.uint8)
        res = est.run_all(img, ["CoM", "SaddlePoint", "Radon", "cSP-11"])
        line = []
        for k, (xy, _) in res.items():
            j = est.match(xy, t)
            e = xy[j[j >= 0]] - t[j >= 0]
            line.append(f"{k} mean {np.round(e.mean(0), 3)} sd {np.round(e.std(0), 3)}")
        print(f"4. coma, roll {geom.roll}: " + "; ".join(line))


if __name__ == "__main__":
    main()
