"""Render the orientation-σ study grid with the independent generator
``tools/refiner_robustness/synth.py`` (supersampled, perspective, shading).

Usage: python gen_synth.py OUT_DIR [--quick]

Writes ``OUT_DIR/img/*.png``, ``OUT_DIR/truth/*.csv`` (``x,y,a1,a2``: corner
position and the two true local axis directions in radians, the image of the
board's u and v directions at that corner) and ``OUT_DIR/list.tsv`` for
``cargo run --release -p chess-corners --example orientation_sigma_study --
files OUT_DIR/list.tsv out.csv``. Everything is synthetic; OUT_DIR belongs
outside the repository (or in a gitignored directory).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "refiner_robustness"))
import synth as S  # noqa: E402

CELLS = (6.0, 10.0, 20.0)
BLURS = (0.5, 1.0, 1.5, 2.5)
ROLLS = (0.0, 15.0, 30.0, 45.0)
# (label, tilt (rx, ry) deg, focal as a multiple of the image width)
VIEWS = (("fronto", (0.0, 0.0), 25.0), ("persp", (40.0, 25.0), 1.2))
NOISES = (0.0, 1.0, 2.0, 4.0, 8.0)
SHADING = (("ramp", dict(illum="ramp", illum_amp=0.35)),
           ("wave", dict(illum="wave", illum_amp=0.2)),
           ("vignette", dict(vignette=0.5)))
SUPERSAMPLE = 12


def truth(geom: S.Geometry, margin: float = 10.0):
    """Corners with the local images of the board u and v directions."""
    hm = geom.homography()
    inv = np.linalg.inv(hm)
    box = np.array([[0, 0, 1], [geom.w, 0, 1], [0, geom.h, 1], [geom.w, geom.h, 1]], float) @ inv.T
    uv = box[:, :2] / box[:, 2:3]
    lo, hi = np.floor(uv.min(0)) - 2, np.ceil(uv.max(0)) + 2
    uu, vv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    uv = np.stack([uu.ravel(), vv.ravel()], 1)

    def img(p):
        q = np.column_stack([p, np.ones(len(p))]) @ hm.T
        ok = q[:, 2] > 0
        x = np.full((len(p), 2), np.nan)
        x[ok] = geom.distort(q[ok, :2] / q[ok, 2:3])
        return x

    eps = 1e-4
    x = img(uv)
    du = img(uv + [eps, 0]) - img(uv - [eps, 0])
    dv = img(uv + [0, eps]) - img(uv - [0, eps])
    a1 = np.arctan2(du[:, 1], du[:, 0])
    a2 = np.arctan2(dv[:, 1], dv[:, 0])
    keep = (np.isfinite(x).all(1) & (x[:, 0] >= margin) & (x[:, 0] <= geom.w - 1 - margin)
            & (x[:, 1] >= margin) & (x[:, 1] <= geom.h - 1 - margin))
    return np.column_stack([x[keep], a1[keep], a2[keep]])


def job(args):
    out, cell, blur, roll, view, shading = args
    label, tilt, focal_mult = view
    size = int(np.clip(cell * 10, 96, 200))
    geom = S.Geometry(w=size, h=size, cell=cell, roll=roll, tilt=tilt,
                      focal=focal_mult * size, phase=(0.31, 0.17))
    sh_label, sh_kw = shading
    optics = S.Optics(psf=S.psf_gaussian(blur, SUPERSAMPLE), **sh_kw)
    lin = S.render(geom, optics, s=SUPERSAMPLE)
    t = truth(geom)
    rows = []
    stem = f"c{cell:g}_b{blur:g}_r{roll:g}_{label}_{sh_label}"
    tpath = out / "truth" / f"{stem}.csv"
    np.savetxt(tpath, t, delimiter=",", header="x,y,a1,a2", comments="", fmt="%.6f")
    shear = 0 if label == "fronto" else 1
    for noise in NOISES:
        for seed in (1, 2):
            if noise == 0 and seed > 1:
                continue
            rng = None if noise == 0 else np.random.default_rng(zlib.crc32(f"{stem}_{noise}_{seed}".encode()))
            im = S.sensor(lin, S.Sensor(noise=noise), rng)
            ipath = out / "img" / f"{stem}_n{noise:g}_s{seed}.png"
            cv2.imwrite(str(ipath), im)
            tag = f"synth,{cell:g},{blur:g},{roll:g},{shear},{noise:g},{sh_label},{seed}"
            rows.append(f"{ipath}\t{tpath}\t{tag}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    a = ap.parse_args()
    (a.out / "img").mkdir(parents=True, exist_ok=True)
    (a.out / "truth").mkdir(parents=True, exist_ok=True)
    none = ("none", {})
    jobs = [(a.out, c, b, r, v, none) for c, b, r, v in itertools.product(CELLS, BLURS, ROLLS, VIEWS)]
    jobs += [(a.out, 10.0, b, 15.0, v, sh) for b, v, sh in itertools.product((1.0, 2.5), VIEWS, SHADING)]
    if a.quick:
        jobs = jobs[:3]
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(job, jobs) for r in rs]
    (a.out / "list.tsv").write_text("\n".join(rows) + "\n")
    print(f"{len(jobs)} renders, {len(rows)} images -> {a.out}")


if __name__ == "__main__":
    main()
