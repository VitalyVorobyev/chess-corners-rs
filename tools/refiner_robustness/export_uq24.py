"""Export noise-free boards from the independent generator (``synth.py``) for
the Rust uncertainty evaluation (``chess-corners-core``'s ``uncertainty_eval``
tests, which add the noise themselves).

Each scene is written as ``<name>.f32`` (little-endian float32 irradiance in
gray levels, row-major), ``<name>.json`` (``{"w", "h", ...}``) and
``<name>.csv`` (true corners ``x,y``, pixel centres on integers). Name
prefixes select the test that reads them:

- ``cal_*``: Gaussian-PSF grid for calibrating the systematic models
  (cells 8 / 12, rolls 0 / 8 / 22 / 45 deg, PSF sigma 0.3-3.0, two phases).
- ``mc_*``: Monte Carlo noise-scatter scenes.
- ``s0_*``: the sigma0^2 grid (cells 8 / 12, rolls 0 / 8 / 45, PSF 0.5 / 1.5 /
  2.5) plus the robustness audit's PSF and illumination conditions.

Usage (writes to ``tools/refiner_robustness/out/uq24``, gitignored)::

    python tools/refiner_robustness/export_uq24.py [--out DIR] [--jobs N]

then ``UQ24_SYNTH_DIR=<DIR> cargo test -p chess-corners-core --release ... -- --ignored``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth  # noqa: E402

S = 16
SIZE = 128
MARGIN = 12


def geometry(cell: float, roll: float, phase, persp: bool = False) -> synth.Geometry:
    if persp:
        return synth.Geometry(SIZE, SIZE, cell=cell, roll=roll, tilt=(35.0, -25.0), focal=400.0, phase=phase, k1=0.15)
    return synth.Geometry(SIZE, SIZE, cell=cell, roll=roll, tilt=(8.0, -6.0), focal=2000.0, phase=phase)


def optics(spec) -> synth.Optics:
    kind = spec["psf"][0]
    if kind == "gauss":
        psf = synth.psf_gaussian(spec["psf"][1], S)
    elif kind == "disk":
        psf = synth.psf_disk(spec["psf"][1], S, apod=spec["psf"][2], apod_angle=-50.0)
    elif kind == "coma":
        psf = synth.psf_coma(spec["psf"][1], spec["psf"][2], spec["psf"][3], S)
    else:
        raise ValueError(kind)
    o = synth.Optics(psf=psf)
    if "illum" in spec:
        o.illum, o.illum_amp = spec["illum"]
    o.vignette = spec.get("vignette", 0.0)
    return o


def scenes():
    out = []
    rng = np.random.default_rng(24)
    for cell in (8.0, 12.0):
        for roll in (0.0, 8.0, 22.0, 45.0):
            for sigma in (0.3, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0):
                for k in range(2):
                    ph = tuple(rng.uniform(0, 1, 2))
                    out.append((f"cal_c{cell:g}_r{roll:g}_b{sigma:g}_{k}", dict(cell=cell, roll=roll, phase=ph, psf=("gauss", sigma))))
    for cell, roll, sigma in ((10.0, 8.0, 1.0), (10.0, 30.0, 2.0), (8.0, 45.0, 0.6)):
        out.append((f"mc_c{cell:g}_r{roll:g}_b{sigma:g}", dict(cell=cell, roll=roll, phase=(0.3, 0.7), psf=("gauss", sigma))))
    out.append(("mc_disk1.5", dict(cell=10.0, roll=17.0, phase=(0.6, 0.2), psf=("disk", 1.5, 0.0))))
    for cell in (8.0, 12.0):
        for roll in (0.0, 8.0, 45.0):
            for sigma in (0.5, 1.5, 2.5):
                ph = tuple(rng.uniform(0, 1, 2))
                out.append((f"s0_grid_c{cell:g}_r{roll:g}_b{sigma:g}", dict(cell=cell, roll=roll, phase=ph, psf=("gauss", sigma))))
    audit = {
        "defocus_disk": dict(psf=("disk", 1.5, 0.0)),
        "decentred_defocus": dict(psf=("disk", 1.5, 0.6)),
        "coma_mild": dict(psf=("coma", 0.8, 1.0, 30.0)),
        "coma": dict(psf=("coma", 0.6, 2.0, 30.0)),
        "vignetting40": dict(psf=("gauss", 1.0), vignette=0.4),
        "ramp20": dict(psf=("gauss", 1.0), illum=("ramp", 0.2)),
        "wave5": dict(psf=("gauss", 1.0), illum=("wave", 0.05)),
        "wave15": dict(psf=("gauss", 1.0), illum=("wave", 0.15)),
        "perspective": dict(psf=("gauss", 1.0), persp=True),
    }
    for name, spec in audit.items():
        for k, (roll, ph) in enumerate(((5.0, (0.37, 0.61)), (17.0, (0.12, 0.83)), (38.0, (0.71, 0.29)))):
            out.append((f"s0_audit_{name}_{k}", dict(cell=14.0 if spec.get("persp") else 10.0, roll=roll, phase=ph, **spec)))
    return out


def render(job):
    name, spec, outdir = job
    geom = geometry(spec["cell"], spec["roll"], spec["phase"], spec.get("persp", False))
    lin = synth.render(geom, optics(spec), s=S).astype(np.float32)
    truth = geom.corners(MARGIN)
    d = Path(outdir)
    lin.astype("<f4").tofile(d / f"{name}.f32")
    np.savetxt(d / f"{name}.csv", truth, fmt="%.6f", delimiter=",")
    meta = dict(w=SIZE, h=SIZE, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()})
    (d / f"{name}.json").write_text(json.dumps(meta))
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "uq24"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, str(out)) for n, s in scenes()]
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, name in enumerate(ex.map(render, jobs)):
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} {name}", flush=True)
    print(f"wrote {len(jobs)} scenes to {out}")


if __name__ == "__main__":
    main()
