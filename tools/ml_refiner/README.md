# ML refiner tools

This folder contains a synthetic data generator for the ML subpixel refiner.
Phase 1 generates grayscale patches with `(dx, dy, conf)` labels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/ml_refiner/requirements.txt
```

`matplotlib` is optional (only needed for `--preview`).

## Generate a dataset

```bash
python tools/ml_refiner/synth/generate_dataset.py \
  --config tools/ml_refiner/configs/synth_v1.yaml \
  --out tools/ml_refiner/data/synth_v1
```

Outputs:

```
tools/ml_refiner/data/synth_v1/
  meta.json
  shard_00000.npz
  shard_00001.npz
  ...
```

Each shard (`npz`) contains:

- `patches`: `uint8` array shaped `[N, P, P]`
- `dx`: `float32` array `[N]`
- `dy`: `float32` array `[N]`
- `conf`: `float32` array `[N]` (confidence in `[0, 1]`)
- `noise_sigma`: `float32` array `[N]`
- `blur_sigma`: `float32` array `[N]`
- `H`: `float32` array `[N, 3, 3]` (homography from canonical plane to patch coords, when enabled)

`meta.json` stores the full config plus a seed for reproducibility.

### Sign convention

The patch is centered on the candidate corner. Labels are the offset from the
candidate to the true corner:

```
true_corner = patch_center + (dx, dy)
```

`dx` is positive to the right, `dy` is positive downward (image coordinates).

## Preview

Save a preview grid (red arrows for `(dx, dy)` plus a green cross at the true corner):

```bash
python tools/ml_refiner/synth/generate_dataset.py \
  --config tools/ml_refiner/configs/synth_v1.yaml \
  --out tools/ml_refiner/data/synth_v1 \
  --preview 16
```

If `matplotlib` is not installed, the preview is skipped with a message.

## Self-test

Quick sanity check:

```bash
python tools/ml_refiner/synth/generate_dataset.py \
  --config tools/ml_refiner/configs/synth_v1.yaml \
  --self-test
```

## Notes

- When homography is enabled, the patch is sampled by inverse-warping a
  canonical corner through a random projective transform; rotation is applied
  in the homography stage.
- Homographies are centered so the canonical origin maps to the patch origin
  before applying the `(dx, dy)` shift.
- The ideal corner is rendered as a smooth saddle pattern with a scale factor
  that adjusts edge sharpness.
- Gaussian blur and noise are sampled per patch; confidence is a deterministic
  function of their strengths.
