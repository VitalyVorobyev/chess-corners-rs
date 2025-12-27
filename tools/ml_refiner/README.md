# ML refiner tools

This folder contains a synthetic data generator for the ML subpixel refiner.
Phase 1 generates grayscale patches with `(dx, dy, conf)` labels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/ml_refiner/requirements.txt
```

`matplotlib` is optional (only needed for `--preview` and calibration plots).

Optional install:

```bash
pip install matplotlib
```

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
- `is_pos`: `uint8` array `[N]` (1 = corner present, 0 = negative)
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
  --config tools/ml_refiner/configs/synth_v2.yaml \
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
- Negatives can be enabled via `neg` in `synth_v1.yaml` to provide non-corner
  patches with `conf=0` and `is_pos=0`.

## Train a baseline model

```bash
python tools/ml_refiner/train.py --config tools/ml_refiner/configs/train_v2.yaml
```

By default, `device: auto` selects MPS if available, otherwise CPU.
Progress logging is controlled by `log_interval` (train) and
`log_interval_val` (validation) in the config.
Regression loss is computed on positives only using `is_pos`, while confidence
is trained on all samples.

Outputs are stored under `tools/ml_refiner/runs/<timestamp>/` with:

- `config.yaml`
- `metrics.jsonl`
- `model_best.pt` and `model_last.pt`

## Evaluate a checkpoint

```bash
python tools/ml_refiner/eval.py \\
  --config tools/ml_refiner/configs/train_v1.yaml \\
  --checkpoint tools/ml_refiner/runs/<run>/model_best.pt
```

Optional calibration plot:

```bash
python tools/ml_refiner/eval.py \\
  --config tools/ml_refiner/configs/train_v1.yaml \\
  --checkpoint tools/ml_refiner/runs/<run>/model_best.pt \\
  --plot tools/ml_refiner/runs/<run>/calibration.png
```

## Compare ML vs OpenCV cornerSubPix

Requires `opencv-python` (see `tools/ml_refiner/requirements.txt`).

```bash
python tools/ml_refiner/eval/compare_opencv.py \\
  --config tools/ml_refiner/configs/compare_opencv_v1.yaml
```

This writes a JSON summary and optional plots under
`tools/ml_refiner/runs/compare_opencv/<timestamp>/`.

## Export to ONNX

Export a trained checkpoint and generate fixtures:

```bash
python tools/ml_refiner/export_onnx.py \
  --checkpoint tools/ml_refiner/runs/<run>/model_best.pt \
  --out assets/ml/chess_refiner_v1.onnx \
  --patch-size 21 \
  --opset 17 \
  --num-fixtures 64 \
  --fixtures-out assets/ml/fixtures
```

If opset 17 conversion fails, the exporter may fall back to opset 18; check the
exporter output for the actual opset.

I/O contract:

- Input `patches`: float32 `[N,1,P,P]`, normalized from uint8 by `/255.0`, clamped to `[0,1]`.
- Output `pred`: float32 `[N,3]` with `[dx, dy, conf_logit]` (apply sigmoid to get confidence).

Run parity test (requires `onnxruntime`):

```bash
python -m pytest tools/ml_refiner/tests/test_onnx_parity.py
```
