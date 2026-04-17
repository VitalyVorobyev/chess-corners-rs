# chess_corners (Python)

Python-first bindings for the `chess-corners` detector.

The installed package is a mixed Rust/Python package:

- `chess_corners` is a pure-Python public API with type hints, docstrings, JSON helpers, and readable config objects.
- `chess_corners._native` is the private PyO3 extension module that runs the detector.

## Quick start

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.ChessConfig.multiscale()
cfg.threshold_value = 0.15
cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

corners = chess_corners.find_chess_corners(img, cfg)
print(corners.shape, corners.dtype)
print(cfg)
```

`find_chess_corners(image, cfg=None)` returns a NumPy `float32` array of shape
`(N, 9)` with columns:

1. `x` — subpixel corner x in input pixels
2. `y` — subpixel corner y in input pixels
3. `response` — raw ChESS response at the detected peak
4. `contrast` — amplitude of the fitted bright/dark structure
5. `fit_rms` — RMS residual of the two-axis intensity fit (gray levels)
6. `axis0_angle` — angle of the first local grid axis, radians in `[0, π)`
7. `axis0_sigma` — 1σ uncertainty of `axis0_angle`, radians
8. `axis1_angle` — angle of the second local grid axis, radians in
   `(axis0_angle, axis0_angle + π)`
9. `axis1_sigma` — 1σ uncertainty of `axis1_angle`, radians

Rotating CCW from `axis0_angle` toward `axis1_angle` (by less than π)
traverses a **dark** sector of the corner; the two grid axes are **not**
assumed to be orthogonal, so this output correctly captures projective
warp and lens distortion.

Input requirements:

- `image` must be a 2D `uint8` NumPy array with shape `(H, W)`
- it must be C-contiguous

The rows are sorted deterministically by `response` descending, then `x`, then `y`.

## Public config API

The public config shape is intentionally flat. There is no `params` section and
no nested `pyramid` object.

```python
cfg = chess_corners.ChessConfig()
cfg.detector_mode = chess_corners.DetectorMode.BROAD
cfg.descriptor_mode = chess_corners.DescriptorMode.FOLLOW_DETECTOR
cfg.threshold_mode = chess_corners.ThresholdMode.RELATIVE
cfg.threshold_value = 0.2
cfg.nms_radius = 2
cfg.min_cluster_size = 2
cfg.pyramid_levels = 3
cfg.pyramid_min_size = 128
cfg.refinement_radius = 3
cfg.merge_radius = 3.0
```

All nested objects are default-initialized, so you can always do:

```python
cfg = chess_corners.ChessConfig()
cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER
cfg.refiner.forstner.max_offset = 2.0
```

Supported enums:

- `DetectorMode`: `canonical`, `broad`
- `DescriptorMode`: `follow_detector`, `canonical`, `broad`
- `ThresholdMode`: `relative`, `absolute`
- `RefinementMethod`: `center_of_mass`, `forstner`, `saddle_point`

`broad` uses the wider, blur-tolerant detector sampling pattern. Leave
`descriptor_mode` at `follow_detector` unless you have a reason to override
descriptor or orientation sampling separately.

## Refiner configuration

`cfg.refiner` always contains every leaf config:

- `cfg.refiner.center_of_mass`
- `cfg.refiner.forstner`
- `cfg.refiner.saddle_point`

Only `cfg.refiner.kind` selects which one is active.

```python
cfg = chess_corners.ChessConfig()
cfg.refiner.kind = chess_corners.RefinementMethod.SADDLE_POINT
cfg.refiner.saddle_point.radius = 3
cfg.refiner.saddle_point.max_offset = 2.0
```

## JSON helpers and printing

Every public config object supports:

- `to_dict()`
- `from_dict(...)`
- `to_json()`
- `from_json(...)`
- `pretty()`
- `print()`

Example:

```python
cfg = chess_corners.ChessConfig.multiscale()
text = cfg.to_json(indent=2)
restored = chess_corners.ChessConfig.from_json(text)

print(restored)
restored.print()
```

If `rich` is installed, `.print()` uses it automatically and the config objects
also expose a Rich render hook.

## Canonical JSON schema

The same algorithm config schema is used by Rust, Python, docs, and the CLI:

```json
{
  "detector_mode": "broad",
  "descriptor_mode": "canonical",
  "threshold_mode": "absolute",
  "threshold_value": 0.5,
  "nms_radius": 3,
  "min_cluster_size": 1,
  "refiner": {
    "kind": "forstner",
    "center_of_mass": {
      "radius": 2
    },
    "forstner": {
      "radius": 3,
      "min_trace": 20.0,
      "min_det": 0.001,
      "max_condition_number": 60.0,
      "max_offset": 2.0
    },
    "saddle_point": {
      "radius": 3,
      "det_margin": 0.002,
      "max_offset": 1.75,
      "min_abs_det": 0.0002
    }
  },
  "pyramid_levels": 3,
  "pyramid_min_size": 96,
  "refinement_radius": 4,
  "merge_radius": 2.5
}
```

Unknown keys are rejected with a clear `ConfigError`.

## Example runners

For a complete Pillow-based example that loads the full config from JSON, run:

```bash
uv run --python .venv/bin/python python crates/chess-corners-py/examples/run_with_full_config.py \
  testimages/mid.png \
  config/chess_algorithm_config_example.json
```

For a complete Pillow-based example that defines the entire config directly in
Python code and only takes the image path as an argument, run:

```bash
uv run --python .venv/bin/python python crates/chess-corners-py/examples/run_with_code_config.py \
  testimages/mid.png
```

Both examples use Pillow only for image loading:

```bash
uv pip install --python .venv/bin/python Pillow
```

## ML refiner

If the bindings are built with the `ml-refiner` feature, the package also exports:

```python
corners = chess_corners.find_chess_corners_with_ml(img, cfg)
```

That toggles the separate ML-backed refinement path. It does not change the
canonical `ChessConfig` schema.
