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

cfg = chess_corners.DetectorConfig.chess_multiscale()
cfg.threshold = chess_corners.Threshold.relative(0.15)
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.forstner()

detector = chess_corners.Detector(cfg)
corners = detector.detect(img)
print(corners.shape, corners.dtype)
print(cfg)
```

`Detector(cfg).detect(image)` returns a NumPy `float32` array of shape
`(N, 7)` with columns:

1. `x` — subpixel corner x in input pixels
2. `y` — subpixel corner y in input pixels
3. `response` — raw detector response at the detected peak
4. `axis0_angle` — angle of the first local grid axis, radians in `[0, π)`
5. `axis0_sigma` — 1σ uncertainty of `axis0_angle`, radians
6. `axis1_angle` — angle of the second local grid axis, radians in
   `(axis0_angle, axis0_angle + π)`
7. `axis1_sigma` — 1σ uncertainty of `axis1_angle`, radians

Rotating CCW from `axis0_angle` toward `axis1_angle` (by less than π)
traverses a **dark** sector of the corner; the two grid axes are **not**
assumed to be orthogonal, so this output correctly captures projective
warp and lens distortion.

Input requirements:

- `image` must be a 2D `uint8` NumPy array with shape `(H, W)`
- it must be C-contiguous

The rows are sorted deterministically by `response` descending, then `x`, then `y`.

## Public config API

`DetectorConfig` is strategy-typed: detector-specific tuning lives
inside a `DetectionStrategy` variant. Top-level fields are
`threshold`, `multiscale`, `upscale`, `orientation_method`, and
`merge_radius`.

```python
cfg = chess_corners.DetectorConfig.chess()  # ChESS, no pyramid
cfg.threshold = chess_corners.Threshold.relative(0.2)
cfg.merge_radius = 3.0

# Enable the coarse-to-fine pyramid (both detectors honour this):
cfg.multiscale = chess_corners.MultiscaleConfig.pyramid(
    levels=3, min_size=128, refinement_radius=3,
)

# Detector-specific knobs live inside the strategy. Nested getters
# return the live shared object, so direct attribute assignment
# propagates back to `cfg` — no rebuild needed:
cfg.strategy.chess.ring = chess_corners.ChessRing.BROAD
cfg.strategy.chess.descriptor_ring = chess_corners.DescriptorRing.FOLLOW_DETECTOR
cfg.strategy.chess.nms_radius = 2
cfg.strategy.chess.min_cluster_size = 2

# Switch the active strategy by assigning a new one:
cfg.strategy = chess_corners.DetectionStrategy.from_radon(
    chess_corners.RadonConfig()
)
```

For one-shot configuration, the chainable `with_chess(**kwargs)` /
`with_radon(**kwargs)` builders return a new config with only the
named fields replaced:

```python
cfg = (
    chess_corners.DetectorConfig.chess_multiscale()
    .with_chess(
        refiner=chess_corners.ChessRefiner.forstner(),
        ring=chess_corners.ChessRing.BROAD,
        nms_radius=2,
    )
)
```

Refiners are per-detector: `ChessRefiner` carries one of
`center_of_mass`, `forstner`, `saddle_point`, or `ml` (with the
`ml-refiner` feature). `RadonRefiner` carries one of `radon_peak` or
`center_of_mass`. The active variant's tuning is reachable via the
`payload` property:

```python
fcfg = chess_corners.ForstnerConfig()
fcfg.max_offset = 2.0
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.forstner(fcfg)

assert cfg.strategy.chess.refiner.kind == "forstner"
assert cfg.strategy.chess.refiner.payload.max_offset == 2.0
```

Tagged classes:

- `Threshold`: `Threshold.absolute(value)` / `Threshold.relative(frac)`;
  read `cfg.threshold.kind` and `cfg.threshold.value`.
- `MultiscaleConfig`: `MultiscaleConfig.single_scale()` /
  `MultiscaleConfig.pyramid(levels=, min_size=, refinement_radius=)`;
  read `cfg.multiscale.kind` and (when `pyramid`) `levels`,
  `min_size`, `refinement_radius`.
- `UpscaleConfig`: `UpscaleConfig.disabled()` /
  `UpscaleConfig.fixed(factor)`; read `cfg.upscale.kind` and (when
  `fixed`) `factor`.
- `ChessRefiner`: `center_of_mass()`, `forstner()`, `saddle_point()`,
  `ml()` (with the `ml-refiner` feature).
- `RadonRefiner`: `radon_peak()`, `center_of_mass()`.

Enums:

- `ChessRing`: `CANONICAL`, `BROAD`
- `DescriptorRing`: `FOLLOW_DETECTOR`, `CANONICAL`, `BROAD`
- `PeakFitMode`: `PARABOLIC`, `GAUSSIAN`
- `OrientationMethod`: `RING_FIT`, `DISK_FIT`

`ChessRing.BROAD` uses the wider radius-10 detector sampling pattern.
Leave `descriptor_ring` at `FOLLOW_DETECTOR` unless you have a reason
to override descriptor sampling separately.

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
cfg = chess_corners.DetectorConfig.chess_multiscale()
text = cfg.to_json(indent=2)
restored = chess_corners.DetectorConfig.from_json(text)

print(restored)
restored.print()
```

If `rich` is installed, `.print()` uses it automatically and the config objects
also expose a Rich render hook.

## Canonical JSON schema

The same algorithm config schema is used by Rust, Python, docs, and the CLI:

```json
{
  "strategy": {
    "chess": {
      "ring": "broad",
      "descriptor_ring": "canonical",
      "nms_radius": 3,
      "min_cluster_size": 1,
      "refiner": {
        "forstner": {
          "radius": 3,
          "min_trace": 20.0,
          "min_det": 0.001,
          "max_condition_number": 60.0,
          "max_offset": 2.0
        }
      }
    }
  },
  "threshold": { "absolute": 0.5 },
  "multiscale": {
    "pyramid": {
      "levels": 3,
      "min_size": 96,
      "refinement_radius": 4
    }
  },
  "upscale": "disabled",
  "orientation_method": "ring_fit",
  "merge_radius": 2.5
}
```

Switch to the Radon strategy by replacing the `strategy` object:

```json
"strategy": {
  "radon": {
    "ray_radius": 4,
    "image_upsample": 2,
    "response_blur_radius": 1,
    "peak_fit": "gaussian",
    "nms_radius": 4,
    "min_cluster_size": 2,
    "refiner": { "radon_peak": {} }
  }
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

If the bindings are built with the `ml-refiner` feature, the ML
pipeline is selected by passing `ChessRefiner.ml()` as the active
variant on the ChESS strategy. The ML refiner runs a small ONNX model
on normalized intensity patches around each candidate.
