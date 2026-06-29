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
cfg.threshold = 60.0  # ChESS: absolute floor on the raw response (default 30)
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.forstner()

detector = chess_corners.Detector(cfg)
det = detector.detect(img)
print(det.xy.shape, det.xy.dtype)  # (N, 2) float32
if det.angles is not None:
    print(det.angles.shape)        # (N, 2) float32
print(cfg)
```

`Detector(cfg).detect(image)` returns a `Detections` object with named
arrays:

- `det.xy` — `(N, 2)` float32, subpixel corner positions (x, y) in input pixels
- `det.response` — `(N,)` float32, raw detector response at each peak
- `det.angles` — `(N, 2)` float32, `[axis0_angle, axis1_angle]` in radians `[0, π)`, or `None` when orientation is disabled
- `det.sigmas` — `(N, 2)` float32, 1σ uncertainty per axis in radians, or `None` when orientation is disabled

Rotating CCW from `axis0_angle` toward `axis1_angle` (by less than π)
traverses a **dark** sector of the corner; the two grid axes are **not**
assumed to be orthogonal, so this output correctly captures projective
warp and lens distortion.

The orientation fit is the dominant per-corner cost, and it is
optional. A pipeline that recovers board geometry from corner
*positions* alone can skip it with `cfg.without_orientation()`; in that
case `det.angles` and `det.sigmas` are `None`.

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
cfg.threshold = 60.0  # plain float; ChESS = absolute response floor (default 30), Radon = fraction of per-frame max (default 0.01)
cfg.merge_radius = 3.0

# Enable the coarse-to-fine pyramid (both detectors honour this):
cfg.multiscale = chess_corners.MultiscaleConfig.pyramid(
    levels=3, min_size=128, refinement_radius=3,
)

# Detector-specific knobs live inside the strategy. Nested getters
# return the live shared object, so direct attribute assignment
# propagates back to `cfg` — no rebuild needed:
cfg.strategy.chess.ring = chess_corners.ChessRing.BROAD
cfg.detection.nms_radius = 2
cfg.detection.min_cluster_size = 2

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
    )
    .with_detection(nms_radius=2, min_cluster_size=2)
)
```

Refiners are per-detector: `ChessRefiner` carries one of
`center_of_mass`, `forstner`, `saddle_point`, or `ml` (with the
`ml-refiner` feature). The Radon detector uses its built-in Gaussian
peak fit (`PeakFitMode`); it does not expose a pluggable refiner.
The active `ChessRefiner` variant's tuning is reachable via the
`payload` property:

```python
fcfg = chess_corners.ForstnerConfig()
fcfg.max_offset = 2.0
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.forstner(fcfg)

assert cfg.strategy.chess.refiner.kind == "forstner"
assert cfg.strategy.chess.refiner.payload.max_offset == 2.0
```

Tagged classes:

- `MultiscaleConfig`: `MultiscaleConfig.single_scale()` /
  `MultiscaleConfig.pyramid(levels=, min_size=, refinement_radius=)`;
  read `cfg.multiscale.kind` and (when `pyramid`) `levels`,
  `min_size`, `refinement_radius`.
- `UpscaleConfig`: `UpscaleConfig.disabled()` /
  `UpscaleConfig.fixed(factor)`; read `cfg.upscale.kind` and (when
  `fixed`) `factor`.
- `ChessRefiner`: `center_of_mass()`, `forstner()`, `saddle_point()`,
  `ml()` (with the `ml-refiner` feature).

Enums:

- `ChessRing`: `CANONICAL`, `BROAD`
- `PeakFitMode`: `PARABOLIC`, `GAUSSIAN`
- `OrientationMethod`: `RING_FIT`, `DISK_FIT`; disable the fit entirely
  with `cfg.without_orientation()` (then `det.angles` and `det.sigmas`
  are `None`)

`ChessRing.BROAD` uses the wider radius-10 detector sampling pattern.
Descriptors always sample at the detector ring radius.

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
  "threshold": 60.0,
  "detection": { "nms_radius": 3, "min_cluster_size": 1 },
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

Switch to the Radon strategy by replacing the `strategy` object and setting the shared detection params:

```json
{
  "strategy": {
    "radon": {
      "ray_radius": 4,
      "image_upsample": 2,
      "response_blur_radius": 1,
      "peak_fit": "gaussian"
    }
  },
  "detection": { "nms_radius": 4, "min_cluster_size": 2 }
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
