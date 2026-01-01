# chess_corners (Python)

Python bindings for the `chess-corners` detector.

## Quick start

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.ChessConfig()
cfg.threshold_rel = 0.2
cfg.min_cluster_size = 1

corners = chess_corners.find_chess_corners(img, cfg)
print(corners.shape, corners.dtype)
```

You can also configure the detector via nested config objects:

```python
cfg = chess_corners.ChessConfig()
cfg.params.threshold_rel = 0.2
cfg.multiscale.pyramid.num_levels = 2
```

## What `find_chess_corners` returns

`find_chess_corners(image, cfg=None)` returns a NumPy `float32` array of shape
`(N, 4)` with columns:

1. `x` – subpixel x coordinate (pixels, image space)
2. `y` – subpixel y coordinate (pixels, image space)
3. `response` – ChESS response strength at the corner
4. `orientation` – local grid axis orientation in radians, in `[0, pi)`

The rows are sorted deterministically by the binding:
`response` (descending), then `x`, then `y`.

Input requirements:
- `image` must be a 2D `uint8` NumPy array with shape `(H, W)`
- it must be C-contiguous (non-contiguous arrays raise `ValueError`)

## ChessConfig parameters

`ChessConfig` mirrors the most important Rust settings. Defaults match the
Rust `ChessConfig::default()`.

Response / detector parameters (`cfg.*`):
- `use_radius10` (bool, default `False`)
  - Use the larger ring radius (r=10) instead of r=5 for response computation.
- `descriptor_use_radius10` (Optional[bool], default `None`)
  - Override the descriptor sampling ring radius; when `None`, uses `use_radius10`.
- `threshold_rel` (float, default `0.2`)
  - Relative threshold as a fraction of the max response.
- `threshold_abs` (Optional[float], default `None`)
  - Absolute threshold; when set, it overrides `threshold_rel`.
- `nms_radius` (int, default `2`)
  - Non-maximum suppression radius in pixels.
- `min_cluster_size` (int, default `2`)
  - Minimum count of positive neighbors in the NMS window to accept a corner.

Multiscale parameters (`cfg.*`):
- `pyramid_num_levels` (int, default `3`)
  - Number of pyramid levels (including base). Set to `1` for single-scale.
- `pyramid_min_size` (int, default `128`)
  - Minimum dimension to keep building the pyramid.
- `refinement_radius` (int, default `3`)
  - Coarse-level ROI radius used for coarse-to-fine refinement.
- `merge_radius` (float, default `3.0`)
  - Merge near-duplicate refined corners within this radius (pixels).

## Full configuration structs

The Python bindings expose all configuration structs directly. You can use the
nested API for clarity:

- `ChessConfig.params` → `ChessParams`
- `ChessConfig.multiscale` → `CoarseToFineParams`
- `CoarseToFineParams.pyramid` → `PyramidParams`

All fields listed above are available in the nested structs as well.

## Classical refiners

Select the classic refiner via `ChessParams.refiner`:

```python
cfg = chess_corners.ChessConfig()
cfg.params.refiner = chess_corners.RefinerKind.forstner(
    chess_corners.ForstnerConfig()
)
```

Available configs:
- `CenterOfMassConfig`
- `ForstnerConfig`
- `SaddlePointConfig`

## ML refiner

If the bindings are built with the `ml-refiner` feature, an ML-backed refiner
is available via `find_chess_corners_with_ml`:

```python
cfg = chess_corners.ChessConfig()
corners = chess_corners.find_chess_corners_with_ml(img, cfg)
```

The ML refiner uses built-in defaults and ignores the model’s confidence
output in the current version.

## Development

```bash
maturin develop -m crates/chess-corners-py/pyproject.toml
```
