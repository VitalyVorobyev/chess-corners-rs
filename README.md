# chess-corners-rs

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)

Fast Rust implementation of the classical
[ChESS](https://arxiv.org/abs/1301.5491) chessboard corner detector.

![](book/src/img/mid_chess.png)

This workspace publishes three crates:

- `chess-corners`: ergonomic public API
- `chess-corners-core`: lower-level detector primitives
- `box-image-pyramid`: standalone fixed-2x grayscale pyramid builder

It also ships:

- a CLI binary
- a Python package (`chess_corners`)
- a WebAssembly package (npm: `@vitavision/chess-corners`) for browser usage
- an optional ML-backed refinement path

## Diligence statement

This project is developed with AI coding assistants as implementation tools.
The author validates algorithmic behavior and numerical results and enforces
quality gates (`fmt`, `clippy`, tests, docs, Python checks) before release.

## Rust quick start

```rust
use chess_corners::{ChessConfig, RefinementMethod, find_chess_corners_image};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();

let mut cfg = ChessConfig::multiscale();
cfg.threshold_value = 0.15;
cfg.refiner.kind = RefinementMethod::Forstner;

let corners = find_chess_corners_image(&img, &cfg);
println!("found {} corners", corners.len());
```

## Flat public config

The public API intentionally uses one canonical flat algorithm config.
The high-level Rust `ChessConfig`, the Python `ChessConfig`, the CLI JSON files,
and the docs all use the same field names:

- `detector_mode`
- `descriptor_mode`
- `threshold_mode`
- `threshold_value`
- `nms_radius`
- `min_cluster_size`
- `refiner`
- `pyramid_levels`
- `pyramid_min_size`
- `refinement_radius`
- `merge_radius`

Only `refiner` stays nested because it is a real subdomain:

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

`detector_mode = "canonical"` is the narrow default. Use `"broad"` for the
more blur-tolerant response mode. `descriptor_mode` can either follow the
detector or override descriptor/orientation sampling explicitly.

Shared sample config:

- `config/chess_algorithm_config_example.json`

CLI sample configs:

- `config/chess_cli_config_example.json`
- `config/chess_cli_config_example_ml.json`

## Python quick start

The Python package is a mixed Rust/Python package with a pure-Python public API
and a private native module.

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.ChessConfig.multiscale()
cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

corners = chess_corners.find_chess_corners(img, cfg)
print(corners.shape)
print(cfg)
```

Every public Python config object supports:

- `to_dict()` / `from_dict(...)`
- `to_json()` / `from_json(...)`
- `pretty()`
- `print()`

## JavaScript / WebAssembly

The `chess-corners-wasm` crate provides browser-ready bindings via `wasm-bindgen`.
The published npm package is **`@vitavision/chess-corners`** (renamed from
the unscoped `chess-corners-wasm` in 0.7.0; the legacy package is deprecated
on npm). Build locally with [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

Or install from npm:

```bash
npm install @vitavision/chess-corners
```

Use in a web app:

```js
import init, { ChessDetector } from '@vitavision/chess-corners';

await init();
const detector = new ChessDetector();
detector.set_pyramid_levels(3);

// From a canvas (webcam frame, loaded image, etc.)
const imageData = ctx.getImageData(0, 0, width, height);
const corners = detector.detect_rgba(imageData.data, width, height);

// corners is a Float32Array with stride 9 per corner:
//   [x, y, response, contrast, fit_rms,
//    axis0_angle, axis0_sigma, axis1_angle, axis1_sigma, ...]
for (let i = 0; i < corners.length; i += 9) {
    console.log(`(${corners[i].toFixed(2)}, ${corners[i+1].toFixed(2)})`);
}

// Raw response map for heatmap visualization
const response = detector.response_rgba(imageData.data, width, height);
```

See `crates/chess-corners-wasm/README.md` for the full API reference.

## CLI

Run the detector from JSON:

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
  run config/chess_cli_config_example.json
```

The CLI composes the canonical algorithm fields with app-level fields such as
`image`, `output_json`, `output_png`, `log_level`, and `ml`.

## Python example

The repository includes a Pillow-based end-to-end example:

```bash
uv run --python .venv/bin/python python crates/chess-corners-py/examples/run_with_full_config.py \
  testimages/mid.png \
  config/chess_algorithm_config_example.json
```

## Corner descriptor

Every detection is a `CornerDescriptor` carrying:

- `x`, `y` — subpixel position in input pixels.
- `response` — raw, unnormalized ChESS score `R = SR − DR − 16·MR`.
  The default detector accepts any `R > 0`, matching the paper.
  `R` is linear in 8-bit pixel values, roughly bounded by
  `[−24·255, 8·255]`; it is data-dependent and not comparable across
  scenes.
- `contrast` — bright/dark amplitude recovered by a parametric
  two-axis tanh fit, in gray levels. Independent from `response`.
- `fit_rms` — RMS residual of that fit, in gray levels.
- `axes[0]`, `axes[1]` — the two local grid axes with per-axis 1σ
  angular uncertainty. Axes are **not** assumed orthogonal, so the
  descriptor captures projective warp faithfully. The axis-0 angle
  lives in `[0, π)`; axis 1 lies in `(axis0, axis0 + π)`, with the
  CCW arc between them spanning a dark sector.

The two-axis fit is a 4-parameter Gauss–Newton solve; the per-axis
`sigma` is the standard Cramér–Rao angle uncertainty from the
residual-scaled inverse of the final `JᵀJ`. See
[Part III of the book](book/src/part-03-core-chess-internals.md) for
the full derivation and algorithm.

## Notes on layering

- `chess-corners` is the stable public Rust API.
- `chess-corners-core` remains a lower-level sharp tool and still exposes
  `ChessParams`, `RefinerKind`, and detector internals.
- `box-image-pyramid::PyramidParams` remains a standalone lower-level type.
- The Python package is intentionally Python-native; it does not mirror every
  Rust struct one-to-one.
