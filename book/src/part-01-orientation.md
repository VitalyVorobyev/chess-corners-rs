<p align="center">
  <a href="https://vitavision.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="img/vv-favicon-dark.svg">
      <img src="img/vv-favicon-light.svg" alt="vitavision.dev" height="64">
    </picture>
  </a>
</p>

# Part I: Orientation

Part of the [vitavision.dev](https://vitavision.dev/) computer-vision
atlas. Self-contained algorithm overviews live at
[vitavision.dev/atlas/chess-corners](https://vitavision.dev/atlas/chess-corners)
and
[vitavision.dev/atlas/duda-radon-corners](https://vitavision.dev/atlas/duda-radon-corners);
this book is the implementation reference for the Rust workspace.

## 1.1 What the library does

`chess-corners-rs` detects the corners of a chessboard pattern — the
X-junctions where four alternating dark/bright cells meet — to
sub-pixel precision. It is the kind of detector that sits at the
front of a camera calibration, pose estimation, or AR alignment
pipeline.

Two independent detectors and five subpixel refiners live behind a
single configuration type:

- **ChESS response detector** — a ring-based kernel from
  [Bennett & Lasenby (2014)](https://arxiv.org/abs/1301.5491). This is
  the default and the fastest preset in the measured clean-image
  benchmark. Covered in [Part III](part-03-chess-detector.md); atlas
  overview at
  [vitavision.dev/atlas/chess-corners](https://vitavision.dev/atlas/chess-corners).
- **Radon response detector** — a ray-based kernel from Duda &
  Frese (2018). Added for cases where the ChESS ring does not produce
  enough seeds, especially the small-cell, blur, and low-contrast
  fixtures in this repository. Covered in
  [Part IV](part-04-radon-detector.md); atlas overview at
  [vitavision.dev/atlas/duda-radon-corners](https://vitavision.dev/atlas/duda-radon-corners).

Both produce the same `CornerDescriptor` output, and both feed the
same [multiscale pipeline](part-07-multiscale-and-pyramids.md).

Once a detector has produced integer-pixel seeds, one of five
subpixel refiners brings the coordinates under a pixel:
`CenterOfMass`, `Förstner`, `SaddlePoint`, `RadonPeak`, or an
ONNX-backed `ML` refiner. Each is selected through the active
strategy's `refiner` field. The refiners are described in
[Part V](part-05-refiners.md) and benchmarked in
[Part VIII](part-08-benchmarks.md).

The same `DetectorConfig` drives a Rust API, a Python package, a
browser WebAssembly package, and a CLI. They call the same Rust
detector pipeline and use the same configuration schema.

## 1.2 Typical use cases

- Camera calibration (mono, stereo, or multi-camera) with printed or
  screen-displayed chessboard targets.
- Pose estimation of calibration rigs and fixtures.
- Robotics and AR setups where a chessboard is a temporary alignment
  target.
- Offline evaluation of external calibration pipelines: the
  detectors here are deterministic and independent of OpenCV.

Compared with other corner pipelines:

- **Versus generic corner detectors** (Harris, Shi–Tomasi, FAST):
  both detectors here are specialized for chessboard X-junctions
  and reject edges, blobs, and texture that generic detectors
  accept.
- **Versus ID-based markers** (AprilTag, ArUco): this library
  detects unlabeled grid corners. It does not decode an ID, so you
  need to know the board layout separately.

## 1.3 Workspace layout

The library is split across six crates; three are user-facing and
three are implementation detail you only need if you want to go
below the facade.

```
┌─ chess-corners-py      (PyO3 bindings, pip package: chess-corners)
├─ chess-corners-wasm    (wasm-bindgen, npm package: @vitavision/chess-corners)
│                        ▲
│                        │
├─ chess-corners         (high-level Rust API, CLI, multiscale pipeline)
│                        ▲
│                        │
├─ chess-corners-core    (low-level algorithms: ChESS + Radon detectors,
│                         refiners, descriptors)
│
├─ box-image-pyramid     (standalone u8 pyramid builder, fully
│                         independent — no chess-specific coupling)
│
└─ chess-corners-ml      (optional ONNX refiner; gated behind
                          `ml-refiner` feature)
```

Layering rules enforced by CI:

- `chess-corners-core` does not depend on `chess-corners`.
- `box-image-pyramid` has no chess-specific code — it is a reusable
  grayscale pyramid builder that happens to be used here.
- Algorithms go in `chess-corners-core`; the facade crate adds the
  public `DetectorConfig` type, CLI, multiscale wiring, and feature
  gates.

Support directories:

- `config/` — example CLI JSON configs.
- `testdata/` — sample images used by tests, examples, and book plots.
- `tools/` — Python scripts for plotting, benchmarking, and the
  ML refiner training pipeline.
- `docs/` — design notes, proposals, and backlog.
- `book/` — this book (mdBook source under `book/src/`).

## 1.4 The `CornerDescriptor` output

Every detector in the workspace returns `Vec<CornerDescriptor>`.
The type lives in `chess-corners-core` and is re-exported by the
facade. Fields:

| Field           | Type               | Meaning                                                                 |
|-----------------|--------------------|-------------------------------------------------------------------------|
| `x`, `y`        | `f32`              | Subpixel position in input image pixels.                                |
| `response`      | `f32`              | Raw detector response at the peak. Scale is detector-specific.          |
| `axes[0, 1]`    | `[AxisEstimate; 2]`| The two local grid axis directions, each with a 1σ angular uncertainty. |

The two axes are **not** assumed orthogonal — projective warp or
lens distortion tilts the sectors independently, and the fit
recovers both directions. Polarity convention:
`axes[0].angle ∈ [0, π)`, `axes[1].angle ∈ (axes[0].angle, axes[0].angle + π)`,
with the CCW arc from axis 0 to axis 1 crossing a dark sector. Full
details and the fit math are in
[Part III §3.4](part-03-chess-detector.md#34-corner-descriptors).

## 1.5 Where to go next

- To run the detector on an image: [Part II](part-02-using-the-detector.md).
- To understand the ChESS response: [Part III](part-03-chess-detector.md).
- To understand the Radon response: [Part IV](part-04-radon-detector.md).
- To pick a refiner: [Part V](part-05-refiners.md) for algorithms,
  [Part VIII](part-08-benchmarks.md) for measurements.
- Orientation methods (`RingFit` / `DiskFit`): [Part VI](part-06-orientation-methods.md).
- Multiscale pipeline and pyramid tuning: [Part VII](part-07-multiscale-and-pyramids.md).
- To contribute: [Part IX](part-09-contributing.md).

Rust API documentation builds alongside this book and is published
at the same site:

- [`chess-corners` API reference](api/chess_corners/index.html)
- [`chess-corners-core` API reference](api/chess_corners_core/index.html)
- [`box-image-pyramid` API reference](api/box_image_pyramid/index.html)
- [`chess-corners-ml` API reference](api/chess_corners_ml/index.html)

## 1.6 Installation and features

### Rust

```toml
[dependencies]
chess-corners = "0.11"
image = "0.25"      # if you want GrayImage integration
```

Feature flags on the `chess-corners` facade:

| Feature       | Effect                                                                 |
|---------------|------------------------------------------------------------------------|
| `image`       | Default. `image::GrayImage` convenience entry points.                  |
| `rayon`       | Parallelize response computation and multiscale refinement over cores. |
| `simd`        | Portable SIMD for the ChESS kernel. Nightly only.                      |
| `par_pyramid` | SIMD / Rayon acceleration inside the pyramid downsampler.              |
| `tracing`     | Structured `tracing` spans for the detector pipeline.                  |
| `ml-refiner`  | Enable the ONNX-backed refiner (`chess-corners-ml` dependency).        |
| `cli`         | Build the `chess-corners` binary.                                      |

All feature combinations produce the same numerical results;
features only affect performance and observability.

### Python

```bash
python -m pip install chess-corners
```

### JavaScript / WebAssembly

```bash
wasm-pack build crates/chess-corners-wasm --target web
# installs an npm package: chess-corners-wasm
```

### CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

Every surface consumes the same `DetectorConfig` JSON schema. Examples
live under `config/`. The next part walks through the public API in
all four surfaces.
