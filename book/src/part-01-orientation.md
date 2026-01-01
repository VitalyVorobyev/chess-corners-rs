# Part I: Orientation

## 1.1 What is ChESS?

ChESS (Chess-board Extraction by Subtraction and Summation) is a classical, ID‑free detector for chessboard **X‑junction** corners. It is designed specifically for grid‑like calibration targets: black–white checkerboards, Charuco‑style boards, and other high‑contrast grids where four alternating quadrants meet at each corner.

Unlike generic corner detectors (Harris, Shi–Tomasi, FAST, etc.), ChESS is tuned to answer the question:

> “Is this pixel the intersection of a checkerboard grid?”

It does this by sampling a **16‑point ring** around each candidate pixel and combining two kinds of evidence:

- A “square” term that compares opposite quadrants on the ring, rewarding alternating dark/bright quadrants.
- A “difference” term that penalizes edge‑like structures where intensity flips but doesn’t form an X‑junction.
- A consistency term that compares the ring mean to a 5‑pixel cross at the center, discouraging isolated blobs.

These ingredients are combined into a single **ChESS response** value per pixel. Strong positive responses correspond to chessboard‑like corners; everything else is suppressed by thresholding, non-maximum suppression, and a small cluster filter. A 5×5 refinement step is then used to estimate a **subpixel** corner position.

Typical use cases:

- Camera calibration (mono or stereo), where you want robust chessboard corners across a variety of lighting conditions and slight blur.
- Pose estimation of calibration rigs and fixtures.
- Robotics and AR setups where chessboards are used as temporary alignment targets.

Compared to other approaches:

- **Versus generic corner detectors**: ChESS is more selective and produces fewer false corners on edges, blobs, or texture; it is specialized but more reliable when you know you’re looking at a chessboard.
- **Versus ID‑based markers (AprilTags, ArUco)**: ChESS detects unlabeled grid corners; there is no embedded ID and no global pattern decoding. That can be an advantage when you already know the pattern layout (e.g., an 8×6 checkerboard) and just need accurate corners.

The `chess-corners-rs` workspace implements ChESS in Rust with an emphasis on clarity, testability, and high performance (scalar, `rayon`, and portable SIMD paths), plus ergonomic wrappers for common Rust image workflows.

If you are reading this book online via GitHub Pages, the generated
Rust API docs for the crates are also available:

- `chess-corners-core` API reference: see `/chess_corners_core/`
- `chess-corners` API reference: see `/chess_corners/`

---

## 1.2 Project layout

This repository is a small Rust workspace with two main library crates and a CLI:

- `chess-corners-core` – **low-level core**
  - Path: `crates/chess-corners-core`
  - Responsibilities:
    - Compute dense ChESS response maps on 8‑bit grayscale images (`response` module).
    - Run the detector pipeline on a response map (`detect` module): thresholding, NMS, cluster filtering, subpixel refinement.
    - Define the core types:
      - `ChessParams` – tunable parameters for the response and detector (ring radius, thresholds, NMS radius, minimum cluster size).
      - `ResponseMap` – a simple `w × h` `Vec<f32>` wrapper for the response.
      - `CornerDescriptor` – a rich corner description with subpixel position, response, and orientation.
    - Stay lean and portable: `no_std` is supported when the `std` feature is disabled.
  - Intended audience:
    - Users who need maximum control, want to integrate with custom image types, or want to experiment with the ChESS math and detector pipeline directly.

- `chess-corners` – **ergonomic facade**
  - Path: `crates/chess-corners`
  - Responsibilities:
    - Re-export core types (`ChessParams`, `CornerDescriptor`, `ResponseMap`) so you can usually depend on this crate alone.
    - Provide a high-level detector configuration:
      - `ChessConfig` – combines `ChessParams` with multiscale tuning (`CoarseToFineParams`).
    - Implement single-scale and multiscale detection:
      - `find_chess_corners_image` – detect corners from an `image::GrayImage` (when the `image` feature is enabled).
      - `find_chess_corners_u8` – detect corners directly from `&[u8]` buffers.
      - Multiscale internals (`multiscale` module):
        - `CoarseToFineParams` – pyramid + ROI + merge tuning.
        - `find_chess_corners_buff` / `find_chess_corners` – coarse‑to‑fine detector using image pyramids.
      - Pyramid internals (`pyramid` module):
        - `ImageView`, `ImageBuffer` – minimal grayscale image view and buffer.
        - `PyramidParams`, `PyramidBuffers`, `build_pyramid` – reusable image pyramid construction with optional SIMD/`rayon` via the `par_pyramid` feature.
    - Optionally ship a CLI binary for batch runs and visualization.
  - Intended audience:
    - Users who want “just detect chessboard corners” in a Rust image pipeline with minimal boilerplate.
    - Users who want multiscale detection and performance tuning without touching the lowest-level core types.

- `chess-corners` CLI – **command-line tool**
  - Path: `crates/chess-corners/bin`
  - Entrypoints:
    - `bin/main.rs`, `bin/commands.rs`, `bin/logger.rs`
  - Responsibilities:
    - Load images and optional config JSON (`config/chess_cli_config_example.json`).
    - Run single-scale or multiscale detection using the library API.
    - Emit JSON summaries of detected corners and visualization PNG overlays.
    - Optionally emit JSON tracing spans for profiling.
  - Intended audience:
    - Quick experiments and debugging.
    - Pipeline testing without writing Rust code (e.g., from scripts).

There are also supporting directories:

- `config/` – example configs for the CLI.
- `testdata/` – sample images used in tests / experiments.
- `tools/` – helper scripts such as `plot_corners.py` for overlay visualization.

---

## 1.3 Installing and enabling features

### 1.3.1 As a library dependency

The easiest way to use ChESS from your own project is to depend on the `chess-corners` facade crate. In your `Cargo.toml`:

```toml
[dependencies]
chess-corners = "0.3.0"
image = "0.25" # if you want GrayImage integration
```

This gives you:

- High-level `ChessConfig` / `ChessParams`.
- `find_chess_corners_image` for `image::GrayImage`.
- `find_chess_corners_u8` for raw `&[u8]` buffers.
- Access to `CornerDescriptor` and `ResponseMap`.

A minimal single‑scale example with `image`:

```rust
use chess_corners::{ChessConfig, ChessParams, find_chess_corners_image};
use image::io::Reader as ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?
        .decode()?
        .to_luma8();

    let mut cfg = ChessConfig::single_scale();
    cfg.params = ChessParams::default();

    let corners = find_chess_corners_image(&img, &cfg);
    println!("found {} corners", corners.len());
    Ok(())
}
```

If you don’t use `image`, you can work directly with raw buffers:

```rust
use chess_corners::{ChessConfig, ChessParams, find_chess_corners_u8};

fn detect(img: &[u8], width: u32, height: u32) {
    let mut cfg = ChessConfig::single_scale();
    cfg.params = ChessParams::default();

    let corners = find_chess_corners_u8(img, width, height, &cfg);
    println!("found {} corners", corners.len());
}
```

For very advanced use cases (custom image layout, special ROI handling, or integrating your own detector stages), you can depend on `chess-corners-core` directly instead and work with `response` / `detect` modules.

### 1.3.2 Feature flags and performance

Both crates use Cargo features to control performance and diagnostics:

- On `chess-corners-core`:
  - `std` (default) – enable standard library usage. Disable for `no_std` + `alloc`.
  - `rayon` – parallelize response computation over rows.
  - `simd` – enable portable SIMD for the response kernel (requires nightly).
  - `tracing` – add `tracing` spans to response / descriptor paths.

- On `chess-corners`:
  - `image` (default) – enable `image::GrayImage` integration and the image-based helpers.
  - `rayon` – forward `rayon` to the core and parallelize multiscale refinement. Combine with `par_pyramid` to parallelize pyramid downsampling.
  - `simd` – forward `simd` to the core and enable SIMD on the response path (nightly). Combine with `par_pyramid` for SIMD downsampling.
  - `par_pyramid` – opt-in gate for SIMD/`rayon` acceleration inside the pyramid builder.
  - `tracing` – enable tracing in the core and multiscale layers.
  - `ml-refiner` – enable the ONNX-backed ML subpixel refiner entry points.
  - `cli` – build the `chess-corners` binary.

In your own `Cargo.toml`, you can opt into specific combinations:

```toml
[dependencies]
chess-corners = { version = "0.3", features = ["image", "rayon"] }
```

For example:

- **Default, single-threaded** (simplest): no extra features beyond `image`.
- **Multi-core scalar**: add `"rayon"` to accelerate response/refinement on multi-core machines; add `"par_pyramid"` as well to parallelize downsampling.
- **SIMD-accelerated**: enable `"simd"` (and use a nightly compiler) to vectorize the inner loops; add `"par_pyramid"` to apply SIMD to pyramid downsampling too.
- **Tracing-enabled**: enable `"tracing"` and run with `RUST_LOG` / `tracing-subscriber` to capture spans.

All combinations are designed to produce the **same numerical results**; features only affect performance and observability.

### 1.3.3 Building and using the CLI

If you’re working inside this workspace, you can build and run the CLI directly:

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
  run config/chess_cli_config_example.json
```

This will:

- Load the image specified in the config.
- Run single-scale or multiscale detection depending on the `pyramid_levels` and `min_size` settings (with `pyramid_levels <= 1` behaving as single-scale).
- Save JSON output with detected corners and optional PNG overlays.

You can treat the CLI as:

- A quick way to sanity‑check your installation.
- A reference implementation of how to wire up the library APIs.
- A convenient debugging/profiling harness when you tweak configuration or features.

---

This orientation part should give you enough context to know what ChESS is, how this workspace is structured, and how to bring the detector into your own project. In the next parts, we can dive deeper into the core ChESS math, the multiscale implementation, and practical tuning strategies.
