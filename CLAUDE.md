# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chess-corners-rs** is a high-performance Rust implementation of the ChESS (Chess-board Extraction by Subtraction and Summation) corner detector. It detects chessboard corners in images with subpixel accuracy. Includes Python bindings via PyO3/maturin and WebAssembly bindings via wasm-bindgen/wasm-pack.

## Build & Test Commands

```bash
# Build
cargo build --workspace
cargo build --release

# Test (default features)
cargo test --workspace

# Test (all features, requires nightly)
cargo test --workspace --all-features

# Test individual crates
cargo test -p chess-corners-core
cargo test -p chess-corners

# Test specific feature combinations
cargo test -p chess-corners-core --features rayon
cargo test -p chess-corners --no-default-features --features simd    # nightly only

# Lint (stable toolchain)
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Python bindings (dev build)
maturin develop -m crates/chess-corners-py/pyproject.toml

# WASM bindings (build npm package)
wasm-pack build crates/chess-corners-wasm --target web

# Run examples
cargo run -p chess-corners --example single_scale_image -- testimages/mid.png
cargo run -p chess-corners --example multiscale_image -- testimages/large.png

# CLI
cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config_example.json
```

**Toolchain:** Nightly Rust (set in `rust-toolchain.toml`).

## Workspace Architecture

Six crates with strict layering (see AGENTS.md for full rules):

```
chess-corners-py    (PyO3 bindings, module name: chess_corners)
chess-corners-wasm  (wasm-bindgen bindings, npm package: chess-corners-wasm)
       ↓
chess-corners       (High-level facade, multiscale pipeline, CLI)
       ↓
chess-corners-core  (Low-level core: response, detection, refinement)

box-image-pyramid   (Standalone u8 pyramid, 2x box-filter downsample)
chess-corners-ml    (ONNX inference, optional via ml-refiner feature)
```

**Dependency rules:**
- `chess-corners-core` must NOT depend on `chess-corners`
- `box-image-pyramid` is fully independent (zero chess-specific coupling, reusable in other projects)
- Core algorithms go in `core`; convenience wrappers, builders, and feature gating go in the facade

### Core Algorithm Pipeline

1. **Response** (`core/response.rs`) — Dense ChESS response using 16-sample rings
2. **Detection** (`core/detect.rs`) — Thresholding + NMS + cluster filtering
3. **Refinement** (`core/refine.rs`) — Pluggable trait with 3 built-in refiners: CenterOfMass, Förstner, SaddlePoint
4. **Descriptors** (`core/descriptor.rs`) — Corner descriptors with orientation

### Multiscale Pipeline (`chess-corners`)

Coarse-to-fine pyramid detection with reusable `PyramidBuffers` for successive frames. Configured via `ChessConfig` → `CoarseToFineParams` → `PyramidParams`.

### Feature Flags

| Feature | Crate | Effect |
|---------|-------|--------|
| `rayon` | core, facade | Parallel response computation |
| `simd` | core, facade | Portable SIMD (nightly only) |
| `image` | facade (default) | `image::GrayImage` integration |
| `ml-refiner` | facade, py | ML-backed refinement via ONNX |
| `tracing` | core, facade | Structured logging spans |
| `par_pyramid` | facade, box-image-pyramid | SIMD/rayon in pyramid downsampling |
| `cli` | facade | CLI-only deps (clap, anyhow, serde, tracing-subscriber) |

## Key Design Constraints (from AGENTS.md)

- **Determinism:** Same inputs → same outputs. Parallel results must be sorted by stable keys.
- **No per-corner allocations** in hot paths. Reuse scratch buffers.
- **Refiners are pluggable** via trait. Default settings must reproduce existing behavior.
- **Zero warnings:** No new warnings, avoid `#[allow(...)]` unless justified.
- **Commit prefixes:** `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`
- **Dependency policy:** Minimal deps in core; feature-gate optional deps in facade.

## CI Matrix

- OS: ubuntu, windows, macos × Toolchain: stable, nightly
- `fmt` and `clippy` run on stable only
- SIMD tests run on nightly only
- Python wheel smoke test on Linux (manylinux 2014, Python 3.10+)
- WASM build smoke test on Linux (wasm-pack, `--target web`)
- npm publish triggered by `wasm-v*` tags
