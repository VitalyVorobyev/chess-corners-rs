# Backlog

## Architecture (from design review, 2026-03-25)

- [x] **HIGH** Deduplicate ML/non-ML coarse-to-fine paths in `multiscale.rs` (~200 lines duplicated). Extract shared function parameterized by detector trait or closure.
- [x] **HIGH** Fix ML path missing rayon parallelization (`multiscale.rs:515-534`): both `#[cfg(rayon)]` and `#[cfg(not(rayon))]` blocks are identical sequential code (copy-paste bug). ML path now explicitly sequential; dead `#[cfg]` block removed.
- [x] **HIGH** Add `#[non_exhaustive]` to public structs/enums: `ChessParams`, `ResponseMap`, `RefinerKind`, `RefineStatus`, `ChessConfig`, `CoarseToFineParams`, `PyramidParams`, `PyramidLevel`, `Pyramid`.
- [x] Make `ResponseMap` fields private with validated constructor and accessors.
- [x] Add `Roi::new()` with ordering validation; make fields private.
- [x] Unify coordinate representation: `Corner` and `RefineResult` now use `x, y` fields matching `CornerDescriptor`.
- [x] Fix temp file leak in ONNX model extraction: skip writes when files already exist with matching content.
- [x] Document magic numbers in `ForstnerConfig` defaults with literature references.

## ML Refiner

- [ ] Validate ML refiner accuracy on real-world calibration images (currently synthetic-only benchmarks)
- [ ] Integrate ML confidence score into the pipeline (model outputs `conf_logit` but it is currently ignored)
- [ ] Optimize ML inference performance (~23 ms for 77 corners is too slow for real-time use)

## Performance

- [ ] Explore stable Rust SIMD support as an alternative to nightly-only `portable_simd`
- [ ] Batch refinement optimization (vectorize across multiple corners in a single pass)
- [ ] Integrate benchmark suite into CI (currently only manual via `tools/perf_bench.py`)

## Algorithm

- [ ] Adaptive refinement selection: choose refiner per-corner based on local image context (e.g., low contrast vs. high gradient)

## Documentation

- [ ] Add advanced tracing/profiling cookbook examples
- [ ] Create worked `no_std` examples for `chess-corners-core`

## Python Bindings

- [ ] Support batch processing with `PyramidBuffers` reuse across frames (avoid re-allocation per call)
