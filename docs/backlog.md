# Backlog

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
