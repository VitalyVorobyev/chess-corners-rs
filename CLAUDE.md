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
chess-corners-wasm  (wasm-bindgen bindings, npm package: @vitavision/chess-corners)
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
4. **Descriptors** (`core/descriptor.rs`) — Corner descriptors with two-axis orientation (see `fit_two_axes`), per-axis 1σ uncertainty, contrast, and fit residual

### Multiscale Pipeline (`chess-corners`)

Coarse-to-fine pyramid detection with reusable `PyramidBuffers` for successive frames. Configured via `ChessConfig` → `CoarseToFineParams` → `PyramidParams`.

Optional pre-pipeline **integer bilinear upscaling** (`chess_corners::upscale`, configured via `ChessConfig.upscale`) runs ahead of the pyramid for low-resolution inputs where target corners lack the 5 px ring support. Output descriptor coordinates are always rescaled back to the original input pixel frame.

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

## Pre-PR quality gates (mandatory)

**Always run the full gate sequence below and fix every warning/error
before opening or updating a pull request.** CI runs the same checks;
local is faster and avoids noisy review cycles.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
mdbook build book
```

Notes:

- `cargo doc` warnings (broken intra-doc links, missing docs on public
  items, links to private items) are blocking — do not push with them.
- Book rewrites should also be visually inspected: `mdbook serve book`.
- Python bindings: `maturin develop -m crates/chess-corners-py/pyproject.toml`
  plus `pytest crates/chess-corners-py/python_tests` when the Python
  surface changes.
- WASM: `wasm-pack build crates/chess-corners-wasm --target web` when
  the JS-facing API changes.

## Subagent-driven workflow

**Benchmark and algorithm-tuning work** (e.g. orientation-fit studies,
refiner comparisons, large parameter sweeps) is delegated to subagents to
keep the main context clean. The pattern:

1. A coding subagent implements scaffolding / variants and writes a short
   summary back.
2. A reviewer subagent (`algo-review`, `calibration-review`, or
   `perf-architect` depending on the surface) audits the implementation
   independently.
3. A long-running subagent (often background) executes the sweeps and
   renders artifacts (`REPORT.md`, plots, `metrics.json`).
4. Decision gates hand back to the user with the data; the user picks the
   next variant.

See `tools/orientation_bench/README.md` for the current bench surface.

### Context-management discipline

The main agent's context window is the scarcest resource in any non-trivial
session. Treat it like a hot-path budget — every chunk of verbose tool
output (build logs, `cargo test` outputs > 100 lines, parameter sweeps,
`metrics.json` dumps) eats into the budget you need for reasoning. Rules:

- **Delegate verbose-output work.** Anything that produces > ~200 lines
  of output and needs only a verdict back (build verification, `maturin
  develop`, full `cargo test` runs, the orientation Python sweep, ONNX
  parity checks) goes to a subagent that summarises in < 400 words.
  Don't read raw `metrics.json` or build logs in the main thread.
- **Background mode for long-running, non-blocking tasks.** A 10–15 min
  Python sweep, an ONNX export, or a `wasm-pack` build that doesn't
  gate the next step should be `run_in_background: true`. You'll be
  notified when it completes; in the meantime, do other work in the
  main thread or kick off a parallel implementation subagent.
- **Foreground for results you need now.** Implementation subagents
  whose output you want to integrate before the next step (e.g. a
  perf optimization where the next phase depends on the new numbers)
  are foreground.
- **Multiple agents in one message run in parallel.** Three independent
  tasks → one tool message with three `Agent` calls. Don't serialize
  unless there's a real dependency.
- **Model selection.** Default to `sonnet` for mechanical work
  (parallelize a loop with rayon, port a function across modules,
  run a bench, produce a metrics summary, follow a clear plan).
  Reach for `opus` when the task requires nuanced threshold/heuristic
  judgement, contested architectural calls, or holding many trade-offs
  in working memory at once (algorithmic gating decisions, API surface
  design, ambiguous root-cause investigations). The `opus` budget is
  worth it when "wrong answer" costs another full subagent round.
- **Worktree isolation when changes might conflict.** Use
  `isolation: "worktree"` if you spawn two agents that could touch the
  same files. Two agents working on different files in the same repo
  do *not* need worktrees — `cargo` will serialize on the target lock,
  but that's a few seconds of extra wall time, not correctness.
- **Avoid duplicating subagent work.** If a subagent is auditing the
  bench, do not also `grep` through `metrics.json` yourself. If a
  subagent is rewriting `disk_sector.rs`, don't open it in parallel.
  Wait for the summary, then verify lightly (run gates, spot-check
  the diff) — don't re-do the analysis.
- **Subagent prompts must stand alone.** The agent has zero context
  from your conversation. Include file paths, current state, hard
  constraints, validation steps, and the report format you want back.
  Terse prompts produce terse, generic work.
- **Cap report length.** Always tell the subagent "report under N
  words" — 250 for routine work, 400 for nuanced work. Without a cap
  agents tend toward overlong narration.

### Typical bench/perf cadence in this repo

For a 4-phase perf optimization (e.g. the 2026-05 disk-sector work):

1. Main agent does the analysis + Phase 1 (lowest-risk, bit-near-identical
   change) inline so the perf shape is clear.
2. Subsequent phases are delegated:
   - Algorithmic gates / threshold work → `opus` foreground subagent.
   - Mechanical wiring (rayon, feature flags, API surface) → `sonnet`
     foreground subagent.
   - Synthetic accuracy validation (`maturin develop` + `python -m
     orientation_bench sweep`) → `sonnet` background subagent.
3. Main agent integrates, runs the four mandatory gates one final time,
   summarises end-to-end speedup and accuracy verdict, asks the user
   for the next decision.

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
