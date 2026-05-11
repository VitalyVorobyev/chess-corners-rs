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

# Python bindings (dev build) — `maturin develop` does NOT accept a
# pyproject.toml via `-m`; that flag wants a Cargo.toml. Either cd in:
(cd crates/chess-corners-py && maturin develop --release)
# …or pass the manifest path explicitly:
maturin develop --release -m crates/chess-corners-py/Cargo.toml

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
4. **Descriptors** (`core/descriptor.rs`) — Corner descriptors lifted from raw detections via `corners_to_descriptors_with_method`. Carries two-axis orientation, per-axis 1σ uncertainty, contrast, and fit residual.
5. **Orientation methods** (`core/orientation/`) — Detector-agnostic. `RingFit` (default) runs a 16-sample ring Gauss-Newton fit; `DiskFit` is a full-disk crossing-line estimator with a lazy-gate fallback to `RingFit`. Both ChESS and Radon detectors share this stage.

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
- Python bindings: `(cd crates/chess-corners-py && maturin develop --release)`
  plus `pytest crates/chess-corners-py/python_tests` when the Python
  surface changes.
- WASM: `wasm-pack build crates/chess-corners-wasm --target web` when
  the JS-facing API changes.

## Documentation conventions

### Public surface hygiene (CRITICAL)

Public surfaces — anything that appears in `cargo doc`, the book, the
README, the CHANGELOG, or binding type stubs — describe **what the
code does**, not how it got there. The user has rejected this kind of
content multiple times and considers it a quality issue, not a style
preference. Banned from public surfaces:

- **Lineage names** as method/variant/field labels: `Baseline`, `V1`,
  `V6`, `v6a`, `Phase N`, "previously known as X". Public enum
  variants should describe what the method does (`RingFit`, `DiskFit`),
  never its development chronology.
- **Origin notes**: "ports the Python prototype `disk_sector_py`",
  "based on the bench branch", "originally written in NumPy".
- **Optimization narratives for brand-new features**: "10× faster
  than the internal prototype", "Phase 1–5 perf wins delivered". For
  a feature that did not previously ship, internal speedup history
  is invisible to users — drop it from the changelog.
- **Phase / sprint references**: "Phase 1", "Phase 2 blur sweep
  showed".

Rule of thumb: if a user of this API needs to know X to use the API
correctly, X is public. Otherwise X is internal context — it can live
in commit messages, internal comments inside `.rs` files (so long as
they don't render in `cargo doc`), or `tools/orientation_bench/REPORT.md`-style
historical records, but not in user-facing docs.

When the user pushes back on dev-history terms, do not just remove
the called-out example — sweep the entire change for the same pattern.

### Decisive cleanup

When removing a public-API entity (enum variant, function, type),
sweep ALL of:

- Rust enum + dispatcher arms.
- Rust public re-exports (`pub use`) in facade and crate roots.
- Python bindings (`crates/chess-corners-py/src/config.rs`,
  Python tests, `python/chess_corners/__init__.py`).
- WASM bindings (`crates/chess-corners-wasm/src/config.rs` — and its
  parametrised tests; renumber discriminants to stay contiguous if
  no external pin is found).
- README, book chapters, CHANGELOG bullets.
- Bench tooling (`tools/orientation_bench/{variants,runner,status_plots,
  __main__,README}.{py,md}`).
- Test files (delete obsolete coverage, rename function names that
  encoded the dropped lineage).

If a method A is strictly dominated by method B at equivalent cost,
drop A — even if it has been there for a while. Diagnostic-only paths
("kept for comparison") that no longer feed active development go too.

### Book (mdBook)

- Source: `book/src/`. Built output: `book/book/` (gitignored).
- Chapter files follow `part-NN-<topic>.md`. Inserting a chapter
  mid-stream means renaming subsequent files (`git mv`) and updating
  ALL cross-references — every chapter has both prose mentions
  ("see Part VI") and link targets (`part-06-...md`). Use targeted
  search-and-replace and verify with grep before declaring done.
- Images live in `book/src/img/`. Render scripts live as
  `tools/render_*_overlays.py` and emit there. Existing renderers
  (`tools/render_book_overlays.py`, `tools/synthimages.py`) demonstrate
  the convention.
- Math rendering is enabled (`[output.html] mathjax-support = true`
  in `book/book.toml`). Use `\\[ ... \\]` for display math and
  `\\(...\\)` for inline. Inside math blocks, use single-backslash
  LaTeX commands (`\theta_1`, `\tanh`, `\frac{a}{b}`). Do NOT put
  equations inside ```text fences — they look like ASCII output, not
  math.
- **Topical organization, not file-organization.** A concept that is
  shared across multiple components (orientation methods consumed by
  both ChESS and Radon detectors) gets its own chapter, not a
  sub-section under one of the detectors. Default to placing
  cross-cutting concerns in their own Part.
- **Algorithm chapters need motivating illustrations**, not just
  step-listings. The user explicitly rejected formal step-listings
  alone — they want the failure modes that motivate the algorithm
  shown visually first, then the formal description. Build the
  visualization tooling (Python wheel via maturin develop, render
  script, save PNG to `book/src/img/`) rather than skipping the
  illustration.
- Each chapter ends with `---` and a `Next: [Part X](...)` link.

### CHANGELOG hygiene

- `CHANGELOG.md` keeps `[Unreleased]` inline plus a "Past releases"
  link index. Per-version notes live under `docs/changelog/X.Y.Z.md`.
  When cutting a release, move the released `[Unreleased]` content
  into a new file under `docs/changelog/` and add a link to the index.
- User-facing entries describe **user-visible** changes only. Drop
  internal optimization narratives, lineage references, architecture
  details that don't touch the user's API. Brand-new features get
  one bullet describing what they do — their internal architecture
  is not changelog content.

## Subagent-driven workflow

**Read [`docs/subagent-workflow.md`](docs/subagent-workflow.md)
first.** It is the canonical guide for when and how to dispatch work
in this workspace. Two named agents live under `.claude/agents/`:

- **`quick-implementer`** (Sonnet) — specifiable mechanical work:
  re-exports, CLI plumbing, applying a pre-described fix, running
  `cargo fmt/clippy/test/doc`, `pytest`, `wasm-pack build`, bench
  matrix orchestration, JSON-to-table aggregation.
- **`deep-implementer`** (Opus) — work where numerical / geometric
  reasoning is on the critical path, the fix is not specifiable in
  advance, the diff has to be defended, or multiple files / crates
  have to change in coordinated ways.

When in doubt, default to `quick-implementer` if you can write the
brief without phrases like "figure out", "diagnose", or "decide".
Stay in main context only when the decision IS the work, the user is
interactively steering, or the slice is two minutes of editing.

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
