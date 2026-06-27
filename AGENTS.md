# AGENTS.md — chess-corners

This repository contains **four published crates**:

* **`chess-corners`** — the *public*, user-facing API crate (stable surface, ergonomic types).
* **`chess-corners-core`** — a *low-level*, performance-oriented core crate (almost internal; minimal deps; sharper edges).
* **`box-image-pyramid`** — a small standalone crate for fixed 2x grayscale pyramid construction with reusable buffers.
* **`chess-corners-ml`** — an *advanced, optional* crate providing ONNX-based ML refiner inference, reached through the facade's `ml-refiner` feature. Published for callers who opt into ML refinement; it pulls in the ONNX runtime, so it is not part of a default install. Keep its surface narrow and semver-clean.

The remaining crates — **`chess-corners-py`** and **`chess-corners-wasm`** — are **not** published to crates.io (`publish = false`) and carry no crates.io semver contract; they are distributed as a Python wheel and an npm package respectively.

The codebase prioritizes:

* **Determinism** (same inputs → same outputs)
* **Performance** (CPU-friendly; minimal allocations; SIMD-friendly)
* **Correctness & robustness** (blur, glare, low contrast, partial boards)
* **API stability** (small, well-documented public API in `chess-corners`)

If you are an automated agent (Codex, etc.), follow these rules strictly.

**Before large work, read the knowledge base** under
[`docs/`](docs/README.md): `docs/ROADMAP.md` (milestones to `v1.0.0`),
`docs/BACKLOG.md` (task registry), and the per-workstream RFCs plus the
algorithm index in `docs/design/`. Route verbose / mechanical /
long-running work to subagents per
[`docs/process/subagent-workflow.md`](docs/process/subagent-workflow.md).

---

## 1) Layering rules (most important)

### Dependency direction

* `chess-corners` **may depend on** `chess-corners-core` and `box-image-pyramid`.
* `chess-corners-core` **must not depend on** `chess-corners`.
* `box-image-pyramid` **must not depend on** chess-specific crates.

### Where code goes

* **Core algorithms / hot path** → `chess-corners-core`
* **Standalone grayscale pyramid construction / reusable pyramid buffers** → `box-image-pyramid`
* **Convenience wrappers, builders, user-friendly enums, feature gating, docs** → `chess-corners`

### API exposure

* `chess-corners` should re-export only what users need.
* `chess-corners-core` is public but a “sharp tools” crate:

  * fewer stability guarantees; prefer `pub(crate)` where possible
  * minimal dependencies
  * truly-internal items that benches, tests, and the facade still need
    across the crate boundary live under `chess_corners_core::unstable`,
    which carries **no semver guarantee**. `ChessParams` and
    `RefinerKind` live there (the facade re-exports them at
    `chess_corners::low_level`).
* The `DenseDetector` and `CornerRefiner` traits are **sealed** — they
  are not external extension points. Add a detector or refiner in-crate
  (a new variant), never via a downstream `impl`.

---

## 2) Project goals and non-goals

### What this crate does

`chess-corners` detects chessboard **corners** with subpixel accuracy.
It does **not** do board topology / grid fitting, ChArUco decoding, or
camera calibration — those are downstream concerns. The pipeline:

1. **Response** — dense per-pixel cornerness (the ChESS 16-sample ring,
   or the Radon SAT detector), behind the `DenseDetector` trait.
2. **Detection** — threshold → non-maximum suppression → cluster filter
   (Radon adds a 3-point Gaussian peak fit).
3. **Refinement** — one of four built-in subpixel refiners chosen via
   `RefinerKind` (see §6).
4. **Orientation** — two-axis grid-direction fit per corner (`RingFit`
   default, or `DiskFit`) with calibrated per-axis uncertainty.
5. **Descriptors** — the output `CornerDescriptor { x, y, response, axes }`.

An optional coarse-to-fine **multiscale pyramid** and an optional
integer **upscale** stage wrap the per-scale pipeline; both detectors
run under the same orchestrator. See
`docs/design/algorithms-index.md` for the full stage map.

### Goals

* Fast and reliable subpixel chessboard-corner detection.
* A single clear corner-strength score (`response`) plus a two-axis
  orientation estimate with meaningful uncertainty.
* Clear separation between candidate generation, refinement, and
  orientation / descriptor estimation.

### Non-goals (unless explicitly requested)

* Board topology / grid fitting, ChArUco decoding, or full camera
  calibration (downstream concerns).
* Heavy ML dependencies in default builds (`ml-refiner` is opt-in).
* Non-deterministic outputs.
* Adding bulky dependencies to `chess-corners-core`.

---

## 3) Build, test, and quality gates

**Toolchain:** nightly is pinned via `rust-toolchain.toml` (the `simd`
feature needs nightly `portable_simd`). The default/stable build has an
MSRV of **Rust 1.88** (`rust-version` in `Cargo.toml`).

Before opening a PR, run the full gate sequence (CLAUDE.md is canonical):

* `cargo fmt --all --check`
* `cargo clippy --workspace --all-targets --all-features -- -D warnings`
* `cargo test --workspace --all-features`
* `cargo doc --workspace --no-deps --all-features` (broken intra-doc
  links and missing docs on public items are blocking)
* `mdbook build book`

When the bindings change, also:

* Python: `(cd crates/chess-corners-py && maturin develop --release)`
  then `pytest crates/chess-corners-py/python_tests`
* WASM: `wasm-pack build crates/chess-corners-wasm --target web`

Minimal builds are still worth a spot check (`cargo test -p
chess-corners-core`, `cargo test -p chess-corners`). If benches exist
and you touched hot-path code, run the relevant bench target (only if
requested or CI requires it).

**Do not** introduce new warnings. Avoid `#[allow(...)]` unless justified in the PR.

### Python workflow

For Python package work in this repository:

* prefer **`uv`** over raw `pip` / ad hoc virtualenv commands
* use the checked-in **`.venv`** for local Python verification unless told otherwise
* run Python commands through `uv run --python .venv/bin/python ...`
* install Python packages into that venv with `uv pip install --python .venv/bin/python ...`
* build the Python wheel with `uv run --python .venv/bin/python maturin build -m crates/chess-corners-py/Cargo.toml ...`

When updating Python bindings, examples, or docs, verify the **installed wheel**
when practical, not only the source tree import.

---

## 4) Coding conventions

### Determinism

* Avoid nondeterministic iteration ordering in outputs.
* If parallelism is enabled, final output ordering must be deterministic (sort by stable keys).

### Allocations / hot path

* No per-corner heap allocations in refinement/detection loops.
* Reuse scratch buffers (caller-provided scratch structs or internal reusable buffers).
* Prefer stack-fixed small matrices (e.g., nalgebra static sizes) for tiny solves.

### Error handling

* Use `Option` for “reject/invalid/out-of-bounds” in hot paths.
* If diagnostics matter, return a small `Status` enum + score.

### Compatibility

* `chess-corners` is the compatibility boundary. Keep its public API stable.
* `box-image-pyramid` should remain a small, stable API for the fixed 2x grayscale pyramid use case.
* If behavior changes, gate it behind configuration or feature flags.

---

## 5) Performance rules

When modifying detection/refinement:

* Avoid repeated expensive ops in loops (`sqrt`, `atan2`, normalization) unless needed.
* Keep memory access contiguous and cache-friendly.
* If adding SIMD, keep a scalar fallback and ensure correctness matches.

Any change that could affect performance should include at least one of:

* a micro-benchmark, or
* a timing log in tests/examples (behind a feature flag), or
* a clear complexity/perf rationale in the PR description.

---

## 6) Subpixel refinement: required design pattern

Refinement is selected via the `RefinerKind` config enum and dispatched
through the `CornerRefiner` trait in `chess-corners-core`, exposed
ergonomically in `chess-corners`. As of the v1.0 surface,
**`CornerRefiner` is sealed**: downstream crates cannot add their own
refiners. "Pluggable" therefore means *choose among the built-ins* — a
new refiner is added as an in-crate variant, not via a downstream `impl`.

**Trait shape:**

* Input: image / response view + initial point + params (+ optional context like orientation)
* Output: refined point + score + status (accepted/rejected/out-of-bounds/ill-conditioned)

The four built-in refiners:

* **CenterOfMass** (default; response-map centroid)
* **Förstner** (structure-tensor)
* **SaddlePoint** (quadratic Hessian fit)
* **RadonPeak** (paired with the Radon detector)

**Rule:** default settings must reproduce existing behavior unless the user opts in.

Each refiner must:

* define acceptance criteria clearly
* output a meaningful score (used for filtering/ranking)
* avoid heap allocations per call

---

## 7) Testing policy

Every algorithmic change must include tests.

Minimum expectations:

* Unit tests for refiners on synthetic patches with known subpixel offsets.
* Edge-case tests:

  * near image borders
  * low-contrast patches
  * noisy/blurred patches
  * partial data / missing neighbors
* Regression tests to ensure default pipeline output is unchanged (within tolerance).

If you change thresholds/scoring:

* document the rationale and adjust tests accordingly.

---

## 8) Documentation expectations

When adding/changing:

* public types
* configuration params / thresholds
* algorithm behavior

You must update:

* rustdoc for affected items
* README/usage docs (at least in `chess-corners`)
* a minimal example snippet showing how to use the new feature/config

Guidance docs should include:

* when to use which option (trade-offs)
* default values and why they’re chosen

**Public-surface hygiene (critical):** anything that renders in
`cargo doc`, the book, the README, the CHANGELOG, or binding type stubs
describes *what the code does*, not how it got there. No lineage names
(`V1`, `Baseline`, “Phase N”), origin notes, or optimization narratives
for brand-new features. Internal context belongs in commit messages or
non-rendered `.rs` comments. CLAUDE.md is canonical on this; the user
treats violations as a quality issue, so sweep the whole change.

---

## 9) Dependency policy

* `chess-corners-core`: keep dependencies minimal. Avoid heavy crates.
* `chess-corners`: may add ergonomic or optional deps, but prefer feature-gating.

Any new dependency must be justified in the PR description and be license-compatible.

---

## 10) Versioning and releases (practical guidance)

Because multiple crates are published:

* Avoid breaking changes in `chess-corners` without a major bump and clear migration notes.
* `chess-corners-core` may evolve faster, but breaking changes still require semver discipline.
* `box-image-pyramid` is also a published crate; keep its API narrow and semver-clean because other projects may depend on it directly.
* If `chess-corners` re-exports `core` types, consider whether that couples versions tightly.

---

## 11) PR/commit expectations (for agents)

* Keep PRs focused (one feature/fix at a time).
* Include: summary, tests run, and any perf notes.
* If behavior changes: state it explicitly and provide a config/flag or migration notes.

Suggested commit prefixes:

* `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`

---

## 12) If you’re unsure

When trade-offs conflict (speed vs accuracy, stability vs cleanup):

* Preserve correctness + backwards compatibility first.
* Add configuration/feature flags for opt-in behavior.
* Add tests and (if needed) a benchmark to justify the change.
