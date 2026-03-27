}

# AGENTS.md — chess-corners

This repository contains **three published crates**:

* **`chess-corners`** — the *public*, user-facing API crate (stable surface, ergonomic types).
* **`chess-corners-core`** — a *low-level*, performance-oriented core crate (almost internal; minimal deps; sharper edges).
* **`box-image-pyramid`** — a small standalone crate for fixed 2x grayscale pyramid construction with reusable buffers.

The codebase prioritizes:

* **Determinism** (same inputs → same outputs)
* **Performance** (CPU-friendly; minimal allocations; SIMD-friendly)
* **Correctness & robustness** (blur, glare, low contrast, partial boards)
* **API stability** (small, well-documented public API in `chess-corners`)

If you are an automated agent (Codex, etc.), follow these rules strictly.

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
* `chess-corners-core` can remain public, but treat it as “sharp tools”:

  * fewer stability guarantees
  * more `pub(crate)` where possible
  * minimal dependencies

---

## 2) Project goals and non-goals

### Goals

* Fast and reliable chessboard / ChArUco-style corner detection.
* Pluggable subpixel refinement + meaningful corner scoring.
* Clear separation between:

  * candidate generation
  * refinement
  * topology/grid fitting

### Non-goals (unless explicitly requested)

* Heavy ML dependencies in default builds.
* Non-deterministic outputs.
* Adding bulky dependencies to `chess-corners-core`.

---

## 3) Build, test, and quality gates

Before opening a PR, run:

* `cargo fmt --all`
* `cargo clippy --workspace --all-targets --all-features -- -D warnings`
* `cargo test --workspace --all-features`

Also check minimal builds where relevant:

* `cargo test -p chess-corners-core`
* `cargo test -p chess-corners`

If benches exist and you touched hot-path code:

* run the relevant bench target (only if requested or CI requires it)

**Do not** introduce new warnings. Avoid `#[allow(...)]` unless justified in the PR.

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

Refinement must be pluggable behind a trait, implemented in (or primarily used by) `chess-corners-core`,
and exposed ergonomically in `chess-corners`.

**Trait shape (guideline):**

* Input: image view + initial point + params (+ optional context like response/orientation)
* Output: refined point + score + status (accepted/rejected/out-of-bounds/ill-conditioned)

Built-in refiners should include:

* **Center-of-mass** (legacy default; must preserve current results)
* **Förstner**
* **Saddle-point (quadratic fit)**

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
