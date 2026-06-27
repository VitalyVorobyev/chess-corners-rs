# RFC: Public API stabilization for `v1.0.0`

**Status:** Draft / design. **Workstream:** `API-*` (ROADMAP milestone **M3**).
**Current version:** `0.11.2`. **Supersedes:** the `0.10→0.11` audit (its
structural recommendations largely shipped in #60; the open items below are
carried forward).

## Goal

Freeze a **minimal, clear, semver-stable** public surface so that the site,
Python/WASM bindings, and the C++/vcpkg port all target a contract that will
not move under them. After this milestone the workspace tags `1.0.0` and is
governed by strict semver, enforced by `cargo-semver-checks` in CI.

Guiding principle (from `CLAUDE.md`): *if a user needs X to use the API
correctly, X is public; otherwise it is internal.* Two extra score fields
and leaked internal types make the surface harder to learn — tighten them.

## Decisions locked this session

### D1 — Drop `contrast` and `fit_rms` from `CornerDescriptor`

The public descriptor becomes:

```rust
pub struct CornerDescriptor {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub axes: [AxisEstimate; 2],
}
```

Rationale: at `crates/chess-corners-core/src/orientation/descriptor.rs:81-82`
both fields are just re-exposed copies of the fit-internal `fit.amp` / `fit.rms`.
The σ-uncertainty LUT (`U1`, `orientation/ring_fit/uncertainty.rs`) consumes
the **fit-internal** values, *not* the descriptor fields — so removing the
public fields does **not** change detection or uncertainty. A single
`response` score is clearer; three scores that "must not be compared to each
other" is a UX trap.

Blast radius (all must land together — `API-01`):
- `CornerDescriptor` struct + `new()` constructor (`detect/mod.rs:94-136`).
- Python binding: 9-column array → **7 columns** (`crates/chess-corners-py/src/lib.rs`, the `.pyi`, docstrings).
- WASM binding: stride-9 `Float32Array` → **stride-7** (`crates/chess-corners-wasm/src/lib.rs` + TS types).
- CLI JSON `CornerOut` (`crates/chess-corners/bin/commands.rs`).
- Snapshot baseline regeneration (`crates/chess-corners/tests/snapshot_regression.rs`).
- Core test assertion `contrast > 0` (`detect/chess/detect.rs`) — drop or rewrite against the fit internals.

### D2 — C++/vcpkg bindings bind against the **frozen** surface

The C ABI (see [`cpp-vcpkg-bindings.md`](cpp-vcpkg-bindings.md)) maps the
post-D1 `CornerDescriptor`, so M3 must complete before M5.

## Open items carried into v1.0

### API-02 — Deduplicate `nms_radius` / `min_cluster_size`

Both `ChessConfig` and `RadonConfig` carry independent `nms_radius` and
`min_cluster_size` with identical semantics at different nesting levels
(`crates/chess-corners/src/config.rs`). Decide one of:
- lift the shared detection knobs to `DetectorConfig`, or
- keep per-strategy but document that they are the *same* concept.

Recommendation: lift to a shared `DetectionParams` reused by both strategies;
strategy structs keep only strategy-specific knobs (`ray_radius`,
`response_blur_radius`, `peak_fit`, `ring`, `descriptor_ring`).

### API-03 — Hide internal types leaking at the core root

`ChessParams` and `RefinerKind` are public at the `chess-corners-core` root
but are internal translation/dispatch details. Make them `pub(crate)` or move
under a clearly unstable namespace. Audit all `pub use` in `core/src/lib.rs`
and `chess-corners/src/lib.rs` for the same pattern (the 0.11 audit listed
`RING5/RING10`, `SatElem`, `box_blur_inplace`, `fit_peak_frac`,
stage modules `detect::chess::ring`, `orientation::descriptor`, etc.).

### API-04 — `ChessRefiner::Ml` honesty

`ChessRefiner::Ml` (feature-gated) silently translates to
`RefinerKind::CenterOfMass` in core (`config.rs:~636`), breaking the 1:1
refiner mapping. Either honor ML selection in the core path, model it as a
post-refiner step in config, or gate the variant out entirely when
`ml-refiner` is off. Decide before freeze — the variant is a public contract.

### API-05 — Python stub parity

Runtime PyO3 exposes `Detector.config()` and `Detector.apply_config(...)`
that the `.pyi` omits; factory names drift (`single_scale`/`multiscale_preset`
vs Rust `chess`/`chess_multiscale`/`radon`/`radon_multiscale`). Ship complete
`.pyi` stubs and a test that checks runtime methods against the stub. Decide
whether to align factory names (binding-breaking) now or document the mapping.

### API-06 — `#[non_exhaustive]` & sealed-trait policy

Write down and apply, uniformly:
- `#[non_exhaustive]` on every public enum/struct that may grow (configs,
  `CornerDescriptor`, `RefineStatus`, …) so additions stay non-breaking.
- Seal `DenseDetector` and `CornerRefiner` if external implementations are
  **not** a supported extension point (they currently look implementable);
  if they *are* supported, document the stability contract instead.
- State MSRV and the nightly-only `simd` boundary explicitly.

### API-07 — Binding unknown-variant handling

Python and WASM both map unknown enum discriminants to a default
(`OrientationMethod`→`RingFit`, `PeakFitMode`→`Gaussian`). Document this as
intended, or harden to an error. Keep WASM discriminants contiguous and
pinned so reordering is caught.

### API-08 / API-09 — Enforcement & release

- `API-08`: add `cargo-semver-checks` to CI (baseline at the 1.0.0 commit).
- `API-09`: tag `1.0.0` once D1 + API-02..07 land and all gates are green.

## Diagnostics & namespacing (from the 0.11 audit — verify current state)

The audit recommended a `chess_corners::diagnostics` namespace for response
maps / heatmaps and pushing low-level pipeline composition to
`chess-corners-core`. Confirm what shipped in 0.11.0 and finish any remainder
as part of M3 (these are additive and low-risk).

## Phased execution

1. **Phase A (additive, no break):** API-05 stub parity; finish diagnostics
   namespace if incomplete; add `#[non_exhaustive]` where missing (API-06).
2. **Phase B (breaking, batched):** D1 (drop fields), API-02 (config dedup),
   API-03 (hide internals), API-04 (ML honesty). Land together with binding
   updates + snapshot regen.
3. **Phase C (freeze):** API-07 binding policy, API-08 semver-checks in CI,
   docs/migration notes, then API-09 tag `1.0.0`.

## Open decisions (need a call before Phase B)

1. `nms_radius`/`min_cluster_size`: lift to shared params, or keep per-strategy?
2. `ChessRefiner::Ml`: honor in core / post-step / gate out?
3. Seal `DenseDetector`/`CornerRefiner`, or support external impls?
4. Python factory-name alignment now (binding-break) or document the mapping?
5. `chess-corners-ml`: supported advanced crate at 1.0, or deprecate? (It is
   published `0.10.0` while repo docs list three published crates.)

## Verification (per phase)

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
cargo semver-checks check-release         # once API-08 lands
# bindings, when touched:
(cd crates/chess-corners-py && maturin develop --release) && pytest crates/chess-corners-py/python_tests
wasm-pack build crates/chess-corners-wasm --target web
```

---

See [`algorithms-index.md`](algorithms-index.md) for the stages behind these
types and [`../BACKLOG.md`](../BACKLOG.md) for the `API-*` task list.
