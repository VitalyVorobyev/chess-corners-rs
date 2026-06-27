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

**Status: landed (`API-01`).** Fields removed from the core struct,
`new()` constructor, and the Python / WASM / CLI surfaces; snapshot
debug-dump updated; docs swept.

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

**Status: landed.** Lifted into a shared `#[non_exhaustive]`
`DetectionParams { nms_radius, min_cluster_size }` on
`DetectorConfig.detection`; both strategy configs lost the duplicated
fields. `chess_params()` / `radon_detector_params()` read the shared
values; a `with_detection` builder was added to the Rust facade and the
Python / WASM bindings (with a top-level `detection` JSON/dict object).
The ChESS preset keeps `nms_radius = 2`; the Radon presets keep
`nms_radius = 4`. Snapshot corner counts unchanged.

Both `ChessConfig` and `RadonConfig` carry independent `nms_radius` and
`min_cluster_size` with identical semantics at different nesting levels
(`crates/chess-corners/src/config.rs`). Decide one of:
- lift the shared detection knobs to `DetectorConfig`, or
- keep per-strategy but document that they are the *same* concept.

Recommendation: lift to a shared `DetectionParams` reused by both strategies;
strategy structs keep only strategy-specific knobs (`ray_radius`,
`response_blur_radius`, `peak_fit`, `ring`, `descriptor_ring`).

### API-03 — Hide internal types leaking at the core root

**Status: landed.** `ChessParams` and `RefinerKind` moved off the
`chess-corners-core` crate root into `chess_corners_core::unstable`
(they cannot be `pub(crate)` because the facade, benches, and core
integration tests consume them across the crate boundary). The facade
re-exports both unchanged at `chess_corners::low_level`. The internal
Radon primitives `ANGLES` / `DIR_COS` / `DIR_SIN` / `fit_peak_frac` /
`box_blur_inplace` and the `SatElem` SAT element type are now
`pub(crate)` (no external consumer); the `detect::chess::ring` and
`detect::radon::primitives` stage modules became `pub(crate) mod`
(`orientation::descriptor` was already private). Kept in `unstable`
because in-repo consumers (benches / core tests / facade) still use
them: the ring tables (`RING5` / `RING10` / `ring_offsets` /
`RingOffsets`), `chess_response_u8_scalar`, `chess_response_u8_patch`,
`detect_peaks_from_response{,_with_refine_radius}`,
`find_corners_u8_with_refiner`, `refine_corners_on_image`, and
`MAX_IMAGE_UPSAMPLE`. The facade crate root was already curated in #60
(no buffers / low-level fns leak there); nothing to move.

Original task: `ChessParams` and `RefinerKind` are public at the
`chess-corners-core` root but are internal translation/dispatch
details. Make them `pub(crate)` or move under a clearly unstable
namespace. Audit all `pub use` in `core/src/lib.rs` and
`chess-corners/src/lib.rs` for the same pattern (the 0.11 audit listed
`RING5/RING10`, `SatElem`, `box_blur_inplace`, `fit_peak_frac`,
stage modules `detect::chess::ring`, `orientation::descriptor`, etc.).

### API-04 — `ChessRefiner::Ml` honesty

**Status: landed.** The variant is gated behind `ml-refiner` (since
0.11.0) and the facade routes it to the ONNX path. The silent
`Ml → CenterOfMass` translation was removed from `chess_params`; the ML
path's coarse-level seed detection now falls through to the core default
refiner explicitly, so no public `ChessRefiner::Ml` selection is ever
silently downgraded to a classic refiner. Default-feature builds carry
no `Ml` variant and no dead fallback.

`ChessRefiner::Ml` (feature-gated) silently translates to
`RefinerKind::CenterOfMass` in core (`config.rs:~636`), breaking the 1:1
refiner mapping. Either honor ML selection in the core path, model it as a
post-refiner step in config, or gate the variant out entirely when
`ml-refiner` is off. Decide before freeze — the variant is a public contract.

### API-05 — Python stub parity

**Status: landed.** On inspection the gap had already been closed: the
shipped `.pyi` documents every `Detector` method (`detect`, `config`,
`apply_config`, `radon_heatmap`), and the `DetectorConfig` factory names
already match Rust (`chess` / `chess_multiscale` / `radon` /
`radon_multiscale`). The earlier "drift" note was stale —
`MultiscaleConfig.single_scale()` is the faithful binding of the Rust
`MultiscaleConfig::SingleScale` variant, not a divergent factory, and
`multiscale_preset` does not exist. No factory rename was needed. Added a
dependency-free pytest parity guard (`test_stub_parity.py`) that
introspects the runtime `Detector` methods and asserts each appears in
the stub's `class Detector:` block, so future drift fails CI (pytest
82 → 83).

Original task: runtime PyO3 exposes `Detector.config()` and
`Detector.apply_config(...)` that the `.pyi` omits; factory names drift.
Ship complete `.pyi` stubs and a test that checks runtime methods against
the stub. Decide whether to align factory names or document the mapping.

### API-06 — `#[non_exhaustive]` & sealed-trait policy

**Status: landed.**

- `#[non_exhaustive]` added to `CenterOfMassConfig`, `ForstnerConfig`,
  `SaddlePointConfig` (they were the only refiner configs missing it;
  `RadonPeakConfig` already had it), to the `RingOffsets` enum, and to
  `ChessBuffers` — the one scratch-buffer carrier with a public field
  (`response`), where a future internal scratch field would otherwise
  break external literal construction; `#[non_exhaustive]` keeps that
  additive while leaving `response` readable.
  Already covered: `ChessParams`, `RefinerKind`, `Refiner`,
  `RefineStatus`, `RefineResult`, `RefineContext`, `OrientationMethod`,
  `PeakFitMode`, `Corner`, `AxisEstimate`, `CornerDescriptor`,
  `AxisFitResult`, `RadonDetectorParams`, and every facade config /
  result / error enum. **Deliberately skipped** (noted for the freeze):
  types whose fields are all private so external literal construction is
  already impossible (`ResponseMap`, `ImageView` — private `origin`,
  `Roi`, the runtime refiner structs, `Detector`,
  `DetectorDiagnostics`); the zero-sized sealed detector markers
  `ChessDetector` / `RadonDetector`; and the all-private scratch-buffer
  carriers `RadonBuffers` / `UpscaleBuffers` (built via `Default`, so
  new fields are already additive).
- `DenseDetector` and `CornerRefiner` are **sealed** via a private
  `mod private { pub trait Sealed {} }` supertrait, impl'd only for the
  in-crate types (`ChessDetector` / `RadonDetector`; the four built-in
  refiners plus the `Refiner` dispatcher). The rustdoc on each trait
  documents the seal. To make the `CornerRefiner` seal possible, the
  facade's only external impl — a no-op refiner in the ML path — was
  removed: it was provably equivalent to a direct
  `unstable::detect_peaks_from_response_with_refine_radius` call, so the
  facade now calls that (behaviour-identical, net LOC down).
- MSRV stated: `rust-version = "1.88"` was already set in
  `[workspace.package]`; the crate-root docs and READMEs now state that
  the default/stable build needs Rust ≥ 1.88 and that the `simd` feature
  needs nightly.

Original task: write down and apply, uniformly: `#[non_exhaustive]` on
every public enum/struct that may grow; seal `DenseDetector` /
`CornerRefiner` if external implementations are not a supported
extension point; state MSRV and the nightly-only `simd` boundary.

### API-07 — Binding unknown-variant handling

**Status: landed (documented, not hardened — by design).** The "map
unknown to default" only happens in the **core → binding** direction, as
the forward-compat shim for `#[non_exhaustive]` core enums (a binding
built against an older core still runs against a newer one). Hardening
that path to an error would defeat forward-compat, so it stays a default
and is now documented (WASM crate docs; Python keeps the same shim).
**Caller** input is already strict: Python rejects unknown keys
(`reject_unknown_keys`) and unknown enum members (PyO3 enum validation);
WASM has no free-form options object — config is built through typed
getters/setters and factory constructors, so there is no untyped key to
misread. WASM discriminants were already explicit (`= 0/1/2`); a
`cargo test` now pins all four `#[wasm_bindgen]` numeric enums
(`ChessRing`, `DescriptorRing`, `PeakFitMode`, `OrientationMethod`) so a
reorder — breaking for JS consumers — fails locally (28 → 29 tests).

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

## Decisions (resolved)

1. `nms_radius`/`min_cluster_size` → **lift to a shared `DetectionParams`** reused by both strategies; strategy structs keep only strategy-specific knobs (API-02).
2. `ChessRefiner::Ml` → **feature-gate the variant** (exists only with `ml-refiner`); drop the silent CenterOfMass fallback (API-04).
3. `DenseDetector` / `CornerRefiner` → **seal both** — no external impls, so the trait signatures stay free to evolve post-1.0 (API-06).
4. `chess-corners-ml` → **keep published, documented as advanced** (it has
   no `publish = false`; AGENTS.md was corrected from "three published
   crates / ml is internal" to four, listing ml as advanced/optional).
5. Python factory names → **no change needed** (resolved in API-05): the
   `DetectorConfig` factories already match Rust
   (`chess`/`chess_multiscale`/`radon`/`radon_multiscale`), and
   `MultiscaleConfig.single_scale()` is the faithful binding of
   `MultiscaleConfig::SingleScale`. The earlier drift note was stale.

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
