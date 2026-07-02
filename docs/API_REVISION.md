# API Revision — chess-corners-rs workspace (pre-1.0.0 freeze audit)

Working document for the v1.0.0 API-contract audit. Drives the guided
execution phase; edit freely — each change below is executed only after
explicit approval. Delete after the release.

## Summary

Current public surface: **74 substantive items** across 4 published crates
(chess-corners 24 · chess-corners-core 42 · box-image-pyramid 6 ·
chess-corners-ml 2), plus 4 binding surfaces (Python, WASM, C ABI, CLI JSON).

**Verdict:** the surface is in strong shape. The M6 hardening (DEBT-01/02,
API-01..07) already did the heavy lifting: stage modules are private, traits
are sealed, growth-prone types carry `#[non_exhaustive]`, diagnostics live in
an opt-in channel with a documented weaker promise, and the core root is an
explicitly documented low-level contract. What remains is a short tail of
genuine leaks: one invariant-bypassing method with zero consumers, one
lowered-params type nothing public can consume, one orientation helper whose
own docs say it exists for unit tests, one public signature that missed the
DEBT-03 `ImageView` harmonization, a partially-encapsulated boundary type,
and three binding-surface drifts.

**Plan:** 9 changes (5 Rust + 4 bindings) in 3 execution phases, plus 4
non-breaking doc/metadata corrections. Semver impact: **major** — batched
into the single 0.11.2 → 1.0.0 bump per 0.x-mode planning (no deprecation
shims; the user has explicitly authorized clean breaks).

## The intended contract

- **chess-corners (facade):** `Detector` (`new`, `with_default`, `config`,
  `set_config`, `detect`, `detect_u8`, `diagnostics`) · `DetectorConfig` and
  its config family (`DetectionStrategy`, `ChessConfig`, `RadonConfig`,
  `ChessRefiner`, `ChessRing`, `DetectionParams`, `MultiscaleConfig`,
  `UpscaleConfig`) · `ChessError`/`UpscaleError` · result types
  (`CornerDescriptor`, `AxisEstimate`) · re-exported config payloads
  (`CenterOfMassConfig`, `ForstnerConfig`, `SaddlePointConfig`,
  `OrientationMethod`, `PeakFitMode`) · lowering methods `chess_params()` /
  `radon_detector_params()` · upscale stage primitives (`upscale_bilinear_u8`,
  `rescale_descriptors_to_input`, `UpscaleBuffers`) · `diagnostics` module.
- **chess-corners-core:** the documented stage contract — response
  (`chess_response_u8`, `chess_response_u8_patch`, `radon_response_u8`,
  `ResponseMap`, `Roi`, `RadonResponseView`), detection (`find_corners_u8`,
  `detect_corners_from_response(_with_refiner)`,
  `detect_peaks_from_response_with_refine_radius`, `detect_peaks_from_radon`,
  `merge_corners_simple`, `Corner`), refinement (`CornerRefiner` sealed,
  `Refiner`, `RefinerKind`, three backends + configs, `RefineContext/Result/
  Status`, `refine_corners_on_image`), orientation (`describe_corners`,
  `fit_axes_at_point`, `AxisFitResult`, `OrientationMethod`), dense-detector
  abstraction (`DenseDetector` sealed, `ChessDetector`, `RadonDetector`,
  `ChessBuffers`, `RadonBuffers`), params (`ChessParams`,
  `RadonDetectorParams`, `PeakFitMode`), `ImageView`, result types
  (`CornerDescriptor`, `AxisEstimate`).
- **box-image-pyramid:** unchanged (6 items, minimal and coherent).
- **chess-corners-ml:** unchanged surface (`ModelSource`, `MlModel`);
  stays 0.x, honest docs (see N2).

## Consumer workflow — before/after

The facade 90% path is already clean and does not change:

```rust
let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::forstner());
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;   // Vec<CornerDescriptor>, honest Option axes
```

The friction sits at the edges. Orientation-at-a-point, before:

```rust
// raw triple; every other stage fn takes ImageView
let fit = fit_axes_at_point(&img, w, h, cx, cy, 5, OrientationMethod::RingFit);
```

after:

```rust
let view = ImageView::from_u8_slice(w, h, &img).ok_or(...)?;
let fit = fit_axes_at_point(view, cx, cy, 5, OrientationMethod::RingFit);
```

Config mutation, before (two doors, one unvalidated):

```rust
detector.config_mut().upscale = upscale_cfg;   // silently skips validation
detector.set_config(cfg)?;                      // validated
```

after (one door): `set_config` only.

JS Radon configuration, before: `withRadon({responseBlurRadius: 2})` throws
"unexpected option"; after: all four Radon knobs accepted, matching Python.

## Classification

Compact form; tier ∈ {Result, Config, Stage (deliberate low-level), Diag,
Internal}. Only items with a non-"keep" target are detailed.

| Item(s) | Tier | Target |
|---|---|---|
| Facade: `Detector` + methods (minus `config_mut`), config family, `ChessError`, `UpscaleError`, `CornerDescriptor`, `AxisEstimate`, refiner configs, `OrientationMethod`, `PeakFitMode` | Result/Config | keep |
| Facade: `upscale_bilinear_u8`, `rescale_descriptors_to_input`, `UpscaleBuffers` | Stage | keep (documented hand-composition surface; benched) |
| Facade: `diagnostics` module, `Detector::diagnostics` | Diag | keep (correctly channeled, weaker promise documented) |
| **Facade: `Detector::config_mut`** | Internal | **remove** (R1) |
| **Facade: `CoarseToFineParams`, `DetectorConfig::coarse_to_fine_params()`** | Internal | **demote to `pub(crate)`** (R2) |
| Core: response/detection/refinement/orientation/dense/params contract (see above) | Stage | keep (documented deliberate contract, book-taught, sealed where needed) |
| **Core: `fit_axes_from_samples`** | Internal | **demote to `pub(crate)`** (R3) |
| **Core: `fit_axes_at_point` signature** | Stage | **take `ImageView` instead of `(img, w, h)`** (R4) |
| **Core: `ImageView` fields** | Stage | **`pub(crate)` fields + public accessors** (R5) |
| box-image-pyramid: all 6 items | Result/Stage | keep |
| chess-corners-ml: `ModelSource`, `MlModel` | Stage | keep; docs corrected (N2); stays 0.x |
| **WASM: `singleScale()`** | back-compat shim | **remove** (B1) |
| **WASM: `withRadon` options** | Config | **add `responseBlurRadius`/`peakFit`** (B2) |
| **Python: `with_radon(refiner=…)` dead key** | back-compat leftover | **reject with error** (B3) |
| **C ABI: reduced flat `cc_config`** | Config | **extend to full `DetectorConfig` parity** (B4) |

## Leak deep-dives

### R1 — `Detector::config_mut` (facade `detector.rs:106-119`)
Zero consumers anywhere in the workspace (not even tests). Hands out
`&mut DetectorConfig`, bypassing the upscale validation that `Detector::new`
and `set_config` enforce — its own doc comment warns callers away from
itself. Classic debugging-session `pub`. **Fix: remove.** Migration: use
`set_config` (validated, drops ML state identically).

### R2 — `CoarseToFineParams` + `coarse_to_fine_params()` (facade)
DEBT-02 exposed three lowering methods. Two lower onto core types with
public stage functions (`chess_params()` → CLI/WASM/examples consume it;
`radon_detector_params()` likewise). The third produces
`CoarseToFineParams`, consumed **only** by the facade's private `multiscale`
module — nothing public accepts the value. This is exactly the incoherence
DEBT-01 removed in core (params public, stages private → dead API).
**Fix: demote both to `pub(crate)`.** Migration: none possible — no
external use was ever functional.

### R3 — `fit_axes_from_samples` (core `orientation/api.rs:96`)
Own doc: "convenient for unit tests that don't want to construct a real
image." Zero consumers outside in-crate unit tests. Additionally
astonishing: asked for `DiskFit`, it silently computes `RingFit`.
**Fix: demote to `pub(crate)`.** In-crate tests keep working.

### R4 — `fit_axes_at_point(img: &[u8], w, h, …)` (core `orientation/api.rs:61`)
DEBT-03 collapsed `(img, w, h)` triples into `ImageView` on the internal hot
paths but this public fn kept the raw triple — the only stage fn that did.
Also carries a dev-history rationale in its public doc ("Public so the
orientation benchmark can drive the fit directly"). **Fix: signature takes
`ImageView<'_>`; reword doc to the consumer-facing contract.** Same pass
rewords `AxisFitResult`'s "public mirror of the (crate-private) TwoAxisFit".

### R5 — core `ImageView` partial encapsulation (core `imageview.rs`)
`data`/`width`/`height` are `pub`, `origin` is `pub(crate)` with checked
constructors "preserving invariants". Halfway encapsulation: external code
cannot literal-construct (good) but can mutate `view.width` on its `Copy`
and make `sample()` index out of bounds. The type carries a real invariant
(`data.len() == w*h`) and coordinate-frame semantics — per
make-illegal-states-unrepresentable it should not expose mutable fields.
**Fix: fields → `pub(crate)` (core internals keep direct access), add
`data()`, `width()`, `height()` accessors beside the existing `origin()`.**
Migration: field reads become accessor calls.
*(box-image-pyramid's `ImageView`/`ImageBuffer` stay plain-data: they are
honest carriers whose only method re-checks the invariant.)*

### Considered and kept (with rationale)
- **`ImageView` name collision (core vs box-image-pyramid):** different
  semantics (core's carries an origin/coordinate frame + sampling; pyramid's
  is a plain borrow). Unifying would couple the deliberately independent
  pyramid crate. The facade bridges in 3 lines. Keep both.
- **`RefinerKind` vs `Refiner` vs concrete refiners:** clean config/runtime
  split (serializable selector → scratch-owning dispatcher → backends), all
  `#[non_exhaustive]`/sealed. SOLID-02 stays parked; freezing this shape is
  safe.
- **`detect_corners_from_response` vs `_with_refiner`:** config-driven vs
  instance-driven (scratch reuse) — both earn their place; docs get a
  cross-reference in the N-pass.
- **`find_corners_u8`, `merge_corners_simple`, dense-detector types, Radon
  types:** documented stage contract, book-taught, facade/example consumers.

## Breaking-change plan

All breaking changes ship together in 1.0.0 (0.x mode → batch; no shims).
Order below keeps the workspace green after every individual change.

### Phase R — Rust surface (semver: major)
- **R1** remove `Detector::config_mut` — breaking — migrate to `set_config`.
- **R2** demote `CoarseToFineParams` + `coarse_to_fine_params()` to
  `pub(crate)`; drop root re-export — breaking — no migration (no consumers).
- **R3** demote `fit_axes_from_samples` to `pub(crate)` — breaking — no
  migration (no consumers).
- **R4** `fit_axes_at_point` takes `ImageView<'_>`; doc rewording — breaking
  — wrap slice via `ImageView::from_u8_slice`.
- **R5** encapsulate core `ImageView` fields behind accessors — breaking —
  `view.data` → `view.data()` etc.

### Phase B — Binding surfaces (semver: major for npm/PyPI)
- **B1** WASM: remove deprecated `singleScale()` — breaking (JS) — use
  `DetectorConfig.chess()` / `multiscale` setters.
- **B2** WASM: `withRadon` accepts `responseBlurRadius`/`peakFit` — additive.
- **B3** Python: `with_radon(refiner=…)` raises instead of silently
  ignoring — breaking only for code that was already broken.
- **B4** C ABI: extend `cc_config` to full `DetectorConfig` parity —
  `merge_radius`, upscale factor, ChESS ring selection, Radon geometry
  (`ray_radius`, `image_upsample`, `response_blur_radius`, `peak_fit`) —
  ABI-breaking (struct layout) — bump `CC_ABI_VERSION`; presets must mirror
  the Rust preset defaults exactly; header/hpp/CMake/parity/example/book
  Part IX all move in lockstep.

### Phase N — Non-breaking corrections (docs/metadata)
- **N1** facade `lib.rs:156-161`: stale "(N, 7) array" Python description →
  `Detections` SoA.
- **N2** chess-corners-ml `lib.rs:1-7`: false "not published to crates.io"
  claim → honest support-crate statement (published, independently versioned
  0.x, surface follows the facade's `ml-refiner` needs).
- **N3** CLAUDE.md architecture section: `corners_to_descriptors_with_method`
  → `describe_corners` (name drift).
- **N4** CHANGELOG `[Unreleased]` + Migration-to-1.0.0 section: entries for
  R1–R5, B1–B3.

## Resolved questions (Gate 1, 2026-07-02)

- **Q1 (R5):** include — `ImageView` fields become `pub(crate)` with public
  accessors.
- **Q2 (C ABI):** user chose **full parity** — `cc_config` gains the
  missing knobs at 1.0 (change B4 above) rather than freezing the reduced
  surface.
- Plan approved as written; execution is per-change via subagents, each
  gated by the workspace quality gates. CHANGELOG entries land as one
  coherent pass in N4 (avoids parallel edits).
