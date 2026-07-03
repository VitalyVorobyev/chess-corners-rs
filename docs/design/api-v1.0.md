# RFC: Public API stabilization for `v1.0.0`

**Status:** Implemented (M3/M6, 2026-06). **Workstream:** `API-*` (ROADMAP
milestone **M3**), with the M6 `DEBT-*` coherence pass. **Supersedes:** the
`0.10→0.11` audit. Per-task status lives in [`../BACKLOG.md`](../BACKLOG.md)
`API-*`; this doc keeps the design rationale and the resolved decisions.

## Goal

Freeze a **minimal, clear, semver-stable** public surface so that the site,
Python/WASM bindings, and the C++/vcpkg port all target a contract that will
not move under them. After this the workspace tags `1.0.0` under strict semver,
enforced by `cargo-semver-checks` in CI.

Guiding principle (from `CLAUDE.md`): *if a user needs X to use the API
correctly, X is public; otherwise it is internal.*

## D1 — the frozen `CornerDescriptor`

`contrast` and `fit_rms` were dropped (`API-01`). The public descriptor is:

```rust
pub struct CornerDescriptor {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub axes: Option<[AxisEstimate; 2]>,
}
```

`axes` is `None` when the per-corner orientation fit is opted out via
`DetectorConfig::without_orientation()`.

**Rationale.** Both dropped fields were re-exposed copies of the fit-internal
`fit.amp` / `fit.rms`. The σ-uncertainty LUT consumes the *fit-internal*
values, not the descriptor fields — so removing them changes neither detection
nor uncertainty. A single `response` score is clearer than three scores that
"must not be compared to each other."

**As shipped across bindings:** the Python surface returns a `Detections`
structure-of-arrays (`.xy`, `.response`, `.angles`, `.sigmas`; `.angles` /
`.sigmas` are `None`, not sentinel-filled, when orientation is off — `PY-02`),
not the positional array originally sketched here. WASM keeps a flat, stride-7
`Float32Array` (axis values `NaN` when orientation is off).

## D2 — C++/vcpkg binds the frozen surface

The C ABI (see [`cpp-vcpkg-bindings.md`](cpp-vcpkg-bindings.md)) maps the post-D1
`CornerDescriptor`, so M3 completes before M5.

## Decisions that govern future work

- **Config dedup (`API-02`).** `nms_radius` / `min_cluster_size` are a single
  shared `#[non_exhaustive]` `DetectionParams` on `DetectorConfig.detection`;
  strategy structs keep only strategy-specific knobs (`ray_radius`,
  `response_blur_radius`, `peak_fit`, `ring`). One concept, one place.
- **Internal-type leak (`API-03` → `DEBT-01/02`).** Hiding
  `ChessParams`/`RefinerKind` in `chess_corners_core::unstable` proved
  incoherent — the root-public `chess_response_u8` / `find_corners_u8` /
  `detect_corners_from_response` *require* `ChessParams`, so the stable contract
  was uncallable without a "no-semver" type. M6 deleted `unstable` and the
  facade `chess_corners::low_level`, promoted the genuinely-needed types to the
  documented crate root, and demoted the rest (ring tables, Radon primitives,
  scalar reference) to `pub(crate)`. **Lesson: don't gate a type behind an
  unstable namespace while a stable entry point requires it.**
- **`ChessRefiner::Ml` honesty (`API-04`).** The variant exists only under
  `ml-refiner`; the silent `Ml → CenterOfMass` fallback was removed. A public
  refiner selection is never silently downgraded.
- **`#[non_exhaustive]` + sealed traits (`API-06`).** Every public enum/struct
  that may grow carries `#[non_exhaustive]`; types whose fields are all private
  (external literal construction already impossible) are deliberately skipped.
  `DenseDetector` and `CornerRefiner` are **sealed** via a private `Sealed`
  supertrait — external impls are not a supported extension point, so the trait
  signatures stay free to evolve post-1.0. MSRV: stable ≥ 1.88; `simd` =
  nightly.
- **Unknown-variant handling (`API-07`).** "Map unknown → documented default"
  happens **only** in the core→binding direction, as the forward-compat shim for
  `#[non_exhaustive]` core enums (an older binding runs against a newer core).
  Caller input stays strict (Python rejects unknown keys / members; WASM has no
  free-form options object). WASM numeric-enum discriminants are pinned by a
  `cargo test` so a reorder — breaking for JS — fails locally.
- **Python factory names (`API-05`).** No rename needed: the `DetectorConfig`
  factories already match Rust (`chess` / `chess_multiscale` / `radon` /
  `radon_multiscale`) and `MultiscaleConfig.single_scale()` faithfully binds
  `MultiscaleConfig::SingleScale`. A stub-vs-runtime parity guard test prevents
  future drift.
- **`chess-corners-ml` stays published**, documented as advanced/optional (no
  `publish = false`).

## Diagnostics namespace

Opt-in `chess_corners::diagnostics` (response maps via `chess_response_u8`,
Radon heatmaps via `radon_heatmap_u8`/`radon_heatmap_image`) plus a
detector-bound `Detector::diagnostics()` accessor, each with an explicitly
weaker stability promise than `Detector::detect`. Low-level pipeline composition
lives in `chess-corners-core` (see `DEBT-02`).

## Enforcement & release

- `API-08`: `cargo-semver-checks` in CI, advisory against `v0.11.2` until 1.0.0
  is the baseline; flips to blocking at `API-09`.
- `API-09`: the release act — tag `1.0.0` and publish (deferred by choice).

---

See [`algorithms-index.md`](algorithms-index.md) for the stages behind these
types and [`../BACKLOG.md`](../BACKLOG.md) for the `API-*` task list.
