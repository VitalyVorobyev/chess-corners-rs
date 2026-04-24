# Whole-Image Radon Detector — Design Document

## 1. Problem framing

Today every corner passes through the ChESS ring kernel in
`crates/chess-corners-core/src/response.rs`: a pixel is only a
candidate when `R = SR − DR − 16·|μₙ − μₗ|` is strictly positive
(contract in `detect_corners_from_response`). That test presumes a
clearly bimodal 5 px (or 10 px) ring, and it fails silently on heavy
motion blur, strong defocus, low-contrast scenes (intensity range of a
few gray levels), or cells smaller than `~2·ring_radius`. The
`RadonPeakRefiner` in `crates/chess-corners-core/src/refine_radon.rs`
already proves that a localised 4-angle Radon response stays healthy
under those conditions (0.04–0.08 px on noisy/blurry sweeps), but it
only runs on seeds that ChESS has already produced. Hard frames yield
zero seeds, so refinement accuracy is moot. A dedicated whole-image
Radon detector is the missing complement: ChESS remains the cheap
default, Radon rescues frames that ChESS can't seed.

## 2. Algorithm sketch

```
 u8 ─► [upscale] ─► [summed-area tables] ─► [dense 4-angle Radon
        (shared       (one O(N) pass)         R(x,y) = (max_α S_α
         stage)                                     − min_α S_α)²]
                                                    │
                                                    ▼
                               [box-blur r=1] ─► [threshold + NMS +
                                                  cluster filter]
                                                    │
                                                    ▼
                                [3-point Gaussian fit on blurred R]
                                                    │
                                                    ▼
                                [rescale_descriptors_to_input] ─►
                                                 CornerDescriptor
```

`DIR_COS`/`DIR_SIN`, `PeakFitMode`, `fit_peak_frac`, and
`box_blur_inplace` already live in `refine_radon.rs`. M1 lifts them
into a new `crates/chess-corners-core/src/radon.rs` consumed by both
the refiner and the new detector. **New code:** a SAT-backed
`radon_response_u8` producing a dense `ResponseMap`, and a
`detect_corners_from_radon` driver that reuses thresholding/NMS from
`detect.rs` but runs the intrinsic Duda-Frese 3-point fit instead of
dispatching to `CornerRefiner`.

**Integer-box-sum speedup.** The four paper angles are
`{0, π/4, π/2, 3π/4}`. Horizontal/vertical rays resolve to separable
1-D prefix sums; diagonals collapse to integer `(1,±1)` directions on
the upsampled grid and have their own diagonal prefix sums. Each ray
sum becomes O(1) per pixel after O(N) preprocessing, so the dense
response costs ≈ 4 SAT passes + a pointwise `(max−min)²`. ChESS on the
same grid performs 21 random-access gathers per pixel. Expected
wallclock: within ~1.5× of ChESS at `image_upsample=2`, and the
kernel parallelises and SIMD-vectorises more cleanly.

## 3. API design

**Recommendation:** grow `DetectorMode` with a `Radon` variant. Users
already treat `DetectorMode` in `crates/chess-corners/src/config.rs`
as "how do I find candidates", so adding a third kernel slots in
without inventing a new top-level surface. Add a sibling struct
`RadonDetectorConfig` on `ChessConfig` carrying `ray_radius`,
`image_upsample`, `response_blur_radius`, `peak_fit`, `min_response`,
and an optional NMS override (suppression scales with upsample).
`ChessConfig::to_chess_params` still emits threshold/cluster params;
the multiscale dispatcher picks the kernel. When Radon is active,
`RefinerConfig` drives only descriptor computation, not localisation.

A parallel `find_chess_corners_radon` entry point is rejected — it
doubles the facade surface and forces CLI/binding duplication.
Low-level users still get `chess_corners_core::radon::{
radon_response_u8, detect_corners_from_radon }` mirroring the
`response.rs` / `detect.rs` split.

## 4. Dependency & feature-flag plan

The kernel lives in **`chess-corners-core`** next to `response.rs`.
No new workspace dependencies. A `RadonBuffers` struct owns the SAT
and a `ResponseMap` scratch, allocated once per `(w,h)` and reused
across frames — same pattern as `PyramidBuffers` and `UpscaleBuffers`.

Feature-flag behaviour:

- `rayon`: SAT passes use `par_chunks_mut`, exactly like
  `compute_response_parallel`.
- `simd`: the pointwise `(max − min)²` over four precomputed sums is
  trivial 4-lane f32, gated the same way as the existing ChESS SIMD
  branch.
- `tracing`: `#[instrument]` on both public entry points.
- `image` / `ml-refiner` / `par_pyramid`: no interaction.

**Python/WASM:** defer to M3. Because the kernel is selected through
`ChessConfig::detector_mode`, bindings inherit it automatically once
the config enum gains the variant. Only serde plumbing and one smoke
test per binding are needed.

## 5. Test & benchmark plan

**Core unit tests** (next to the new `radon.rs`): SAT identity/ramp/
checkerboard; `radon_response_u8` must equal the per-candidate
response from `refine_radon.rs` pixel-for-pixel on a synthetic patch
(golden test that pins the refactor); detector end-to-end must recover
the `recovers_ideal_subpixel_offset` fixture without a seed.

**Facade integration** (M2): rename/extend
`crates/chess-corners/tests/refiner_benchmark.rs` into a detector
comparison that runs every sweep with both `DetectorMode::Canonical`
and `DetectorMode::Radon`, printing accuracy + throughput side by
side. Add a **ChESS-hostile fixture**: aliased board at 3 px cells,
σ = 2.0 blur, σ = 15 noise, contrast compressed to [90, 165]. Assert
ChESS returns 0 and Radon returns ≥ N − 2 corners. A determinism test
locks bit-identical output across `rayon` on/off.

**Criterion** (`crates/chess-corners-core/benches/radon_response.rs`):
throughput at 640×480 / 1280×720 / 1920×1080 × `image_upsample ∈ {1, 2}`
× `ray_radius ∈ {2, 3}`, compared against `chess_response_u8`.

## 6. Roll-out

- **M1 (minimum shippable):** Extract shared primitives from
  `refine_radon.rs` into `core/src/radon.rs`; add
  `RadonDetectorParams`, `RadonBuffers`, `radon_response_u8`,
  `detect_corners_from_radon`, unit tests, hostile fixture. No facade
  changes — callers reach the new module directly. Ships as a core
  addition only.
- **M2:** `DetectorMode::Radon` + `RadonDetectorConfig` in the facade;
  multiscale integration (Radon at coarse level, `RadonPeakRefiner` at
  base); detector-comparison benchmark; Criterion.
- **M3:** Python/WASM serde variant + one smoke test each; CLI JSON
  example + `book/` update.

## 7. Risks and open questions

Please confirm before I start:

1. `DetectorMode::Radon` variant vs. separate config struct. I picked
   the variant; switch to a separate struct if you expect Radon to
   diverge far from ChESS's knob set.
2. Extracting `refine_radon.rs` primitives into `core/radon.rs` and
   re-importing from the refiner — or duplicate and keep the refiner
   untouched.
3. SAT element type: `i64` (always safe) vs `u32` (more SIMD lanes,
   16 MP cap).

Implementation details I'll decide unless you object: SAT layout
`(w+1)·(h+1)` with zero padding for branchless queries; diagonals
computed via bilinear gather first, promoted to a second SAT only if
benches demand it; default `ThresholdMode::Relative` at ~1 % (Radon's
`(max−min)²` is non-negative, so ChESS's `R > 0` contract doesn't
carry over); default NMS radius = `3·image_upsample`.
