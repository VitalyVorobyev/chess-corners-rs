# API 2.0 — refinement stage, uncertainty tier, removals

Status: accepted · Workstream: API / UQ · Target: 2.0.0 (breaking) ·
Tasks: `UQ-43`, `UQ-19`, `UQ-06`, Förstner removal.

## 1. Why a major release

Four defects in the 1.x surface cannot be fixed additively:

1. **An ignored field.** `DetectorConfig::noise_sigma` is ignored under
   `Uncertainty::Off`, the default since 1.7.0. The facade already lowers the
   pair to one shape (`CovarianceStage::Reported { noise: NoiseLevel }`).
2. **Two sources of truth for the covariance.** `Corner` and `RefineResult`
   carry a public `covariance_per_noise_variance` field that mirrors the
   private model, and `CovarianceModel::effective` reconciles the two by
   float equality. Writing the field silently drops the systematic term.
3. **Refinement is configured per detector, with the same choice spelled
   twice.** `ChessRefiner::SaddlePoint(SaddlePointConfig)` and
   `RadonRefiner::SaddlePoint(SaddlePointConfig)` are the same stage. A
   refiner choice does not survive switching the detection strategy, and
   the ML refiner is unreachable from Radon seeds although it only reads the
   image.
4. **Dominated and dead surface.** Förstner is dominated by SaddlePoint (more
   accurate on 18 of 19 synthetic conditions and on every real set; real
   floor 0.110 vs 0.053 px). `DenseDetector::refines_on_image` is always
   `true`. `attach_radon_peak_covariance` is a hidden public helper. All
   three are `#[deprecated]` in 1.x (Förstner and the helper since 1.7.0).

## 2. Decisions

### 2.1 Uncertainty carries its noise level

```rust
#[non_exhaustive]
pub enum Uncertainty {
    #[default]
    Off,
    Propagated { noise: NoiseLevel },
}

#[non_exhaustive]
pub enum NoiseLevel {
    /// Estimate σ per frame (`estimate_noise_sigma_u8`).
    #[default]
    Estimate,
    /// Total noise of the 8-bit image, rounding included.
    Known(f32),
}
```

- `DetectorConfig::noise_sigma` and `with_noise_sigma` are removed; a
  `Known(σ)` that is not finite and ≥ 0 is rejected when the `Detector` is
  built (`ChessError::InvalidNoiseSigma`, as today).
- JSON: `"uncertainty": "off"`, `{"propagated": {"noise": "estimate"}}`,
  `{"propagated": {"noise": {"known": 1.2}}}`.
- Python / WASM: `Uncertainty.off()`, `Uncertainty.propagated(noise_sigma=None)`
  (JS `Uncertainty.propagated(sigma?)`).
- C: `cc_config.uncertainty` keeps a tag plus `has_noise_sigma` /
  `noise_sigma`, read only under `CC_UNCERTAINTY_PROPAGATED`. The tags are
  renumbered so a zero-initialised `cc_config` is `Off`, the same as every
  other binding (`OFF = 0`, `PROPAGATED = 1`); `cc_abi_version` bumps.

### 2.2 The covariance model is the only state

`covariance_per_noise_variance` fields and `CovarianceModel::effective` are
removed; `noise_covariance_per_unit_variance()` and `position_covariance(σ)`
read the model. `attach_radon_peak_covariance` and
`DenseDetector::refines_on_image` are removed.

### 2.3 One image-refinement stage

Each detector keeps the subpixel localizer that works on its own response
map; image refinement is one optional, detector-independent stage:

| | Response-map localizer (strategy config) | Image refiner (`DetectorConfig::refiner`) |
|---|---|---|
| ChESS | center of mass (`ChessConfig::center_of_mass`) | `None` (default) · `SaddlePoint(SaddlePointConfig)` · `Ml` |
| Radon | peak fit (`RadonConfig::peak_fit`) | `None` (default) · `SaddlePoint(SaddlePointConfig)` · `Ml` |

```rust
pub struct DetectorConfig { /* … */ pub refiner: Option<ImageRefiner>, /* … */ }

#[non_exhaustive]
pub enum ImageRefiner {
    SaddlePoint(SaddlePointConfig),
    #[cfg(feature = "ml-refiner")]
    Ml,
}
```

- This makes every combination legal by construction: there is no refiner
  that needs a response map it cannot have. `ChessRefiner` and
  `RadonRefiner` are removed.
- Defaults reproduce 1.x output: ChESS with center of mass, Radon with the
  peak fit.
- Seeds and fallbacks stay per detector, as in 1.x, and are documented on
  the stage: an image refiner starts from the ChESS integer peak and drops a
  corner it rejects; on Radon it starts from the peak fit and keeps the peak
  fit when it does not converge. Outputs of every 1.x configuration that
  still exists are bit-identical.
- Core mirrors the split: the center-of-mass localizer is ChESS's,
  `RefinerKind` keeps only image refiners, and the facade lowers
  `DetectorConfig::refiner` in one place.
- `Uncertainty` applies to whichever localizer produced the final position,
  as today.

Rejected: one `DetectorConfig::refiner` enum including `CenterOfMass` and
`PeakFit` (the `UQ-19` wording). Half its variants would be invalid for one
of the two detectors, so it would trade a duplicated choice for an illegal
state checked at run time.

### 2.4 Frame noise σ stays a detector getter

`Detector::detect*` keep returning `Vec<CornerDescriptor>`;
`Detector::last_noise_sigma()` stays. The covariance is opt-in and `Off` by
default, so changing the return type of every detection call for metadata
most callers never read costs more than the temporal coupling of a getter
on an already stateful detector. Python (`Detections.noise_sigma`) and C
(`cc_result`) already return it with the corners.

### 2.5 Orientation σ

Decided by the `UQ-06` evidence (`reference/orientation-sigma.md`): either
one calibrated definition for RingFit and DiskFit, or the field is renamed
to what it is (a fit-quality score). calib-targets-rs gates and weights on
it, so the 2.0 changelog states the scale change and the threshold mapping.

## 3. Versioning while 2.0 is developed

After `v1.7.0` is tagged, `main` is 2.0 development: the first breaking PR
sets the workspace version to `2.0.0` (publishing runs only on tags, and
`cargo-semver-checks` accepts breaking changes against the 1.7.0 baseline
once the major version moves). A 1.7.x fix, if one is needed, is cut from a
`release/1.7` branch at `v1.7.0`.

## 4. Phases

| PR | Scope |
|---|---|
| 1 | Version 2.0.0; remove the deprecated surface (§2.2, Förstner everywhere: core, facade, bindings, schema, book, tools) |
| 2 | §2.1 in every binding, the schema and the book; C tags renumbered |
| 3 | §2.3 in core, facade, every binding, the schema and the book |
| 4 | §2.5 after the evidence |
| 5 | Migration guide (book + `docs/changelog/2.0.0.md`), release |

Each PR runs the full gate set plus Python, WASM and C header checks, and
proves bit-identical positions and covariances for every surviving 1.x
configuration.
