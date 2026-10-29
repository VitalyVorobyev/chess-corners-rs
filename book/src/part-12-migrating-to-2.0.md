# Part XII: Migrating from 1.x to 2.0

2.0 removes the parts of the 1.x API that let a configuration say one
thing and do another, and the refiners that a better one dominates.
Positions are unchanged: every 1.x configuration that still exists in
2.0 gives the same corners and covariances, bit for bit. The axis-fit
score is the one output whose values change (§12.6).

This chapter lists every breaking change with before and after code
for each binding. Most programs need only §12.2 and §12.3.

| Change | Affects | What to do |
|---|---|---|
| [§12.2](#122-the-uncertainty-setting-carries-its-noise-level) The noise level moved into `Uncertainty` | Code that reports position covariance | Pass σ inside `Uncertainty::Propagated` |
| [§12.3](#123-one-image-refinement-stage) One image-refinement stage | Code that picks a refiner | Move the refiner to `DetectorConfig::refiner` |
| [§12.4](#124-the-förstner-refiner-is-removed) Förstner refiner removed | Förstner users | Use the saddle-point refiner |
| [§12.5](#125-the-ml-refiner-is-removed) ML refiner removed | ML refiner users, `ml-refiner` builds | Use the saddle-point refiner; drop the feature |
| [§12.6](#126-orientation-reports-fit_std_error) `sigma` → `fit_std_error` | Code that reads or gates on axis scores | Rename; rescale a gate by about ×0.65 |
| [§12.7](#127-configuration-files) Stricter configuration files; CLI nests the detector | Saved JSON configs, CLI config files | Rewrite the moved keys; move detector keys under `"detector"` |
| [§12.8](#128-c-and-c-abi-version-8) C ABI version 8 | C and C++ | Rebuild; update renamed tags and fields |
| [§12.9](#129-chess-corners-core) Core accessors and helpers | Direct `chess-corners-core` users | Use the accessor methods |

Update the dependency first:

```toml
[dependencies]
chess-corners = "2.1"
```

```bash
python -m pip install --upgrade "chess-corners>=2,<3"
npm install @vitavision/chess-corners@2
```

## 12.1 What did not change

- Detection, the presets, `threshold`, `detection`, `multiscale`,
  `upscale`, `orientation_method` and `merge_radius`.
- Corner positions and position covariances for every configuration
  that maps across.
- `Uncertainty::Off` is the default, as in 1.7.
- `Detector::last_noise_sigma()` (Python `Detections.noise_sigma`, JS
  `ChessDetector.noiseSigma()`, C `cc_result.noise_sigma`).

## 12.2 The uncertainty setting carries its noise level

A noise level only matters when a covariance is reported, so it now
lives inside that setting. In 1.x a `noise_sigma` next to
`Uncertainty::Off` was silently ignored; in 2.0 it cannot be written.

`Uncertainty` is `Off` or `Propagated { noise }`, and `noise` is
`NoiseLevel::Estimate` (estimate σ from each image, as 1.x did without
`noise_sigma`) or `NoiseLevel::Known(σ)`. `DetectorConfig::noise_sigma`
and `with_noise_sigma` are removed. A `Known(σ)` that is negative or not
finite is still rejected by `Detector::new`
(`ChessError::InvalidNoiseSigma`).

Rust:

```rust,ignore
// 1.x
let cfg = DetectorConfig::chess().with_uncertainty(Uncertainty::Propagated);
let cfg = DetectorConfig::chess()
    .with_uncertainty(Uncertainty::Propagated)
    .with_noise_sigma(1.2);

// 2.0
use chess_corners::{DetectorConfig, NoiseLevel, Uncertainty};

let cfg = DetectorConfig::chess().with_uncertainty(Uncertainty::Propagated {
    noise: NoiseLevel::Estimate,
});
let cfg = DetectorConfig::chess().with_uncertainty(Uncertainty::Propagated {
    noise: NoiseLevel::Known(1.2),
});
```

Python: `Uncertainty` is an immutable value instead of an enum.

```python
# 1.x
cfg.uncertainty = Uncertainty.PROPAGATED
cfg.noise_sigma = 1.2                      # or cfg.with_noise_sigma(1.2)

# 2.0
cfg.uncertainty = Uncertainty.propagated()                 # estimate per image
cfg.uncertainty = Uncertainty.propagated(noise_sigma=1.2)  # known σ
cfg = cfg.with_uncertainty(Uncertainty.propagated(noise_sigma=1.2))
```

Replace `Uncertainty.OFF` with `Uncertainty.off()`. To test the
setting, read `cfg.uncertainty.kind` (`"off"` or `"propagated"`) and
`cfg.uncertainty.noise_sigma`.

JavaScript: likewise a value, and `noiseSigma` / `withNoiseSigma` are
gone from `DetectorConfig`.

```js
// 1.x
cfg = cfg.withUncertainty(Uncertainty.Propagated).withNoiseSigma(1.2);

// 2.0
cfg = cfg.withUncertainty(Uncertainty.propagated());     // estimate per image
cfg = cfg.withUncertainty(Uncertainty.propagated(1.2));  // known σ
```

`Uncertainty.Off` becomes `Uncertainty.off()`; read
`cfg.uncertainty.kind` and `cfg.uncertainty.noiseSigma`.

JSON configuration:

```json
{ "uncertainty": "propagated" }
{ "uncertainty": "propagated", "noise_sigma": 1.2 }
```

becomes

```json
{ "uncertainty": { "propagated": { "noise": "estimate" } } }
{ "uncertainty": { "propagated": { "noise": { "known": 1.2 } } } }
```

`"uncertainty": "off"` is unchanged. C and C++ are in §12.8.

## 12.3 One image-refinement stage

In 1.x each detector had its own refiner enum, `ChessRefiner` and
`RadonRefiner`, and the same saddle-point choice was spelled once per
detector. A refiner did not survive switching detectors. In 2.0 each
detector keeps the localizer that works on its own response map, and
image refinement is one optional stage that works with either detector:

| | Detector's own localizer (strategy config) | Image refiner (`DetectorConfig::refiner`) |
|---|---|---|
| ChESS | center of mass, `ChessConfig::center_of_mass` | `None` (default) or `ImageRefiner::SaddlePoint(_)` |
| Radon | peak fit, `RadonConfig::peak_fit` | `None` (default) or `ImageRefiner::SaddlePoint(_)` |

`ChessRefiner`, `RadonRefiner`, `ChessConfig::refiner` and
`RadonConfig::refiner` are removed. The mapping:

| 1.x | 2.0 |
|---|---|
| `ChessRefiner::CenterOfMass(cfg)` (the ChESS default) | `refiner: None` (the default); tune with `ChessConfig::center_of_mass = cfg` |
| `RadonRefiner::PeakFit` (the Radon default) | `refiner: None` (the default) |
| `ChessRefiner::SaddlePoint(cfg)` | `refiner: Some(ImageRefiner::SaddlePoint(cfg))` |
| `RadonRefiner::SaddlePoint(cfg)` | `refiner: Some(ImageRefiner::SaddlePoint(cfg))` |
| `ChessRefiner::Forstner(_)` | removed, see §12.4 |
| `ChessRefiner::Ml` | removed, see §12.5 |

What a refiner does is unchanged: on ChESS it replaces the center of
mass, starts from the integer peak and drops a corner it rejects; on
Radon it starts from the peak fit and keeps the peak fit where it does
not converge ([Part V](part-05-refiners.md)).

Rust:

```rust,ignore
// 1.x
let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::saddle_point());
let cfg = DetectorConfig::radon().with_radon(|r| r.refiner = RadonRefiner::saddle_point());
let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::SaddlePoint(sp));
let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::CenterOfMass(com));
let cfg = DetectorConfig::radon().with_radon(|r| r.refiner = RadonRefiner::PeakFit);

// 2.0
use chess_corners::{DetectorConfig, ImageRefiner};

let cfg = DetectorConfig::chess().with_refiner(ImageRefiner::saddle_point());
let cfg = DetectorConfig::radon().with_refiner(ImageRefiner::saddle_point());
let cfg = DetectorConfig::chess().with_refiner(ImageRefiner::SaddlePoint(sp));
let cfg = DetectorConfig::chess().with_chess(|c| c.center_of_mass = com);
let cfg = DetectorConfig::radon().without_refiner(); // also the default
```

`sp` is a `SaddlePointConfig` and `com` a `CenterOfMassConfig`, both
unchanged.

Python:

```python
# 1.x
cfg.strategy.chess.refiner = ChessRefiner.saddle_point()
cfg = DetectorConfig.radon().with_radon(refiner=RadonRefiner.saddle_point())
cfg.strategy.chess.refiner = ChessRefiner.center_of_mass(com)
cfg.strategy.radon.refiner = RadonRefiner.peak_fit()

# 2.0
cfg.refiner = ImageRefiner.saddle_point()          # or ImageRefiner.saddle_point(sp)
cfg = DetectorConfig.radon().with_refiner(ImageRefiner.saddle_point())
cfg.strategy.chess.center_of_mass = com
cfg = cfg.without_refiner()                         # or cfg.refiner = None
```

JavaScript:

```js
// 1.x
cfg = cfg.withChessRefiner(ChessRefiner.withSaddlePoint(sp));
cfg = cfg.withRadonRefiner(RadonRefiner.withSaddlePoint(sp));
cfg = cfg.withChessRefiner(ChessRefiner.withCenterOfMass(com));
cfg = cfg.withRadonRefiner(RadonRefiner.withPeakFit());

// 2.0
cfg = cfg.withRefiner(ImageRefiner.saddlePoint());   // default tuning
cfg = cfg.withRefiner(ImageRefiner.saddlePointWith(sp));
cfg.strategy.chess.centerOfMass = com;
cfg = cfg.withoutRefiner();
```

`cfg.refiner` is read-only in JavaScript: an `ImageRefiner`, or
`undefined` when none is set.

JSON configuration: the refiner leaves the strategy object and becomes
a top-level key; the center-of-mass tuning stays in the ChESS strategy
under its own name.

```json
{ "strategy": { "chess": { "refiner": { "center_of_mass": { "radius": 2 } } } } }
{ "strategy": { "chess": { "refiner": { "saddle_point": { "radius": 2 } } } } }
{ "strategy": { "radon": { "refiner": "peak_fit" } } }
```

becomes

```json
{ "strategy": { "chess": { "center_of_mass": { "radius": 2 } } } }
{ "strategy": { "chess": {} }, "refiner": { "saddle_point": { "radius": 2 } } }
{ "strategy": { "radon": {} }, "refiner": null }
```

An absent `"refiner"` is `null`. CLI: `--chess-refiner
center_of_mass|saddle_point` becomes `--refiner none|saddle_point`,
for either detector.

## 12.4 The Förstner refiner is removed

The saddle-point refiner is more accurate on real calibration images
and on almost every synthetic condition, so the Förstner refiner is
removed everywhere: `ChessRefiner::Forstner` and
`ChessRefiner::forstner()`, `ForstnerConfig` (Rust, Python, JS), core
`RefinerKind::Forstner`, `Refiner::Forstner` and `ForstnerRefiner`, C
`CC_REFINER_FORSTNER`, CLI `--chess-refiner forstner` and the JSON
`"forstner"` refiner. Use the saddle-point refiner:

| 1.x | 2.0 |
|---|---|
| Rust `ChessRefiner::forstner()` | `ImageRefiner::saddle_point()` |
| Python `ChessRefiner.forstner()` | `ImageRefiner.saddle_point()` |
| JS `ChessRefiner.withForstner(fc)` | `ImageRefiner.saddlePoint()` |
| C `CC_REFINER_FORSTNER` | `CC_IMAGE_REFINER_SADDLE_POINT` |
| JSON `"refiner": { "forstner": {...} }` in `strategy.chess` | top-level `"refiner": { "saddle_point": {} }` |

`ForstnerConfig` has no counterpart; start the saddle-point refiner
from its defaults.

## 12.5 The ML refiner is removed

The saddle-point refiner is more accurate in 15 of the 16 synthetic
conditions of the refiner benchmark (the exception is the heaviest noise
row, 0.100 against 0.093 px) and on rotated boards, where the ML model was
0.64–0.90 px RMS against at most 0.057 px, and it costs about 800× less
per corner. The ML
refiner is removed: Rust `ChessRefiner::Ml`, Python
`ChessRefiner.ml()`, JS `ChessRefiner.withMl()`, the CLI `"ml": true`
setting and the JSON `"ml"` refiner. Select the saddle-point refiner
instead (§12.3).

A build that enabled the `ml-refiner` feature drops it:

```toml
# 1.x
chess-corners = { version = "1.7", features = ["ml-refiner"] }
```

```toml
# 2.0
chess-corners = "2.1"
```

The Python wheel no longer includes the ML refiner.

## 12.6 Orientation reports `fit_std_error`

Each orientation axis now reports `fit_std_error`, the residual-based
least-squares standard error of the axis fit in radians, in place of
`sigma`. The new name says what the value is: a fit-quality score for
ranking and gating axes, not a calibrated error bound. Both orientation
methods now report this same quantity.

| Binding | 1.x | 2.0 |
|---|---|---|
| Rust | `AxisEstimate::sigma` | `AxisEstimate::fit_std_error` |
| Python | `Detections.sigmas` | `Detections.fit_std_errors` |
| JS | `axis0_sigma`, `axis1_sigma` slots | `axis0_fitStdError`, `axis1_fitStdError` (same offsets 4 and 6 of the stride-7 array) |
| C | `cc_axis::sigma` | `cc_axis::fit_std_error` |
| C++ | `Axis::sigma` | `Axis::fit_std_error` |
| CLI JSON output | `axes[].sigma` | `axes[].fit_std_error` |

The values change too:

- `RingFit` values are the 1.x ones divided by 1.0–1.25.
- `DiskFit` computes the standard error over the disk pixels instead of
  a residual heuristic, and its two axes now get separate values.
- The two `RingFit` values now follow their axes when the fit relabels
  the axes; in 1.x they could be attached to the wrong axis.
- Unchanged: the 10° minimum when `DiskFit` and `RingFit` disagree, π
  as "no information", and the axis angles.

A gate tuned on the 1.x values rejects about half as many axes in 2.0.
Multiply its threshold by about 0.65 to keep its rejection rate:

```rust,ignore
// 1.x
let ok = a0.sigma <= 0.05 && a1.sigma <= 0.05;

// 2.0
let max_err = 0.65 * 0.05;
let ok = a0.fit_std_error <= max_err && a1.fit_std_error <= max_err;
```

```python
# 1.x
keep = (det.sigmas <= 0.05).all(axis=1)

# 2.0
keep = (det.fit_std_errors <= 0.65 * 0.05).all(axis=1)
```

## 12.7 Configuration files

Stale 1.x keys now fail loudly instead of being dropped. Every loader
rejects an unknown key at every level of the configuration, with an
error that names it:

- **Rust** (`serde_json` and any other serde format):
  ``unknown field `noise_sigma`, expected one of `strategy`, ...``; a
  1.x per-strategy refiner gives ``unknown field `refiner`, expected
  `ring` or `center_of_mass` ``.
- **Python** (`DetectorConfig.from_dict`, `from_json`,
  `from_json_file`): `config has unknown keys: ...`, and a stale
  `noise_sigma` names its new place in `uncertainty`.
- **JavaScript** (`DetectorConfig.fromJson`): the Rust message, in a
  thrown `Error`. In 1.x it dropped unknown keys.
- **CLI**: the Rust message, at the top level and inside `detector`.

The 1.x `"uncertainty": "propagated"` is an error in every loader.

A complete 2.0 configuration with every key that moved:

```json
{
  "strategy": { "chess": { "ring": "canonical", "center_of_mass": { "radius": 2 } } },
  "refiner": { "saddle_point": {} },
  "uncertainty": { "propagated": { "noise": { "known": 1.2 } } },
  "threshold": 30.0
}
```

`schemas/detector_config.json` describes the full 2.0 shape. It sets
`additionalProperties: false` on every object, so a schema validator
rejects the same keys.

### The CLI config nests the detector settings

A 1.x CLI config file put the detector keys beside `image`,
`output_json`, `output_png` and `log_level`. In 2.0 they go in a
`detector` object, which has the shape of a `DetectorConfig` file.
In 1.x:

```json
{
  "image": "board.png",
  "output_json": "board.corners.json",
  "threshold": 30.0,
  "multiscale": "single_scale"
}
```

In 2.0:

```json
{
  "image": "board.png",
  "output_json": "board.corners.json",
  "detector": {
    "threshold": 30.0,
    "multiscale": "single_scale"
  }
}
```

A 1.x file fails with ``unknown field `threshold`, expected one of
`image`, `output_json`, `output_png`, `log_level`, `detector` ``. The
JSON summary the CLI writes echoes the configuration under `detector`
instead of `algorithm`.

## 12.8 C and C++: ABI version 8

`cc_abi_version()` returns 8. Rebuild against the new header: a C++
program built against the 1.x header throws on the version check
instead of reading a stale layout.

| 1.x | 2.0 |
|---|---|
| `cc_refiner_t` | `cc_image_refiner_t` |
| `CC_REFINER_CENTER_OF_MASS` (0), `CC_REFINER_PEAK_FIT` (3) | `CC_IMAGE_REFINER_NONE` (0) |
| `CC_REFINER_SADDLE_POINT` (2) | `CC_IMAGE_REFINER_SADDLE_POINT` (1), valid for either strategy |
| `CC_REFINER_FORSTNER` (1) | removed (§12.4) |
| `CC_UNCERTAINTY_PROPAGATED` (0) | `CC_UNCERTAINTY_PROPAGATED` (1) |
| `CC_UNCERTAINTY_OFF` (1) | `CC_UNCERTAINTY_OFF` (0) |
| `cc_axis::sigma` | `cc_axis::fit_std_error` |

The constants are renumbered, so code that stored the numbers in place
of the names must be updated. A zero-initialised `cc_config` now
selects `CC_UNCERTAINTY_OFF` and `CC_IMAGE_REFINER_NONE`, matching every
preset and every other binding. In 1.x it selected propagated
uncertainty and `CC_REFINER_CENTER_OF_MASS`, which a Radon config
rejected. Still start from a preset: a zeroed config has a zero
threshold and NMS radius.

`has_noise_sigma` and `noise_sigma` keep their names and are read only
under `CC_UNCERTAINTY_PROPAGATED`:

```c
/* 1.x */
cc_config cfg = cc_config_radon();
cfg.refiner = CC_REFINER_SADDLE_POINT;
cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
cfg.has_noise_sigma = 1;
cfg.noise_sigma = 1.2f;
/* ... */
if (c->axes[0].sigma <= 0.05f) { /* ... */ }

/* 2.0: the same source with the renamed constant and field */
cc_config cfg = cc_config_radon();
cfg.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
cfg.has_noise_sigma = 1;    /* 0: estimate σ per image */
cfg.noise_sigma = 1.2f;
/* ... */
if (c->axes[0].fit_std_error <= 0.65f * 0.05f) { /* ... */ }
```

C++: `Config::uncertainty` is a `chess_corners::Uncertainty` value that
carries the noise level, in place of the tag plus
`std::optional<float> Config::noise_sigma`. `Config::refiner` is a
`cc_image_refiner_t`, and `Axis::sigma` is `Axis::fit_std_error`.
`CHESS_CORNERS_ABI_VERSION` is 8.

```cpp
// 1.x
auto cfg = chess_corners::Config::chess();
cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
cfg.noise_sigma = 1.2f;
cfg.refiner = CC_REFINER_SADDLE_POINT;

// 2.0
auto cfg = chess_corners::Config::chess();
cfg.uncertainty = chess_corners::Uncertainty::propagated(1.2f);  // or propagated()
cfg.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
```

## 12.9 `chess-corners-core`

These changes affect programs that depend on `chess-corners-core`
directly. The facade types (`CornerDescriptor::position_covariance`,
`AxisEstimate`) cover the common cases.

**Covariance fields.** `Corner` and `RefineResult` no longer have a
public `covariance_per_noise_variance` field. The field mirrored the
private covariance model, and writing it dropped the systematic term.
Read the model through the methods, as 1.7 already recommended:

```rust,ignore
// 1.x
let c_noise = corner.covariance_per_noise_variance;

// 2.0
let c_noise = corner.noise_covariance_per_unit_variance(); // px² per gray level²
let sigma_xy = corner.position_covariance(1.2);            // full covariance, px²
```

**Removed helpers.**

| 1.x | 2.0 |
|---|---|
| `attach_radon_peak_covariance(...)` | `DenseDetector::attach_peak_covariance` on `RadonDetector` |
| `DenseDetector::refines_on_image()` | removed: both detectors always refine on the image; drop the call |
| `Uncertainty` (core) | moved to `chess-corners`; the core refiners take `Refiner::with_covariance(bool)` in place of `Refiner::with_uncertainty(tier)` |
| `ChessParams::refiner` | `ChessParams::center_of_mass`; `RefinerKind` holds only image refiners, and `Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg))` builds the ChESS localizer |
| `AxisEstimate::new(angle, sigma)` | `AxisEstimate::new(angle, fit_std_error)` |
| `AxisFitResult::sigma_theta1` / `sigma_theta2` | `fit_std_error_theta1` / `fit_std_error_theta2` |

---

Next: [Part II](part-02-using-the-detector.md) walks through the 2.0
API on every binding.
