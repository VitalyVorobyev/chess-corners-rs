use box_image_pyramid::PyramidParams;
use chess_corners_core::{
    unstable::{ChessParams, RefinerKind},
    CenterOfMassConfig, ForstnerConfig, OrientationMethod, PeakFitMode, RadonDetectorParams,
    SaddlePointConfig,
};
use serde::{Deserialize, Serialize};

use crate::multiscale::CoarseToFineParams;
use crate::upscale::UpscaleConfig;

// ---------------------------------------------------------------------------
// Detector kernel / ring selection
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Selects the `r=5` (canonical) or `r=10`
/// (broad) ring used by the dense response kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChessRing {
    /// Paper-default radius-5 ring (16 samples).
    #[default]
    Canonical,
    /// Radius-10 ring. Larger support window for callers that want the
    /// detector to sample farther from the candidate center.
    Broad,
}

// ---------------------------------------------------------------------------
// Refiner enums (one per detector)
// ---------------------------------------------------------------------------

/// Subpixel refiner selection for the ChESS detector.
///
/// Each variant carries its own tuning struct as a payload: there is
/// no shared discriminator + parallel-tuning-struct shape, so
/// switching variants can never leave a stale config field behind.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChessRefiner {
    /// Center-of-mass (intensity centroid) refinement on the response
    /// map. Cheapest refiner in the shipped benchmark; the library default.
    CenterOfMass(CenterOfMassConfig),
    /// Förstner structure-tensor refinement on the image patch.
    Forstner(ForstnerConfig),
    /// Quadratic surface fit at the saddle point.
    SaddlePoint(SaddlePointConfig),
    /// ML-backed subpixel refinement. Runs a small ONNX model on a
    /// normalized intensity patch around each candidate. Requires the
    /// `ml-refiner` feature.
    #[cfg(feature = "ml-refiner")]
    Ml,
}

impl Default for ChessRefiner {
    fn default() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
}

impl ChessRefiner {
    /// Center-of-mass refinement with default tuning.
    pub fn center_of_mass() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
    /// Förstner structure-tensor refinement with default tuning.
    pub fn forstner() -> Self {
        Self::Forstner(ForstnerConfig::default())
    }
    /// Saddle-point quadratic fit with default tuning.
    pub fn saddle_point() -> Self {
        Self::SaddlePoint(SaddlePointConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Multiscale configuration
// ---------------------------------------------------------------------------

/// Coarse-to-fine multiscale configuration.
///
/// JSON shape mirrors [`UpscaleConfig`]:
///
/// - `{ "single_scale": null }` — run the detector once on the full image.
/// - `{ "pyramid": { "levels": 3, "min_size": 128, "refinement_radius": 3 } }`
///   — build an image pyramid, detect seeds on the coarsest level, and
///   refine each seed into the base image. Honoured by both ChESS and
///   Radon strategies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MultiscaleConfig {
    /// Single-scale detection (no pyramid).
    #[default]
    SingleScale,
    /// Coarse-to-fine pyramid detection.
    Pyramid {
        /// Number of pyramid levels (≥ 1). Level 0 is the base image;
        /// each subsequent level is a 2× box-filter downsample.
        levels: u8,
        /// Minimum short-edge length in pixels. The pyramid stops once
        /// the next level would fall below this size.
        min_size: usize,
        /// ROI half-radius at the coarse level used to refine each seed
        /// into the base image, in coarse-level pixels.
        refinement_radius: u32,
    },
}

impl MultiscaleConfig {
    /// Three-level pyramid with library defaults (`min_size = 128`, `refinement_radius = 3`).
    /// Equivalent to the multiscale preset used by [`DetectorConfig::chess_multiscale`]
    /// and [`DetectorConfig::radon_multiscale`].
    pub const fn pyramid_default() -> Self {
        Self::Pyramid {
            levels: 3,
            min_size: 128,
            refinement_radius: 3,
        }
    }
    /// Pyramid with caller-supplied parameters.
    pub const fn pyramid(levels: u8, min_size: usize, refinement_radius: u32) -> Self {
        Self::Pyramid {
            levels,
            min_size,
            refinement_radius,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-strategy configs
// ---------------------------------------------------------------------------

/// Configuration for the ChESS detector branch of [`DetectionStrategy`].
///
/// Carries the detector ring choice and the subpixel refiner. The shared
/// NMS / clustering thresholds ([`DetectionParams`]), multiscale, and
/// upscale live at the top level of [`DetectorConfig`] and apply to both
/// strategies. Descriptors always sample at the detector ring radius.
///
/// # Common knobs
///
/// - [`ring`](ChessConfig::ring) — choose the detector kernel radius.
/// - [`refiner`](ChessConfig::refiner) — select and configure the
///   subpixel refinement backend.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessConfig {
    /// Detector ring radius. `Canonical` selects the paper's `r=5`,
    /// `Broad` selects `r=10`. Descriptors sample at this same radius.
    pub ring: ChessRing,
    /// Subpixel refiner. Each variant carries its tuning struct.
    pub refiner: ChessRefiner,
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self {
            ring: ChessRing::Canonical,
            refiner: ChessRefiner::default(),
        }
    }
}

/// Configuration for the whole-image Radon detector branch of
/// [`DetectionStrategy`].
///
/// All radii and counts are in **working-resolution** pixels (i.e.
/// after `image_upsample`). The shared NMS / clustering thresholds
/// ([`DetectionParams`]), multiscale, and upscale live at the top level
/// of [`DetectorConfig`] and apply to both strategies.
///
/// # Common knobs
///
/// - [`image_upsample`](RadonConfig::image_upsample) — `2` (the default)
///   reproduces the paper's 2× supersampled detection; `1` is faster but
///   less accurate on low-resolution inputs.
///
/// # Advanced tuning
///
/// The remaining fields control low-level detection behaviour. The
/// defaults reproduce the paper's recommended settings and work well
/// for typical camera images. Adjust them only when you have a specific
/// reason (e.g. a non-standard image resolution or SNR budget).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RadonConfig {
    /// Advanced tuning. Half-length of each Radon ray in
    /// working-resolution pixels. The ray has `2·ray_radius + 1`
    /// samples. Paper default at `image_upsample = 2` is `ray_radius = 4`.
    /// Shorter rays are faster but integrate less signal; longer rays are
    /// more discriminating but may cross into neighbouring cells.
    pub ray_radius: u32,
    /// Image-level supersampling factor applied before ray integration.
    /// `1` operates on the input grid; `2` (paper default) is equivalent
    /// to bilinearly upsampling the input first, giving sub-pixel ray
    /// positioning. Values ≥ 3 are clamped to 2 by the core detector.
    pub image_upsample: u32,
    /// Advanced tuning. Half-size of the box blur applied to the Radon
    /// response map after integration. `0` disables blurring; `1`
    /// (default) yields a 3×3 box, smoothing quantisation noise in the
    /// response. Increase only on very high-SNR images where extra
    /// smoothing is unwanted.
    pub response_blur_radius: u32,
    /// Advanced tuning. Peak-fit mode for the 3-point subpixel
    /// refinement of the response-map argmax. `Gaussian` (default) fits
    /// on log-response (more accurate near the peak); `Parabolic` fits
    /// directly on the response values. See [`PeakFitMode`].
    pub peak_fit: PeakFitMode,
}

impl Default for RadonConfig {
    fn default() -> Self {
        Self {
            ray_radius: 4,
            image_upsample: 2,
            response_blur_radius: 1,
            peak_fit: PeakFitMode::Gaussian,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared detection params
// ---------------------------------------------------------------------------

/// Shared non-maximum-suppression and peak-clustering thresholds.
///
/// These two knobs have identical meaning for the ChESS and Radon
/// detectors, so they live once at the [`DetectorConfig`] level rather
/// than being duplicated inside each strategy config. Both are expressed
/// in the detector's working-resolution pixels (for Radon, that is after
/// `image_upsample`). Tune them through
/// [`DetectorConfig::with_detection`].
///
/// The defaults match the ChESS presets; the Radon presets raise
/// `nms_radius` to `4` to suit the wider Radon response peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DetectionParams {
    /// Non-maximum-suppression half-radius in working-resolution pixels.
    /// Only the highest-response pixel within this radius is kept.
    /// Reduce when corners are packed closer than `2·nms_radius` pixels;
    /// increase to suppress near-duplicate detections on blurry images.
    pub nms_radius: u32,
    /// Minimum number of positive-response neighbours within the NMS
    /// window that a candidate must have to be accepted. Increase to
    /// require a stronger local cluster of response, suppressing isolated
    /// noise peaks at the cost of potentially missing weak corners near
    /// image boundaries.
    pub min_cluster_size: u32,
}

impl Default for DetectionParams {
    fn default() -> Self {
        // Matches the ChESS presets; `DetectorConfig::default()` is `chess()`.
        Self {
            nms_radius: 2,
            min_cluster_size: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// DetectionStrategy
// ---------------------------------------------------------------------------

/// Top-level detector dispatch. Selects between the ChESS kernel
/// pipeline and the Radon whole-image detector. The chosen variant
/// carries all detector-specific tuning; settings that don't apply to
/// the active detector are simply unreachable, so the type system
/// enforces correctness instead of silently ignoring fields.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DetectionStrategy {
    /// ChESS kernel detection with optional coarse-to-fine multiscale.
    Chess(ChessConfig),
    /// Whole-image Radon (Duda-Frese) detection.
    Radon(RadonConfig),
}

impl Default for DetectionStrategy {
    fn default() -> Self {
        DetectionStrategy::Chess(ChessConfig::default())
    }
}

// ---------------------------------------------------------------------------
// DetectorConfig
// ---------------------------------------------------------------------------

/// High-level detection configuration.
///
/// Build one with [`DetectorConfig::chess`],
/// [`DetectorConfig::chess_multiscale`], [`DetectorConfig::radon`], or
/// [`DetectorConfig::radon_multiscale`] and tweak only the fields you need.
/// The detector translates this into the low-level [`ChessParams`] /
/// [`RadonDetectorParams`] consumed by `chess-corners-core` at the detection
/// boundary.
///
/// # Common knobs
///
/// These fields are the primary surface for most callers:
///
/// - [`strategy`](DetectorConfig::strategy) — choose ChESS or Radon and
///   configure its parameters.
/// - [`threshold`](DetectorConfig::threshold) — control how many corners are
///   returned: lower → more candidates, higher → fewer and stronger. ChESS
///   reads it as an absolute response floor; Radon as a fraction of the
///   per-frame maximum.
/// - [`multiscale`](DetectorConfig::multiscale) — enable coarse-to-fine
///   pyramid detection (`Pyramid`) or keep it off (`SingleScale`).
/// - [`upscale`](DetectorConfig::upscale) — pre-pipeline integer bilinear
///   upscaling for low-resolution inputs where corners have fewer than 5 px
///   of ring support. `Disabled` by default.
/// - [`orientation_method`](DetectorConfig::orientation_method) — how corner
///   axis orientations are estimated when building descriptors.
///
/// # Advanced tuning
///
/// - [`detection`](DetectorConfig::detection) — shared NMS / clustering
///   thresholds applied by both strategies. See [`DetectionParams`].
/// - [`merge_radius`](DetectorConfig::merge_radius) — duplicate-suppression
///   radius across pyramid levels. See the field docs below.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DetectorConfig {
    /// Detector dispatch: ChESS or Radon, each carrying its own tuning.
    pub strategy: DetectionStrategy,
    /// Detector acceptance threshold.
    ///
    /// ChESS reads it as an absolute floor on the raw response
    /// `R = SR − DR − 16·MR`: a candidate is kept when `R` exceeds it.
    /// Useful floors run roughly `30..=300` depending on image contrast;
    /// the [`chess`](Self::chess) preset defaults to `30`, which suppresses
    /// texture noise while keeping well-formed corners.
    /// Radon reads it as a fraction in `[0.0, 1.0]` of the per-frame
    /// maximum response, because Radon's `(max − min)²` score scales
    /// with image size and has no portable absolute scale.
    pub threshold: f32,
    /// Shared non-maximum-suppression and peak-clustering thresholds.
    /// Honoured by both strategies. See [`DetectionParams`].
    pub detection: DetectionParams,
    /// Coarse-to-fine multiscale configuration. `SingleScale` skips
    /// the pyramid entirely. Honoured by both strategies.
    pub multiscale: MultiscaleConfig,
    /// Pre-pipeline integer upscaling. `Disabled` skips the stage.
    pub upscale: UpscaleConfig,
    /// Orientation-fit method used when building corner descriptors, or
    /// `None` to skip the per-corner fit entirely. When `None`, every
    /// descriptor carries `axes: None`; positions and responses are
    /// unaffected. Skipping orientation is the cheaper path for consumers
    /// that derive board geometry themselves.
    pub orientation_method: Option<OrientationMethod>,
    /// Advanced tuning. Merge radius in base-image pixels for
    /// cross-level and cross-seed duplicate suppression. After seeds
    /// detected at coarser pyramid levels are refined into the base
    /// image, any two refined positions within this radius are merged
    /// into a single output corner. Default is `3.0` px. Increase if
    /// you see duplicate detections near the same physical corner;
    /// decrease if distinct corners closer than `2·merge_radius` pixels
    /// are being merged.
    pub merge_radius: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self::chess()
    }
}

impl DetectorConfig {
    /// Single-scale ChESS preset.
    pub fn chess() -> Self {
        Self {
            strategy: DetectionStrategy::Chess(ChessConfig::default()),
            // Absolute floor on the ChESS response: suppresses texture
            // noise while keeping well-formed corners. See the field doc.
            threshold: 30.0,
            detection: DetectionParams::default(),
            multiscale: MultiscaleConfig::SingleScale,
            upscale: UpscaleConfig::Disabled,
            orientation_method: Some(OrientationMethod::default()),
            merge_radius: 3.0,
        }
    }

    /// Three-level coarse-to-fine ChESS preset.
    pub fn chess_multiscale() -> Self {
        Self {
            multiscale: MultiscaleConfig::pyramid_default(),
            ..Self::chess()
        }
    }

    /// Whole-image Radon detector preset.
    /// Single-scale; use [`Self::radon_multiscale`] for coarse-to-fine
    /// Radon detection on larger frames.
    pub fn radon() -> Self {
        Self {
            strategy: DetectionStrategy::Radon(RadonConfig::default()),
            threshold: 0.01,
            detection: DetectionParams {
                nms_radius: 4,
                min_cluster_size: 2,
            },
            multiscale: MultiscaleConfig::SingleScale,
            ..Self::chess()
        }
    }

    /// Coarse-to-fine Radon preset. Measure against [`Self::radon`] on
    /// your target frame sizes; this preset trades more configuration
    /// machinery for less full-resolution detector work on large frames.
    pub fn radon_multiscale() -> Self {
        Self {
            strategy: DetectionStrategy::Radon(RadonConfig::default()),
            threshold: 0.01,
            detection: DetectionParams {
                nms_radius: 4,
                min_cluster_size: 2,
            },
            multiscale: MultiscaleConfig::pyramid_default(),
            ..Self::chess()
        }
    }

    /// Set the active strategy to ChESS and apply `f` to the nested config.
    /// If the current strategy is already ChESS, mutate it in place.
    /// Otherwise, replace the strategy with [`ChessConfig::default`] and apply `f`.
    ///
    /// Top-level fields (threshold, multiscale, upscale, orientation_method,
    /// merge_radius) are untouched. When switching strategies, prefer the
    /// preset constructors — ChESS reads `threshold` as an absolute response
    /// floor, Radon as a fraction of the per-frame maximum.
    pub fn with_chess<F: FnOnce(&mut ChessConfig)>(mut self, f: F) -> Self {
        let mut chess = match self.strategy {
            DetectionStrategy::Chess(c) => c,
            DetectionStrategy::Radon(_) => ChessConfig::default(),
        };
        f(&mut chess);
        self.strategy = DetectionStrategy::Chess(chess);
        self
    }

    /// Mirror of [`Self::with_chess`] for the Radon strategy.
    pub fn with_radon<F: FnOnce(&mut RadonConfig)>(mut self, f: F) -> Self {
        let mut radon = match self.strategy {
            DetectionStrategy::Radon(r) => r,
            DetectionStrategy::Chess(_) => RadonConfig::default(),
        };
        f(&mut radon);
        self.strategy = DetectionStrategy::Radon(radon);
        self
    }

    /// Replace the acceptance threshold. See [`DetectorConfig::threshold`]
    /// for the per-detector interpretation.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
    /// Replace the multiscale configuration.
    pub fn with_multiscale(mut self, multiscale: MultiscaleConfig) -> Self {
        self.multiscale = multiscale;
        self
    }
    /// Replace the upscale configuration.
    pub fn with_upscale(mut self, upscale: UpscaleConfig) -> Self {
        self.upscale = upscale;
        self
    }
    /// Replace the orientation-fit method used when building descriptors.
    pub fn with_orientation_method(mut self, method: OrientationMethod) -> Self {
        self.orientation_method = Some(method);
        self
    }
    /// Skip the per-corner orientation fit. Descriptors are still produced
    /// with subpixel positions and responses, but carry `axes: None`. Use
    /// this when you derive board geometry yourself and don't need the
    /// per-corner axes — it removes the dominant per-corner cost.
    pub fn without_orientation(mut self) -> Self {
        self.orientation_method = None;
        self
    }
    /// Replace the merge radius for cross-level duplicate suppression.
    pub fn with_merge_radius(mut self, radius: f32) -> Self {
        self.merge_radius = radius;
        self
    }

    /// Apply `f` to the shared [`DetectionParams`] (NMS / clustering
    /// thresholds honoured by both strategies) and return the updated
    /// config.
    pub fn with_detection<F: FnOnce(&mut DetectionParams)>(mut self, f: F) -> Self {
        f(&mut self.detection);
        self
    }

    /// Translate this config into the low-level [`ChessParams`] consumed
    /// by `chess-corners-core`. Only meaningful when
    /// [`Self::strategy`] is the ChESS variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Radon`], the
    /// ChESS-specific fields fall back to their [`ChessParams::default()`]
    /// values; callers should route through
    /// [`Self::radon_detector_params`] instead.
    pub(crate) fn chess_params(&self) -> ChessParams {
        let mut params = ChessParams::default();
        params.nms_radius = self.detection.nms_radius;
        params.min_cluster_size = self.detection.min_cluster_size;
        if let DetectionStrategy::Chess(chess) = &self.strategy {
            params.use_radius10 = matches!(chess.ring, ChessRing::Broad);
            match chess.refiner {
                ChessRefiner::CenterOfMass(cfg) => params.refiner = RefinerKind::CenterOfMass(cfg),
                ChessRefiner::Forstner(cfg) => params.refiner = RefinerKind::Forstner(cfg),
                ChessRefiner::SaddlePoint(cfg) => params.refiner = RefinerKind::SaddlePoint(cfg),
                // The ML refiner runs in the facade, not core: `Detector`
                // routes `Ml` to the ONNX pipeline. Core only produces the
                // candidates (and coarse-level seeds) the ML pass consumes,
                // using its default refiner — so leave `params.refiner` at
                // the core default rather than substituting an unrelated
                // kind.
                #[cfg(feature = "ml-refiner")]
                ChessRefiner::Ml => {}
            }
        }
        // ChESS interprets `threshold` as an absolute floor on the raw response.
        params.threshold_abs = Some(self.threshold);
        params.orientation_method = self.orientation_method;
        params
    }

    /// Translate this config into the low-level [`RadonDetectorParams`]
    /// consumed by `chess-corners-core`. Only meaningful when
    /// [`Self::strategy`] is the Radon variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Chess`], the
    /// Radon-specific fields fall back to their
    /// [`RadonDetectorParams::default()`] values; callers should route
    /// through [`Self::chess_params`] instead.
    pub(crate) fn radon_detector_params(&self) -> RadonDetectorParams {
        let mut params = RadonDetectorParams::default();
        params.nms_radius = self.detection.nms_radius;
        params.min_cluster_size = self.detection.min_cluster_size;
        if let DetectionStrategy::Radon(radon) = &self.strategy {
            params.ray_radius = radon.ray_radius;
            params.image_upsample = radon.image_upsample;
            params.response_blur_radius = radon.response_blur_radius;
            params.peak_fit = radon.peak_fit;
        }
        // Radon interprets `threshold` as a fraction of the per-frame maximum.
        params.threshold_abs = None;
        params.threshold_rel = self.threshold;
        params
    }

    /// Translate this config into the [`CoarseToFineParams`] that drive
    /// the multiscale pipeline. Returns `None` when [`Self::multiscale`]
    /// is [`MultiscaleConfig::SingleScale`]. Both ChESS and Radon honour
    /// the same top-level multiscale settings.
    pub(crate) fn coarse_to_fine_params(&self) -> Option<CoarseToFineParams> {
        let MultiscaleConfig::Pyramid {
            levels,
            min_size,
            refinement_radius,
        } = self.multiscale
        else {
            return None;
        };
        let mut cfg = CoarseToFineParams::default();
        let mut pyramid = PyramidParams::default();
        pyramid.num_levels = levels;
        pyramid.min_size = min_size;
        cfg.pyramid = pyramid;
        cfg.refinement_radius = refinement_radius;
        cfg.merge_radius = self.merge_radius;
        Some(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_strategy_chess(cfg: &DetectorConfig) -> &ChessConfig {
        match &cfg.strategy {
            DetectionStrategy::Chess(c) => c,
            other => panic!("expected ChESS strategy, got {other:?}"),
        }
    }

    fn assert_strategy_radon(cfg: &DetectorConfig) -> &RadonConfig {
        match &cfg.strategy {
            DetectionStrategy::Radon(r) => r,
            other => panic!("expected Radon strategy, got {other:?}"),
        }
    }

    #[test]
    fn default_is_single_scale_chess_with_denoise_floor() {
        let cfg = DetectorConfig::default();
        let chess = assert_strategy_chess(&cfg);
        assert_eq!(chess.ring, ChessRing::Canonical);
        assert_eq!(
            chess.refiner,
            ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
        );
        assert_eq!(cfg.detection.nms_radius, 2);
        assert_eq!(cfg.detection.min_cluster_size, 2);
        assert_eq!(cfg.multiscale, MultiscaleConfig::SingleScale);
        assert_eq!(cfg.upscale, UpscaleConfig::Disabled);
        assert_eq!(cfg.threshold, 30.0);
        assert_eq!(cfg.merge_radius, 3.0);
        assert!(cfg.coarse_to_fine_params().is_none());

        let params = cfg.chess_params();
        assert!(!params.use_radius10);
        assert_eq!(params.threshold_abs, Some(30.0));
        assert_eq!(params.nms_radius, 2);
        assert_eq!(params.min_cluster_size, 2);
        assert_eq!(
            params.refiner,
            RefinerKind::CenterOfMass(CenterOfMassConfig::default())
        );
    }

    #[test]
    fn chess_threshold_lowers_to_absolute_floor() {
        let cfg = DetectorConfig {
            threshold: 7.5,
            ..DetectorConfig::chess()
        };
        let params = cfg.chess_params();
        assert_eq!(params.threshold_abs, Some(7.5));
    }

    #[test]
    fn radon_threshold_lowers_to_relative_fraction() {
        let cfg = DetectorConfig {
            threshold: 0.15,
            ..DetectorConfig::radon()
        };
        let params = cfg.radon_detector_params();
        assert_eq!(params.threshold_abs, None);
        assert!((params.threshold_rel - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn chess_multiscale_preset_carries_pyramid_params() {
        let cfg = DetectorConfig::chess_multiscale();
        let MultiscaleConfig::Pyramid {
            levels,
            min_size,
            refinement_radius,
        } = cfg.multiscale
        else {
            panic!("chess_multiscale preset must carry Pyramid params");
        };
        assert_eq!(levels, 3);
        assert_eq!(min_size, 128);
        assert_eq!(refinement_radius, 3);

        let cf = cfg
            .coarse_to_fine_params()
            .expect("chess_multiscale config must produce CoarseToFineParams");
        assert_eq!(cf.pyramid.num_levels, 3);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn radon_preset_uses_radon_config_and_relative_threshold() {
        let cfg = DetectorConfig::radon();
        let radon = assert_strategy_radon(&cfg);
        assert_eq!(radon.ray_radius, 4);
        assert_eq!(radon.image_upsample, 2);
        assert_eq!(radon.response_blur_radius, 1);
        assert_eq!(radon.peak_fit, PeakFitMode::Gaussian);
        assert_eq!(cfg.detection.nms_radius, 4);
        assert_eq!(cfg.detection.min_cluster_size, 2);
        assert_eq!(cfg.threshold, 0.01);
        assert_eq!(cfg.multiscale, MultiscaleConfig::SingleScale);
        assert!(cfg.coarse_to_fine_params().is_none());

        let radon_params = cfg.radon_detector_params();
        assert_eq!(radon_params.ray_radius, 4);
        assert_eq!(radon_params.image_upsample, 2);
        assert_eq!(radon_params.nms_radius, 4);
        assert_eq!(radon_params.min_cluster_size, 2);
        assert_eq!(radon_params.threshold_abs, None);
        assert!((radon_params.threshold_rel - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn radon_multiscale_preset_carries_pyramid_params() {
        let cfg = DetectorConfig::radon_multiscale();
        assert_strategy_radon(&cfg);
        assert_eq!(cfg.threshold, 0.01);
        let MultiscaleConfig::Pyramid {
            levels,
            min_size,
            refinement_radius,
        } = cfg.multiscale
        else {
            panic!("radon_multiscale preset must carry Pyramid params");
        };
        assert_eq!(levels, 3);
        assert_eq!(min_size, 128);
        assert_eq!(refinement_radius, 3);

        let cf = cfg
            .coarse_to_fine_params()
            .expect("radon_multiscale config must produce CoarseToFineParams");
        assert_eq!(cf.pyramid.num_levels, 3);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn broad_ring_and_forstner_refiner_propagate_to_params() {
        // `ForstnerConfig` is `#[non_exhaustive]`; build via `Default`
        // and set the field under test.
        let mut forstner = ForstnerConfig::default();
        forstner.max_offset = 2.0;
        let cfg = DetectorConfig {
            strategy: DetectionStrategy::Chess(ChessConfig {
                ring: ChessRing::Broad,
                refiner: ChessRefiner::Forstner(forstner),
                ..ChessConfig::default()
            }),
            ..DetectorConfig::chess()
        };

        let params = cfg.chess_params();
        assert!(params.use_radius10);
        assert_eq!(params.refiner, RefinerKind::Forstner(forstner));
    }

    #[test]
    fn chess_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::chess();
        let json = serde_json::to_string(&cfg).expect("serialize chess config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize chess config");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn chess_multiscale_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::chess_multiscale();
        let json = serde_json::to_string(&cfg).expect("serialize chess_multiscale config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize chess_multiscale config");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn radon_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::radon();
        let json = serde_json::to_string(&cfg).expect("serialize radon config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize radon config");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn radon_multiscale_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::radon_multiscale();
        let json = serde_json::to_string(&cfg).expect("serialize radon_multiscale config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize radon_multiscale config");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn threshold_serializes_as_a_bare_number() {
        let cfg = DetectorConfig::chess().with_threshold(3.5);
        let json = serde_json::to_string(&cfg).expect("serialize config");
        assert!(json.contains("\"threshold\":3.5"));
        let decoded: DetectorConfig = serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(decoded.threshold, 3.5);
    }

    #[test]
    fn multiscale_config_round_trips_with_externally_tagged_payload() {
        let single = MultiscaleConfig::SingleScale;
        let single_json = serde_json::to_string(&single).expect("serialize single-scale");
        assert!(single_json.contains("single_scale"));
        let decoded: MultiscaleConfig =
            serde_json::from_str(&single_json).expect("deserialize single-scale");
        assert_eq!(decoded, single);

        let pyramid = MultiscaleConfig::Pyramid {
            levels: 3,
            min_size: 128,
            refinement_radius: 3,
        };
        let pyramid_json = serde_json::to_string(&pyramid).expect("serialize pyramid");
        assert!(pyramid_json.contains("pyramid"));
        let decoded: MultiscaleConfig =
            serde_json::from_str(&pyramid_json).expect("deserialize pyramid");
        assert_eq!(decoded, pyramid);
    }

    #[test]
    fn chess_refiner_round_trips_each_variant() {
        let variants = [
            ChessRefiner::CenterOfMass(CenterOfMassConfig::default()),
            ChessRefiner::Forstner(ForstnerConfig::default()),
            ChessRefiner::SaddlePoint(SaddlePointConfig::default()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).expect("serialize chess refiner");
            let decoded: ChessRefiner =
                serde_json::from_str(&json).expect("deserialize chess refiner");
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn unit_enum_variants_serialize_as_bare_strings() {
        // Codifies the externally-tagged serde encoding for unit variants.
        // The Python from_dict paths must accept these bare strings produced
        // by serde so that Rust→JSON→Python round-trips work end-to-end.
        let json = serde_json::to_string(&MultiscaleConfig::SingleScale).unwrap();
        assert_eq!(json, "\"single_scale\"");

        let json = serde_json::to_string(&UpscaleConfig::Disabled).unwrap();
        assert_eq!(json, "\"disabled\"");
    }

    #[test]
    fn with_chess_mutates_in_place_when_strategy_is_chess() {
        let cfg = DetectorConfig::chess().with_chess(|c| c.ring = ChessRing::Broad);
        let chess = assert_strategy_chess(&cfg);
        assert_eq!(chess.ring, ChessRing::Broad);
        // Other chess fields untouched
        assert_eq!(
            chess.refiner,
            ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
        );
    }

    #[test]
    fn with_chess_replaces_radon_preserves_threshold() {
        let cfg = DetectorConfig::radon()
            .with_threshold(5.0)
            .with_chess(|c| c.ring = ChessRing::Broad);
        // Strategy replaced with chess
        let chess = assert_strategy_chess(&cfg);
        assert_eq!(chess.ring, ChessRing::Broad);
        // Top-level threshold preserved
        assert_eq!(cfg.threshold, 5.0);
    }

    #[test]
    fn with_radon_mutates_in_place_when_strategy_is_radon() {
        let cfg = DetectorConfig::radon().with_radon(|r| r.ray_radius = 9);
        let radon = assert_strategy_radon(&cfg);
        assert_eq!(radon.ray_radius, 9);
        // Other radon fields untouched
        assert_eq!(radon.image_upsample, 2);
    }

    #[test]
    fn with_radon_replaces_chess_preserves_threshold() {
        let cfg = DetectorConfig::chess()
            .with_threshold(0.5)
            .with_radon(|r| r.ray_radius = 6);
        let radon = assert_strategy_radon(&cfg);
        assert_eq!(radon.ray_radius, 6);
        // Threshold preserved
        assert_eq!(cfg.threshold, 0.5);
    }

    #[test]
    fn with_detection_sets_shared_params_for_both_strategies() {
        // ChESS strategy: shared detection params flow into ChessParams.
        let cfg = DetectorConfig::chess().with_detection(|d| {
            d.nms_radius = 7;
            d.min_cluster_size = 4;
        });
        assert_eq!(cfg.detection.nms_radius, 7);
        assert_eq!(cfg.detection.min_cluster_size, 4);
        let params = cfg.chess_params();
        assert_eq!(params.nms_radius, 7);
        assert_eq!(params.min_cluster_size, 4);

        // Radon strategy: same shared params flow into RadonDetectorParams.
        let cfg = DetectorConfig::radon().with_detection(|d| d.nms_radius = 9);
        assert_eq!(cfg.detection.nms_radius, 9);
        let params = cfg.radon_detector_params();
        assert_eq!(params.nms_radius, 9);
    }

    #[test]
    fn chained_builder_produces_expected_state() {
        let cfg = DetectorConfig::chess()
            .with_threshold(0.15)
            .with_chess(|c| c.refiner = ChessRefiner::forstner());
        assert_eq!(cfg.threshold, 0.15);
        let chess = assert_strategy_chess(&cfg);
        assert_eq!(
            chess.refiner,
            ChessRefiner::Forstner(ForstnerConfig::default())
        );
    }

    #[test]
    fn with_multiscale_sets_multiscale() {
        let cfg = DetectorConfig::chess().with_multiscale(MultiscaleConfig::pyramid_default());
        assert_eq!(
            cfg.multiscale,
            MultiscaleConfig::Pyramid {
                levels: 3,
                min_size: 128,
                refinement_radius: 3
            }
        );
    }

    #[test]
    fn with_upscale_sets_upscale() {
        let cfg = DetectorConfig::chess().with_upscale(UpscaleConfig::Fixed(2));
        assert_eq!(cfg.upscale, UpscaleConfig::Fixed(2));
    }

    #[test]
    fn with_orientation_method_sets_method() {
        let method = OrientationMethod::DiskFit;
        let cfg = DetectorConfig::chess().with_orientation_method(method);
        assert_eq!(cfg.orientation_method, Some(method));
    }

    #[test]
    fn without_orientation_clears_method() {
        let cfg = DetectorConfig::chess().without_orientation();
        assert_eq!(cfg.orientation_method, None);
        // Lowering carries the `None` through to the low-level params.
        assert_eq!(cfg.chess_params().orientation_method, None);
    }

    #[test]
    fn default_orientation_method_is_some_ring_fit() {
        let cfg = DetectorConfig::chess();
        assert_eq!(cfg.orientation_method, Some(OrientationMethod::RingFit));
    }

    #[test]
    fn with_merge_radius_sets_radius() {
        let cfg = DetectorConfig::chess().with_merge_radius(5.0);
        assert!((cfg.merge_radius - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn chess_refiner_shortcuts_equal_full_constructors() {
        assert_eq!(
            ChessRefiner::center_of_mass(),
            ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
        );
        assert_eq!(
            ChessRefiner::forstner(),
            ChessRefiner::Forstner(ForstnerConfig::default())
        );
        assert_eq!(
            ChessRefiner::saddle_point(),
            ChessRefiner::SaddlePoint(SaddlePointConfig::default())
        );
    }

    #[test]
    fn multiscale_config_pyramid_default_equals_literal() {
        assert_eq!(
            MultiscaleConfig::pyramid_default(),
            MultiscaleConfig::Pyramid {
                levels: 3,
                min_size: 128,
                refinement_radius: 3
            }
        );
    }

    #[cfg(feature = "ml-refiner")]
    #[test]
    fn chess_refiner_ml_serializes_as_bare_string() {
        let json = serde_json::to_string(&ChessRefiner::Ml).unwrap();
        assert_eq!(json, "\"ml\"");
        let decoded: ChessRefiner = serde_json::from_str(&json).expect("deserialize ml refiner");
        assert_eq!(decoded, ChessRefiner::Ml);
    }
}
