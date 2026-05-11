use box_image_pyramid::PyramidParams;
use chess_corners_core::{
    CenterOfMassConfig, ChessParams, ForstnerConfig, OrientationMethod, PeakFitMode,
    RadonDetectorParams, RadonPeakConfig, RefinerKind, SaddlePointConfig,
};
use serde::{Deserialize, Serialize};

use crate::multiscale::CoarseToFineParams;
use crate::upscale::UpscaleConfig;

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

/// Detector acceptance threshold.
///
/// A single, mode-aware enum that replaces the previous `(threshold_mode,
/// threshold_value)` pair. Both the ChESS and Radon pipelines route through
/// the same enum, so the user can't set a relative value while the active
/// detector reads it as absolute.
///
/// - For ChESS the response is the paper's `R = SR − DR − 16·MR`.
///   `Absolute(0.0)` encodes the paper's `R > 0` acceptance contract.
/// - For Radon the response is the squared range `(max − min)²` of the
///   ray-sum range across orientations; pick a positive `Absolute(_)` floor
///   or a `Relative(_)` fraction of the per-frame maximum.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Threshold {
    /// Accept responses `≥ value` in the detector's native score units.
    Absolute(f32),
    /// Accept responses `≥ frac · max(response)` in the current frame.
    /// `frac` is a fraction in `[0.0, 1.0]`.
    Relative(f32),
}

impl Default for Threshold {
    fn default() -> Self {
        // Paper's ChESS contract: any strictly positive response is a corner.
        // Radon presets override this to `Relative(0.01)`.
        Threshold::Absolute(0.0)
    }
}

// ---------------------------------------------------------------------------
// Detector kernel / ring selection
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Selects the `r=5` (canonical) or `r=10`
/// (broad) ring used by the dense response kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChessRing {
    /// Paper-default radius-5 ring (16 samples). Fast and stable for cell
    /// pitches `≥ ~12 px`.
    #[default]
    Canonical,
    /// Radius-10 ring. Larger support window; useful for low-resolution or
    /// heavily blurred imagery where the canonical ring under-samples.
    Broad,
}

/// Descriptor sampling ring selection. Independent of the detector ring
/// chosen by [`ChessRing`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DescriptorMode {
    /// Use the same ring radius as the detector.
    #[default]
    FollowDetector,
    /// Force the descriptor ring to `r=5`.
    Canonical,
    /// Force the descriptor ring to `r=10`.
    Broad,
}

// ---------------------------------------------------------------------------
// Refiner config (unchanged surface)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RefinementMethod {
    #[default]
    CenterOfMass,
    Forstner,
    SaddlePoint,
    RadonPeak,
    /// ML-backed subpixel refinement (feature `ml-refiner`). Runs a
    /// small ONNX model on a normalized intensity patch around each
    /// candidate. Requires the `ml-refiner` feature.
    #[cfg(feature = "ml-refiner")]
    Ml,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RefinerConfig {
    pub kind: RefinementMethod,
    pub center_of_mass: CenterOfMassConfig,
    pub forstner: ForstnerConfig,
    pub saddle_point: SaddlePointConfig,
    pub radon_peak: RadonPeakConfig,
}

impl RefinerConfig {
    /// Construct a [`RefinerConfig`] with all fields specified.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        kind: RefinementMethod,
        center_of_mass: CenterOfMassConfig,
        forstner: ForstnerConfig,
        saddle_point: SaddlePointConfig,
        radon_peak: RadonPeakConfig,
    ) -> Self {
        Self {
            kind,
            center_of_mass,
            forstner,
            saddle_point,
            radon_peak,
        }
    }

    /// Preset that selects the center-of-mass (intensity centroid) refiner.
    /// Fast and stable; best when corners have clear ring support and moderate
    /// blur. This is the library default.
    pub fn center_of_mass() -> Self {
        Self {
            kind: RefinementMethod::CenterOfMass,
            ..Self::default()
        }
    }

    /// Preset that selects the Förstner corner refiner.
    /// Uses a structure-tensor moment approach; more accurate than
    /// center-of-mass on anisotropic corners.
    pub fn forstner() -> Self {
        Self {
            kind: RefinementMethod::Forstner,
            ..Self::default()
        }
    }

    /// Preset that selects the saddle-point refiner.
    /// Fits a local quadratic and locates the saddle; very accurate on
    /// clean, symmetric chessboard corners.
    pub fn saddle_point() -> Self {
        Self {
            kind: RefinementMethod::SaddlePoint,
            ..Self::default()
        }
    }

    /// Preset that selects the Radon-peak refiner.
    /// Reconstructs the corner by projecting intensity along candidate
    /// axes; robust to heavy blur and low contrast.
    pub fn radon_peak() -> Self {
        Self {
            kind: RefinementMethod::RadonPeak,
            ..Self::default()
        }
    }

    /// Convert this config into the lower-level [`RefinerKind`] used by
    /// `chess-corners-core`. Each variant carries its own tuning struct
    /// taken from the corresponding field of this config.
    ///
    /// The ML variant ([`RefinementMethod::Ml`], gated on the
    /// `ml-refiner` feature) does not map to a core [`RefinerKind`]
    /// variant — the ML refiner lives in the facade, not the core
    /// crate. When `kind == Ml`, this function returns the
    /// center-of-mass fallback that the multiscale pipeline uses for
    /// the coarse pass; the actual ML refinement is dispatched by
    /// [`crate::Detector`] before this method is consulted.
    pub fn to_refiner_kind(&self) -> RefinerKind {
        match self.kind {
            RefinementMethod::CenterOfMass => RefinerKind::CenterOfMass(self.center_of_mass),
            RefinementMethod::Forstner => RefinerKind::Forstner(self.forstner),
            RefinementMethod::SaddlePoint => RefinerKind::SaddlePoint(self.saddle_point),
            RefinementMethod::RadonPeak => RefinerKind::RadonPeak(self.radon_peak),
            #[cfg(feature = "ml-refiner")]
            RefinementMethod::Ml => RefinerKind::CenterOfMass(self.center_of_mass),
        }
    }
}

// ---------------------------------------------------------------------------
// Multiscale parameters
// ---------------------------------------------------------------------------

/// Multiscale (coarse-to-fine) pipeline parameters, attached to a
/// [`ChessStrategy`] when the user opts into pyramid detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct MultiscaleParams {
    /// Number of pyramid levels (≥ 1). Level 0 is the base image; each
    /// subsequent level is a 2× box-filter downsample.
    pub pyramid_levels: u8,
    /// Minimum short-edge length in pixels. The pyramid stops once the
    /// next level would fall below this size.
    pub pyramid_min_size: usize,
    /// ROI half-radius at the coarse level used to refine each seed into
    /// the base image, in coarse-level pixels.
    pub refinement_radius: u32,
}

impl Default for MultiscaleParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            pyramid_min_size: 128,
            refinement_radius: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy structs
// ---------------------------------------------------------------------------

/// Configuration for the ChESS detector branch of [`DetectionStrategy`].
///
/// Carries the ring choice and NMS / clustering thresholds (in input-image
/// pixels). Multiscale settings are now at the top level of
/// [`DetectorConfig`] and apply to both ChESS and Radon strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessStrategy {
    pub ring: ChessRing,
    /// Non-maximum-suppression half-radius, in input-image pixels.
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbours in the NMS window
    /// required to accept a peak.
    pub min_cluster_size: u32,
}

impl Default for ChessStrategy {
    fn default() -> Self {
        Self {
            ring: ChessRing::Canonical,
            nms_radius: 2,
            min_cluster_size: 2,
        }
    }
}

/// Configuration for the whole-image Radon detector branch of
/// [`DetectionStrategy`].
///
/// All radii and counts are in **working-resolution** pixels (i.e. after
/// `image_upsample`). Multiscale settings are at the top level of
/// [`DetectorConfig`] and apply to both strategies.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RadonStrategy {
    /// Half-length of each ray (working-resolution pixels). The ray has
    /// `2·ray_radius + 1` samples. Paper default at `image_upsample = 2`
    /// is `ray_radius = 4`.
    pub ray_radius: u32,
    /// Image-level supersampling factor. `1` operates on the input grid;
    /// `2` bilinearly upsamples first (paper default). Values ≥ 3 are
    /// clamped to 2 by the core detector.
    pub image_upsample: u32,
    /// Half-size of the box blur applied to the response map. `0` disables
    /// blurring; `1` yields a 3×3 box.
    pub response_blur_radius: u32,
    /// Peak-fit mode for the 3-point subpixel refinement.
    pub peak_fit: PeakFitMode,
    /// Non-maximum-suppression half-radius, in working-resolution pixels.
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbours in the NMS window
    /// required to accept a peak.
    pub min_cluster_size: u32,
}

impl Default for RadonStrategy {
    fn default() -> Self {
        Self {
            ray_radius: 4,
            image_upsample: 2,
            response_blur_radius: 1,
            peak_fit: PeakFitMode::Gaussian,
            nms_radius: 4,
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
    Chess(ChessStrategy),
    /// Whole-image Radon (Duda-Frese) detection. Currently single-scale.
    Radon(RadonStrategy),
}

impl Default for DetectionStrategy {
    fn default() -> Self {
        DetectionStrategy::Chess(ChessStrategy::default())
    }
}

// ---------------------------------------------------------------------------
// DetectorConfig
// ---------------------------------------------------------------------------

/// High-level detection configuration.
///
/// Build one with [`DetectorConfig::single_scale`],
/// [`DetectorConfig::multiscale`], [`DetectorConfig::radon`], or
/// [`DetectorConfig::radon_multiscale`] and tweak only the fields you need.
/// The detector translates this into the low-level [`ChessParams`] /
/// [`RadonDetectorParams`] consumed by `chess-corners-core` at the detection
/// boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DetectorConfig {
    /// Detector dispatch: ChESS or Radon, each carrying its own tuning.
    pub strategy: DetectionStrategy,
    /// Acceptance threshold. Same enum is honoured by both detectors.
    pub threshold: Threshold,
    /// Optional coarse-to-fine multiscale settings. `None` = single-scale.
    /// Honoured by both the ChESS and Radon strategies.
    pub multiscale: Option<MultiscaleParams>,
    /// Subpixel refiner selection and per-variant tuning.
    pub refiner: RefinerConfig,
    /// Orientation-fit method used when building corner descriptors.
    pub orientation_method: OrientationMethod,
    /// Descriptor ring choice (independent of the detector ring).
    pub descriptor_mode: DescriptorMode,
    /// Optional pre-pipeline integer upscaling. Disabled by default.
    pub upscale: UpscaleConfig,
    /// Merge radius (base-image pixels) for cross-level / cross-seed
    /// duplicate suppression. Honoured by both detectors.
    pub merge_radius: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self::single_scale()
    }
}

impl DetectorConfig {
    /// Single-scale ChESS preset. Recommended for images where the cell
    /// size comfortably exceeds the canonical ring's ~12 px support.
    pub fn single_scale() -> Self {
        Self {
            strategy: DetectionStrategy::Chess(ChessStrategy::default()),
            threshold: Threshold::Absolute(0.0),
            multiscale: None,
            refiner: RefinerConfig::default(),
            orientation_method: OrientationMethod::default(),
            descriptor_mode: DescriptorMode::default(),
            upscale: UpscaleConfig::default(),
            merge_radius: 3.0,
        }
    }

    /// Three-level coarse-to-fine ChESS preset. Recommended for images
    /// ≥ 1 MP or with cell sizes varying significantly across the frame.
    pub fn multiscale() -> Self {
        Self {
            strategy: DetectionStrategy::Chess(ChessStrategy::default()),
            multiscale: Some(MultiscaleParams {
                pyramid_levels: 3,
                pyramid_min_size: 128,
                refinement_radius: 3,
            }),
            ..Self::single_scale()
        }
    }

    /// Whole-image Radon detector preset. Useful for heavy blur, low
    /// contrast, or cells smaller than the ChESS ring support.
    /// Single-scale; use [`Self::radon_multiscale`] for coarse-to-fine
    /// Radon detection on larger frames.
    pub fn radon() -> Self {
        Self {
            strategy: DetectionStrategy::Radon(RadonStrategy::default()),
            threshold: Threshold::Relative(0.01),
            multiscale: None,
            ..Self::single_scale()
        }
    }

    /// Coarse-to-fine Radon preset. Useful for blurry / low-contrast
    /// imagery on large frames where the SAT-based response benefits
    /// from coarse seeding. Single-scale Radon ([`Self::radon`]) remains
    /// the right pick for small frames or tight per-frame latency budgets.
    pub fn radon_multiscale() -> Self {
        Self {
            strategy: DetectionStrategy::Radon(RadonStrategy::default()),
            threshold: Threshold::Relative(0.01),
            multiscale: Some(MultiscaleParams::default()),
            ..Self::single_scale()
        }
    }

    /// Translate this config into the low-level [`ChessParams`] consumed
    /// by `chess-corners-core`. Only meaningful when
    /// [`Self::strategy`] is the ChESS variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Radon`], the
    /// ChESS-specific fields (`use_radius10`, NMS, cluster size) fall
    /// back to their `ChessParams::default()` values; callers should
    /// route through [`Self::to_radon_detector_params`] instead.
    pub fn to_chess_params(&self) -> ChessParams {
        let mut params = ChessParams::default();
        if let DetectionStrategy::Chess(chess) = &self.strategy {
            params.use_radius10 = matches!(chess.ring, ChessRing::Broad);
            params.nms_radius = chess.nms_radius;
            params.min_cluster_size = chess.min_cluster_size;
        }
        params.descriptor_use_radius10 = match self.descriptor_mode {
            DescriptorMode::FollowDetector => None,
            DescriptorMode::Canonical => Some(false),
            DescriptorMode::Broad => Some(true),
        };
        apply_threshold(&mut params, self.threshold);
        params.refiner = self.refiner.to_refiner_kind();
        params.orientation_method = self.orientation_method;
        params
    }

    /// Translate this config into the low-level [`RadonDetectorParams`]
    /// consumed by `chess-corners-core`. Only meaningful when
    /// [`Self::strategy`] is the Radon variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Chess`], the
    /// Radon-specific fields fall back to their `RadonDetectorParams::
    /// default()` values; callers should route through
    /// [`Self::to_chess_params`] instead.
    pub fn to_radon_detector_params(&self) -> RadonDetectorParams {
        let mut params = RadonDetectorParams::default();
        if let DetectionStrategy::Radon(radon) = &self.strategy {
            params.ray_radius = radon.ray_radius;
            params.image_upsample = radon.image_upsample;
            params.response_blur_radius = radon.response_blur_radius;
            params.peak_fit = radon.peak_fit;
            params.nms_radius = radon.nms_radius;
            params.min_cluster_size = radon.min_cluster_size;
        }
        apply_threshold_radon(&mut params, self.threshold);
        params
    }

    /// Translate this config into the [`CoarseToFineParams`] that drive
    /// the multiscale pipeline. Returns `None` when `self.multiscale` is
    /// `None` (single-scale). Both ChESS and Radon honour the same
    /// top-level multiscale settings.
    pub fn to_coarse_to_fine_params(&self) -> Option<CoarseToFineParams> {
        let ms = self.multiscale?;
        let mut cfg = CoarseToFineParams::default();
        let mut pyramid = PyramidParams::default();
        pyramid.num_levels = ms.pyramid_levels;
        pyramid.min_size = ms.pyramid_min_size;
        cfg.pyramid = pyramid;
        cfg.refinement_radius = ms.refinement_radius;
        cfg.merge_radius = self.merge_radius;
        Some(cfg)
    }
}

/// Renamed to [`DetectorConfig`]; this alias keeps existing call sites
/// compiling and will be removed in 0.11.0.
pub type ChessConfig = DetectorConfig;

// ---------------------------------------------------------------------------
// Threshold → core param translation
// ---------------------------------------------------------------------------

/// Translate a [`Threshold`] into the core's `(threshold_abs,
/// threshold_rel)` pair used by [`ChessParams`].
///
/// `Absolute(v)` sets `threshold_abs = Some(v)` (overrides relative);
/// `Relative(f)` sets `threshold_abs = None` and `threshold_rel = f`.
fn apply_threshold(params: &mut ChessParams, threshold: Threshold) {
    match threshold {
        Threshold::Absolute(value) => {
            params.threshold_abs = Some(value);
        }
        Threshold::Relative(frac) => {
            params.threshold_abs = None;
            params.threshold_rel = frac;
        }
    }
}

/// Translate a [`Threshold`] into the core's `(threshold_abs,
/// threshold_rel)` pair used by [`RadonDetectorParams`].
fn apply_threshold_radon(params: &mut RadonDetectorParams, threshold: Threshold) {
    match threshold {
        Threshold::Absolute(value) => {
            params.threshold_abs = Some(value);
        }
        Threshold::Relative(frac) => {
            params.threshold_abs = None;
            params.threshold_rel = frac;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_strategy_chess(cfg: &DetectorConfig) -> &ChessStrategy {
        match &cfg.strategy {
            DetectionStrategy::Chess(c) => c,
            other => panic!("expected ChESS strategy, got {other:?}"),
        }
    }

    fn assert_strategy_radon(cfg: &DetectorConfig) -> &RadonStrategy {
        match &cfg.strategy {
            DetectionStrategy::Radon(r) => r,
            other => panic!("expected Radon strategy, got {other:?}"),
        }
    }

    #[test]
    fn default_is_single_scale_chess_with_paper_threshold() {
        let cfg = DetectorConfig::default();
        let chess = assert_strategy_chess(&cfg);
        assert_eq!(chess.ring, ChessRing::Canonical);
        assert_eq!(chess.nms_radius, 2);
        assert_eq!(chess.min_cluster_size, 2);
        assert!(cfg.multiscale.is_none());
        assert_eq!(cfg.threshold, Threshold::Absolute(0.0));
        assert_eq!(cfg.descriptor_mode, DescriptorMode::FollowDetector);
        assert_eq!(cfg.merge_radius, 3.0);
        assert!(cfg.to_coarse_to_fine_params().is_none());

        let params = cfg.to_chess_params();
        assert!(!params.use_radius10);
        assert_eq!(params.descriptor_use_radius10, None);
        assert_eq!(params.threshold_abs, Some(0.0));
        assert_eq!(params.nms_radius, 2);
        assert_eq!(params.min_cluster_size, 2);
        assert_eq!(
            params.refiner,
            RefinerKind::CenterOfMass(CenterOfMassConfig::default())
        );
    }

    #[test]
    fn relative_threshold_clears_absolute() {
        let cfg = DetectorConfig {
            threshold: Threshold::Relative(0.15),
            ..DetectorConfig::single_scale()
        };
        let params = cfg.to_chess_params();
        assert_eq!(params.threshold_abs, None);
        assert!((params.threshold_rel - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn absolute_threshold_overrides_relative() {
        let cfg = DetectorConfig {
            threshold: Threshold::Absolute(7.5),
            ..DetectorConfig::single_scale()
        };
        let params = cfg.to_chess_params();
        assert_eq!(params.threshold_abs, Some(7.5));
    }

    #[test]
    fn multiscale_preset_carries_pyramid_params() {
        let cfg = DetectorConfig::multiscale();
        // multiscale is now at the top level, not nested in the strategy
        let ms = cfg
            .multiscale
            .expect("multiscale preset must carry MultiscaleParams");
        assert_eq!(ms.pyramid_levels, 3);
        assert_eq!(ms.pyramid_min_size, 128);
        assert_eq!(ms.refinement_radius, 3);

        let cf = cfg
            .to_coarse_to_fine_params()
            .expect("multiscale config must produce CoarseToFineParams");
        assert_eq!(cf.pyramid.num_levels, 3);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn radon_preset_uses_radon_strategy_and_relative_threshold() {
        let cfg = DetectorConfig::radon();
        let radon = assert_strategy_radon(&cfg);
        assert_eq!(radon.ray_radius, 4);
        assert_eq!(radon.image_upsample, 2);
        assert_eq!(radon.response_blur_radius, 1);
        assert_eq!(radon.peak_fit, PeakFitMode::Gaussian);
        assert_eq!(radon.nms_radius, 4);
        assert_eq!(radon.min_cluster_size, 2);
        assert_eq!(cfg.threshold, Threshold::Relative(0.01));
        assert!(cfg.multiscale.is_none());
        assert!(cfg.to_coarse_to_fine_params().is_none());

        let radon_params = cfg.to_radon_detector_params();
        assert_eq!(radon_params.ray_radius, 4);
        assert_eq!(radon_params.image_upsample, 2);
        assert_eq!(radon_params.threshold_abs, None);
        assert!((radon_params.threshold_rel - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn radon_multiscale_preset_carries_pyramid_params() {
        let cfg = DetectorConfig::radon_multiscale();
        assert_strategy_radon(&cfg);
        assert_eq!(cfg.threshold, Threshold::Relative(0.01));
        let ms = cfg
            .multiscale
            .expect("radon_multiscale preset must carry MultiscaleParams");
        assert_eq!(ms.pyramid_levels, 3);
        assert_eq!(ms.pyramid_min_size, 128);
        assert_eq!(ms.refinement_radius, 3);

        let cf = cfg
            .to_coarse_to_fine_params()
            .expect("radon_multiscale config must produce CoarseToFineParams");
        assert_eq!(cf.pyramid.num_levels, 3);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn broad_ring_and_forstner_refiner_propagate_to_params() {
        let cfg = DetectorConfig {
            strategy: DetectionStrategy::Chess(ChessStrategy {
                ring: ChessRing::Broad,
                ..ChessStrategy::default()
            }),
            descriptor_mode: DescriptorMode::Canonical,
            refiner: RefinerConfig {
                kind: RefinementMethod::Forstner,
                forstner: ForstnerConfig {
                    max_offset: 2.0,
                    ..ForstnerConfig::default()
                },
                ..RefinerConfig::default()
            },
            ..DetectorConfig::single_scale()
        };

        let params = cfg.to_chess_params();
        assert!(params.use_radius10);
        assert_eq!(params.descriptor_use_radius10, Some(false));
        assert_eq!(
            params.refiner,
            RefinerKind::Forstner(ForstnerConfig {
                max_offset: 2.0,
                ..ForstnerConfig::default()
            })
        );
    }

    #[test]
    fn single_scale_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::single_scale();
        let json = serde_json::to_string(&cfg).expect("serialize single-scale config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize single-scale config");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn multiscale_preset_round_trips_through_serde() {
        let cfg = DetectorConfig::multiscale();
        let json = serde_json::to_string(&cfg).expect("serialize multiscale config");
        let decoded: DetectorConfig =
            serde_json::from_str(&json).expect("deserialize multiscale config");
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
    fn threshold_round_trips_with_externally_tagged_payload() {
        let abs = Threshold::Absolute(3.5);
        let abs_json = serde_json::to_string(&abs).expect("serialize absolute threshold");
        assert!(abs_json.contains("absolute"));
        let abs_decoded: Threshold =
            serde_json::from_str(&abs_json).expect("deserialize absolute threshold");
        assert_eq!(abs_decoded, abs);

        let rel = Threshold::Relative(0.42);
        let rel_json = serde_json::to_string(&rel).expect("serialize relative threshold");
        assert!(rel_json.contains("relative"));
        let rel_decoded: Threshold =
            serde_json::from_str(&rel_json).expect("deserialize relative threshold");
        assert_eq!(rel_decoded, rel);
    }
}
