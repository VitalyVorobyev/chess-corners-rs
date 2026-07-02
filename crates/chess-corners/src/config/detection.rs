use box_image_pyramid::PyramidParams;
use chess_corners_core::{ChessParams, OrientationMethod, RadonDetectorParams, RefinerKind};
use serde::{Deserialize, Serialize};

use crate::multiscale::CoarseToFineParams;
use crate::upscale::UpscaleConfig;

use super::{ChessConfig, ChessRefiner, ChessRing, MultiscaleConfig, RadonConfig};

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
            threshold: RadonDetectorParams::DEFAULT_THRESHOLD_REL,
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
            threshold: RadonDetectorParams::DEFAULT_THRESHOLD_REL,
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

    /// Lower this config into the [`ChessParams`] consumed by the
    /// `chess-corners-core` response and detection stages. Use this when
    /// driving the core stage functions directly instead of through
    /// [`Detector`](crate::Detector). Only meaningful when
    /// [`Self::strategy`] is the ChESS variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Radon`], the
    /// ChESS-specific fields fall back to their [`ChessParams::default()`]
    /// values; callers should route through
    /// [`Self::radon_detector_params`] instead.
    pub fn chess_params(&self) -> ChessParams {
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
        params.threshold = self.threshold;
        params.orientation_method = self.orientation_method;
        params
    }

    /// Lower this config into the [`RadonDetectorParams`] consumed by the
    /// `chess-corners-core` Radon response and detection stages. Use this
    /// when driving the core stage functions directly instead of through
    /// [`Detector`](crate::Detector). Only meaningful when
    /// [`Self::strategy`] is the Radon variant.
    ///
    /// When the active strategy is [`DetectionStrategy::Chess`], the
    /// Radon-specific fields fall back to their
    /// [`RadonDetectorParams::default()`] values; callers should route
    /// through [`Self::chess_params`] instead.
    pub fn radon_detector_params(&self) -> RadonDetectorParams {
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
        params.threshold_rel = self.threshold;
        params
    }

    /// Lower this config into the [`CoarseToFineParams`] that drive the
    /// multiscale pipeline. Returns `None` when [`Self::multiscale`]
    /// is [`MultiscaleConfig::SingleScale`]. Both ChESS and Radon honour
    /// the same top-level multiscale settings. Use this when composing
    /// the multiscale stages directly instead of through
    /// [`Detector`](crate::Detector).
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
