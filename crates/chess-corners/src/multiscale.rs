//! Unified corner detection (single or multiscale).
//!
//! This module implements the coarse-to-fine detector used by the
//! `chess-corners` facade. It can:
//!
//! - run a single-scale detection when `pyramid.num_levels <= 1`, or
//! - build an image pyramid, run a coarse detector on the smallest
//!   level, and refine each seed in the base image (coarse-to-fine)
//!   before merging duplicates.
//!
//! The generic driver [`detect_multiscale`] is parameterised over
//! [`DenseDetector`], so both the ChESS and Radon detectors flow
//! through the same orchestrator. The dispatch behind
//! [`crate::Detector::detect_u8`] selects the detector ZST per
//! [`crate::DetectionStrategy`]. End users should reach detection
//! through [`crate::Detector`].

#[cfg(feature = "ml-refiner")]
use crate::ml_refiner;
use crate::{DetectionStrategy, DetectorConfig};
use box_image_pyramid::{build_pyramid, PyramidBuffers, PyramidParams};
use chess_corners_core::{describe_corners, merge_corners_simple, Corner};
use chess_corners_core::{ChessBuffers, ChessDetector, CornerDescriptor, DenseDetector};
#[cfg(feature = "ml-refiner")]
use chess_corners_core::{ChessParams, ResponseMap};
use chess_corners_core::{
    CornerRefiner, ImageView, OrientationMethod, RadonBuffers, RadonDetector, Refiner, RefinerKind,
};

/// Bridge from `chess_corners_core::ImageView` to `box_image_pyramid::ImageView`.
fn to_pyramid_view(v: ImageView<'_>) -> box_image_pyramid::ImageView<'_> {
    // invariant: v was already validated as a coherent ImageView, so the pyramid view cannot fail.
    box_image_pyramid::ImageView::new(v.width(), v.height(), v.data()).unwrap()
}
#[cfg(feature = "tracing")]
use tracing::info_span;

/// Parameters controlling the coarse-to-fine multiscale detector.
///
/// The default keeps `num_levels = 1`, so callers start in the single-scale
/// regime unless they explicitly opt into a pyramid.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub(crate) struct CoarseToFineParams {
    /// Image pyramid shape and construction parameters.
    pub pyramid: PyramidParams,
    /// ROI radius at the coarse level (ignored when `pyramid.num_levels <= 1`).
    /// Expressed in coarse-level pixels and automatically scaled to the base
    /// image, with a minimum enforced to keep refinement away from borders.
    pub refinement_radius: u32,
    /// Radius (in base-image pixels) used to merge near-duplicate refined
    /// corners after coarse-to-fine refinement.
    pub merge_radius: f32,
}

impl Default for CoarseToFineParams {
    fn default() -> Self {
        Self {
            pyramid: PyramidParams::default(),
            // Smaller coarse-level ROI around each coarse prediction. With the
            // default 3-level pyramid this maps to roughly a 12px radius
            // (~25px window) at the base resolution.
            refinement_radius: 3,
            // merge duplicates within ~3 pixels
            merge_radius: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ML-refiner adapter: kept as a closure-driven specialisation because it
// is ChESS-only and threads mutable state through the per-seed loop.
// ---------------------------------------------------------------------------

#[cfg(feature = "ml-refiner")]
fn detect_with_ml_refiner(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    ml_state: &mut ml_refiner::MlRefinerState,
) -> Vec<Corner> {
    ml_refiner::detect_corners_with_ml(resp, params, image, ml_state)
}

fn refiner_radius(refiner_kind: &RefinerKind) -> i32 {
    Refiner::from_kind(refiner_kind.clone()).radius()
}

// ---------------------------------------------------------------------------
// Shared coarse-to-fine pipeline
// ---------------------------------------------------------------------------

/// Pre-computed parameters for per-seed ROI refinement.
struct RoiContext {
    inv_scale: f32,
    border: i32,
    safe_margin: i32,
    roi_r: i32,
    base_w_i: i32,
    base_h_i: i32,
}

impl RoiContext {
    /// Compute a clamped, validated ROI around a coarse seed projected to base
    /// image coordinates. Returns `None` if the seed is too close to the border
    /// or the resulting ROI is too small.
    fn compute_roi(&self, c: &Corner) -> Option<(i32, i32, i32, i32)> {
        let cx = (c.x * self.inv_scale).round() as i32;
        let cy = (c.y * self.inv_scale).round() as i32;

        if cx < self.safe_margin
            || cy < self.safe_margin
            || cx >= self.base_w_i - self.safe_margin
            || cy >= self.base_h_i - self.safe_margin
        {
            return None;
        }

        let mut x0 = cx - self.roi_r;
        let mut y0 = cy - self.roi_r;
        let mut x1 = cx + self.roi_r + 1;
        let mut y1 = cy + self.roi_r + 1;

        let min_xy = self.border;
        let max_x = self.base_w_i - self.border;
        let max_y = self.base_h_i - self.border;

        if x0 < min_xy {
            x0 = min_xy;
        }
        if y0 < min_xy {
            y0 = min_xy;
        }
        if x1 > max_x {
            x1 = max_x;
        }
        if y1 > max_y {
            y1 = max_y;
        }

        if x1 - x0 <= 2 * self.border || y1 - y0 <= 2 * self.border {
            return None;
        }

        Some((x0, y0, x1, y1))
    }
}

fn make_roi_context(
    base: ImageView<'_>,
    coarse_scale: f32,
    detector_border: i32,
    refine_border: i32,
    cf: &CoarseToFineParams,
) -> RoiContext {
    let border = (detector_border + refine_border).max(0);
    let safe_margin = border + 1;
    let roi_r_base = (cf.refinement_radius as f32 / coarse_scale).ceil() as i32;
    let min_roi_r = border + 2;

    RoiContext {
        inv_scale: 1.0 / coarse_scale,
        border,
        safe_margin,
        roi_r: roi_r_base.max(min_roi_r),
        base_w_i: base.width() as i32,
        base_h_i: base.height() as i32,
    }
}

// ---------------------------------------------------------------------------
// Shared coarse-to-fine traversal
// ---------------------------------------------------------------------------

/// The per-stage detect/refine operations that vary between the two
/// multiscale pipelines (detector-driven vs ML-refiner). The traversal
/// *around* these operations — pyramid build, single-scale /
/// single-level / coarse-to-fine branching, ROI carving, base-frame
/// shifting, merge and describe — is shared in [`traverse`], so the two
/// callers cannot drift apart on anything but these three steps.
trait TraversalStep {
    /// Produce final corners over a full image view. Used by the
    /// single-scale and single-level branches. Positions are in the
    /// view's own frame.
    fn detect_full(&mut self, view: ImageView<'_>) -> Vec<Corner>;

    /// Produce coarse seeds over the coarsest pyramid level. Seeds only
    /// need positions accurate enough to carve a base-image ROI, so the
    /// ML pipeline seeds with the classic refiner here rather than the
    /// (expensive) model. Positions are in the coarse-level frame.
    fn detect_seeds(&mut self, view: ImageView<'_>) -> Vec<Corner>;

    /// Refine one ROI `(x0, y0, x1, y1)` of `base` (bounds in
    /// base-image pixels). Returned corner positions are **patch-local**
    /// (origin at the ROI's top-left); [`traverse`] shifts them back
    /// into base-image coordinates.
    fn refine_roi(&mut self, base: ImageView<'_>, roi: (i32, i32, i32, i32)) -> Vec<Corner>;
}

/// Post-detection shaping shared by every traversal branch: the ROI
/// margins, the descriptor sampling ring, the orientation method, and
/// the merge radius applied when running without a pyramid.
struct TraversalShape {
    detector_border: i32,
    refine_border: i32,
    descriptor_ring_radius: u32,
    orientation_method: Option<OrientationMethod>,
    /// Merge radius for the single-scale (no-pyramid) branch. The
    /// pyramid branches read [`CoarseToFineParams::merge_radius`], which
    /// the config lowering keeps equal to this value.
    single_scale_merge_radius: f32,
}

/// Shared coarse-to-fine traversal. Runs a single-scale detection when
/// `multiscale` is `None` or resolves to a 1-level pyramid; otherwise
/// builds the pyramid, seeds on the coarsest level, and refines each
/// seed inside a base-image ROI before merging duplicates and producing
/// descriptors. All numerical variation lives in `step`.
fn traverse<S: TraversalStep>(
    base: ImageView<'_>,
    multiscale: Option<&CoarseToFineParams>,
    pyramid_buffers: &mut PyramidBuffers,
    shape: &TraversalShape,
    step: &mut S,
) -> Vec<CornerDescriptor> {
    let base_view = ImageView::from_u8_slice(base.width(), base.height(), base.data())
        .expect("base image dimensions must match buffer length");

    // Single-scale path: run the detector once on the full base view,
    // merge, describe.
    let Some(cf) = multiscale else {
        let mut corners = step.detect_full(base_view);
        let merged = merge_corners_simple(&mut corners, shape.single_scale_merge_radius);
        return describe_corners(
            base.data(),
            base.width(),
            base.height(),
            shape.descriptor_ring_radius,
            merged,
            shape.orientation_method,
        );
    };

    let pyramid = build_pyramid(to_pyramid_view(base), &cf.pyramid, pyramid_buffers);
    if pyramid.levels.is_empty() {
        return Vec::new();
    }

    // Single-level pyramid: same as single-scale but on the pyramid
    // level's data (which equals the base for num_levels==1 / no
    // downsampling).
    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        let lvl_view = ImageView::from_u8_slice(lvl.img.width, lvl.img.height, lvl.img.data)
            .expect("pyramid level dimensions must match buffer length");
        let mut corners = step.detect_full(lvl_view);
        let merged = merge_corners_simple(&mut corners, cf.merge_radius);
        return describe_corners(
            lvl.img.data,
            lvl.img.width,
            lvl.img.height,
            shape.descriptor_ring_radius,
            merged,
            shape.orientation_method,
        );
    }

    // --- Coarse-to-fine path ---

    // invariant: pyramid was built from a validated input image, so at least one level exists.
    let coarse_lvl = pyramid.levels.last().unwrap();
    let coarse_w = coarse_lvl.img.width;
    let coarse_h = coarse_lvl.img.height;

    #[cfg(feature = "tracing")]
    let coarse_span = info_span!("coarse_detect", w = coarse_w, h = coarse_h).entered();
    // invariant: coarse level dimensions come from the pyramid which already validated them.
    let coarse_view = ImageView::from_u8_slice(coarse_w, coarse_h, coarse_lvl.img.data).unwrap();
    let coarse_corners = step.detect_seeds(coarse_view);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let roi_ctx = make_roi_context(
        base,
        coarse_lvl.scale,
        shape.detector_border,
        shape.refine_border,
        cf,
    );

    #[cfg(feature = "tracing")]
    let refine_span = info_span!(
        "refine",
        seeds = coarse_corners.len(),
        roi_r = roi_ctx.roi_r
    )
    .entered();

    // Per-seed refinement runs sequentially: the detector's patch
    // response computation mutates shared scratch buffers, and rayon
    // parallelism over seeds would require cloning those buffers per
    // worker for little gain (the heavy per-seed work is a small ROI).
    let mut refined: Vec<Corner> = Vec::new();
    for c in coarse_corners {
        let Some(roi_bounds) = roi_ctx.compute_roi(&c) else {
            continue;
        };
        let (x0, y0, _x1, _y1) = roi_bounds;
        let mut patch = step.refine_roi(base, roi_bounds);
        // Shift patch-local refined positions to base coords.
        for pc in &mut patch {
            pc.x += x0 as f32;
            pc.y += y0 as f32;
        }
        refined.extend(patch);
    }

    #[cfg(feature = "tracing")]
    drop(refine_span);

    #[cfg(feature = "tracing")]
    let merge_span = info_span!(
        "merge",
        merge_radius = cf.merge_radius,
        candidates = refined.len()
    )
    .entered();
    let merged = merge_corners_simple(&mut refined, cf.merge_radius);
    #[cfg(feature = "tracing")]
    drop(merge_span);

    describe_corners(
        base.data(),
        base.width(),
        base.height(),
        shape.descriptor_ring_radius,
        merged,
        shape.orientation_method,
    )
}

/// [`TraversalStep`] for the generic detector-driven path. Both
/// `detect_full` and `detect_seeds` are the same operation (compute
/// response → detect peaks → image-domain refine); only the ML pipeline
/// distinguishes them. The single owned [`Refiner`] is reused across
/// coarse seeds and every ROI, matching the "no per-corner allocation"
/// contract.
struct DetectorStep<'a, D: DenseDetector> {
    detector: &'a D,
    params: &'a D::Params,
    buffers: &'a mut D::Buffers,
    refiner: Refiner,
    refine_border: i32,
}

impl<D: DenseDetector> TraversalStep for DetectorStep<'_, D> {
    fn detect_full(&mut self, view: ImageView<'_>) -> Vec<Corner> {
        let resp = self
            .detector
            .compute_response(view, self.params, self.buffers);
        let peaks = self
            .detector
            .detect_corners(&resp, self.params, self.refine_border);
        self.detector
            .refine_peaks_on_image(peaks, view, &resp, &mut self.refiner)
    }

    fn detect_seeds(&mut self, view: ImageView<'_>) -> Vec<Corner> {
        // Identical to `detect_full` for the generic path: coarse seeds
        // use the same detector and image-domain refiner.
        self.detect_full(view)
    }

    fn refine_roi(&mut self, base: ImageView<'_>, roi: (i32, i32, i32, i32)) -> Vec<Corner> {
        let (x0, y0, _x1, _y1) = roi;
        let resp = self
            .detector
            .compute_response_patch(base, roi, self.params, self.buffers);
        let peaks = self
            .detector
            .detect_corners(&resp, self.params, self.refine_border);
        if peaks.is_empty() {
            return Vec::new();
        }
        // Image-domain refinement over the base image with origin
        // [x0, y0]: the refiner sees patch-local seed coords and samples
        // base pixels at (cx + x0, cy + y0). The detector decides
        // whether its response is forwardable to the refiner (ChESS →
        // yes; Radon → no, returns peaks as-is).
        let patch_image =
            ImageView::with_origin(base.width(), base.height(), base.data(), [x0, y0])
                .expect("base image dimensions must match buffer length");
        self.detector
            .refine_peaks_on_image(peaks, patch_image, &resp, &mut self.refiner)
    }
}

// ---------------------------------------------------------------------------
// Generic multiscale orchestrator (driven by DenseDetector)
// ---------------------------------------------------------------------------

/// Pixel-shape arguments common to every detector path (descriptor
/// sampling + orientation + post-detection merge). Lets the generic
/// orchestrator stay parameter-symmetric over [`DenseDetector`].
struct DetectorShape<'r> {
    refiner_kind: &'r RefinerKind,
    descriptor_ring_radius: u32,
    orientation_method: Option<OrientationMethod>,
    merge_radius: f32,
}

/// Generic multiscale corner detection driver.
///
/// Runs a single-scale detection when `multiscale` is `None` or
/// resolves to a 1-level pyramid; otherwise builds the pyramid,
/// detects seeds on the coarsest level, and refines each seed inside
/// a base-image ROI before merging duplicates and producing
/// descriptors.
///
/// The detector is selected through the [`DenseDetector`] trait, so
/// both ChESS and Radon share the same control flow. Detector-domain
/// peak extraction stays inside [`DenseDetector::detect_corners`]
/// (which returns peaks with the response-map subpixel position only
/// — e.g. the ChESS quadratic / Radon 3-point Gaussian); image-domain
/// refinement (`CenterOfMassRefiner`, `ForstnerRefiner`, …) runs as a
/// separate post-detection stage via
/// [`refine_corners_on_image`](chess_corners_core::refine_corners_on_image).
///
/// `descriptor_ring_radius` and `orientation_method` are sourced from
/// the ChESS-derived params even when the active detector is Radon —
/// descriptor sampling and orientation are detector-agnostic stages
/// downstream of peak extraction.
fn detect_multiscale<D: DenseDetector>(
    base: ImageView<'_>,
    detector: &D,
    params: &D::Params,
    detector_buffers: &mut D::Buffers,
    pyramid_buffers: &mut PyramidBuffers,
    multiscale: Option<&CoarseToFineParams>,
    shape: &DetectorShape<'_>,
) -> Vec<CornerDescriptor> {
    // The refiner only contributes to the ROI margin when the
    // detector actually consumes it. Detectors whose
    // `refine_peaks_on_image` is a no-op (today: Radon) declare
    // `refines_on_image() == false` so that switching the active
    // strategy's refiner between e.g. `CenterOfMass` and
    // `SaddlePoint` doesn't silently shrink Radon's valid seed
    // area near the image border.
    let refine_border = if detector.refines_on_image() {
        refiner_radius(shape.refiner_kind)
    } else {
        0
    };

    let traversal = TraversalShape {
        detector_border: detector.roi_border(params),
        refine_border,
        descriptor_ring_radius: shape.descriptor_ring_radius,
        orientation_method: shape.orientation_method,
        single_scale_merge_radius: shape.merge_radius,
    };
    let mut step = DetectorStep {
        detector,
        params,
        buffers: detector_buffers,
        refiner: Refiner::from_kind(shape.refiner_kind.clone()),
        refine_border,
    };
    traverse(base, multiscale, pyramid_buffers, &traversal, &mut step)
}

// ---------------------------------------------------------------------------
// Detector-typed entry points (called by the facade Detector dispatch)
// ---------------------------------------------------------------------------

/// Detect corners through the generic orchestrator. The `cfg.strategy`
/// selects between [`ChessDetector`] and [`RadonDetector`]; both
/// flow through the same control flow.
pub(crate) fn detect_with_buffers(
    base: ImageView<'_>,
    cfg: &DetectorConfig,
    pyramid_buffers: &mut PyramidBuffers,
    chess_buffers: &mut ChessBuffers,
    radon_buffers: &mut RadonBuffers,
) -> Vec<CornerDescriptor> {
    let multiscale = cfg.coarse_to_fine_params();

    match &cfg.strategy {
        DetectionStrategy::Chess(_) => {
            let chess_params = cfg.chess_params();
            let refiner_kind = chess_params.refiner.clone();
            let shape = DetectorShape {
                refiner_kind: &refiner_kind,
                descriptor_ring_radius: chess_params.ring_radius(),
                orientation_method: chess_params.orientation_method,
                merge_radius: cfg.merge_radius,
            };
            detect_multiscale(
                base,
                &ChessDetector,
                &chess_params,
                chess_buffers,
                pyramid_buffers,
                multiscale.as_ref(),
                &shape,
            )
        }
        DetectionStrategy::Radon(_) => {
            let radon_params = cfg.radon_detector_params();
            // Radon's subpixel is its built-in Gaussian peak fit:
            // `refine_peaks_on_image` is a no-op and `refines_on_image()`
            // is false, so this refiner kind is constructed but never
            // applied. Use the cheapest default.
            let refiner_kind = RefinerKind::default();
            // Radon strategies don't carry a descriptor-ring knob;
            // descriptors sample the canonical r=5 ring downstream.
            // Orientation method is top-level on DetectorConfig.
            let shape = DetectorShape {
                refiner_kind: &refiner_kind,
                descriptor_ring_radius: chess_corners_core::ChessParams::default().ring_radius(),
                orientation_method: cfg.orientation_method,
                merge_radius: cfg.merge_radius,
            };
            detect_multiscale(
                base,
                &RadonDetector,
                &radon_params,
                radon_buffers,
                pyramid_buffers,
                multiscale.as_ref(),
                &shape,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// ML refiner path (ChESS-only specialisation)
// ---------------------------------------------------------------------------

/// ML-refiner detection path. ChESS-only: the ML model expects
/// ChESS-shaped intensity patches, so Radon+ML is a category error
/// and falls back to the generic Radon path.
///
/// Kept as a separate specialisation because the ML refiner threads
/// per-frame mutable state (`MlRefinerState`) through the per-seed
/// loop, which the generic [`detect_multiscale`] driver intentionally
/// does not.
#[cfg(feature = "ml-refiner")]
pub(crate) fn detect_with_ml(
    base: ImageView<'_>,
    cfg: &DetectorConfig,
    pyramid_buffers: &mut PyramidBuffers,
    chess_buffers: &mut ChessBuffers,
    radon_buffers: &mut RadonBuffers,
    ml: &ml_refiner::MlRefinerParams,
    ml_state: &mut ml_refiner::MlRefinerState,
) -> Vec<CornerDescriptor> {
    // ML pairs only with ChESS-style patches; fall back to the
    // generic Radon path otherwise.
    if matches!(&cfg.strategy, DetectionStrategy::Radon(_)) {
        return detect_with_buffers(base, cfg, pyramid_buffers, chess_buffers, radon_buffers);
    }

    let _ = (radon_buffers,); // unused on the ChESS branch but kept in the signature for symmetry.

    let params = cfg.chess_params();
    let ml_border = ml_refiner::patch_radius(ml);
    coarse_to_fine_with_ml(
        base,
        cfg,
        pyramid_buffers,
        chess_buffers,
        &params,
        ml_border,
        &mut |resp, p, image| detect_with_ml_refiner(resp, p, image, ml_state),
    )
}

/// Sequential coarse-to-fine driver for the ChESS+ML path. Threads a
/// `&mut FnMut(...)` so the caller can hold mutable ML state without
/// the borrow-checker conflict that would arise from splitting the
/// closure across coarse and per-seed call sites.
///
/// The traversal skeleton is shared with the generic detector path via
/// [`traverse`]; the ML-specific behaviour lives entirely in
/// [`MlStep`]: `detect_full` / `refine_roi` run the model, while
/// `detect_seeds` uses the classic ChESS refiner (coarse seeds only
/// need to be accurate enough to carve an ROI).
#[cfg(feature = "ml-refiner")]
fn coarse_to_fine_with_ml<R>(
    base: ImageView<'_>,
    cfg: &DetectorConfig,
    pyramid_buffers: &mut PyramidBuffers,
    chess_buffers: &mut ChessBuffers,
    params: &ChessParams,
    refine_border: i32,
    detect_fn: &mut R,
) -> Vec<CornerDescriptor>
where
    R: FnMut(&ResponseMap, &ChessParams, Option<ImageView<'_>>) -> Vec<Corner>,
{
    let seed_refiner = Refiner::from_kind(params.refiner.clone());
    let seed_refine_border = seed_refiner.radius();
    let traversal = TraversalShape {
        detector_border: ChessDetector.roi_border(params),
        refine_border,
        descriptor_ring_radius: params.ring_radius(),
        orientation_method: params.orientation_method,
        single_scale_merge_radius: cfg.merge_radius,
    };
    let mut step = MlStep {
        params,
        buffers: chess_buffers,
        seed_refiner,
        seed_refine_border,
        detect_fn,
    };
    traverse(
        base,
        cfg.coarse_to_fine_params().as_ref(),
        pyramid_buffers,
        &traversal,
        &mut step,
    )
}

/// [`TraversalStep`] for the ChESS+ML path. `detect_full` and
/// `refine_roi` run the ML model (via `detect_fn`), while `detect_seeds`
/// runs the classic ChESS refiner: coarse seeds only need positions
/// accurate enough to carve a base-image ROI, so the (expensive) model
/// is reserved for the fine per-seed step.
#[cfg(feature = "ml-refiner")]
struct MlStep<'a, R> {
    params: &'a ChessParams,
    buffers: &'a mut ChessBuffers,
    seed_refiner: Refiner,
    seed_refine_border: i32,
    detect_fn: &'a mut R,
}

#[cfg(feature = "ml-refiner")]
impl<R> TraversalStep for MlStep<'_, R>
where
    R: FnMut(&ResponseMap, &ChessParams, Option<ImageView<'_>>) -> Vec<Corner>,
{
    fn detect_full(&mut self, view: ImageView<'_>) -> Vec<Corner> {
        let resp = ChessDetector.compute_response(view, self.params, self.buffers);
        (self.detect_fn)(resp, self.params, Some(view))
    }

    fn detect_seeds(&mut self, view: ImageView<'_>) -> Vec<Corner> {
        // Coarse seeds via the classic ChESS refiner, not the model.
        let resp = ChessDetector.compute_response(view, self.params, self.buffers);
        let peaks = ChessDetector.detect_corners(&resp, self.params, self.seed_refine_border);
        ChessDetector.refine_peaks_on_image(peaks, view, &resp, &mut self.seed_refiner)
    }

    fn refine_roi(&mut self, base: ImageView<'_>, roi: (i32, i32, i32, i32)) -> Vec<Corner> {
        let (x0, y0, _x1, _y1) = roi;
        let resp = ChessDetector.compute_response_patch(base, roi, self.params, self.buffers);
        let refine_view =
            ImageView::with_origin(base.width(), base.height(), base.data(), [x0, y0])
                .expect("base image dimensions must match buffer length");
        (self.detect_fn)(resp, self.params, Some(refine_view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use box_image_pyramid::ImageBuffer;

    #[test]
    fn default_coarse_to_fine_config_is_single_scale() {
        let cfg = CoarseToFineParams::default();
        assert_eq!(cfg.pyramid.num_levels, 1);
        assert_eq!(cfg.pyramid.min_size, 128);
        assert_eq!(cfg.refinement_radius, 3);
        assert_eq!(cfg.merge_radius, 3.0);
    }

    #[test]
    fn chess_config_multiscale_preset_has_expected_pyramid() {
        let cfg = DetectorConfig::chess_multiscale();
        let cf = cfg
            .coarse_to_fine_params()
            .expect("chess_multiscale preset must produce CoarseToFineParams");
        assert_eq!(cf.pyramid.num_levels, 3);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn coarse_to_fine_trace_reports_timings() {
        let buf = ImageBuffer::new(32, 32);
        let view = ImageView::from_u8_slice(buf.width, buf.height, &buf.data)
            .expect("dimensions must match");
        let cfg = DetectorConfig::default();
        let mut pyramid = PyramidBuffers::default();
        let mut chess_buffers = ChessBuffers::default();
        let mut radon_buffers = RadonBuffers::default();
        let corners = detect_with_buffers(
            view,
            &cfg,
            &mut pyramid,
            &mut chess_buffers,
            &mut radon_buffers,
        );
        assert!(corners.is_empty());
    }
}
