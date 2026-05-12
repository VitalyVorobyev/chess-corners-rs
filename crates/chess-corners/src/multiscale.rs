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
//! through the same orchestrator. The dispatch in
//! [`crate::Detector::detect_view`] selects the detector ZST per
//! [`crate::DetectionStrategy`]. End users should reach detection
//! through [`crate::Detector`].

#[cfg(feature = "ml-refiner")]
use crate::ml_refiner;
#[cfg(feature = "ml-refiner")]
use crate::ChessParams;
use crate::{DetectionStrategy, DetectorConfig};
use box_image_pyramid::{build_pyramid, PyramidBuffers, PyramidParams};
#[cfg(feature = "ml-refiner")]
use chess_corners_core::detect::chess::response::{
    chess_response_u8, chess_response_u8_patch, Roi,
};
#[cfg(feature = "ml-refiner")]
use chess_corners_core::detect::detect_corners_from_response_with_refiner;
use chess_corners_core::detect::merge_corners_simple;
use chess_corners_core::detect::Corner;
use chess_corners_core::orientation::describe_corners;
#[cfg(feature = "ml-refiner")]
use chess_corners_core::ResponseMap;
use chess_corners_core::{ChessBuffers, ChessDetector, CornerDescriptor, DenseDetector};
use chess_corners_core::{
    CornerRefiner, ImageView, OrientationMethod, RadonBuffers, RadonDetector, Refiner, RefinerKind,
};

/// Bridge from `chess_corners_core::ImageView` to `box_image_pyramid::ImageView`.
fn to_pyramid_view(v: ImageView<'_>) -> box_image_pyramid::ImageView<'_> {
    // invariant: v was already validated as a coherent ImageView, so the pyramid view cannot fail.
    box_image_pyramid::ImageView::new(v.width, v.height, v.data).unwrap()
}
#[cfg(feature = "tracing")]
use tracing::info_span;

/// Parameters controlling the coarse-to-fine multiscale detector.
///
/// The default keeps `num_levels = 1`, so callers start in the single-scale
/// regime unless they explicitly opt into a pyramid.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CoarseToFineParams {
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

impl CoarseToFineParams {
    pub fn new() -> Self {
        Self::default()
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

#[cfg(feature = "ml-refiner")]
fn detect_with_refiner_kind(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    refiner_kind: &RefinerKind,
) -> Vec<Corner> {
    let mut refiner = Refiner::from_kind(refiner_kind.clone());
    detect_corners_from_response_with_refiner(resp, params, image, &mut refiner)
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
        base_w_i: base.width as i32,
        base_h_i: base.height as i32,
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
    orientation_method: OrientationMethod,
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
/// [`refine_corners_on_image`](chess_corners_core::detect::refine_corners_on_image).
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
    let base_view = ImageView::from_u8_slice(base.width, base.height, base.data)
        .expect("base image dimensions must match buffer length");

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

    // Single-scale path: no pyramid, run detector once on the full
    // base view, refine through the detector's image-domain step,
    // merge, describe.
    let Some(cf) = multiscale else {
        let resp = detector.compute_response(base, params, detector_buffers);
        let peaks = detector.detect_corners(&resp, params, refine_border);
        let mut refiner = Refiner::from_kind(shape.refiner_kind.clone());
        let mut corners = detector.refine_peaks_on_image(peaks, base_view, &resp, &mut refiner);
        let merged = merge_corners_simple(&mut corners, shape.merge_radius);
        return describe_corners(
            base.data,
            base.width,
            base.height,
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
        let resp = detector.compute_response(lvl_view, params, detector_buffers);
        let peaks = detector.detect_corners(&resp, params, refine_border);
        let mut refiner = Refiner::from_kind(shape.refiner_kind.clone());
        let mut corners = detector.refine_peaks_on_image(peaks, lvl_view, &resp, &mut refiner);
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
    let coarse_resp = detector.compute_response(coarse_view, params, detector_buffers);
    let coarse_peaks = detector.detect_corners(&coarse_resp, params, refine_border);
    let mut refiner = Refiner::from_kind(shape.refiner_kind.clone());
    let coarse_corners =
        detector.refine_peaks_on_image(coarse_peaks, coarse_view, &coarse_resp, &mut refiner);
    // Drop the response borrow before reusing detector_buffers for the
    // per-seed patch path.
    drop(coarse_resp);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let detector_border = detector.roi_border(params);
    let roi_ctx = make_roi_context(base, coarse_lvl.scale, detector_border, refine_border, cf);

    #[cfg(feature = "tracing")]
    let refine_span = info_span!(
        "refine",
        seeds = coarse_corners.len(),
        roi_r = roi_ctx.roi_r
    )
    .entered();

    // Per-seed refinement runs sequentially in the generic path: the
    // detector's `compute_response_patch` mutates `detector_buffers`,
    // and rayon parallelism over seeds would require cloning the
    // buffers per worker. The cost-benefit here flips toward
    // simplicity — the heavy per-seed work for ChESS is a small ROI
    // and parallelism gained little; for Radon, the SAT build inside
    // each patch dominates.
    let mut refined: Vec<Corner> = Vec::new();
    for c in coarse_corners {
        let Some(roi_bounds) = roi_ctx.compute_roi(&c) else {
            continue;
        };
        let (x0, y0, _x1, _y1) = roi_bounds;
        let patch_resp =
            detector.compute_response_patch(base, roi_bounds, params, detector_buffers);
        // Width/height inferred from the response's shape: ChESS
        // emits a ResponseMap sized to the ROI (with reach-outside
        // border math); Radon emits a working-resolution
        // RadonResponseView whose pixels are in the *patch* coord
        // frame. detector.detect_corners returns corners in the same
        // patch-local frame.
        let patch_peaks = detector.detect_corners(&patch_resp, params, refine_border);
        if patch_peaks.is_empty() {
            continue;
        }

        // Image-domain refinement over the base image, with origin
        // [x0, y0]: the refiner sees patch-local seed coords and
        // samples base pixels at (cx + x0, cy + y0). The detector
        // decides whether its response is forwardable to the refiner
        // (ChESS → yes, ResponseMap; Radon → no, returns peaks as-is).
        let patch_image = ImageView::with_origin(base.width, base.height, base.data, [x0, y0])
            .expect("base image dimensions must match buffer length");
        let mut patch_refined =
            detector.refine_peaks_on_image(patch_peaks, patch_image, &patch_resp, &mut refiner);

        // Drop the patch_resp borrow so detector_buffers is free for
        // the next iteration.
        drop(patch_resp);

        // Shift patch-local refined positions to base coords.
        for pc in &mut patch_refined {
            pc.x += x0 as f32;
            pc.y += y0 as f32;
        }
        refined.extend(patch_refined);
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
        base.data,
        base.width,
        base.height,
        shape.descriptor_ring_radius,
        merged,
        shape.orientation_method,
    )
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
    let multiscale = cfg.to_coarse_to_fine_params();

    match &cfg.strategy {
        DetectionStrategy::Chess(_) => {
            let chess_params = cfg.to_chess_params();
            let refiner_kind = chess_params.refiner.clone();
            let shape = DetectorShape {
                refiner_kind: &refiner_kind,
                descriptor_ring_radius: chess_params.descriptor_ring_radius(),
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
            let radon_params = cfg.to_radon_detector_params();
            let refiner_kind = radon_params.refiner.clone();
            // Radon strategies don't carry a descriptor-ring knob;
            // descriptors sample the canonical r=5 ring downstream.
            // Orientation method is top-level on DetectorConfig.
            let shape = DetectorShape {
                refiner_kind: &refiner_kind,
                descriptor_ring_radius: chess_corners_core::ChessParams::default()
                    .descriptor_ring_radius(),
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

    let params = cfg.to_chess_params();
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
    // Single-scale ChESS+ML.
    let Some(cf) = cfg.to_coarse_to_fine_params() else {
        let detector = ChessDetector;
        let resp = detector.compute_response(base, params, chess_buffers);
        let view = ImageView::from_u8_slice(base.width, base.height, base.data)
            .expect("image dimensions must match buffer length");
        let mut raw = detect_fn(resp, params, Some(view));
        let merged = merge_corners_simple(&mut raw, cfg.merge_radius);
        return describe_corners(
            base.data,
            base.width,
            base.height,
            params.descriptor_ring_radius(),
            merged,
            params.orientation_method,
        );
    };

    let pyramid = build_pyramid(to_pyramid_view(base), &cf.pyramid, pyramid_buffers);
    if pyramid.levels.is_empty() {
        return Vec::new();
    }

    // Single-scale fallback for ChESS+ML when num_levels=1.
    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        let resp = chess_response_u8(lvl.img.data, lvl.img.width, lvl.img.height, params);
        let view = ImageView::from_u8_slice(lvl.img.width, lvl.img.height, lvl.img.data)
            .expect("image dimensions must match buffer length");
        let mut raw = detect_fn(&resp, params, Some(view));
        let merged = merge_corners_simple(&mut raw, cf.merge_radius);
        return describe_corners(
            lvl.img.data,
            lvl.img.width,
            lvl.img.height,
            params.descriptor_ring_radius(),
            merged,
            params.orientation_method,
        );
    }

    // Coarse-to-fine: coarse seeds via the classic ChESS refiner; ROI
    // refinement via the ML pipeline.
    let coarse_lvl = pyramid.levels.last().unwrap();
    let coarse_w = coarse_lvl.img.width;
    let coarse_h = coarse_lvl.img.height;

    #[cfg(feature = "tracing")]
    let coarse_span = info_span!("coarse_detect", w = coarse_w, h = coarse_h).entered();
    let coarse_resp = chess_response_u8(coarse_lvl.img.data, coarse_w, coarse_h, params);
    let coarse_view = ImageView::from_u8_slice(coarse_w, coarse_h, coarse_lvl.img.data).unwrap();
    let coarse_corners =
        detect_with_refiner_kind(&coarse_resp, params, Some(coarse_view), &params.refiner);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let detector_border = ChessDetector.roi_border(params);
    let roi_ctx = make_roi_context(base, coarse_lvl.scale, detector_border, refine_border, &cf);

    #[cfg(feature = "tracing")]
    let refine_span = info_span!(
        "refine",
        seeds = coarse_corners.len(),
        roi_r = roi_ctx.roi_r
    )
    .entered();

    let mut refined: Vec<Corner> = Vec::new();
    for c in coarse_corners {
        let Some((x0, y0, x1, y1)) = roi_ctx.compute_roi(&c) else {
            continue;
        };
        let roi = match Roi::new(x0 as usize, y0 as usize, x1 as usize, y1 as usize) {
            Some(r) => r,
            None => continue,
        };
        let patch_resp = chess_response_u8_patch(base.data, base.width, base.height, params, roi);
        if patch_resp.width() == 0 || patch_resp.height() == 0 {
            continue;
        }
        let refine_view = ImageView::with_origin(base.width, base.height, base.data, [x0, y0])
            .expect("base image dimensions must match buffer length");
        let mut patch_corners = detect_fn(&patch_resp, params, Some(refine_view));
        for pc in &mut patch_corners {
            pc.x += x0 as f32;
            pc.y += y0 as f32;
        }
        refined.extend(patch_corners);
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
        base.data,
        base.width,
        base.height,
        params.descriptor_ring_radius(),
        merged,
        params.orientation_method,
    )
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
        let cfg = DetectorConfig::multiscale();
        let cf = cfg
            .to_coarse_to_fine_params()
            .expect("multiscale preset must produce CoarseToFineParams");
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
