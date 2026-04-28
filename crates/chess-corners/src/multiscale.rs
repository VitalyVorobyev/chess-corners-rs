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
//! The main entry points are:
//!
//! - [`find_chess_corners`] – convenience wrapper that allocates
//!   pyramid buffers internally and returns [`CornerDescriptor`]
//!   values in base-image coordinates.
//! - [`find_chess_corners_buff`] – lower-level helper that accepts a
//!   caller-provided [`PyramidBuffers`] so you can reuse allocations
//!   across frames in a tight loop.
//! - ML-backed refinement variants (feature `ml-refiner`):
//!   `find_chess_corners_with_ml` / `find_chess_corners_buff_with_ml`.

#[cfg(feature = "ml-refiner")]
use crate::ml_refiner;
use crate::{ChessConfig, ChessParams, DetectorMode};
use box_image_pyramid::{build_pyramid, PyramidBuffers, PyramidParams};
use chess_corners_core::descriptor::{corners_to_descriptors, Corner};
use chess_corners_core::detect::{detect_corners_from_response_with_refiner, merge_corners_simple};
use chess_corners_core::response::{chess_response_u8, chess_response_u8_patch, Roi};
use chess_corners_core::{
    detect_corners_from_radon, radon_response_u8, CornerDescriptor, CornerRefiner, RadonBuffers,
};
use chess_corners_core::{ImageView, Refiner, RefinerKind, ResponseMap};

/// Bridge from `chess_corners_core::ImageView` to `box_image_pyramid::ImageView`.
fn to_pyramid_view(v: ImageView<'_>) -> box_image_pyramid::ImageView<'_> {
    box_image_pyramid::ImageView::new(v.width, v.height, v.data).unwrap()
}
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::{info_span, instrument};

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
// Detector helpers: thin wrappers that adapt the classic and ML detectors
// to a common call signature.
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

/// Compute the response patch for a seed ROI and return offset corners.
///
/// This is the shared "inner loop" body for both the classic and ML refine
/// paths. It computes the ChESS response inside the ROI, runs the provided
/// `detect` closure, and shifts patch-local coordinates back into base-image
/// space.
fn refine_seed_in_roi(
    base: ImageView<'_>,
    params: &ChessParams,
    roi_bounds: (i32, i32, i32, i32),
    mut detect: impl FnMut(&ResponseMap, &ChessParams, Option<ImageView<'_>>) -> Vec<Corner>,
) -> Option<Vec<Corner>> {
    let (x0, y0, x1, y1) = roi_bounds;
    let base_w = base.width;
    let base_h = base.height;

    let roi = Roi::new(x0 as usize, y0 as usize, x1 as usize, y1 as usize)?;
    let patch_resp = chess_response_u8_patch(base.data, base_w, base_h, params, roi);

    if patch_resp.width() == 0 || patch_resp.height() == 0 {
        return None;
    }

    let refine_view = ImageView::with_origin(base_w, base_h, base.data, [x0, y0])
        .expect("base image dimensions must match buffer length");
    let mut patch_corners = detect(&patch_resp, params, Some(refine_view));

    for pc in &mut patch_corners {
        pc.x += x0 as f32;
        pc.y += y0 as f32;
    }

    if patch_corners.is_empty() {
        None
    } else {
        Some(patch_corners)
    }
}

/// Merge refined corners and convert to descriptors.
fn merge_and_describe(
    base: ImageView<'_>,
    params: &ChessParams,
    merge_radius: f32,
    refined: &mut Vec<Corner>,
) -> Vec<CornerDescriptor> {
    #[cfg(feature = "tracing")]
    let merge_span = info_span!(
        "merge",
        merge_radius = merge_radius,
        candidates = refined.len()
    )
    .entered();
    let merged = merge_corners_simple(refined, merge_radius);
    #[cfg(feature = "tracing")]
    drop(merge_span);

    corners_to_descriptors(
        base.data,
        base.width,
        base.height,
        params.descriptor_ring_radius(),
        merged,
    )
}

// ---------------------------------------------------------------------------
// Shared coarse-to-fine driver
// ---------------------------------------------------------------------------

/// Sequential coarse-to-fine driver parameterised over per-seed refinement.
// Only called from the ML path; gate it so the compiler sees no dead code when
// the `ml-refiner` feature is not enabled.
#[cfg(feature = "ml-refiner")]
///
/// Used by the ML path (and non-rayon builds). The classic path with rayon
/// parallelism is inlined in [`find_chess_corners_buff_with_refiner`] to avoid
/// ownership conflicts from splitting `detect_fn` across two closures.
///
/// Callers supply:
/// - `coarse_detect` — [`RefinerKind`] for coarse-level detection. The ML path
///   passes `params.refiner` because the ML model is not reliable at coarse
///   resolution.
/// - `detect_fn` — called for the single-scale fallback and for each base-level
///   ROI in the coarse-to-fine path (same closure, called in mutually exclusive
///   code paths so no aliasing).
/// - `refine_border` — pixel border the refiner requires; influences the ROI
///   context built from the coarse scale.
fn coarse_to_fine_with<R>(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
    coarse_detect: &RefinerKind,
    refine_border: i32,
    detect_fn: &mut R,
) -> Vec<CornerDescriptor>
where
    R: FnMut(&ResponseMap, &ChessParams, Option<ImageView<'_>>) -> Vec<Corner>,
{
    let params = cfg.to_chess_params();
    let cf = cfg.to_coarse_to_fine_params();

    let pyramid = build_pyramid(to_pyramid_view(base), &cf.pyramid, buffers);
    if pyramid.levels.is_empty() {
        return Vec::new();
    }

    // Single-scale fallback: run directly on the sole pyramid level.
    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        let resp = chess_response_u8(lvl.img.data, lvl.img.width, lvl.img.height, &params);
        let view = ImageView::from_u8_slice(lvl.img.width, lvl.img.height, lvl.img.data)
            .expect("image dimensions must match buffer length");
        let mut raw = detect_fn(&resp, &params, Some(view));
        let merged = merge_corners_simple(&mut raw, cf.merge_radius);
        return corners_to_descriptors(
            lvl.img.data,
            lvl.img.width,
            lvl.img.height,
            params.descriptor_ring_radius(),
            merged,
        );
    }

    // --- Coarse-to-fine path ---

    let coarse_lvl = pyramid.levels.last().unwrap();
    let coarse_w = coarse_lvl.img.width;
    let coarse_h = coarse_lvl.img.height;

    #[cfg(feature = "tracing")]
    let coarse_span = info_span!("coarse_detect", w = coarse_w, h = coarse_h).entered();
    let coarse_resp = chess_response_u8(coarse_lvl.img.data, coarse_w, coarse_h, &params);
    let coarse_view = ImageView::from_u8_slice(coarse_w, coarse_h, coarse_lvl.img.data).unwrap();
    let coarse_corners =
        detect_with_refiner_kind(&coarse_resp, &params, Some(coarse_view), coarse_detect);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let roi_ctx = make_roi_context(base, coarse_lvl.scale, &params, refine_border, &cf);

    #[cfg(feature = "tracing")]
    let refine_span = info_span!(
        "refine",
        seeds = coarse_corners.len(),
        roi_r = roi_ctx.roi_r
    )
    .entered();

    let mut refined: Vec<Corner> = coarse_corners
        .into_iter()
        .filter_map(|c| {
            let roi_bounds = roi_ctx.compute_roi(&c)?;
            refine_seed_in_roi(base, &params, roi_bounds, &mut *detect_fn)
        })
        .flatten()
        .collect();

    #[cfg(feature = "tracing")]
    drop(refine_span);

    merge_and_describe(base, &params, cf.merge_radius, &mut refined)
}

// ---------------------------------------------------------------------------
// Classic (RefinerKind) path
// ---------------------------------------------------------------------------

/// Detect corners using a caller-provided pyramid buffer.
///
/// - When `cfg.pyramid_levels <= 1`, this behaves as a
///   single-scale detector on `base`.
/// - Otherwise, it builds a pyramid into `buffers`, runs a coarse
///   detector on the smallest level, refines each coarse seed inside a
///   base-image ROI, merges near-duplicate corners, and finally
///   converts them into [`CornerDescriptor`] values sampled at the
///   full resolution.
pub fn find_chess_corners_buff(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
) -> Vec<CornerDescriptor> {
    let refiner = cfg.refiner.to_refiner_kind();
    find_chess_corners_buff_with_refiner(base, cfg, buffers, &refiner)
}

/// Variant of [`find_chess_corners_buff`] that accepts an explicit refiner selection.
///
/// When the `rayon` feature is enabled, per-seed ROI refinement in the
/// coarse-to-fine path runs in parallel. The ML path uses
/// [`find_chess_corners_buff_with_ml`] instead, which is always sequential
/// due to mutable ML state.
pub fn find_chess_corners_buff_with_refiner(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    // Radon detector has its own response + NMS + peak-fit pipeline
    // and does not use the ChESS response map or `refiner`. Dispatch
    // early so the ChESS path below can assume Canonical/Broad mode.
    if matches!(cfg.detector_mode, DetectorMode::Radon) {
        return detect_with_radon(base, cfg);
    }

    let params = cfg.to_chess_params();
    let cf = cfg.to_coarse_to_fine_params();
    let border = refiner_radius(refiner);

    let pyramid = build_pyramid(to_pyramid_view(base), &cf.pyramid, buffers);
    if pyramid.levels.is_empty() {
        return Vec::new();
    }

    // Single-scale fallback.
    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        let resp = chess_response_u8(lvl.img.data, lvl.img.width, lvl.img.height, &params);
        let refine_view = ImageView::from_u8_slice(lvl.img.width, lvl.img.height, lvl.img.data)
            .expect("image dimensions must match buffer length");
        let mut raw = detect_with_refiner_kind(&resp, &params, Some(refine_view), refiner);
        let merged = merge_corners_simple(&mut raw, cf.merge_radius);
        return corners_to_descriptors(
            lvl.img.data,
            lvl.img.width,
            lvl.img.height,
            params.descriptor_ring_radius(),
            merged,
        );
    }

    // --- Coarse-to-fine path (with optional rayon parallelism) ---

    let coarse_lvl = pyramid.levels.last().unwrap();
    let coarse_w = coarse_lvl.img.width;
    let coarse_h = coarse_lvl.img.height;

    #[cfg(feature = "tracing")]
    let coarse_span = info_span!("coarse_detect", w = coarse_w, h = coarse_h).entered();
    let coarse_resp = chess_response_u8(coarse_lvl.img.data, coarse_w, coarse_h, &params);
    let coarse_view = ImageView::from_u8_slice(coarse_w, coarse_h, coarse_lvl.img.data).unwrap();
    let coarse_corners =
        detect_with_refiner_kind(&coarse_resp, &params, Some(coarse_view), refiner);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let roi_ctx = make_roi_context(base, coarse_lvl.scale, &params, border, &cf);

    #[cfg(feature = "tracing")]
    let refine_span = info_span!(
        "refine",
        seeds = coarse_corners.len(),
        roi_r = roi_ctx.roi_r
    )
    .entered();

    let refine_one = |c: Corner| -> Option<Vec<Corner>> {
        let roi_bounds = roi_ctx.compute_roi(&c)?;
        refine_seed_in_roi(base, &params, roi_bounds, |resp, p, image| {
            detect_with_refiner_kind(resp, p, image, refiner)
        })
    };

    #[cfg(feature = "rayon")]
    let mut refined: Vec<Corner> = coarse_corners
        .into_par_iter()
        .filter_map(refine_one)
        .flatten()
        .collect();

    #[cfg(not(feature = "rayon"))]
    let mut refined: Vec<Corner> = coarse_corners
        .into_iter()
        .filter_map(refine_one)
        .flatten()
        .collect();

    #[cfg(feature = "tracing")]
    drop(refine_span);

    merge_and_describe(base, &params, cf.merge_radius, &mut refined)
}

// ---------------------------------------------------------------------------
// Radon detector path
// ---------------------------------------------------------------------------

/// Run the whole-image Duda-Frese Radon detector on `base` and
/// produce [`CornerDescriptor`] values in base-image coordinates.
///
/// Single-scale only — the Radon pipeline is already fast enough at
/// base resolution (SAT-based O(1) ray sums) that building a pyramid
/// for coarse-to-fine is not net faster on calibration frames. If
/// `cfg.pyramid_levels > 1` that field is ignored and the detector
/// runs on `base`.
///
/// The Radon detector applies its own threshold / NMS / 3-point
/// Gaussian peak-fit internally, so `cfg.refiner` is not consulted.
/// Descriptor sampling still honours
/// [`ChessConfig::descriptor_mode`](crate::ChessConfig::descriptor_mode).
fn detect_with_radon(base: ImageView<'_>, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    #[cfg(feature = "tracing")]
    let span = info_span!(
        "radon_detect",
        w = base.width,
        h = base.height,
        upsample = cfg.radon_detector.image_upsample,
    )
    .entered();

    // Allocate RadonBuffers on the stack of this call. The detector
    // buffers reuse pattern (mirroring PyramidBuffers) is deferred —
    // callers who need zero-alloc framing can call the core-level
    // `radon_response_u8` / `detect_corners_from_radon` directly.
    let mut rb = RadonBuffers::new();
    let resp = radon_response_u8(
        base.data,
        base.width,
        base.height,
        &cfg.radon_detector,
        &mut rb,
    );
    let corners = detect_corners_from_radon(&resp, &cfg.radon_detector);

    if corners.is_empty() {
        #[cfg(feature = "tracing")]
        drop(span);
        return Vec::new();
    }

    // Descriptor sampling uses the ChESS ring at the resolution
    // configured by `descriptor_mode`. Corners are already in
    // base-image coordinates.
    let params = cfg.to_chess_params();
    let mut merged = corners;
    let merged = merge_corners_simple(&mut merged, cfg.merge_radius);
    let out = corners_to_descriptors(
        base.data,
        base.width,
        base.height,
        params.descriptor_ring_radius(),
        merged,
    );
    #[cfg(feature = "tracing")]
    drop(span);
    out
}

// ---------------------------------------------------------------------------
// ML refiner path
// ---------------------------------------------------------------------------

/// Variant of [`find_chess_corners_buff`] that uses the ML refiner pipeline.
#[cfg(feature = "ml-refiner")]
pub fn find_chess_corners_buff_with_ml(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
) -> Vec<CornerDescriptor> {
    let ml_params = ml_refiner::MlRefinerParams::default();
    let fallback_refiner = cfg.refiner.to_refiner_kind();
    let mut ml_state = ml_refiner::MlRefinerState::new(&ml_params, &fallback_refiner);
    find_chess_corners_buff_with_ml_state(base, cfg, buffers, &ml_params, &mut ml_state)
}

#[cfg(feature = "ml-refiner")]
fn find_chess_corners_buff_with_ml_state(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
    ml: &ml_refiner::MlRefinerParams,
    ml_state: &mut ml_refiner::MlRefinerState,
) -> Vec<CornerDescriptor> {
    // The Radon detector produces corners through its own internal
    // peak-fit and does not emit ChESS-response seeds, so pairing it
    // with the ML refiner is a category error. Fall back to the
    // Radon detector's native output in that case — the user can
    // still pick the ML refiner by switching `detector_mode` back to
    // `Canonical` or `Broad`.
    if matches!(cfg.detector_mode, DetectorMode::Radon) {
        return detect_with_radon(base, cfg);
    }

    let params = cfg.to_chess_params();
    let ml_border = ml_refiner::patch_radius(ml);
    // Coarse detection always uses the classic refiner (ML model is not
    // reliable at coarse resolution). ROI refinement uses the ML path.
    // `ml_state` is mutable but used in a single closure; `coarse_to_fine_with`
    // runs sequentially so there is no aliasing.
    coarse_to_fine_with(
        base,
        cfg,
        buffers,
        &params.refiner.clone(),
        ml_border,
        &mut |resp, p, image| detect_with_ml_refiner(resp, p, image, ml_state),
    )
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_roi_context(
    base: ImageView<'_>,
    coarse_scale: f32,
    params: &ChessParams,
    refine_border: i32,
    cf: &CoarseToFineParams,
) -> RoiContext {
    let ring_r = params.ring_radius() as i32;
    let nms_r = params.nms_radius as i32;
    let border = (ring_r + nms_r + refine_border).max(0);
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
// Convenience wrappers (allocate pyramid buffers internally)
// ---------------------------------------------------------------------------

/// Detect corners from a base-level grayscale view, allocating
/// pyramid storage internally.
///
/// This is the high-level entry point used by
/// [`crate::find_chess_corners_u8`] and the `image` helpers. For
/// repeated calls on successive frames, prefer
/// [`find_chess_corners_buff`] with a reusable [`PyramidBuffers`] to
/// avoid repeated allocations.
#[must_use]
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "info",
        skip(base, cfg),
        fields(levels = cfg.pyramid_levels, min_size = cfg.pyramid_min_size)
    )
)]
pub fn find_chess_corners(base: ImageView<'_>, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    let refiner = cfg.refiner.to_refiner_kind();
    find_chess_corners_with_refiner(base, cfg, &refiner)
}

/// Single-call helper that lets callers pick the refiner.
#[must_use]
pub fn find_chess_corners_with_refiner(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    let mut buffers = PyramidBuffers::with_capacity(cfg.pyramid_levels);
    find_chess_corners_buff_with_refiner(base, cfg, &mut buffers, refiner)
}

/// Single-call helper that runs the ML refiner pipeline.
#[cfg(feature = "ml-refiner")]
#[must_use]
pub fn find_chess_corners_with_ml(base: ImageView<'_>, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    let mut buffers = PyramidBuffers::with_capacity(cfg.pyramid_levels);
    find_chess_corners_buff_with_ml(base, cfg, &mut buffers)
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
        let cfg = ChessConfig::multiscale();
        assert_eq!(cfg.pyramid_levels, 3);
        assert_eq!(cfg.pyramid_min_size, 128);
        assert_eq!(cfg.refinement_radius, 3);
        assert_eq!(cfg.merge_radius, 3.0);
    }

    #[test]
    fn coarse_to_fine_trace_reports_timings() {
        let buf = ImageBuffer::new(32, 32);
        let view = ImageView::from_u8_slice(buf.width, buf.height, &buf.data)
            .expect("dimensions must match");
        let cfg = ChessConfig::default();
        let corners = find_chess_corners(view, &cfg);
        assert!(corners.is_empty());
    }
}
