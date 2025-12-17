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

use crate::pyramid::{build_pyramid, PyramidBuffers, PyramidParams};
use crate::{ChessConfig, ChessParams};
use chess_corners_core::descriptor::{corners_to_descriptors, Corner};
use chess_corners_core::detect::{detect_corners_from_response_with_refiner, merge_corners_simple};
use chess_corners_core::response::{chess_response_u8, chess_response_u8_patch, Roi};
use chess_corners_core::CornerDescriptor;
use chess_corners_core::{CornerRefiner, ImageView, Refiner, RefinerKind, ResponseMap};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::{info_span, instrument};

/// Parameters controlling the coarse-to-fine multiscale detector.
#[derive(Clone, Debug)]
pub struct CoarseToFineParams {
    /// Image pyramid shape and construction parameters.
    pub pyramid: PyramidParams,
    /// ROI radius at the coarse level (ignored when `pyramid.num_levels <= 1`).
    /// Expressed in coarse-level pixels and automatically scaled to the base
    /// image, with a minimum enforced to keep refinement away from borders.
    pub roi_radius: u32,
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
            roi_radius: 3,
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

/// Detect corners using a caller-provided pyramid buffer.
///
/// - When `cfg.multiscale.pyramid.num_levels <= 1`, this behaves as a
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
    find_chess_corners_buff_with_refiner(base, cfg, buffers, &cfg.params.refiner)
}

/// Variant of [`find_chess_corners_buff`] that accepts an explicit refiner selection.
pub fn find_chess_corners_buff_with_refiner(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    let params = &cfg.params;
    let cf = &cfg.multiscale;

    let pyramid = build_pyramid(base, &cf.pyramid, buffers);
    if pyramid.levels.is_empty() {
        return Vec::new();
    }

    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        #[cfg(feature = "tracing")]
        let single_span =
            info_span!("single_scale", w = lvl.img.width, h = lvl.img.height).entered();
        let resp = chess_response_u8(lvl.img.data, lvl.img.width, lvl.img.height, params);
        let refine_view = ImageView::from_u8_slice(lvl.img.width, lvl.img.height, lvl.img.data)
            .expect("image dimensions must match buffer length");
        let mut raw = detect_with_refiner_kind(&resp, params, Some(refine_view), refiner);
        let merged = merge_corners_simple(&mut raw, cf.merge_radius);
        let desc = corners_to_descriptors(
            lvl.img.data,
            lvl.img.width,
            lvl.img.height,
            params.descriptor_ring_radius(),
            merged,
        );
        #[cfg(feature = "tracing")]
        drop(single_span);
        return desc;
    }

    let base_w = base.width;
    let base_h = base.height;
    let base_w_i = base_w as i32;
    let base_h_i = base_h as i32;

    // Use the last (smallest) level as the coarse detector input.
    let coarse_lvl = pyramid
        .levels
        .last()
        .expect("pyramid levels are non-empty after earlier check");

    let coarse_w = coarse_lvl.img.width;
    let coarse_h = coarse_lvl.img.height;

    #[cfg(feature = "tracing")]
    let coarse_span = info_span!("coarse_detect", w = coarse_w, h = coarse_h).entered();
    // Full detection on coarse level
    let coarse_resp = chess_response_u8(coarse_lvl.img.data, coarse_w, coarse_h, params);
    let coarse_view = ImageView::from_u8_slice(coarse_w, coarse_h, coarse_lvl.img.data).unwrap();
    let coarse_corners = detect_with_refiner_kind(&coarse_resp, params, Some(coarse_view), refiner);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return Vec::new();
    }

    let inv_scale = 1.0 / coarse_lvl.scale;

    // Compute the same "border" margin as the core detector uses.
    let ring_r = params.ring_radius() as i32;
    let nms_r = params.nms_radius as i32;
    let refine_r = refiner_radius(refiner);
    let border = (ring_r + nms_r + refine_r).max(0);
    // Require a bit of breathing room inside the image
    let safe_margin = border + 1;

    // Convert the user-provided ROI radius (expressed in coarse-level pixels)
    // to base-image pixels. Enforce a minimum radius that leaves interior room
    // beyond the detector's own border margin so refinement can run.
    let roi_r_base = (cf.roi_radius as f32 / coarse_lvl.scale).ceil() as i32;
    let min_roi_r = border + 2;
    let roi_r = roi_r_base.max(min_roi_r);

    let refine_one = |c: Corner| -> Option<Vec<Corner>> {
        // Project coarse coordinate to base image
        let cx_base = c.xy[0] * inv_scale;
        let cy_base = c.xy[1] * inv_scale;

        let cx = cx_base.round() as i32;
        let cy = cy_base.round() as i32;

        // Skip coarse seeds that are too close to the image border to safely
        // build an ROI and run refinement.
        if cx < safe_margin
            || cy < safe_margin
            || cx >= base_w_i - safe_margin
            || cy >= base_h_i - safe_margin
        {
            return None;
        }

        // Initial ROI proposal around the coarse prediction.
        let mut x0 = cx - roi_r;
        let mut y0 = cy - roi_r;
        let mut x1 = cx + roi_r + 1;
        let mut y1 = cy + roi_r + 1;

        // Clamp ROI to stay inside the area where full ring + refinement
        // footprint are safe. This mirrors the detector's own border logic.
        let min_xy = border;
        let max_x = base_w_i - border;
        let max_y = base_h_i - border;

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

        // Ensure ROI is still large enough to run NMS + refinement.
        if x1 - x0 <= 2 * border || y1 - y0 <= 2 * border {
            return None;
        }

        let x0u = x0 as usize;
        let y0u = y0 as usize;
        let x1u = x1 as usize;
        let y1u = y1 as usize;

        // Compute response only inside this ROI at base level.
        let patch_resp = chess_response_u8_patch(
            base.data,
            base_w,
            base_h,
            params,
            Roi {
                x0: x0u,
                y0: y0u,
                x1: x1u,
                y1: y1u,
            },
        );

        if patch_resp.w == 0 || patch_resp.h == 0 {
            return None;
        }

        // Run the standard detector on the patch response. It treats the patch
        // as an independent image with its own (0,0) origin.
        let refine_view = ImageView::with_origin(base_w, base_h, base.data, [x0, y0])
            .expect("base image dimensions must match buffer length");
        let mut patch_corners =
            detect_with_refiner_kind(&patch_resp, params, Some(refine_view), refiner);

        for pc in &mut patch_corners {
            pc.xy[0] += x0 as f32;
            pc.xy[1] += y0 as f32;
        }

        if patch_corners.is_empty() {
            None
        } else {
            Some(patch_corners)
        }
    };

    #[cfg(feature = "tracing")]
    let refine_span = info_span!("refine", seeds = coarse_corners.len(), roi_r = roi_r).entered();

    #[cfg(feature = "rayon")]
    let mut refined: Vec<Corner> = coarse_corners
        .into_par_iter()
        .filter_map(refine_one)
        .flatten()
        .collect();

    #[cfg(not(feature = "rayon"))]
    let mut refined: Vec<Corner> = {
        let mut acc = Vec::new();
        for c in coarse_corners {
            if let Some(mut v) = refine_one(c) {
                acc.append(&mut v);
            }
        }
        acc
    };

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

    let desc_radius = params.descriptor_ring_radius();
    corners_to_descriptors(base.data, base_w, base_h, desc_radius, merged)
}

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
        fields(levels = cfg.multiscale.pyramid.num_levels, min_size = cfg.multiscale.pyramid.min_size)
    )
)]
pub fn find_chess_corners(base: ImageView<'_>, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    find_chess_corners_with_refiner(base, cfg, &cfg.params.refiner)
}

/// Single-call helper that lets callers pick the refiner.
#[must_use]
pub fn find_chess_corners_with_refiner(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    let mut buffers = PyramidBuffers::with_capacity(cfg.multiscale.pyramid.num_levels);
    find_chess_corners_buff_with_refiner(base, cfg, &mut buffers, refiner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::ImageBuffer;

    #[test]
    fn coarse_to_fine_trace_reports_timings() {
        let img = ImageBuffer::new(32, 32);
        let cfg = ChessConfig::default();
        let corners = find_chess_corners(img.as_view(), &cfg);
        assert!(corners.is_empty());
    }
}
