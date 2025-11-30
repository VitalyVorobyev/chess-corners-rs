//! Unified corner detection (single or multiscale).
//!
//! - If `pyramid.num_levels <= 1`, runs single-scale detection on the base.
//! - Otherwise, runs a coarse detection on the smallest pyramid level, then
//!   refines each seed in the base image (coarse-to-fine) and merges duplicates.

use crate::pyramid::{build_pyramid, ImageView, PyramidBuffers, PyramidParams};
use crate::ChessConfig;
use chess_corners_core::descriptor::{corners_to_descriptors, Corner};
use chess_corners_core::detect::detect_corners_from_response;
use chess_corners_core::response::{chess_response_u8, chess_response_u8_patch, Roi};
use chess_corners_core::CornerDescriptor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::{debug_span, instrument};

#[derive(Clone, Debug)]
pub struct CoarseToFineParams {
    pub pyramid: PyramidParams,
    /// ROI radius at the coarse level (ignored when `pyramid.num_levels <= 1`).
    pub roi_radius: u32,
    pub merge_radius: f32,
}

/// Timing breakdown for the detector.
#[derive(Clone, Debug)]
pub struct CoarseToFineResult {
    pub corners: Vec<CornerDescriptor>,
    pub coarse_cols: usize,
    pub coarse_rows: usize,
}
// Profiling helper; prefer enabling the `tracing` feature. Timing fields are
// best-effort and may change.

impl Default for CoarseToFineParams {
    fn default() -> Self {
        Self {
            pyramid: PyramidParams::default(),
            // smaller ROI (2*12+1 = 25px window) *at the coarse level* around
            // the prediction. At the base level this is upscaled according to
            // the coarse pyramid scale.
            roi_radius: 12,
            // merge duplicates within ~2 pixels
            merge_radius: 2.0,
        }
    }
}

impl CoarseToFineParams {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "debug",
        skip(base, buffers),
        fields(levels = cfg.multiscale.pyramid.num_levels, min_size = cfg.multiscale.pyramid.min_size)
    )
)]
pub fn find_chess_corners_buff(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
) -> CoarseToFineResult {
    let params = &cfg.params;
    let cf = &cfg.multiscale;

    let pyramid = build_pyramid(base, &cf.pyramid, buffers);
    if pyramid.levels.is_empty() {
        return CoarseToFineResult {
            corners: Vec::new(),
            coarse_cols: 0,
            coarse_rows: 0,
        };
    }

    if pyramid.levels.len() == 1 {
        let lvl = &pyramid.levels[0];
        let resp = chess_response_u8(
            lvl.img.data,
            lvl.img.width as usize,
            lvl.img.height as usize,
            params,
        );
        let raw = detect_corners_from_response(&resp, params);
        let desc = corners_to_descriptors(
            lvl.img.data,
            lvl.img.width as usize,
            lvl.img.height as usize,
            params.descriptor_ring_radius(),
            raw,
        );
        return CoarseToFineResult {
            corners: desc,
            coarse_cols: lvl.img.width as usize,
            coarse_rows: lvl.img.height as usize,
        };
    }

    let base_w = base.width as usize;
    let base_h = base.height as usize;
    let base_w_i = base_w as i32;
    let base_h_i = base_h as i32;

    // Use the last (smallest) level as the coarse detector input.
    let coarse_lvl = pyramid
        .levels
        .last()
        .expect("pyramid levels are non-empty after earlier check");

    let coarse_w = coarse_lvl.img.width as usize;
    let coarse_h = coarse_lvl.img.height as usize;

    // Full detection on coarse level
    #[cfg(feature = "tracing")]
    let coarse_span = debug_span!("coarse").entered();
    let coarse_resp = chess_response_u8(coarse_lvl.img.data, coarse_w, coarse_h, params);
    let coarse_corners = detect_corners_from_response(&coarse_resp, params);
    #[cfg(feature = "tracing")]
    drop(coarse_span);

    if coarse_corners.is_empty() {
        return CoarseToFineResult {
            corners: Vec::new(),
            coarse_cols: coarse_w,
            coarse_rows: coarse_h,
        };
    }

    let inv_scale = 1.0 / coarse_lvl.scale;

    // Compute the same "border" margin as the core detector uses.
    let ring_r = params.ring_radius() as i32;
    let nms_r = params.nms_radius as i32;
    let refine_r = 2i32; // 5x5 refinement window
    let border = (ring_r + nms_r + refine_r).max(0);
    // Require a bit of breathing room inside the image
    let safe_margin = border + 1;

    // Convert the user-provided ROI radius (expressed in coarse-level pixels)
    // to base-image pixels. Enforce a minimum radius that leaves interior room
    // beyond the detector's own border margin so refinement can run.
    let roi_r_base = (cf.roi_radius as f32 / coarse_lvl.scale).ceil() as i32;
    let min_roi_r = border + 2;
    let roi_r = roi_r_base.max(min_roi_r);

    #[cfg(feature = "tracing")]
    let refine_span = debug_span!("refine").entered();

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

        // Clamp ROI to stay inside the area where full ring + 5x5 refinement
        // are safe. This mirrors the detector's own border logic.
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
        let mut patch_corners = detect_corners_from_response(&patch_resp, params);

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
    let merge_span = debug_span!("merge").entered();
    let merged = merge_corners_simple(&mut refined, cf.merge_radius);
    #[cfg(feature = "tracing")]
    drop(merge_span);

    let desc_radius = params.descriptor_ring_radius();
    let descriptors = corners_to_descriptors(base.data, base_w, base_h, desc_radius, merged);

    CoarseToFineResult {
        corners: descriptors,
        coarse_cols: coarse_w,
        coarse_rows: coarse_h,
    }
}

fn merge_corners_simple(corners: &mut Vec<Corner>, radius: f32) -> Vec<Corner> {
    let r2 = radius * radius;
    let mut out: Vec<Corner> = Vec::new();

    // naive O(N^2) for now; N is small for single chessboard
    'outer: for c in corners.drain(..) {
        for o in &mut out {
            let dx = c.xy[0] - o.xy[0];
            let dy = c.xy[1] - o.xy[1];
            if dx * dx + dy * dy <= r2 {
                // keep the stronger
                if c.strength > o.strength {
                    *o = c;
                }
                continue 'outer;
            }
        }
        out.push(c);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::ImageBuffer;

    #[test]
    fn merge_corners_prefers_stronger_entries() {
        let mut corners = vec![
            Corner {
                xy: [10.0, 10.0],
                strength: 1.0,
            },
            Corner {
                xy: [11.0, 11.0],
                strength: 5.0,
            },
            Corner {
                xy: [20.0, 20.0],
                strength: 3.0,
            },
        ];
        let merged = merge_corners_simple(&mut corners, 2.5);
        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|c| (c.xy[0] - 11.0).abs() < 1e-6
            && (c.xy[1] - 11.0).abs() < 1e-6
            && (c.strength - 5.0).abs() < 1e-6));
        assert!(merged
            .iter()
            .any(|c| (c.xy[0] - 20.0).abs() < 1e-6 && (c.xy[1] - 20.0).abs() < 1e-6));
    }

    #[test]
    fn coarse_to_fine_trace_reports_timings() {
        let img = ImageBuffer::new(32, 32);
        let mut buffers = PyramidBuffers::new();
        let cfg = ChessConfig::default();
        let res = find_chess_corners_buff(img.as_view(), &cfg, &mut buffers);
        assert!(res.corners.is_empty());
    }
}
