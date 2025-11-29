//! Multiscale and coarse-to-fine corner detection helpers.
//!
//! The multiscale path builds an image pyramid, runs the ChESS detector on the
//! coarsest level to generate seeds, then refines each seed in a base-image
//! ROI using a dense response patch and NMS + subpixel refinement. An
//! adaptive ROI radius is used: `CoarseToFineParams::roi_radius` is specified
//! in coarse-level pixels and converted to base-image pixels based on the
//! pyramid scale, with a minimum margin derived from the detector's own
//! border.

use crate::pyramid::{build_pyramid, PyramidBuffers, PyramidParams};
use chess_core::detect::{self, detect_corners_from_response};
use chess_core::response::{chess_response_u8_patch, Roi};
use chess_core::ChessParams;
use image::GrayImage;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::time::Instant;

/// Corner detected in a multiscale run, reported in base-level coordinates.
pub struct MultiscaleCorner {
    /// Position in the original (level 0) image coordinates.
    pub xy: [f32; 2],
    pub strength: f32,
}

/// Parameters controlling coarse-to-fine refinement on an image pyramid.
///
/// - `pyramid`: how the pyramid is built (levels, scale factor, min size).
/// - `roi_radius`: half-size of the refinement ROI, expressed in *coarse-level
///   pixels* (i.e., the level used for seeding). The implementation converts
///   this to a radius in base-image pixels based on the coarse level's scale
///   and enforces a minimum margin derived from the detector's own border.
/// - `merge_radius`: radius used to merge duplicate refined corners.
pub struct CoarseToFineParams {
    pub pyramid: PyramidParams,
    pub roi_radius: u32,
    pub merge_radius: f32,
}

/// Timing breakdown for the coarse-to-fine detector.
pub struct CoarseToFineResult {
    pub corners: Vec<MultiscaleCorner>,
    /// Time to build the pyramid (ms).
    pub build_ms: f64,
    /// Time spent on the coarse detection at the smallest level (ms).
    pub coarse_ms: f64,
    /// Time spent refining ROIs at the base resolution (ms).
    pub refine_ms: f64,
    /// Time spent merging overlapping refined corners (ms).
    pub merge_ms: f64,
    pub coarse_cols: usize,
    pub coarse_rows: usize,
}

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

/// Detect corners across an image pyramid and merge nearby detections.
///
/// Coordinates are rescaled back to the base image so consumers can treat the
/// output as a single set. Corners closer than `2` pixels are merged with
/// simple strength-based suppression. Pass a persistent `PyramidBuffers` to
/// avoid per-call allocations.
///
/// # Example
/// ```rust
/// use chess::{find_corners_multiscale_image, ChessParams, PyramidBuffers, PyramidParams};
/// use image::GrayImage;
///
/// let img = GrayImage::from_pixel(64, 64, image::Luma([0u8]));
/// let params = ChessParams::default();
/// let pyramid = PyramidParams::default();
/// let mut buffers = PyramidBuffers::new();
///
/// let corners = find_corners_multiscale_image(&img, &params, &pyramid, &mut buffers);
/// assert!(corners.is_empty());
/// ```
pub fn find_corners_multiscale_image(
    img: &GrayImage,
    p: &ChessParams,
    py: &PyramidParams,
    buffers: &mut PyramidBuffers,
) -> Vec<MultiscaleCorner> {
    let pyramid = build_pyramid(img, py, buffers);

    let mut all = Vec::new();

    for lvl in pyramid.levels.iter() {
        let w = lvl.img.width() as usize;
        let h = lvl.img.height() as usize;

        let cores = detect::find_corners_u8(lvl.img.as_raw(), w, h, p);

        let inv_scale = 1.0 / lvl.scale;

        for c in cores {
            all.push(MultiscaleCorner {
                xy: [c.xy[0] * inv_scale, c.xy[1] * inv_scale],
                strength: c.strength,
            });
        }
    }

    // TODO: merge duplicates (see next point)
    merge_corners_simple(&mut all, 2.0)
}

/// Coarse-to-fine corner detection on an image pyramid.
///
/// Algorithm:
/// 1. Build a pyramid according to `cf.pyramid`.
/// 2. Run full ChESS detection on the coarsest level only.
/// 3. Upscale each coarse corner to base-level coordinates.
/// 4. Around each predicted location, compute a ChESS response patch at the
///    base level and run NMS + subpixel refinement inside that patch.
/// 5. Merge duplicates within `cf.merge_radius`.
///
/// This trades a small amount of complexity for a significant speed-up on
/// large images, since the dense response is never computed for the full
/// base-resolution frame. Pass a persistent `PyramidBuffers` to reuse memory
/// across frames.
pub fn find_corners_coarse_to_fine_image(
    img: &GrayImage,
    params: &ChessParams,
    cf: &CoarseToFineParams,
    buffers: &mut PyramidBuffers,
) -> Vec<MultiscaleCorner> {
    find_corners_coarse_to_fine_image_trace(img, params, cf, buffers).corners
}

/// Coarse-to-fine corner detection with timing stats.
///
/// See [`find_corners_coarse_to_fine_image`] for the algorithm outline. This
/// variant additionally reports per-stage timings so you can profile the
/// multiscale pipeline.
pub fn find_corners_coarse_to_fine_image_trace(
    img: &GrayImage,
    params: &ChessParams,
    cf: &CoarseToFineParams,
    buffers: &mut PyramidBuffers,
) -> CoarseToFineResult {
    let build_started = Instant::now();
    let pyramid = build_pyramid(img, &cf.pyramid, buffers);
    let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
    if pyramid.levels.is_empty() {
        return CoarseToFineResult {
            corners: Vec::new(),
            build_ms,
            coarse_ms: 0.0,
            refine_ms: 0.0,
            merge_ms: 0.0,
            coarse_cols: 0,
            coarse_rows: 0,
        };
    }

    let base_w = img.width() as usize;
    let base_h = img.height() as usize;
    let base_w_i = base_w as i32;
    let base_h_i = base_h as i32;

    // Use the last (smallest) level as the coarse detector input.
    let coarse_lvl = pyramid
        .levels
        .last()
        .expect("pyramid levels are non-empty after earlier check");

    let coarse_w = coarse_lvl.img.width() as usize;
    let coarse_h = coarse_lvl.img.height() as usize;

    // Full detection on coarse level
    let coarse_started = Instant::now();
    let coarse_corners =
        detect::find_corners_u8(coarse_lvl.img.as_raw(), coarse_w, coarse_h, params);
    let coarse_ms = coarse_started.elapsed().as_secs_f64() * 1000.0;

    if coarse_corners.is_empty() {
        return CoarseToFineResult {
            corners: Vec::new(),
            build_ms,
            coarse_ms,
            refine_ms: 0.0,
            merge_ms: 0.0,
            coarse_cols: coarse_w,
            coarse_rows: coarse_h,
        };
    }

    let inv_scale = 1.0 / coarse_lvl.scale;

    // Compute the same "border" margin as the core detector uses.
    let ring_r = params.radius as i32;
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

    let refine_started = Instant::now();

    let refine_one = |c: detect::Corner| -> Option<Vec<MultiscaleCorner>> {
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
            img.as_raw(),
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

        let mut out = Vec::with_capacity(patch_corners.len());
        for pc in patch_corners.drain(..) {
            out.push(MultiscaleCorner {
                xy: [pc.xy[0] + x0 as f32, pc.xy[1] + y0 as f32],
                strength: pc.strength,
            });
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    };

    #[cfg(feature = "rayon")]
    let mut refined: Vec<MultiscaleCorner> = coarse_corners
        .into_par_iter()
        .filter_map(refine_one)
        .flatten()
        .collect();

    #[cfg(not(feature = "rayon"))]
    let mut refined: Vec<MultiscaleCorner> = {
        let mut acc = Vec::new();
        for c in coarse_corners {
            if let Some(mut v) = refine_one(c) {
                acc.append(&mut v);
            }
        }
        acc
    };

    let refine_ms = refine_started.elapsed().as_secs_f64() * 1000.0;

    let merge_started = Instant::now();
    let merged = merge_corners_simple(&mut refined, cf.merge_radius);
    let merge_ms = merge_started.elapsed().as_secs_f64() * 1000.0;

    CoarseToFineResult {
        corners: merged,
        build_ms,
        coarse_ms,
        refine_ms,
        merge_ms,
        coarse_cols: coarse_w,
        coarse_rows: coarse_h,
    }
}

fn merge_corners_simple(corners: &mut Vec<MultiscaleCorner>, radius: f32) -> Vec<MultiscaleCorner> {
    let r2 = radius * radius;
    let mut out: Vec<MultiscaleCorner> = Vec::new();

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
    use image::Luma;

    #[test]
    fn merge_corners_prefers_stronger_entries() {
        let mut corners = vec![
            MultiscaleCorner {
                xy: [10.0, 10.0],
                strength: 1.0,
            },
            MultiscaleCorner {
                xy: [11.0, 11.0],
                strength: 5.0,
            },
            MultiscaleCorner {
                xy: [20.0, 20.0],
                strength: 3.0,
            },
        ];
        let merged = merge_corners_simple(&mut corners, 2.5);
        assert_eq!(merged.len(), 2);
        assert!(merged
            .iter()
            .any(|c| c.xy == [11.0, 11.0] && (c.strength - 5.0).abs() < 1e-6));
        assert!(merged.iter().any(|c| c.xy == [20.0, 20.0]));
    }

    #[test]
    fn multiscale_path_runs_on_blank_image() {
        let img = GrayImage::from_pixel(32, 32, Luma([0u8]));
        let params = ChessParams::default();
        let pyramid = PyramidParams::default();
        let mut buffers = PyramidBuffers::new();
        let res = find_corners_multiscale_image(&img, &params, &pyramid, &mut buffers);
        assert!(res.is_empty());
    }

    #[test]
    fn coarse_to_fine_trace_reports_timings() {
        let img = GrayImage::from_pixel(32, 32, Luma([0u8]));
        let params = ChessParams::default();
        let cf = CoarseToFineParams::default();
        let mut buffers = PyramidBuffers::new();
        let res = find_corners_coarse_to_fine_image_trace(&img, &params, &cf, &mut buffers);
        assert!(res.build_ms >= 0.0);
        assert!(res.coarse_ms >= 0.0);
        assert!(res.refine_ms >= 0.0);
        assert!(res.merge_ms >= 0.0);
        assert!(res.corners.is_empty());
    }
}
