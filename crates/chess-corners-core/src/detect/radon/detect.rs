//! Peak detection over a Radon response map.
//!
//! Applies threshold / NMS / min-cluster rejection on the
//! working-resolution response from
//! [`super::response::radon_response_u8`], then a 3-point peak fit on
//! the blurred response to get subpixel offsets. Output coordinates
//! are divided by `image_upsample` so they sit in the **input pixel
//! frame**.

use super::primitives::fit_peak_frac;
use super::response::{RadonDetectorParams, RadonResponseView};
use crate::detect::{count_positive_neighbors, is_local_max, Corner};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Stage 1 of Radon detection: threshold + NMS + cluster-filter +
/// 3-point Gaussian peak-fit on the response map.
///
/// The 3-point Gaussian peak-fit is a **response-map** subpixel
/// operation (not image-domain refinement), so it stays inside this
/// stage. The output is already subpixel in the input-image frame
/// (positions are divided by `image_upsample`). Image-domain
/// refinement against the input pixels is the caller's choice and
/// lives in [`crate::unstable::refine_corners_on_image`].
pub fn detect_peaks_from_radon(
    resp: &RadonResponseView<'_>,
    params: &RadonDetectorParams,
) -> Vec<Corner> {
    let w = resp.width();
    let h = resp.height();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let data = resp.data();

    let mut max_r = f32::NEG_INFINITY;
    for &v in data {
        if v > max_r {
            max_r = v;
        }
    }
    if !max_r.is_finite() {
        return Vec::new();
    }

    let thr = params
        .threshold_abs
        .unwrap_or(params.threshold_rel * max_r)
        .max(0.0);

    let nms_r = params.nms_radius as i32;
    let ray_r = params.ray_radius_clamped() as i32;
    // We already zeroed pixels < ray_r from the border; leave 1 extra
    // pixel for the 3-point peak fit.
    let border = (ray_r + nms_r + 1).max(0) as usize;
    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    let inv_up = 1.0 / (params.image_upsample_clamped() as f32);
    let min_cluster = params.min_cluster_size;
    let peak_fit = params.peak_fit;

    #[inline(always)]
    fn at(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
        data[y * w + x]
    }

    let process_row = |y: usize, sink: &mut Vec<Corner>| {
        for x in border..(w - border) {
            let v = at(data, w, x, y);
            if v <= thr {
                continue;
            }
            if !is_local_max(data, w, h, x, y, nms_r, v) {
                continue;
            }
            if count_positive_neighbors(data, w, h, x, y, nms_r) < min_cluster {
                continue;
            }

            let r_c = v;
            let r_xm = at(data, w, x - 1, y);
            let r_xp = at(data, w, x + 1, y);
            let r_ym = at(data, w, x, y - 1);
            let r_yp = at(data, w, x, y + 1);
            let fx = fit_peak_frac(r_xm, r_c, r_xp, peak_fit);
            let fy = fit_peak_frac(r_ym, r_c, r_yp, peak_fit);

            let gx = (x as f32 + fx) * inv_up;
            let gy = (y as f32 + fy) * inv_up;
            sink.push(Corner {
                x: gx,
                y: gy,
                strength: v,
            });
        }
    };

    #[cfg(feature = "rayon")]
    {
        let row_results: Vec<Vec<Corner>> = (border..(h - border))
            .into_par_iter()
            .map(|y| {
                let mut sink = Vec::new();
                process_row(y, &mut sink);
                sink
            })
            .collect();
        let total: usize = row_results.iter().map(|r| r.len()).sum();
        let mut out = Vec::with_capacity(total);
        for row in row_results {
            out.extend(row);
        }
        out
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut out = Vec::new();
        for y in border..(h - border) {
            process_row(y, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::response::{radon_response_u8, RadonBuffers, RadonDetectorParams};
    use super::super::test_fixtures::synthetic_chessboard_aa;
    use super::*;

    #[test]
    fn detector_recovers_interior_corner_without_seed() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);

        let params = RadonDetectorParams {
            image_upsample: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_peaks_from_radon(&resp, &params);

        assert!(
            !corners.is_empty(),
            "expected at least one corner, got none"
        );

        let expected_x =
            offset.0 + (((SIZE as f32 / 2.0 - offset.0) / CELL as f32).round() * CELL as f32);
        let expected_y =
            offset.1 + (((SIZE as f32 / 2.0 - offset.1) / CELL as f32).round() * CELL as f32);

        let (best, err) = corners
            .iter()
            .map(|c| {
                let dx = c.x - expected_x;
                let dy = c.y - expected_y;
                (c, (dx * dx + dy * dy).sqrt())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty");
        assert!(
            err < 0.2,
            "closest corner err={:.4} at ({:.3}, {:.3}), expected ({:.3}, {:.3}); {} corners found",
            err,
            best.x,
            best.y,
            expected_x,
            expected_y,
            corners.len(),
        );
    }

    #[test]
    fn detector_upsample_1_still_produces_corners() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 1,
            ray_radius: 2,
            nms_radius: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_peaks_from_radon(&resp, &params);
        assert!(!corners.is_empty(), "upsample=1 produced no corners");
    }
}
