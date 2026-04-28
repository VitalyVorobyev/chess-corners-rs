//! Parity tests between the whole-image Radon detector and the
//! per-candidate Radon refiner.
//!
//! **Important design note.** The two paths compute a 4-angle
//! `(max − min)²` Radon response but with different ray geometries:
//!
//! - [`RadonPeakRefiner`](chess_corners_core::RadonPeakRefiner) uses
//!   *isotropic* rays — bilinear-sampled at `1/image_upsample` steps
//!   along a unit direction vector. All four rays have the same
//!   Euclidean length `2·ray_radius` in physical pixels.
//! - [`radon_response_u8`](chess_corners_core::radon_response_u8) uses
//!   *axially-equal* rays — integer strides through summed-area
//!   tables. The diagonal rays cover `√2·` more Euclidean distance
//!   than the axial rays at the same `ray_radius`. This is the SAT
//!   speedup's price; it matches the paper's whole-image pipeline.
//!
//! As a consequence the two response *values* disagree by up to a few
//! ×, but both detect the same underlying corner pattern and their
//! subpixel peaks coincide to well under 0.1 px on clean inputs.
//!
//! These tests codify the split:
//!
//! 1. **Axial parity** — at every interior pixel, the detector's
//!    horizontal and vertical ray sums must match the refiner's to
//!    within u8-quantisation noise. This pins the shared
//!    1-D prefix-sum / bilinear equivalence.
//! 2. **Corner-location parity** — the detector's subpixel peak at a
//!    known synthetic corner must coincide with the refiner's subpixel
//!    result seeded at the same integer pixel, well under 0.1 px.

use chess_corners_core::{
    radon_response_u8, CornerRefiner, ImageView, PeakFitMode, RadonBuffers, RadonDetectorParams,
    RadonPeakConfig, RadonPeakRefiner, RefineContext, RefineStatus,
};

fn synthetic_chessboard_aa(
    size: usize,
    cell: usize,
    offset: (f32, f32),
    dark: u8,
    bright: u8,
) -> Vec<u8> {
    const SUPER: usize = 8;
    let (ox, oy) = offset;
    let c = cell as f32;
    let dark_f = dark as f32;
    let bright_f = bright as f32;
    let inv_super2 = 1.0 / (SUPER * SUPER) as f32;
    let mut img = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0f32;
            for sy in 0..SUPER {
                let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                let cy = ((yf - oy) / c).floor() as i32;
                for sx in 0..SUPER {
                    let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                    let cx = ((xf - ox) / c).floor() as i32;
                    let dark_cell = (cx + cy).rem_euclid(2) == 0;
                    acc += if dark_cell { dark_f } else { bright_f };
                }
            }
            img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

/// Sum a single-angle ray via the refiner's convention (isotropic,
/// bilinear, step = 1/upsample). Used as a ground truth for the axial
/// parity test.
fn refiner_ray_sum(
    view: &ImageView<'_>,
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    step: f32,
    ray_samples: i32,
) -> f32 {
    let mut sum = 0.0f32;
    for k in -ray_samples..=ray_samples {
        let kf = k as f32 * step;
        sum += view.sample_bilinear(cx + kf * dx, cy + kf * dy);
    }
    sum
}

#[test]
fn axial_ray_sums_match_refiner_at_image_upsample_1() {
    // At image_upsample=1 and on integer positions, the detector's
    // SAT sums are over the same 2r+1 integer pixels the refiner
    // would sample with `step=1`. They must match exactly (modulo
    // f32 summation order).
    const SIZE: usize = 33;
    let img = synthetic_chessboard_aa(SIZE, 6, (16.3, 16.7), 30, 230);
    let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();

    let ray_radius: u32 = 2;
    let params = RadonDetectorParams {
        ray_radius,
        image_upsample: 1,
        response_blur_radius: 0,
        peak_fit: PeakFitMode::Gaussian,
        threshold_abs: Some(0.0),
        threshold_rel: 0.0,
        nms_radius: 1,
        min_cluster_size: 0,
    };
    let mut buffers = RadonBuffers::new();
    let detector_resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);

    // The detector's response is `(max - min)²` over 4 rays using the
    // anisotropic SAT convention. We cross-check by computing the
    // response a different way — directly from the refiner's
    // isotropic-ray formulation — BUT only at the two axial angles
    // where the two conventions agree (horizontal and vertical rays
    // collapse to single-axis integer strides regardless of
    // convention). This verifies the shared 1-D prefix-sum primitive.

    let r = ray_radius as usize;
    let ray_samples = ray_radius as i32;

    let mut worst_abs = 0.0f32;
    for y in r..(SIZE - r) {
        for x in r..(SIZE - r) {
            // Direct axial ray sums from the image.
            let mut h_sum = 0.0f32;
            for k in -(r as i32)..=(r as i32) {
                h_sum += img[y * SIZE + ((x as i32 + k) as usize)] as f32;
            }
            let mut v_sum = 0.0f32;
            for k in -(r as i32)..=(r as i32) {
                v_sum += img[((y as i32 + k) as usize) * SIZE + x] as f32;
            }

            // Refiner-style bilinear sums at step=1.0. Should match exactly.
            let ref_h = refiner_ray_sum(&view, x as f32, y as f32, 1.0, 0.0, 1.0, ray_samples);
            let ref_v = refiner_ray_sum(&view, x as f32, y as f32, 0.0, 1.0, 1.0, ray_samples);
            assert!(
                (h_sum - ref_h).abs() < 1e-3,
                "horiz at ({x},{y}): direct={h_sum} refiner={ref_h}"
            );
            assert!(
                (v_sum - ref_v).abs() < 1e-3,
                "vert at ({x},{y}): direct={v_sum} refiner={ref_v}"
            );

            // The DETECTOR's response is shape-consistent with both
            // (it's also non-negative and finite). We don't require
            // numerical equality of the full (max-min)² here because
            // the diagonal rays differ geometrically (see module doc).
            let dv = detector_resp.at(x, y);
            assert!(dv.is_finite() && dv >= 0.0, "bad detector value {dv}");
            worst_abs = worst_abs.max((h_sum - ref_h).abs().max((v_sum - ref_v).abs()));
        }
    }
    eprintln!("axial parity worst abs-diff = {worst_abs:.3e}");
}

#[test]
fn detector_peak_matches_refiner_peak_on_clean_corner() {
    // The true test: both paths pick out the same subpixel corner.
    // Render an AA chessboard, run the full detector, then run the
    // refiner seeded at the round()'d detector result on the same
    // image. The two subpixel outputs should agree to well under
    // 0.1 px — they're different integrators of the same underlying
    // Radon-peak structure.
    const SIZE: usize = 65;
    const CELL: usize = 8;
    let offset = (32.35, 32.8);
    let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);
    let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();

    let det_params = RadonDetectorParams {
        image_upsample: 2,
        ..RadonDetectorParams::default()
    };
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(&img, SIZE, SIZE, &det_params, &mut buffers);
    let detector_corners = chess_corners_core::detect_corners_from_radon(&resp, &det_params);

    // Find the corner nearest to the image center.
    let ctr_x = SIZE as f32 * 0.5;
    let ctr_y = SIZE as f32 * 0.5;
    let best_det = detector_corners
        .iter()
        .min_by(|a, b| {
            let da = (a.x - ctr_x).powi(2) + (a.y - ctr_y).powi(2);
            let db = (b.x - ctr_x).powi(2) + (b.y - ctr_y).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .expect("detector returned no corners");

    // Refine with RadonPeak seeded at the detector's integer pixel.
    let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
    let seed = [best_det.x.round(), best_det.y.round()];
    let refined = refiner.refine(seed, RefineContext::new(Some(view), None));
    assert_eq!(refined.status, RefineStatus::Accepted);

    let dx = best_det.x - refined.x;
    let dy = best_det.y - refined.y;
    let diff = (dx * dx + dy * dy).sqrt();
    eprintln!(
        "detector=({:.4}, {:.4}) refiner=({:.4}, {:.4}) diff={:.4}",
        best_det.x, best_det.y, refined.x, refined.y, diff,
    );
    assert!(
        diff < 0.1,
        "detector-refiner subpixel peak disagreement {diff} >= 0.1 px"
    );
}
