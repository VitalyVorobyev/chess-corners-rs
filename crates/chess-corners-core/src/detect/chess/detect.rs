//! Corner detection utilities built on top of the dense ChESS response map.
use super::response::chess_response_u8;
use crate::detect::{count_positive_neighbors, is_local_max, Corner, CornerDescriptor};
use crate::imageview::ImageView;
use crate::orientation::describe_corners;
use crate::refine::{CornerRefiner, RefineContext, RefineStatus, Refiner};
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Compute corners starting from an 8-bit grayscale image.
///
/// This is a convenience that combines:
/// - chess_response_u8 (dense response map)
/// - thresholding + NMS
/// - subpixel refinement driven by [`ChessParams::refiner`]
pub fn find_corners_u8(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> Vec<CornerDescriptor> {
    let mut refiner = Refiner::from_kind(params.refiner.clone());
    find_corners_u8_with_refiner(img, w, h, params, &mut refiner)
}

/// Compute corners starting from an 8-bit grayscale image using a custom refiner.
pub fn find_corners_u8_with_refiner(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
    refiner: &mut dyn CornerRefiner,
) -> Vec<CornerDescriptor> {
    let resp = chess_response_u8(img, w, h, params);
    let image =
        ImageView::from_u8_slice(w, h, img).expect("image dimensions must match buffer length");
    let corners = detect_corners_from_response_with_refiner(&resp, params, Some(image), refiner);
    let desc_radius = params.ring_radius();
    describe_corners(img, w, h, desc_radius, corners, params.orientation_method)
}

/// Core detector: run NMS + refinement on an existing response map.
///
/// Useful if you want to reuse the response map for debugging or tuning. Honors
/// the absolute response threshold, enforces the configurable NMS radius, and
/// rejects isolated responses via `min_cluster_size`.
pub fn detect_corners_from_response(resp: &ResponseMap, params: &ChessParams) -> Vec<Corner> {
    let mut refiner = Refiner::from_kind(params.refiner.clone());
    detect_corners_from_response_with_refiner(resp, params, None, &mut refiner)
}

/// Detector variant that accepts a user-provided refiner implementation.
///
/// Wires [`detect_peaks_from_response`] (stage 1: threshold + NMS +
/// cluster-filter on the response map) into [`refine_corners_on_image`]
/// (stage 2: image-domain subpixel refinement). The two stages are
/// available individually for callers that want to inspect or replace
/// either half.
pub fn detect_corners_from_response_with_refiner(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    refiner: &mut dyn CornerRefiner,
) -> Vec<Corner> {
    let peaks = detect_peaks_from_response_with_refine_radius(resp, params, refiner.radius());
    refine_corners_on_image(peaks, image, Some(resp), refiner)
}

/// Stage 1 of ChESS detection: threshold + NMS + cluster-filter on the
/// response map.
///
/// Returns peaks at integer coordinates (cast to `f32`) with the raw
/// response value as `strength`. The refiner is **not** consulted at
/// this stage — image-domain subpixel refinement runs separately in
/// [`refine_corners_on_image`].
///
/// The border margin accounts for the ring radius and the NMS window
/// only; if a downstream refiner needs additional border, the fused
/// helper [`detect_corners_from_response_with_refiner`] handles that
/// by deferring to a private variant that also takes the refiner
/// radius.
#[cfg_attr(
    feature = "tracing",
    instrument(level = "debug", skip(resp, params), fields(w = resp.w, h = resp.h))
)]
pub fn detect_peaks_from_response(resp: &ResponseMap, params: &ChessParams) -> Vec<Corner> {
    detect_peaks_from_response_with_refine_radius(resp, params, 0)
}

/// Stage 1 of ChESS detection: same as [`detect_peaks_from_response`]
/// but extends the border margin by `refine_radius` extra pixels so
/// that an image-domain refiner with the given patch half-width can
/// safely operate on every accepted peak.
///
/// Used by the [`DenseDetector`](crate::DenseDetector) trait
/// implementor for ChESS, which threads the refiner radius from the
/// orchestrator into peak detection so the response and refinement
/// stages address the same valid pixel region. See
/// [`detect_corners_from_response_with_refiner`] for the fused
/// variant that combines both stages.
#[cfg_attr(
    feature = "tracing",
    instrument(level = "debug", skip(resp, params), fields(w = resp.w, h = resp.h))
)]
pub fn detect_peaks_from_response_with_refine_radius(
    resp: &ResponseMap,
    params: &ChessParams,
    refine_radius: i32,
) -> Vec<Corner> {
    let w = resp.w;
    let h = resp.h;

    if w == 0 || h == 0 {
        return Vec::new();
    }

    // Absolute response floor. The paper's acceptance criterion is
    // "R > 0", so we use a strict comparison below; the default
    // `threshold = 0.0` reduces to "accept any strictly positive
    // response", the paper's contract.
    let thr = params.threshold.max(0.0);

    let nms_r = params.nms_radius as i32;
    let ring_r = params.ring_radius() as i32;

    // We need to stay away from the borders enough to:
    // - have a full NMS window
    // - have a full refinement window (when chained with a refiner)
    // The response map itself is valid in [ring_r .. w-ring_r), but
    // we don't want to sample outside [0..w/h) during refinement.
    let border = (ring_r + nms_r + refine_radius).max(0) as usize;

    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    let mut corners = Vec::new();

    for y in border..(h - border) {
        for x in border..(w - border) {
            let v = resp.at(x, y);
            if v <= thr {
                continue;
            }

            // Local maximum in NMS window
            if !is_local_max(resp.data(), resp.w, resp.h, x, y, nms_r, v) {
                continue;
            }

            // Reject isolated pixels: require a minimum number of positive
            // neighbors in the same NMS window.
            let cluster_size = count_positive_neighbors(resp.data(), resp.w, resp.h, x, y, nms_r);
            if cluster_size < params.min_cluster_size {
                continue;
            }

            corners.push(Corner {
                x: x as f32,
                y: y as f32,
                strength: v,
            });
        }
    }

    corners
}

/// Stage 2 of detection: image-domain subpixel refinement.
///
/// Detector-agnostic: works on any `Vec<Corner>` regardless of whether
/// the peaks came from the ChESS or Radon detector. Each input peak
/// is fed to `refiner` with a [`RefineContext`] containing the image
/// view and the optional response map. Peaks the refiner rejects
/// (status not [`RefineStatus::Accepted`]) are dropped from the
/// output; accepted peaks are emitted with their refined subpixel
/// `(x, y)` and the **input** `strength` (the refiner does not
/// rescore the peak strength).
///
/// Iteration order matches the order of the input vector — necessary
/// for downstream stages that assume a stable scan order.
#[cfg_attr(
    feature = "tracing",
    instrument(level = "debug", skip(corners, image, response, refiner))
)]
pub fn refine_corners_on_image(
    corners: Vec<Corner>,
    image: Option<ImageView<'_>>,
    response: Option<&ResponseMap>,
    refiner: &mut dyn CornerRefiner,
) -> Vec<Corner> {
    if corners.is_empty() {
        return Vec::new();
    }

    let ctx = RefineContext { image, response };
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let seed_xy = [c.x, c.y];
        let res = refiner.refine(seed_xy, ctx);
        if matches!(res.status, RefineStatus::Accepted) {
            out.push(Corner {
                x: res.x,
                y: res.y,
                strength: c.strength,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refine::{
        CenterOfMassConfig, CenterOfMassRefiner, RefineContext, RefineStatus, RefinerKind,
    };
    use image::{GrayImage, Luma};

    fn make_quadrant_corner(size: u32, dark: u8, bright: u8) -> GrayImage {
        let mut img = GrayImage::from_pixel(size, size, Luma([dark]));
        let mid = size / 2;
        for y in 0..size {
            for x in 0..size {
                let in_top = y < mid;
                let in_left = x < mid;
                if in_top ^ in_left {
                    img.put_pixel(x, y, Luma([bright]));
                }
            }
        }
        img
    }

    #[test]
    fn descriptors_report_two_axes_stable() {
        use core::f32::consts::{FRAC_PI_2, PI};

        let size = 32u32;
        let params = ChessParams::default();

        let img = make_quadrant_corner(size, 20, 220);
        let corners = find_corners_u8(img.as_raw(), size as usize, size as usize, &params);
        assert!(!corners.is_empty(), "expected at least one descriptor");

        let best = corners
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .expect("non-empty");
        let best_axes = best.axes.expect("orientation enabled (default config)");

        // axes[0] in [0, π), axes[1] in (axes[0], axes[0] + π)
        assert!(best_axes[0].angle >= 0.0 && best_axes[0].angle < PI);
        assert!(
            best_axes[1].angle > best_axes[0].angle && best_axes[1].angle < best_axes[0].angle + PI
        );

        // The quadrant corner has one axis horizontal (line angle 0)
        // and one vertical (line angle π/2). Accept a generous tolerance
        // because the 32×32 synthetic image is aliased.
        let near_line = |x: f32, target: f32| -> f32 {
            let xr = x.rem_euclid(PI);
            let tr = target.rem_euclid(PI);
            let d = (xr - tr).abs();
            d.min(PI - d)
        };
        // One of the two axes matches horizontal (line 0), the other vertical (line π/2).
        let horiz = near_line(best_axes[0].angle, 0.0).min(near_line(best_axes[1].angle, 0.0));
        let vert =
            near_line(best_axes[0].angle, FRAC_PI_2).min(near_line(best_axes[1].angle, FRAC_PI_2));
        assert!(
            horiz < 0.35,
            "horiz line miss: {horiz}, axes {:?}",
            best_axes
        );
        assert!(vert < 0.35, "vert line miss: {vert}, axes {:?}", best_axes);

        // Brightness shift stability: both axes survive a global
        // intensity offset.
        let mut brighter = img.clone();
        for p in brighter.pixels_mut() {
            p[0] = p[0].saturating_add(5);
        }

        let brighter_corners =
            find_corners_u8(brighter.as_raw(), size as usize, size as usize, &params);
        assert!(!brighter_corners.is_empty());
        let best_brighter = brighter_corners
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .expect("non-empty brighter");
        let best_brighter_axes = best_brighter
            .axes
            .expect("orientation enabled (default config)");

        assert!((best.x - best_brighter.x).abs() < 0.5 && (best.y - best_brighter.y).abs() < 0.5);

        let da0 = near_line(best_axes[0].angle, best_brighter_axes[0].angle);
        let da1 = near_line(best_axes[1].angle, best_brighter_axes[1].angle);
        assert!(da0 < 0.35, "axis0 delta after brightness shift: {da0}");
        assert!(da1 < 0.35, "axis1 delta after brightness shift: {da1}");
    }

    #[test]
    fn default_refiner_matches_center_of_mass() {
        let mut resp = ResponseMap {
            w: 32,
            h: 32,
            data: vec![0.0; 32 * 32],
        };

        let cx = 16usize;
        let cy = 16usize;
        let w = resp.w;

        resp.data[cy * w + cx] = 10.0;
        resp.data[cy * w + (cx + 1)] = 6.0;
        resp.data[(cy + 1) * w + cx] = 5.0;
        resp.data[(cy + 1) * w + (cx + 1)] = 4.0;

        let params = ChessParams::default();

        let mut refiner = CenterOfMassRefiner::new(CenterOfMassConfig::default());
        let ctx = RefineContext {
            image: None,
            response: Some(&resp),
        };
        let expected = refiner.refine([cx as f32, cy as f32], ctx);
        assert_eq!(expected.status, RefineStatus::Accepted);

        let corners = detect_corners_from_response(&resp, &params);
        assert_eq!(corners.len(), 1);
        let c = &corners[0];
        assert!((c.x - expected.x).abs() < 1e-6);
        assert!((c.y - expected.y).abs() < 1e-6);
    }

    #[test]
    fn params_refiner_controls_margin() {
        let mut resp = ResponseMap {
            w: 30,
            h: 30,
            data: vec![0.0; 30 * 30],
        };

        let cx = 10usize;
        let cy = 10usize;
        let w = resp.w;

        resp.data[cy * w + cx] = 10.0;
        resp.data[cy * w + (cx + 1)] = 1.0;
        resp.data[(cy + 1) * w + cx] = 1.0;

        let mut params = ChessParams {
            threshold: 0.5,
            ..Default::default()
        };

        let baseline = detect_corners_from_response(&resp, &params);
        assert_eq!(baseline.len(), 1, "expected baseline detection");

        params.refiner = RefinerKind::CenterOfMass(CenterOfMassConfig { radius: 4 });
        let shrunk = detect_corners_from_response(&resp, &params);
        assert!(
            shrunk.is_empty(),
            "larger refiner radius should increase border and skip the corner"
        );
    }
}
