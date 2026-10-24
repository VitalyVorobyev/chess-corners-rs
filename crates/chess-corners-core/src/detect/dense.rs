//! Two-stage dense corner detector abstraction.
//!
//! [`DenseDetector`] is the contract the multiscale orchestrator drives
//! over: each implementor computes a dense per-pixel response over an
//! [`ImageView`] (stage 1) and extracts subpixel corner peaks from it
//! (stage 2); [`DenseDetector::refine_peaks_on_image`] then refines the
//! peaks on the input image with one of the built-in refiners.
//!
//! Two zero-sized implementors live alongside the trait:
//!
//! - [`ChessDetector`] — wraps the ChESS response kernel
//!   ([`crate::chess_response_u8`]) and the
//!   threshold + NMS + cluster-filter stage
//!   ([`crate::detect_peaks_from_response_with_refine_radius`]).
//! - [`RadonDetector`] — wraps the whole-image Duda-Frese Radon
//!   response ([`crate::radon_response_u8`]) and the
//!   threshold + NMS + 3-point Gaussian peak-fit stage
//!   ([`crate::detect_peaks_from_radon`]); image refinement keeps the
//!   peak-fit position unless the refiner reports a covariance.
//!
//! The free functions named above remain public; the trait is an
//! additive uniform interface, not a replacement.
//!
//! # Toolchain note
//!
//! [`DenseDetector::Response`] is a generic associated type (GAT);
//! downstream consumers need Rust 1.65 or newer. This workspace
//! already pins nightly via `rust-toolchain.toml`, so the trait is
//! always available here.

use super::{
    chess::{
        detect::{detect_peaks_from_response_with_refine_radius, refine_corners_on_image},
        response::{chess_response_u8, chess_response_u8_patch, Roi},
    },
    radon::{
        detect::retain_above_relative_threshold, detect_peaks_from_radon, radon_response_u8,
        RadonBuffers, RadonDetectorParams, RadonResponseView,
    },
    Corner,
};
use crate::imageview::ImageView;
use crate::refine::{CornerRefiner, RefineContext, RefineStatus};
use crate::{ChessParams, ResponseMap};

/// Sealing supertrait for [`DenseDetector`]. Implemented only for the
/// in-crate detector markers, so downstream crates cannot add their own
/// [`DenseDetector`] implementations.
mod private {
    pub trait Sealed {}
}

/// Two-stage dense corner detector contract.
///
/// Implementors compute a dense per-pixel response over the input
/// image (`compute_response`, stage 1) and extract subpixel corner
/// peaks from it (`detect_corners`, stage 2). The two stages share
/// reusable scratch through [`Self::Buffers`] so a multiscale
/// orchestrator can amortise allocations across frames.
///
/// Subpixel refinement on the *input image* (center-of-mass,
/// Förstner, saddle-point) runs on the detected peaks through
/// [`Self::refine_peaks_on_image`].
///
/// # Toolchain
///
/// Uses generic associated types ([`Self::Response`]); downstream
/// consumers require Rust 1.65 or newer.
///
/// # Stability
///
/// This trait is **sealed** via a private supertrait bound and cannot
/// be implemented outside this crate. The only implementors are
/// [`ChessDetector`] and [`RadonDetector`]; it is not a public
/// extension point, so its method set may evolve without a breaking
/// release.
pub trait DenseDetector: private::Sealed {
    /// Detector-specific tuning parameters.
    type Params;
    /// Reusable scratch buffers. Allocated once via
    /// [`Default::default`] and reused across `compute_response`
    /// calls to avoid per-frame allocation.
    type Buffers: Default;
    /// Native response representation. May be owned (a borrow of an
    /// owned [`ResponseMap`] in [`Self::Buffers`]) or a transient
    /// view ([`RadonResponseView`]) over the same scratch. The
    /// borrow lifetime ties the response back to the buffers that
    /// produced it.
    type Response<'a>
    where
        Self: 'a,
        Self::Buffers: 'a;

    /// Compute the dense response over `view`, writing into
    /// `buffers` and returning a borrowed handle.
    fn compute_response<'a>(
        &self,
        view: ImageView<'_>,
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a>;

    /// Extract corner peaks from `response`. Positions are in the
    /// input-image frame (the same frame `view` was passed in at
    /// [`Self::compute_response`]).
    ///
    /// `refine_border` is an *additional* base-image-pixel margin
    /// the implementor must keep around each accepted peak so that a
    /// downstream image-domain refiner with that patch half-width
    /// has full support. Passing `0` selects "no extra refiner
    /// margin" — appropriate when refinement happens through a
    /// separate stage that does its own bounds check. Whether the
    /// implementor extends its own border (ChESS) or ignores the
    /// argument (Radon — its NMS + Gaussian peak-fit already enforce
    /// the support needed) is detector-specific.
    fn detect_corners(
        &self,
        response: &Self::Response<'_>,
        params: &Self::Params,
        refine_border: i32,
    ) -> Vec<Corner>;

    /// Compute the dense response over the sub-rectangle
    /// `[x0..x1) × [y0..y1)` of `base`, where the ROI is given as the
    /// tuple `(x0, y0, x1, y1)`. The returned response is sized to
    /// the ROI; any [`Corner`] positions produced from it by
    /// [`Self::detect_corners`] are **patch-local** (origin = ROI's
    /// top-left), and the caller is responsible for shifting them
    /// back into base-image coordinates by adding `(x0, y0)`.
    ///
    /// Implementors may reach outside the ROI to compute responses
    /// near its borders (so that an ROI tile produces values
    /// numerically identical to the full-frame response on the
    /// overlapping interior). The shared `buffers` is reused across
    /// calls.
    fn compute_response_patch<'a>(
        &self,
        base: ImageView<'_>,
        roi: (i32, i32, i32, i32),
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a>;

    /// Apply the frame-level acceptance test to the union of the peaks
    /// [`Self::detect_corners`] returned for every ROI patch of one
    /// frame (see [`Self::compute_response_patch`]).
    ///
    /// A patch can only judge a peak against itself. A detector whose
    /// threshold is relative to the frame's strongest response (Radon)
    /// therefore admits, inside a patch without a strong corner, peaks
    /// far below the frame threshold; this pass re-applies the threshold
    /// against the strongest peak of the union — the frame's strongest
    /// response wherever the coarse level seeded it. A per-patch threshold
    /// is never stricter than that, so the result is the set of peaks the
    /// detector accepts against that maximum.
    ///
    /// The default keeps every corner: an absolute threshold (ChESS)
    /// already decided each peak inside its patch.
    fn retain_frame_peaks(&self, corners: &mut Vec<Corner>, params: &Self::Params) {
        let _ = (corners, params);
    }

    /// Detector-specific safety border (in **base-image pixels**) the
    /// orchestrator must keep around each seed when carving an ROI.
    /// Typically the sum of the detector's response-support radius
    /// (e.g. ChESS ring radius, Radon ray radius) and its NMS
    /// half-window — i.e. the minimum margin needed for
    /// [`Self::compute_response_patch`] + [`Self::detect_corners`] to
    /// return a non-trivial peak inside the ROI.
    fn roi_border(&self, params: &Self::Params) -> i32;

    /// Refine the peaks produced by [`Self::detect_corners`] on the
    /// input image with `refiner`.
    ///
    /// ChESS forwards its [`ResponseMap`] together with the image to
    /// [`crate::refine_corners_on_image`], so every refiner applies, and
    /// drops the corners the refiner does not accept.
    ///
    /// Radon's [`RadonResponseView`] is not a [`ResponseMap`] in the
    /// peak positions' frame, so the refiner sees only the image (the
    /// center-of-mass refiner, which reads the response map, rejects
    /// every seed). Radon peaks are already subpixel, so a corner keeps
    /// its peak-fit position (and covariance) unless the refiner
    /// converges on it — the saddle-point refiner does so when its
    /// re-centred fit converged — and is never dropped.
    fn refine_peaks_on_image(
        &self,
        corners: Vec<Corner>,
        image: ImageView<'_>,
        response: &Self::Response<'_>,
        refiner: &mut dyn CornerRefiner,
    ) -> Vec<Corner>;

    /// Whether [`Self::refine_peaks_on_image`] consumes the refiner:
    /// `true` for both built-in detectors.
    #[deprecated(
        since = "1.6.0",
        note = "both built-in detectors refine on the image; this is always `true`"
    )]
    fn refines_on_image(&self) -> bool {
        true
    }
}

/// Reusable scratch for [`ChessDetector`]. Wraps an owned
/// [`ResponseMap`]; the ChESS response kernel currently allocates its
/// output, and this struct keeps the latest map alive so the trait's
/// `Response<'a> = &'a ResponseMap` can borrow it across the two
/// stages.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct ChessBuffers {
    /// Dense ChESS response from the most recent
    /// [`ChessDetector::compute_response`] call.
    pub response: ResponseMap,
}

/// Zero-sized [`DenseDetector`] implementor for the ChESS kernel.
///
/// Wraps the canonical 16-sample ring response
/// ([`crate::chess_response_u8`]) and the
/// threshold + NMS + cluster-filter peak detector
/// ([`crate::detect_peaks_from_response_with_refine_radius`]). Subpixel
/// refinement (center-of-mass, Förstner, saddle-point) is a separate
/// detector-agnostic stage; see
/// [`crate::refine_corners_on_image`].
#[derive(Debug, Default, Clone, Copy)]
pub struct ChessDetector;

impl private::Sealed for ChessDetector {}

impl DenseDetector for ChessDetector {
    type Params = ChessParams;
    type Buffers = ChessBuffers;
    type Response<'a> = &'a ResponseMap;

    fn compute_response<'a>(
        &self,
        view: ImageView<'_>,
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a> {
        // chess_response_u8 returns an owned ResponseMap. Swap it into
        // the buffer so the borrow returned to the caller lives as long
        // as `buffers`. The previous map (likely from the prior frame)
        // is dropped; its backing Vec capacity is reclaimed at next
        // allocation. A future `chess_response_u8_into(.., &mut Vec)`
        // helper could keep the allocation, but the snapshot-pinned
        // numerical contract has to stay bit-identical first.
        buffers.response = chess_response_u8(view.data, view.width, view.height, params);
        &buffers.response
    }

    fn detect_corners(
        &self,
        response: &Self::Response<'_>,
        params: &Self::Params,
        refine_border: i32,
    ) -> Vec<Corner> {
        detect_peaks_from_response_with_refine_radius(response, params, refine_border)
    }

    fn compute_response_patch<'a>(
        &self,
        base: ImageView<'_>,
        roi: (i32, i32, i32, i32),
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a> {
        // ChESS has a dedicated patch kernel that reaches outside the
        // ROI to compute response values near its borders, so the
        // patch output overlaps numerically with the full-frame
        // response inside the ROI. Copy-then-compute would break that
        // invariant (and the snapshot regression that pins it).
        let (x0, y0, x1, y1) = roi;
        let roi_obj = Roi::new(
            x0.max(0) as usize,
            y0.max(0) as usize,
            x1.max(0) as usize,
            y1.max(0) as usize,
        );
        buffers.response = match roi_obj {
            Some(r) => chess_response_u8_patch(base.data, base.width, base.height, params, r),
            None => ResponseMap::default(),
        };
        &buffers.response
    }

    fn roi_border(&self, params: &Self::Params) -> i32 {
        (params.ring_radius() as i32 + params.nms_radius as i32).max(0)
    }

    fn refine_peaks_on_image(
        &self,
        corners: Vec<Corner>,
        image: ImageView<'_>,
        response: &Self::Response<'_>,
        refiner: &mut dyn CornerRefiner,
    ) -> Vec<Corner> {
        // ChESS's response IS a `ResponseMap`, so we can forward it
        // straight through. This restores the bit-for-bit numerical
        // behaviour of the legacy fused path
        // (`detect_corners_from_response_with_refiner`), which always
        // passed `Some(resp)` to the refiner.
        refine_corners_on_image(corners, Some(image), Some(response), refiner)
    }
}

/// Zero-sized [`DenseDetector`] implementor for the whole-image
/// Duda-Frese Radon kernel.
///
/// Wraps [`crate::radon_response_u8`] (SAT-based dense
/// response) and [`crate::detect_peaks_from_radon`]
/// (threshold, NMS, 3-point Gaussian peak-fit on the working-resolution
/// map). Output [`Corner`] positions are in the input-image frame: the
/// Radon peak detector divides by `image_upsample` internally.
#[derive(Debug, Default, Clone, Copy)]
pub struct RadonDetector;

impl private::Sealed for RadonDetector {}

impl DenseDetector for RadonDetector {
    type Params = RadonDetectorParams;
    type Buffers = RadonBuffers;
    type Response<'a> = RadonResponseView<'a>;

    fn compute_response<'a>(
        &self,
        view: ImageView<'_>,
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a> {
        radon_response_u8(view.data, view.width, view.height, params, buffers)
    }

    fn detect_corners(
        &self,
        response: &Self::Response<'_>,
        params: &Self::Params,
        _refine_border: i32,
    ) -> Vec<Corner> {
        // Radon's working-resolution border already absorbs the
        // ray_radius + nms + 3-point-fit margin internally, and the
        // returned peak positions are in input-image coordinates
        // (post `image_upsample` division). An additional refiner
        // border argument expressed in *base-image pixels* doesn't
        // map cleanly onto the working-resolution peak detector and
        // is ignored.
        detect_peaks_from_radon(response, params)
    }

    fn compute_response_patch<'a>(
        &self,
        base: ImageView<'_>,
        roi: (i32, i32, i32, i32),
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a> {
        // Radon has no ROI-border tricks (the SAT-based kernel reads
        // only inside its input slice), so a copy-then-compute path
        // is numerically equivalent to a hypothetical patch kernel.
        //
        // The ROI patch is allocated locally; `radon_response_u8`
        // borrows `&mut buffers` for the SAT / response scratch, and
        // returning the view ties that borrow to the caller's `'a`,
        // so we cannot also keep the ROI pixels inside `buffers`.
        // The Vec is small (ROI pixel count, hundreds of bytes in
        // typical multiscale use) and the allocation cost is
        // dominated by the SAT build per ROI.
        let (x0, y0, x1, y1) = roi;
        let x0 = (x0.max(0) as usize).min(base.width);
        let y0 = (y0.max(0) as usize).min(base.height);
        let x1 = (x1.max(0) as usize).min(base.width);
        let y1 = (y1.max(0) as usize).min(base.height);
        if x1 <= x0 || y1 <= y0 {
            // Produce a degenerate 0×0 response that detect_corners
            // will see as empty.
            return radon_response_u8(&[], 0, 0, params, buffers);
        }
        let roi_w = x1 - x0;
        let roi_h = y1 - y0;
        let mut scratch = vec![0u8; roi_w * roi_h];
        for py in 0..roi_h {
            let src_off = (y0 + py) * base.width + x0;
            let dst_off = py * roi_w;
            scratch[dst_off..dst_off + roi_w].copy_from_slice(&base.data[src_off..src_off + roi_w]);
        }
        radon_response_u8(&scratch, roi_w, roi_h, params, buffers)
    }

    fn retain_frame_peaks(&self, corners: &mut Vec<Corner>, params: &Self::Params) {
        retain_above_relative_threshold(corners, params.threshold_rel);
    }

    fn roi_border(&self, params: &Self::Params) -> i32 {
        // Radon operates at working resolution; the ROI is sliced in
        // base-image pixels, so the border-in-base-pixels accounts
        // for the working-resolution support divided by the
        // upsample. ray_radius and nms_radius are in working pixels;
        // the extra +1 is the pixel `detect_peaks_from_radon` reserves
        // for its 3-point peak fit — its scan starts at
        // `ray + nms + 1`, so the margin must cover that pixel too or
        // a peak on the ROI boundary falls in the excluded border.
        let up = params.image_upsample_clamped() as i32;
        let ray = params.ray_radius_clamped() as i32;
        let nms = params.nms_radius as i32;
        (((ray + nms + 1) + up - 1) / up).max(0)
    }

    fn refine_peaks_on_image(
        &self,
        mut corners: Vec<Corner>,
        image: ImageView<'_>,
        _response: &Self::Response<'_>,
        refiner: &mut dyn CornerRefiner,
    ) -> Vec<Corner> {
        // The working-resolution Radon response is not a `ResponseMap`
        // in the peaks' frame, so the refiner sees the image only. A
        // refined position replaces the peak fit only when the refiner
        // converged: the saddle-point refiner's integer-grid fallback
        // (re-centring did not converge) is less accurate than the peak
        // fit it would replace.
        let ctx = RefineContext::new(Some(image), None);
        for c in &mut corners {
            let res = refiner.refine([c.x, c.y], ctx);
            if res.status == RefineStatus::Accepted && res.converged {
                c.x = res.x;
                c.y = res.y;
                c.set_model(res.covariance_model());
            }
        }
        corners
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::radon::test_fixtures::synthetic_chessboard_aa;

    /// Smoke test: both detectors return at least one corner on the
    /// shared synthetic anti-aliased chessboard fixture, end to end
    /// through the [`DenseDetector`] trait. The snapshot test in the
    /// facade is the real numerical regression gate; this only
    /// guards the trait wiring.
    #[test]
    fn dense_detector_trait_drives_both_implementors() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let img = synthetic_chessboard_aa(SIZE, CELL, (32.35, 32.8), 30, 230);
        let view = ImageView::from_u8_slice(SIZE, SIZE, &img).expect("view");

        // ChESS path.
        let chess_params = ChessParams::default();
        let mut chess_buffers = ChessBuffers::default();
        let chess = ChessDetector;
        let resp = chess.compute_response(view, &chess_params, &mut chess_buffers);
        let chess_corners = chess.detect_corners(&resp, &chess_params, 0);
        assert!(
            !chess_corners.is_empty(),
            "ChessDetector returned no corners on synthetic chessboard"
        );

        // Radon path.
        let radon_params = RadonDetectorParams {
            image_upsample: 2,
            ..RadonDetectorParams::default()
        };
        let mut radon_buffers = RadonBuffers::default();
        let radon = RadonDetector;
        let resp = radon.compute_response(view, &radon_params, &mut radon_buffers);
        let radon_corners = radon.detect_corners(&resp, &radon_params, 0);
        assert!(
            !radon_corners.is_empty(),
            "RadonDetector returned no corners on synthetic chessboard"
        );
    }

    /// Radon image refinement never drops a peak: the saddle-point refiner
    /// moves the corners it refines to convergence (and attaches their
    /// covariance), every other corner — and every corner under a refiner
    /// that reports no covariance — keeps its peak-fit position.
    #[test]
    fn radon_refinement_keeps_peaks_without_a_covariance() {
        use crate::refine::{CenterOfMassConfig, Refiner, RefinerKind, SaddlePointConfig};
        const SIZE: usize = 65;
        let img = synthetic_chessboard_aa(SIZE, 8, (32.35, 32.8), 30, 230);
        let view = ImageView::from_u8_slice(SIZE, SIZE, &img).expect("view");
        let params = RadonDetectorParams::default();
        let mut buffers = RadonBuffers::default();
        let resp = RadonDetector.compute_response(view, &params, &mut buffers);
        let peaks = RadonDetector.detect_corners(&resp, &params, 0);
        assert!(peaks.len() > 10);

        let mut sp = Refiner::from_kind(RefinerKind::SaddlePoint(SaddlePointConfig::default()));
        let refined = RadonDetector.refine_peaks_on_image(peaks.clone(), view, &resp, &mut sp);
        assert_eq!(refined.len(), peaks.len());
        let mut moved = 0;
        for (p, r) in peaks.iter().zip(&refined) {
            assert_eq!(p.strength, r.strength);
            if r.covariance_per_noise_variance.is_some() {
                moved += 1;
                assert!((p.x - r.x).abs() < 0.5 && (p.y - r.y).abs() < 0.5);
            } else {
                assert_eq!((p.x, p.y), (r.x, r.y));
            }
        }
        assert!(
            moved * 2 > peaks.len(),
            "{moved} of {} refined",
            peaks.len()
        );

        let mut com = Refiner::from_kind(RefinerKind::CenterOfMass(CenterOfMassConfig::default()));
        let kept = RadonDetector.refine_peaks_on_image(peaks.clone(), view, &resp, &mut com);
        assert_eq!(kept.len(), peaks.len());
        for (p, k) in peaks.iter().zip(&kept) {
            assert_eq!((p.x, p.y), (k.x, k.y));
            assert!(k.covariance_per_noise_variance.is_none());
        }
    }

    /// A second smoke test that pins the buffer-reuse semantics: a
    /// fresh `ChessBuffers::default()` must produce a valid response
    /// before any explicit initialisation, and a second call on the
    /// same buffer must continue to work.
    #[test]
    fn chess_buffers_default_supports_first_use_and_reuse() {
        const SIZE: usize = 33;
        const CELL: usize = 6;
        let img_a = synthetic_chessboard_aa(SIZE, CELL, (16.4, 16.1), 30, 230);
        let img_b = synthetic_chessboard_aa(SIZE, CELL, (16.7, 16.2), 30, 230);
        let view_a = ImageView::from_u8_slice(SIZE, SIZE, &img_a).expect("view");
        let view_b = ImageView::from_u8_slice(SIZE, SIZE, &img_b).expect("view");
        let params = ChessParams::default();
        let mut buffers = ChessBuffers::default();
        let chess = ChessDetector;

        let resp_a = chess.compute_response(view_a, &params, &mut buffers);
        let n_a = chess.detect_corners(&resp_a, &params, 0).len();
        assert!(n_a > 0);

        let resp_b = chess.compute_response(view_b, &params, &mut buffers);
        let n_b = chess.detect_corners(&resp_b, &params, 0).len();
        assert!(n_b > 0);
    }
}
