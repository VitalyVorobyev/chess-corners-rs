//! Two-stage dense corner detector abstraction.
//!
//! [`DenseDetector`] is the contract the multiscale orchestrator drives
//! over: each implementor computes a dense per-pixel response over an
//! [`ImageView`] (stage 1) and extracts subpixel corner peaks from it
//! (stage 2). Image-domain subpixel refinement (center-of-mass,
//! Förstner, saddle-point, …) is **not** part of this trait — it runs
//! detector-agnostically via
//! [`crate::unstable::refine_corners_on_image`].
//!
//! Two zero-sized implementors live alongside the trait:
//!
//! - [`ChessDetector`] — wraps the ChESS response kernel
//!   ([`crate::chess_response_u8`]) and the
//!   threshold + NMS + cluster-filter stage
//!   ([`crate::unstable::detect_peaks_from_response`]).
//! - [`RadonDetector`] — wraps the whole-image Duda-Frese Radon
//!   response ([`crate::radon_response_u8`]) and the
//!   threshold + NMS + 3-point Gaussian peak-fit stage
//!   ([`crate::detect_peaks_from_radon`]).
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
        detect_peaks_from_radon, radon_response_u8, RadonBuffers, RadonDetectorParams,
        RadonResponseView,
    },
    Corner,
};
use crate::imageview::ImageView;
use crate::refine::CornerRefiner;
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
/// Subpixel refinement on the *input image* (Förstner, saddle-point,
/// center-of-mass, …) is NOT part of this trait — that runs as a
/// post-detection stage via
/// [`crate::unstable::refine_corners_on_image`], which is
/// detector-agnostic.
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

    /// Detector-specific safety border (in **base-image pixels**) the
    /// orchestrator must keep around each seed when carving an ROI.
    /// Typically the sum of the detector's response-support radius
    /// (e.g. ChESS ring radius, Radon ray radius) and its NMS
    /// half-window — i.e. the minimum margin needed for
    /// [`Self::compute_response_patch`] + [`Self::detect_corners`] to
    /// return a non-trivial peak inside the ROI.
    fn roi_border(&self, params: &Self::Params) -> i32;

    /// Apply a detector-appropriate image-domain refinement step to
    /// the peaks produced by [`Self::detect_corners`].
    ///
    /// The default subpixel refiners ([`crate::Refiner`]
    /// variants) expect a [`ResponseMap`] (center-of-mass,
    /// Förstner) or an image patch (saddle-point, Radon-peak) keyed
    /// to the detector's response. ChESS forwards its
    /// [`ResponseMap`] straight through; Radon's
    /// [`RadonResponseView`] does not fit the [`ResponseMap`]
    /// contract (working-resolution layout, different coordinate
    /// frame than the peak positions), so the Radon implementor
    /// keeps the 3-point Gaussian peak fit from
    /// [`Self::detect_corners`] as the subpixel position and skips
    /// further refinement.
    fn refine_peaks_on_image(
        &self,
        corners: Vec<Corner>,
        image: ImageView<'_>,
        response: &Self::Response<'_>,
        refiner: &mut dyn CornerRefiner,
    ) -> Vec<Corner>;

    /// Whether [`Self::refine_peaks_on_image`] actually consumes the
    /// orchestrator-supplied refiner. When `false`, the orchestrator
    /// must not include the refiner's patch radius in the per-seed
    /// ROI margin — otherwise a no-op refiner choice would still
    /// shrink the valid seed area near the image border (a tunable
    /// silently coupling to an unused setting).
    ///
    /// Default `true` matches the ChESS-style "refine on image"
    /// contract; the Radon impl returns `false` because its
    /// [`Self::refine_peaks_on_image`] is a no-op.
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
/// ([`crate::unstable::detect_peaks_from_response`]). Subpixel
/// refinement (center-of-mass, Förstner, saddle-point) is a separate
/// detector-agnostic stage; see
/// [`crate::unstable::refine_corners_on_image`].
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

    fn roi_border(&self, params: &Self::Params) -> i32 {
        // Radon operates at working resolution; the ROI is sliced in
        // base-image pixels, so the border-in-base-pixels accounts
        // for the working-resolution support divided by the
        // upsample. ray_radius and nms_radius are in working pixels.
        let up = params.image_upsample_clamped() as i32;
        let ray = params.ray_radius_clamped() as i32;
        let nms = params.nms_radius as i32;
        ((ray + nms + up - 1) / up).max(0)
    }

    fn refine_peaks_on_image(
        &self,
        corners: Vec<Corner>,
        _image: ImageView<'_>,
        _response: &Self::Response<'_>,
        _refiner: &mut dyn CornerRefiner,
    ) -> Vec<Corner> {
        // Radon's `detect_corners` already runs a 3-point Gaussian
        // peak fit on the response map, which IS the subpixel
        // refinement step for this detector. The default
        // `Refiner::CenterOfMass` consumes a `ResponseMap` keyed to
        // the seed's coordinate frame; Radon's response is a
        // `RadonResponseView` at working resolution (different
        // coordinate frame and different element layout), so wiring
        // the refiner here would either reject every corner (if we
        // pass `None`) or silently sample the wrong frame. We keep
        // the peaks as the detector emitted them.
        corners
    }

    fn refines_on_image(&self) -> bool {
        // Paired with the no-op `refine_peaks_on_image` above: the
        // orchestrator must not pad the per-seed ROI with the
        // refiner's patch radius, since the refiner is never
        // consulted on the Radon path.
        false
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
