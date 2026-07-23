//! High-level chessboard-corner detector with reusable scratch buffers.
//!
//! [`Detector`] is the primary entry point for the `chess-corners`
//! crate. It owns the [`DetectorConfig`] and the scratch buffers
//! (pyramid, upscale, …) required to run detection without
//! re-allocating across frames. It dispatches to either the ChESS or
//! the Radon strategy depending on the active [`DetectorConfig::strategy`].
//!
//! ```
//! use chess_corners::{Detector, DetectorConfig};
//!
//! // 8×8 black/white checkerboard of 16-pixel squares (128×128).
//! let mut img = vec![0u8; 128 * 128];
//! for y in 0..128 {
//!     for x in 0..128 {
//!         if ((x / 16) + (y / 16)) % 2 == 0 {
//!             img[y * 128 + x] = 255;
//!         }
//!     }
//! }
//!
//! let mut detector = Detector::new(DetectorConfig::chess_multiscale())?;
//! let corners = detector.detect_u8(&img, 128, 128)?;
//! assert!(!corners.is_empty());
//! # Ok::<(), chess_corners::ChessError>(())
//! ```

use box_image_pyramid::PyramidBuffers;
use chess_corners_core::{ChessBuffers, RadonBuffers};

#[cfg(feature = "ml-refiner")]
use crate::ml_refiner;
use crate::multiscale;
use crate::upscale::{self, UpscaleBuffers};
use crate::{ChessError, CornerDescriptor, DetectorConfig, Roi};
use chess_corners_core::ImageView;

/// High-level chessboard-corner detector.
///
/// Owns the pyramid and detector-specific scratch buffers so the
/// caller can reuse them across successive frames.
pub struct Detector {
    cfg: DetectorConfig,
    pyramid: PyramidBuffers,
    chess_buffers: ChessBuffers,
    radon_buffers: RadonBuffers,
    upscale: UpscaleBuffers,
    #[cfg(feature = "ml-refiner")]
    ml_state: Option<ml_refiner::MlRefinerState>,
    #[cfg(feature = "ml-refiner")]
    ml_params: ml_refiner::MlRefinerParams,
}

impl Detector {
    /// Build a detector with the given config.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::Upscale`] when the [`DetectorConfig::upscale`]
    /// configuration is invalid.
    pub fn new(cfg: DetectorConfig) -> Result<Self, ChessError> {
        cfg.upscale.validate()?;
        Ok(Self {
            cfg,
            pyramid: PyramidBuffers::default(),
            chess_buffers: ChessBuffers::default(),
            radon_buffers: RadonBuffers::default(),
            upscale: UpscaleBuffers::new(),
            #[cfg(feature = "ml-refiner")]
            ml_state: None,
            #[cfg(feature = "ml-refiner")]
            ml_params: ml_refiner::MlRefinerParams::default(),
        })
    }

    /// Build a detector with the default config.
    pub fn with_default() -> Self {
        // DetectorConfig::default() always has a valid upscale config
        // (`Off`), so `new` cannot fail here.
        Self::new(DetectorConfig::default()).expect("default DetectorConfig is always valid")
    }

    /// Borrow the active config.
    pub fn config(&self) -> &DetectorConfig {
        &self.cfg
    }

    /// Replace the active config.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::Upscale`] when the new config's upscale
    /// section is invalid.
    pub fn set_config(&mut self, cfg: DetectorConfig) -> Result<(), ChessError> {
        cfg.upscale.validate()?;
        self.cfg = cfg;
        // Drop ML state on config change so the next `detect` call
        // re-builds it against the (possibly new) fallback refiner.
        #[cfg(feature = "ml-refiner")]
        {
            self.ml_state = None;
        }
        Ok(())
    }

    /// Detect chessboard corners from a raw 8-bit grayscale buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::DimensionMismatch`] if `img.len() !=
    /// width * height`. Returns [`ChessError::Upscale`] if the upscale
    /// configuration becomes invalid (this should not normally
    /// happen — [`Detector::new`] / [`Detector::set_config`] validate
    /// up-front).
    pub fn detect_u8(
        &mut self,
        img: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<CornerDescriptor>, ChessError> {
        let src_w = width as usize;
        let src_h = height as usize;
        let expected = src_w * src_h;
        if img.len() != expected {
            return Err(ChessError::DimensionMismatch {
                expected,
                actual: img.len(),
            });
        }

        let factor = self.cfg.upscale.effective_factor();
        if factor <= 1 {
            let view =
                ImageView::from_u8_slice(src_w, src_h, img).expect("dimensions were checked above");
            return Ok(Self::detect_view_inner(
                &self.cfg,
                &mut self.pyramid,
                &mut self.chess_buffers,
                &mut self.radon_buffers,
                #[cfg(feature = "ml-refiner")]
                &mut self.ml_state,
                #[cfg(feature = "ml-refiner")]
                &self.ml_params,
                view,
            ));
        }

        // Split-borrow: each field is borrowed independently so
        // `upscaled` (which borrows `self.upscale`) and the
        // detect_view_inner call (which borrows other fields) don't
        // conflict.
        let upscaled = upscale::upscale_bilinear_u8(img, src_w, src_h, factor, &mut self.upscale)?;
        let mut corners = Self::detect_view_inner(
            &self.cfg,
            &mut self.pyramid,
            &mut self.chess_buffers,
            &mut self.radon_buffers,
            #[cfg(feature = "ml-refiner")]
            &mut self.ml_state,
            #[cfg(feature = "ml-refiner")]
            &self.ml_params,
            upscaled,
        );
        upscale::rescale_descriptors_to_input(&mut corners, factor);
        Ok(corners)
    }

    /// Detect chessboard corners from an [`image::GrayImage`].
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::Upscale`] if the upscale configuration
    /// becomes invalid.
    #[cfg(feature = "image")]
    pub fn detect(&mut self, img: &image::GrayImage) -> Result<Vec<CornerDescriptor>, ChessError> {
        self.detect_u8(img.as_raw(), img.width(), img.height())
    }

    /// Detect chessboard corners inside a rectangular region `roi` of a
    /// raw 8-bit grayscale image.
    ///
    /// Returns the corners whose detected peak lies inside `roi`, each
    /// refined and described exactly as [`Detector::detect_u8`] would: the
    /// same configured refiner and the same orientation/descriptor stage.
    /// Coordinates are in the full input-image pixel frame. Both detection
    /// strategies (ChESS and Radon) are supported.
    ///
    /// # Single-scale only
    ///
    /// ROI detection is single-scale local detection by definition: the
    /// [`multiscale`](DetectorConfig::multiscale) and
    /// [`upscale`](DetectorConfig::upscale) sections of the active config
    /// do **not** apply on this path. For whole-image detection — including
    /// the coarse-to-fine pyramid and the pre-pipeline upscaling stage —
    /// use [`Detector::detect_u8`] / [`Detector::detect`].
    ///
    /// # ROI clamping
    ///
    /// A `roi` extending past the image is clamped to the image bounds
    /// (matching the multiscale ROI-carving behaviour). A fully
    /// out-of-range or degenerate post-clamp `roi` returns `Ok(vec![])`,
    /// not an error.
    ///
    /// # Parity with `detect_u8`
    ///
    /// Parity is defined against a **single-scale, non-upscaled**
    /// [`Detector::detect_u8`] run — [`multiscale`](DetectorConfig::multiscale)
    /// set to `SingleScale` and [`upscale`](DetectorConfig::upscale)
    /// disabled, as in the [`DetectorConfig::chess`] and
    /// [`DetectorConfig::radon`] presets. When a pyramid or upscale
    /// section is active, `detect_u8` detects on resampled images that
    /// this path never builds, so its output is not comparable
    /// corner-for-corner.
    ///
    /// Against that run, every **ChESS** corner whose peak lies more
    /// than `ring_radius + nms_radius` pixels inside the clamped `roi`
    /// (on every side) is returned here with a **bit-identical
    /// `response`** and a position that agrees to within floating-point
    /// rounding (well under `1e-3` px). The integer peak detection is
    /// exact; only the sub-pixel refinement rounds differently, because
    /// it runs in the ROI-local coordinate frame — the same numerical
    /// relationship a coarse-to-fine multiscale run has to a full-frame
    /// single-scale run. The **Radon** strategy recomputes its response
    /// over the carved patch (prefix-sum accumulation restarts at the
    /// patch origin), so its parity is approximate: interior corners
    /// reappear, but response values and subpixel positions carry a
    /// small floating-point drift rather than being bit-exact. Corners
    /// nearer the ROI edge — or nearer than the detector support to the
    /// image border — may differ or be absent under either strategy.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::DimensionMismatch`] if `img.len() != width *
    /// height`. With the `ml-refiner` feature, returns
    /// `ChessError::RoiRefinerUnsupported` when the configuration selects
    /// the ML refiner: the ML refiner runs a whole-frame model pipeline
    /// this path does not carry, and the refiner selection is never
    /// silently downgraded — use [`Detector::detect_u8`] for ML refinement.
    pub fn detect_u8_roi(
        &mut self,
        img: &[u8],
        width: u32,
        height: u32,
        roi: Roi,
    ) -> Result<Vec<CornerDescriptor>, ChessError> {
        let src_w = width as usize;
        let src_h = height as usize;
        let expected = src_w * src_h;
        if img.len() != expected {
            return Err(ChessError::DimensionMismatch {
                expected,
                actual: img.len(),
            });
        }

        // The ML refiner has no ROI-path plumbing; refuse rather than
        // silently downgrading to the core default refiner.
        #[cfg(feature = "ml-refiner")]
        if Self::is_ml_refiner(&self.cfg) {
            return Err(ChessError::RoiRefinerUnsupported);
        }

        let view =
            ImageView::from_u8_slice(src_w, src_h, img).expect("dimensions were checked above");
        Ok(multiscale::detect_roi_with_buffers(
            view,
            roi,
            &self.cfg,
            &mut self.chess_buffers,
            &mut self.radon_buffers,
        ))
    }

    /// Detect chessboard corners inside a rectangular region `roi` of an
    /// [`image::GrayImage`].
    ///
    /// See [`Detector::detect_u8_roi`] for the ROI contract, the
    /// single-scale caveat, the clamping behaviour, and the parity
    /// guarantee.
    ///
    /// # Errors
    ///
    /// Same as [`Detector::detect_u8_roi`].
    #[cfg(feature = "image")]
    pub fn detect_roi(
        &mut self,
        img: &image::GrayImage,
        roi: Roi,
    ) -> Result<Vec<CornerDescriptor>, ChessError> {
        self.detect_u8_roi(img.as_raw(), img.width(), img.height(), roi)
    }

    /// Borrow a detector-bound diagnostics accessor.
    ///
    /// The returned [`DetectorDiagnostics`](crate::diagnostics::DetectorDiagnostics)
    /// exposes intermediate
    /// detector outputs — the dense ChESS response map and the Radon
    /// heatmap — sourced from this detector's already-configured
    /// [`DetectorConfig`], so a caller holding a configured `Detector`
    /// need not re-supply a config to obtain diagnostic data.
    ///
    /// This is the detector-bound half of the diagnostics channel; the
    /// free functions in [`crate::diagnostics`] serve stateless
    /// callers. Both share the same **opt-in, looser-stability**
    /// contract: diagnostic outputs are advisory and may change as the
    /// detector internals evolve, independently of the
    /// [`Detector::detect`] result contract.
    pub fn diagnostics(&self) -> crate::diagnostics::DetectorDiagnostics<'_> {
        crate::diagnostics::DetectorDiagnostics::new(self)
    }

    fn detect_view_inner(
        cfg: &DetectorConfig,
        pyramid: &mut PyramidBuffers,
        chess_buffers: &mut ChessBuffers,
        radon_buffers: &mut RadonBuffers,
        #[cfg(feature = "ml-refiner")] ml_state: &mut Option<ml_refiner::MlRefinerState>,
        #[cfg(feature = "ml-refiner")] ml_params: &ml_refiner::MlRefinerParams,
        view: ImageView<'_>,
    ) -> Vec<CornerDescriptor> {
        #[cfg(feature = "ml-refiner")]
        if Self::is_ml_refiner(cfg) {
            if ml_state.is_none() {
                let fallback = chess_corners_core::RefinerKind::CenterOfMass(
                    chess_corners_core::CenterOfMassConfig::default(),
                );
                *ml_state = Some(ml_refiner::MlRefinerState::new(ml_params, &fallback));
            }
            let state = ml_state.as_mut().expect("ml_state initialised above");
            return multiscale::detect_with_ml(
                view,
                cfg,
                pyramid,
                chess_buffers,
                radon_buffers,
                ml_params,
                state,
            );
        }

        multiscale::detect_with_buffers(view, cfg, pyramid, chess_buffers, radon_buffers)
    }

    /// Whether the active config selects the ML refiner. Only true on
    /// the ChESS path, since the Radon strategy carries a separate
    /// refiner enum that has no ML variant.
    #[cfg(feature = "ml-refiner")]
    #[inline]
    fn is_ml_refiner(cfg: &DetectorConfig) -> bool {
        matches!(
            &cfg.strategy,
            crate::DetectionStrategy::Chess(c) if matches!(c.refiner, crate::ChessRefiner::Ml)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UpscaleConfig;
    use chess_corners_testutil::{aa_chessboard, gaussian_blur};

    fn synthetic_board(size: usize) -> Vec<u8> {
        aa_chessboard(size, 12, (0.0, 0.0), 20, 220)
    }

    fn roi(x0: usize, y0: usize, x1: usize, y1: usize) -> Roi {
        Roi::new(x0, y0, x1, y1).expect("valid roi")
    }

    fn bit_eq(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn detect_u8_reports_dimension_mismatch() {
        let mut det = Detector::with_default();
        let img = vec![0u8; 10];
        let err = det.detect_u8(&img, 8, 8).unwrap_err();
        match err {
            ChessError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 10);
            }
            other => panic!("expected ChessError::DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn set_config_valid_swap_changes_detection_outcome() {
        let size = 96usize;
        let img = synthetic_board(size);

        let mut det = Detector::new(DetectorConfig::chess().with_threshold(30.0)).unwrap();
        let low_threshold = det.detect_u8(&img, size as u32, size as u32).unwrap();
        assert!(
            !low_threshold.is_empty(),
            "expected corners at the default threshold"
        );

        det.set_config(DetectorConfig::chess().with_threshold(5000.0))
            .expect("valid upscale config");
        let high_threshold = det.detect_u8(&img, size as u32, size as u32).unwrap();
        assert!(
            high_threshold.len() < low_threshold.len(),
            "raising the response threshold far above real corner strengths \
             must suppress detections: low={} high={}",
            low_threshold.len(),
            high_threshold.len()
        );
    }

    #[test]
    fn set_config_rejects_invalid_upscale_and_leaves_detector_usable() {
        let mut det = Detector::new(DetectorConfig::chess()).unwrap();
        let original_upscale = det.config().upscale;

        let bad = DetectorConfig::chess().with_upscale(UpscaleConfig::fixed(5));
        let err = det.set_config(bad).unwrap_err();
        assert!(matches!(err, ChessError::Upscale(_)));

        // The failed swap must not have mutated the active config.
        assert_eq!(det.config().upscale, original_upscale);

        // The detector must still be usable after the rejected swap.
        let img = synthetic_board(64);
        let corners = det.detect_u8(&img, 64, 64).unwrap();
        assert!(!corners.is_empty());
    }

    #[test]
    fn detect_u8_roi_reports_dimension_mismatch() {
        let mut det = Detector::with_default();
        let img = vec![0u8; 10];
        let err = det.detect_u8_roi(&img, 8, 8, roi(0, 0, 4, 4)).unwrap_err();
        assert!(matches!(err, ChessError::DimensionMismatch { .. }));
    }

    /// Full-frame ChESS corners lying more than `ring_radius + nms_radius`
    /// (= 7 for the default ring) inside an interior ROI must reappear from
    /// `detect_u8_roi` with a bit-identical response and a position that
    /// matches to floating-point rounding. The inflated patch reproduces
    /// the full-frame response bit-for-bit, so the integer peak and its
    /// response strength are exact; the sub-pixel centroid runs in the
    /// ROI-local frame and rounds by ~1 ULP relative to the global-frame
    /// centroid (the same effect the coarse-to-fine path exhibits).
    #[test]
    fn chess_roi_matches_full_frame_on_interior() {
        let size = 100usize;
        let img = aa_chessboard(size, 12, (0.35, 0.7), 20, 220);
        let mut det = Detector::new(DetectorConfig::chess()).unwrap();

        let full = det.detect_u8(&img, size as u32, size as u32).unwrap();
        let sub = det
            .detect_u8_roi(&img, size as u32, size as u32, roi(20, 20, 80, 80))
            .unwrap();

        // ring_radius(5) + nms_radius(2) for the default ChESS config.
        let border = 7.0f32;
        let (rx0, ry0, rx1, ry1) = (20.0f32, 20.0, 80.0, 80.0);
        let mut checked = 0usize;
        for fc in &full {
            let strictly_inside = fc.x > rx0 + border
                && fc.x < rx1 - border
                && fc.y > ry0 + border
                && fc.y < ry1 - border;
            if !strictly_inside {
                continue;
            }
            // Response (raw peak strength) is frame-independent → exact.
            // Position agrees to floating-point rounding of the ROI shift.
            let found = sub.iter().any(|sc| {
                bit_eq(sc.response, fc.response)
                    && (sc.x - fc.x).abs() < 1e-3
                    && (sc.y - fc.y).abs() < 1e-3
            });
            assert!(
                found,
                "interior corner ({:.4},{:.4}) r={:.4} missing or outside parity tolerance in ROI result",
                fc.x, fc.y, fc.response
            );
            checked += 1;
        }
        assert!(
            checked >= 4,
            "expected several strictly-interior corners to compare, got {checked}"
        );
    }

    /// A ROI whose top-left is far from the origin must return corners in
    /// the global (base-image) frame. If the coordinates were patch-local
    /// they would not coincide with any full-frame corner position.
    #[test]
    fn chess_roi_coordinates_are_global() {
        let size = 100usize;
        let img = aa_chessboard(size, 12, (0.35, 0.7), 20, 220);
        let mut det = Detector::new(DetectorConfig::chess()).unwrap();

        let full = det.detect_u8(&img, size as u32, size as u32).unwrap();
        let sub = det
            .detect_u8_roi(&img, size as u32, size as u32, roi(48, 48, 92, 92))
            .unwrap();

        assert!(!sub.is_empty(), "expected corners in the interior ROI");
        for sc in &sub {
            let matched = full
                .iter()
                .any(|fc| (fc.x - sc.x).abs() < 0.01 && (fc.y - sc.y).abs() < 0.01);
            assert!(
                matched,
                "ROI corner ({:.3},{:.3}) does not match any full-frame (global) corner",
                sc.x, sc.y
            );
        }
    }

    #[test]
    fn roi_edges_and_out_of_bounds() {
        let size = 100usize;
        let img = aa_chessboard(size, 12, (0.35, 0.7), 20, 220);
        let mut det = Detector::new(DetectorConfig::chess()).unwrap();
        let w = size as u32;

        // ROI flush against the top-left image border: no panic, still
        // finds interior corners away from the border.
        let border_roi = det.detect_u8_roi(&img, w, w, roi(0, 0, 45, 45)).unwrap();
        assert!(
            !border_roi.is_empty(),
            "expected corners in a border-touching ROI"
        );

        // ROI smaller than the detector support: Ok(empty), no panic.
        let tiny = det.detect_u8_roi(&img, w, w, roi(50, 50, 53, 53)).unwrap();
        assert!(tiny.is_empty(), "sub-support ROI must yield no corners");

        // Partially out-of-range ROI clamps to the image bounds (must not
        // panic or error).
        let _clamped = det
            .detect_u8_roi(&img, w, w, roi(80, 80, 300, 300))
            .unwrap();

        // Fully out-of-range ROI is degenerate after clamping → empty.
        let gone = det
            .detect_u8_roi(&img, w, w, roi(150, 150, 300, 300))
            .unwrap();
        assert!(
            gone.is_empty(),
            "fully out-of-range ROI must yield no corners"
        );
    }

    #[test]
    fn roi_respects_orientation_config() {
        let size = 100usize;
        let img = aa_chessboard(size, 12, (0.35, 0.7), 20, 220);
        let w = size as u32;
        let region = roi(24, 24, 84, 84);

        let mut with_axes = Detector::new(DetectorConfig::chess()).unwrap();
        let described = with_axes.detect_u8_roi(&img, w, w, region).unwrap();
        assert!(!described.is_empty());
        assert!(
            described.iter().all(|c| c.axes.is_some()),
            "default config must attach orientation axes"
        );

        let mut no_axes = Detector::new(DetectorConfig::chess().without_orientation()).unwrap();
        let bare = no_axes.detect_u8_roi(&img, w, w, region).unwrap();
        assert!(!bare.is_empty());
        assert!(
            bare.iter().all(|c| c.axes.is_none()),
            "without_orientation() must skip the orientation fit"
        );
    }

    #[test]
    fn roi_detection_is_deterministic() {
        let size = 100usize;
        let img = aa_chessboard(size, 12, (0.35, 0.7), 20, 220);
        let w = size as u32;
        let region = roi(20, 20, 80, 80);
        let mut det = Detector::new(DetectorConfig::chess()).unwrap();

        let a = det.detect_u8_roi(&img, w, w, region).unwrap();
        let b = det.detect_u8_roi(&img, w, w, region).unwrap();
        assert_eq!(a.len(), b.len(), "corner count must be stable");
        for (ca, cb) in a.iter().zip(&b) {
            assert!(
                bit_eq(ca.x, cb.x) && bit_eq(ca.y, cb.y) && bit_eq(ca.response, cb.response),
                "ROI detection must be bit-deterministic"
            );
        }
    }

    /// Radon smoke: ROI detection is non-empty and every ROI corner lies
    /// close to a full-frame corner from the same region. Radon computes
    /// its response on the copied ROI, so patch and full-frame subpixel
    /// positions agree only within a small tolerance (not bit-exact).
    #[test]
    fn radon_roi_smoke_matches_full_frame() {
        let size = 129usize;
        let mut img = aa_chessboard(size, 12, (0.4, 0.6), 30, 220);
        gaussian_blur(&mut img, size, 1.2);
        let w = size as u32;
        let mut det = Detector::new(DetectorConfig::radon()).unwrap();

        let full = det.detect_u8(&img, w, w).unwrap();
        let sub = det.detect_u8_roi(&img, w, w, roi(36, 36, 96, 96)).unwrap();

        assert!(!sub.is_empty(), "Radon ROI detection must find corners");
        for sc in &sub {
            let near = full
                .iter()
                .any(|fc| (fc.x - sc.x).abs() < 2.0 && (fc.y - sc.y).abs() < 2.0);
            assert!(
                near,
                "Radon ROI corner ({:.2},{:.2}) has no nearby full-frame corner",
                sc.x, sc.y
            );
        }
    }

    /// A Radon corner whose integer peak sits exactly ON the ROI
    /// boundary (subpixel position just inside it) must still be
    /// returned. The carved patch has to reserve the extra pixel
    /// `detect_peaks_from_radon` keeps for its 3-point peak fit —
    /// with a margin of only `ray + nms` the boundary peak lands in
    /// the excluded patch border and silently disappears (regression
    /// for the `RadonDetector::roi_border` off-by-one).
    #[test]
    fn radon_roi_keeps_boundary_peaks() {
        let size = 129usize;
        let mut img = aa_chessboard(size, 12, (0.4, 0.6), 30, 220);
        gaussian_blur(&mut img, size, 1.2);
        let w = size as u32;

        for upsample in [1u32, 2] {
            let cfg = DetectorConfig::radon().with_radon(|r| r.image_upsample = upsample);
            let mut det = Detector::new(cfg).unwrap();
            let full = det.detect_u8(&img, w, w).unwrap();
            assert!(!full.is_empty(), "full-frame Radon must find corners");

            // Put an ROI boundary exactly on each corner's integer
            // peak, on the side that keeps the subpixel position
            // inside the half-open ROI: the min edge when the position
            // sits right of / below the peak, the max edge otherwise.
            // Corners whose position is nearly centred on the peak are
            // skipped — their half-open membership is ambiguous.
            let mut checked = 0usize;
            for fc in &full {
                let px = fc.x.round();
                let py = fc.y.round();
                let (fx, fy) = (fc.x - px, fc.y - py);
                if fx.abs() < 0.05 || fy.abs() < 0.05 {
                    continue;
                }
                let (px, py) = (px as usize, py as usize);
                let (x0, x1) = if fx > 0.0 {
                    (px, (px + 50).min(size))
                } else {
                    ((px + 1).saturating_sub(50), px + 1)
                };
                let (y0, y1) = if fy > 0.0 {
                    (py, (py + 50).min(size))
                } else {
                    ((py + 1).saturating_sub(50), py + 1)
                };
                let region = roi(x0, y0, x1, y1);
                let sub = det.detect_u8_roi(&img, w, w, region).unwrap();
                let found = sub
                    .iter()
                    .any(|sc| (sc.x - fc.x).abs() < 0.5 && (sc.y - fc.y).abs() < 0.5);
                assert!(
                    found,
                    "upsample={upsample}: boundary corner ({:.3},{:.3}) dropped from ROI ({x0},{y0})-({x1},{y1})",
                    fc.x, fc.y
                );
                checked += 1;
            }
            assert!(
                checked >= 3,
                "upsample={upsample}: expected several boundary-peak corners to check, got {checked}"
            );
        }
    }

    #[cfg(feature = "ml-refiner")]
    #[test]
    fn roi_rejects_ml_refiner_without_silent_downgrade() {
        let size = 64usize;
        let img = synthetic_board(size);
        let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = crate::ChessRefiner::Ml);
        let mut det = Detector::new(cfg).unwrap();
        let err = det
            .detect_u8_roi(&img, size as u32, size as u32, roi(10, 10, 50, 50))
            .unwrap_err();
        assert!(matches!(err, ChessError::RoiRefinerUnsupported));
    }
}
