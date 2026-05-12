//! High-level chessboard-corner detector with reusable scratch buffers.
//!
//! [`Detector`] is the primary entry point for the `chess-corners`
//! crate. It owns the [`DetectorConfig`] and the scratch buffers
//! (pyramid, upscale, …) required to run detection without
//! re-allocating across frames.
//!
//! ```no_run
//! use chess_corners::{DetectorConfig, Detector};
//! use image::io::Reader as ImageReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let img = ImageReader::open("board.png")?.decode()?.to_luma8();
//!
//! let mut detector = Detector::new(DetectorConfig::multiscale())?;
//! let corners = detector.detect(&img)?;
//! println!("found {} corners", corners.len());
//! # Ok(()) }
//! ```

use box_image_pyramid::PyramidBuffers;
use chess_corners_core::{ChessBuffers, CornerDescriptor, ImageView, RadonBuffers};

#[cfg(feature = "ml-refiner")]
use crate::ml_refiner;
use crate::multiscale;
use crate::upscale::{self, UpscaleBuffers};
use crate::{ChessError, DetectorConfig};

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

    /// Mutable access to the active config for ad-hoc tweaks. The
    /// caller is responsible for keeping the config valid; callers
    /// that change [`DetectorConfig::upscale`] should use
    /// [`Self::set_config`] instead so the upscale invariants are
    /// re-validated.
    pub fn config_mut(&mut self) -> &mut DetectorConfig {
        // Drop ML state on raw mutation; the next detect call rebuilds
        // it against whatever fallback refiner the new config implies.
        #[cfg(feature = "ml-refiner")]
        {
            self.ml_state = None;
        }
        &mut self.cfg
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

    /// Detect chessboard corners from a borrowed [`ImageView`].
    ///
    /// Lower-level than [`Self::detect_u8`] / [`Self::detect`]:
    /// upscaling is not applied here. Use this when you have a
    /// pre-upscaled image or you don't want the upscale pipeline at
    /// all.
    pub fn detect_view(&mut self, view: ImageView<'_>) -> Vec<CornerDescriptor> {
        Self::detect_view_inner(
            &self.cfg,
            &mut self.pyramid,
            &mut self.chess_buffers,
            &mut self.radon_buffers,
            #[cfg(feature = "ml-refiner")]
            &mut self.ml_state,
            #[cfg(feature = "ml-refiner")]
            &self.ml_params,
            view,
        )
    }

    /// Compute the dense Radon response map for `img` at working
    /// resolution. Convenience wrapper for visualisation; the detector
    /// itself uses the response internally and never returns it.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::DimensionMismatch`] if `img.len() !=
    /// width * height`. Returns [`ChessError::Upscale`] if the upscale
    /// configuration is invalid.
    pub fn radon_heatmap_u8(
        &mut self,
        img: &[u8],
        width: u32,
        height: u32,
    ) -> Result<chess_corners_core::ResponseMap, ChessError> {
        crate::radon::radon_heatmap_u8(img, width, height, &self.cfg)
    }

    /// Compute the dense Radon response map from an
    /// [`image::GrayImage`]. See [`Self::radon_heatmap_u8`].
    ///
    /// # Errors
    ///
    /// Inherits the error contract of [`Self::radon_heatmap_u8`].
    #[cfg(feature = "image")]
    pub fn radon_heatmap(
        &mut self,
        img: &image::GrayImage,
    ) -> Result<chess_corners_core::ResponseMap, ChessError> {
        self.radon_heatmap_u8(img.as_raw(), img.width(), img.height())
    }

    #[allow(clippy::too_many_arguments)]
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
