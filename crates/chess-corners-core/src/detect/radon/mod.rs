//! Whole-image Duda-Frese Radon detector.
//!
//! An alternative to the ChESS detector in [`super::chess`] for frames
//! where the 16-sample ring does not produce enough seeds in the target
//! data. The detector computes a dense 4-angle localized Radon response
//! `R(x, y) = (max_α S_α − min_α S_α)²` using summed-area tables for
//! `O(1)`-per-pixel ray sums, then applies a peak-fit pipeline:
//! threshold / NMS / box-blur / 3-point Gaussian fit in x and y.
//!
//! Submodules:
//!
//! - [`primitives`] — angular basis, `fit_peak_frac`, `box_blur_inplace`.
//! - [`response`] — `radon_response_u8` and the SAT / response-map
//!   buffers and types.
//! - [`detect`] — `detect_peaks_from_radon` (peak detection on the
//!   response map, including the response-map 3-point Gaussian
//!   subpixel fit).

pub mod detect;
pub(crate) mod primitives;
pub mod response;

pub use detect::detect_peaks_from_radon;
pub use response::{radon_response_u8, RadonBuffers, RadonDetectorParams, RadonResponseView};

#[cfg(test)]
pub(super) mod test_fixtures {
    /// Anti-aliased synthetic chessboard renderer shared between the
    /// `response.rs` and `detect.rs` test modules. Re-exported from the
    /// workspace `chess-corners-testutil` fixtures so the anti-aliased board
    /// has a single implementation (SOLID-03); the local alias keeps the
    /// `test_fixtures::synthetic_chessboard_aa` path stable for callers.
    pub(crate) use chess_corners_testutil::aa_chessboard as synthetic_chessboard_aa;
}
