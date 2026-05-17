//! Whole-image Duda-Frese Radon detector.
//!
//! An alternative to the ChESS detector in [`super::chess`] for frames
//! where the 16-sample ring does not produce enough seeds in the target
//! data. The detector computes a dense 4-angle localized Radon response
//! `R(x, y) = (max_α S_α − min_α S_α)²` using summed-area tables for
//! `O(1)`-per-pixel ray sums, then applies the same peak-fit pipeline as
//! [`crate::RadonPeakRefiner`]: threshold / NMS /
//! box-blur / 3-point Gaussian fit in x and y.
//!
//! Submodules:
//!
//! - [`primitives`] — angular basis, `fit_peak_frac`, `box_blur_inplace`
//!   shared with [`crate::RadonPeakRefiner`].
//! - [`response`] — `radon_response_u8` and the SAT / response-map
//!   buffers and types.
//! - [`detect`] — `detect_peaks_from_radon` (peak detection on the
//!   response map, including the response-map 3-point Gaussian
//!   subpixel fit).

pub mod detect;
pub mod primitives;
pub mod response;

pub use detect::detect_peaks_from_radon;
pub use response::{radon_response_u8, RadonBuffers, RadonDetectorParams, RadonResponseView};

#[cfg(test)]
pub(super) mod test_fixtures {
    /// Anti-aliased synthetic chessboard renderer shared between
    /// `response.rs` and `detect.rs` test modules.
    pub(crate) fn synthetic_chessboard_aa(
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
}
