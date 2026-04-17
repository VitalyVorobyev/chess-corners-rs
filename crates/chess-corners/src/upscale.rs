//! Optional pre-pipeline image upscaling.
//!
//! Low-resolution inputs — typical of small ChArUco crops — leave
//! target corners inside the ChESS ring margin (5 px for the canonical
//! detector), where the response is zeroed out and corners are lost.
//! This module adds a first-class integer upscaling stage that runs
//! ahead of the pyramid. Output corner coordinates are always rescaled
//! back to input-image pixel coordinates by the facade, so callers do
//! not need to be aware of the stage.
//!
//! Supported factors: 2, 3, 4 (bilinear only in v1).

use chess_corners_core::{CornerDescriptor, ImageView};
use serde::{Deserialize, Serialize};

/// Upscaling mode, encoded into JSON as `"disabled"` or `"fixed"`.
///
/// Kept as a separate enum for forward compatibility — future modes
/// (auto-fit, non-integer factors) can be added without rewriting
/// callers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpscaleMode {
    /// Do not upscale (default).
    #[default]
    Disabled,
    /// Upscale by a fixed integer factor (allowed: 2, 3, 4).
    Fixed,
}

/// Upscaling configuration exposed through [`crate::ChessConfig`].
///
/// JSON shape: `{ "mode": "disabled" }` or `{ "mode": "fixed", "factor": 2 }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpscaleConfig {
    /// Selected upscale mode. Default: [`UpscaleMode::Disabled`].
    pub mode: UpscaleMode,
    /// Integer factor used when `mode == Fixed`. Ignored otherwise.
    /// Must be 2, 3, or 4.
    pub factor: u32,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            mode: UpscaleMode::Disabled,
            factor: 2,
        }
    }
}

impl UpscaleConfig {
    /// Construct a disabled configuration (no upscaling).
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Construct a fixed-factor configuration. Does not validate.
    pub fn fixed(factor: u32) -> Self {
        Self {
            mode: UpscaleMode::Fixed,
            factor,
        }
    }

    /// Return the effective integer factor, or 1 when disabled.
    #[inline]
    pub fn effective_factor(&self) -> u32 {
        match self.mode {
            UpscaleMode::Disabled => 1,
            UpscaleMode::Fixed => self.factor,
        }
    }

    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<(), UpscaleError> {
        if matches!(self.mode, UpscaleMode::Fixed) && !matches!(self.factor, 2..=4) {
            return Err(UpscaleError::InvalidFactor(self.factor));
        }
        Ok(())
    }
}

/// Errors returned by upscaling setup or execution.
#[derive(Debug, PartialEq, Eq)]
pub enum UpscaleError {
    /// The requested factor is not in the supported set {2, 3, 4}.
    InvalidFactor(u32),
    /// Upscaled dimensions would overflow `usize`.
    DimensionOverflow { src: (usize, usize), factor: u32 },
}

impl core::fmt::Display for UpscaleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidFactor(k) => {
                write!(f, "upscale factor {k} not supported (expected 2, 3, or 4)")
            }
            Self::DimensionOverflow { src, factor } => write!(
                f,
                "upscaled dimensions overflow: {}x{} * {} exceeds usize",
                src.0, src.1, factor
            ),
        }
    }
}

impl std::error::Error for UpscaleError {}

/// Reusable scratch buffer for the upscaling stage.
///
/// Reuses its allocation across frames. The buffer grows on demand
/// when dimensions change; it never shrinks, matching the
/// `box-image-pyramid` buffer strategy.
#[derive(Debug, Default, Clone)]
pub struct UpscaleBuffers {
    buf: Vec<u8>,
    w: usize,
    h: usize,
}

impl UpscaleBuffers {
    /// Create an empty buffer. Allocation happens lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        let needed = w.saturating_mul(h);
        if self.buf.len() < needed {
            self.buf.resize(needed, 0);
        }
    }

    /// Current width of the upscaled buffer (0 before first use).
    pub fn width(&self) -> usize {
        self.w
    }

    /// Current height of the upscaled buffer (0 before first use).
    pub fn height(&self) -> usize {
        self.h
    }
}

/// Bilinear upscaling by an integer factor into the provided buffer.
///
/// Uses the half-pixel-center convention (consistent with OpenCV's
/// `INTER_LINEAR` and `box-image-pyramid`'s downsampler).
pub fn upscale_bilinear_u8<'a>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    factor: u32,
    buffers: &'a mut UpscaleBuffers,
) -> Result<ImageView<'a>, UpscaleError> {
    if !matches!(factor, 2..=4) {
        return Err(UpscaleError::InvalidFactor(factor));
    }
    let k = factor as usize;
    let dst_w = src_w
        .checked_mul(k)
        .ok_or(UpscaleError::DimensionOverflow {
            src: (src_w, src_h),
            factor,
        })?;
    let dst_h = src_h
        .checked_mul(k)
        .ok_or(UpscaleError::DimensionOverflow {
            src: (src_w, src_h),
            factor,
        })?;

    assert!(
        src.len() == src_w * src_h,
        "src dimensions / buffer mismatch"
    );
    buffers.ensure(dst_w, dst_h);

    if src_w == 0 || src_h == 0 {
        return Ok(ImageView::from_u8_slice(dst_w, dst_h, &buffers.buf[..dst_w * dst_h]).unwrap());
    }

    let inv_k = 1.0f32 / factor as f32;
    let max_x = src_w as i32 - 1;
    let max_y = src_h as i32 - 1;

    // Precompute per-column (x0, x1, wx). The pattern is periodic with
    // period k, so we only need k entries; but for clarity we compute
    // one per output column.
    let mut xw: Vec<(usize, usize, f32)> = Vec::with_capacity(dst_w);
    for x_out in 0..dst_w {
        let xf = (x_out as f32 + 0.5) * inv_k - 0.5;
        let x0 = xf.floor() as i32;
        let wx = xf - x0 as f32;
        let x0c = x0.clamp(0, max_x) as usize;
        let x1c = (x0 + 1).clamp(0, max_x) as usize;
        xw.push((x0c, x1c, wx));
    }

    for y_out in 0..dst_h {
        let yf = (y_out as f32 + 0.5) * inv_k - 0.5;
        let y0 = yf.floor() as i32;
        let wy = yf - y0 as f32;
        let y0c = y0.clamp(0, max_y) as usize;
        let y1c = (y0 + 1).clamp(0, max_y) as usize;
        let row0 = y0c * src_w;
        let row1 = y1c * src_w;
        let dst_row = y_out * dst_w;

        for (x_out, &(x0, x1, wx)) in xw.iter().enumerate().take(dst_w) {
            let i00 = src[row0 + x0] as f32;
            let i10 = src[row0 + x1] as f32;
            let i01 = src[row1 + x0] as f32;
            let i11 = src[row1 + x1] as f32;
            let top = i00 + (i10 - i00) * wx;
            let bot = i01 + (i11 - i01) * wx;
            let v = top + (bot - top) * wy;
            // Round-half-away-from-zero then clamp to u8.
            let rounded = v + 0.5;
            buffers.buf[dst_row + x_out] = rounded.clamp(0.0, 255.0) as u8;
        }
    }

    let slice = &buffers.buf[..dst_w * dst_h];
    Ok(ImageView::from_u8_slice(dst_w, dst_h, slice).expect("dims match"))
}

/// Rescale corner positions from an upscaled image back to the
/// original input-image pixel frame.
///
/// Uses the inverse of the forward half-pixel-center mapping from
/// [`upscale_bilinear_u8`]:
///
/// ```text
/// forward : x_out = (x_src + 0.5) · k − 0.5
/// inverse : x_src = (x_out + 0.5) / k − 0.5
///         = x_out / k − (k − 1) / (2k)
/// ```
///
/// A naive `x /= k` biases returned coordinates by `(k − 1) / (2k)`
/// pixels (+0.25 px at k = 2). Axis angles and sigmas are
/// scale-invariant and are left untouched.
pub fn rescale_descriptors_to_input(descriptors: &mut [CornerDescriptor], factor: u32) {
    if factor <= 1 {
        return;
    }
    let inv = 1.0f32 / factor as f32;
    let shift = 0.5 * (1.0 - inv);
    for d in descriptors.iter_mut() {
        d.x = d.x * inv - shift;
        d.y = d.y * inv - shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_disabled() {
        let cfg = UpscaleConfig::default();
        assert_eq!(cfg.effective_factor(), 1);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_invalid_factors() {
        for bad in [0u32, 1, 5, 8] {
            let cfg = UpscaleConfig::fixed(bad);
            assert_eq!(cfg.validate(), Err(UpscaleError::InvalidFactor(bad)));
        }
    }

    #[test]
    fn config_accepts_valid_factors() {
        for good in [2u32, 3, 4] {
            let cfg = UpscaleConfig::fixed(good);
            assert!(cfg.validate().is_ok());
            assert_eq!(cfg.effective_factor(), good);
        }
    }

    #[test]
    fn upscale_factor_2_uniform_image_is_uniform() {
        let src = vec![42u8; 8 * 6];
        let mut buffers = UpscaleBuffers::new();
        let view = upscale_bilinear_u8(&src, 8, 6, 2, &mut buffers).unwrap();
        assert_eq!(view.width, 16);
        assert_eq!(view.height, 12);
        assert!(view.data.iter().all(|&v| v == 42));
    }

    #[test]
    fn upscale_factor_2_of_1x1_fills_buffer() {
        let src = [77u8];
        let mut buffers = UpscaleBuffers::new();
        let view = upscale_bilinear_u8(&src, 1, 1, 2, &mut buffers).unwrap();
        assert_eq!(view.width, 2);
        assert_eq!(view.height, 2);
        assert!(view.data.iter().all(|&v| v == 77));
    }

    #[test]
    fn upscale_preserves_linear_gradient_factor_2() {
        // Horizontal ramp: src[i] = i * 10 for i in 0..8.
        let src: Vec<u8> = (0..8).map(|i| i * 10).collect();
        let src = {
            let mut row = Vec::with_capacity(8 * 3);
            for _ in 0..3 {
                row.extend_from_slice(&src);
            }
            row
        };
        let mut buffers = UpscaleBuffers::new();
        let view = upscale_bilinear_u8(&src, 8, 3, 2, &mut buffers).unwrap();
        // The upscaled image should stay monotonic along each row.
        for r in 0..view.height {
            let row = &view.data[r * view.width..(r + 1) * view.width];
            for w in row.windows(2) {
                assert!(w[1] >= w[0].saturating_sub(1), "non-monotonic row: {row:?}");
            }
        }
    }

    #[test]
    fn upscale_factor_3_doubles_dimensions_correctly() {
        let src = vec![128u8; 5 * 4];
        let mut buffers = UpscaleBuffers::new();
        let view = upscale_bilinear_u8(&src, 5, 4, 3, &mut buffers).unwrap();
        assert_eq!(view.width, 15);
        assert_eq!(view.height, 12);
        assert_eq!(view.data.len(), 180);
    }

    #[test]
    fn buffers_are_reused_across_calls() {
        let src1 = vec![10u8; 4 * 4];
        let src2 = vec![200u8; 4 * 4];
        let mut buffers = UpscaleBuffers::new();
        let _ = upscale_bilinear_u8(&src1, 4, 4, 2, &mut buffers).unwrap();
        let cap1 = buffers.buf.capacity();
        let _ = upscale_bilinear_u8(&src2, 4, 4, 2, &mut buffers).unwrap();
        assert_eq!(buffers.buf.capacity(), cap1, "buffer should be reused");
    }

    #[test]
    fn rejects_invalid_factor_at_runtime() {
        let src = vec![0u8; 4];
        let mut buffers = UpscaleBuffers::new();
        let err = upscale_bilinear_u8(&src, 2, 2, 5, &mut buffers).unwrap_err();
        assert_eq!(err, UpscaleError::InvalidFactor(5));
    }

    #[test]
    fn rescale_inverts_half_pixel_upscale() {
        use chess_corners_core::{AxisEstimate, CornerDescriptor};

        // Forward mapping in `upscale_bilinear_u8`:
        //   x_out = (x_src + 0.5) * k - 0.5
        // For a corner at source position (7.25, 3.0) and factor k = 2,
        // the upscaled detection should land at (14.5, 6.5). Running
        // that through `rescale_descriptors_to_input` must return
        // exactly the original source position, not x_out / k.
        fn desc(x: f32, y: f32) -> CornerDescriptor {
            CornerDescriptor {
                x,
                y,
                response: 1.0,
                contrast: 0.0,
                fit_rms: 0.0,
                axes: [
                    AxisEstimate {
                        angle: 0.0,
                        sigma: 0.0,
                    },
                    AxisEstimate {
                        angle: 0.0,
                        sigma: 0.0,
                    },
                ],
            }
        }

        for &(k, x_src, y_src) in &[
            (2u32, 7.25f32, 3.0f32),
            (3u32, 4.0f32, 8.5f32),
            (4u32, 0.5f32, 12.25f32),
        ] {
            let kf = k as f32;
            let x_out = (x_src + 0.5) * kf - 0.5;
            let y_out = (y_src + 0.5) * kf - 0.5;

            let mut d = [desc(x_out, y_out)];
            rescale_descriptors_to_input(&mut d, k);
            assert!(
                (d[0].x - x_src).abs() < 1e-5,
                "k={k}: x {} != expected {x_src}",
                d[0].x
            );
            assert!(
                (d[0].y - y_src).abs() < 1e-5,
                "k={k}: y {} != expected {y_src}",
                d[0].y
            );
        }
    }

    #[test]
    fn rescale_is_noop_for_factor_1() {
        use chess_corners_core::{AxisEstimate, CornerDescriptor};
        let mut d = [CornerDescriptor {
            x: 2.5,
            y: 3.75,
            response: 1.0,
            contrast: 0.0,
            fit_rms: 0.0,
            axes: [
                AxisEstimate {
                    angle: 0.0,
                    sigma: 0.0,
                },
                AxisEstimate {
                    angle: 0.0,
                    sigma: 0.0,
                },
            ],
        }];
        rescale_descriptors_to_input(&mut d, 1);
        assert_eq!(d[0].x, 2.5);
        assert_eq!(d[0].y, 3.75);
    }
}
