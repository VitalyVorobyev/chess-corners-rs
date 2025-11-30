//! Simple image pyramid utilities used by the multiscale corner finder.
//!
//! The API is allocation-friendly: construct a [`PyramidBuffers`] once, then
//! reuse it to build pyramids for successive frames without re-allocating
//! intermediate levels. When the `simd` feature is enabled, the 2× box
//! downsample uses portable SIMD for higher throughput.

use image::GrayImage;
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Reusable backing storage for pyramid construction.
///
/// Call [`PyramidBuffers::prepare_for_image`] once per frame to ensure the
/// internal buffers match the planned pyramid shape, then pass the same
/// instance to [`build_pyramid`] to fill those buffers.
pub struct PyramidBuffers {
    levels: Vec<GrayImage>,
}

impl Default for PyramidBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl PyramidBuffers {
    /// Create an empty buffer set.
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Create a buffer set with capacity reserved for `num_levels`.
    pub fn with_capacity(num_levels: u8) -> Self {
        Self {
            levels: Vec::with_capacity(num_levels.saturating_sub(1) as usize),
        }
    }

    /// Ensure buffers exist for all downsampled levels that will be produced
    /// for `base` and `params`. Returns the number of downsampled levels
    /// (excluding the base) that were sized.
    pub fn prepare_for_image(&mut self, base: &GrayImage, params: &PyramidParams) -> usize {
        if params.num_levels == 0
            || base.width() < params.min_size
            || base.height() < params.min_size
        {
            return 0;
        }

        let mut w = base.width();
        let mut h = base.height();
        let mut prepared = 0usize;

        for _ in 1..params.num_levels {
            if w < 2 || h < 2 {
                break;
            }

            w /= 2;
            h /= 2;

            if w < params.min_size || h < params.min_size {
                break;
            }

            self.ensure_level_shape(prepared, w, h);
            prepared += 1;
        }

        prepared
    }

    fn ensure_level_shape(&mut self, idx: usize, w: u32, h: u32) {
        if idx >= self.levels.len() {
            self.levels.resize_with(idx + 1, || GrayImage::new(w, h));
        }

        let level = &mut self.levels[idx];
        if level.width() != w || level.height() != h {
            *level = GrayImage::new(w, h);
        }
    }
}

/// A single pyramid level. The `scale` is relative to the base image.
pub struct PyramidLevel<'a> {
    pub img: &'a GrayImage,
    pub scale: f32, // relative to base (e.g. 1.0, 0.5, 0.25, ...)
}

/// A top-down pyramid where `levels[0]` is the base (full resolution).
pub struct Pyramid<'a> {
    pub levels: Vec<PyramidLevel<'a>>, // levels[0] is base
}

/// Parameters controlling pyramid generation.
pub struct PyramidParams {
    /// Maximum number of levels (including the base).
    pub num_levels: u8,
    /// Stop building when either dimension falls below this value.
    pub min_size: u32,
}

impl Default for PyramidParams {
    fn default() -> Self {
        Self {
            num_levels: 3,
            min_size: 128,
        }
    }
}

/// Build a top-down image pyramid using fixed 2× downsampling.
///
/// The base image is always included as level 0. Each subsequent level is a
/// 2× downsampled copy (box filter) written into `buffers`. Construction stops
/// when:
/// - either dimension would fall below `min_size`, or
/// - `num_levels` is reached.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "debug",
        skip(base, params, buffers),
        fields(levels = params.num_levels, min_size = params.min_size)
    )
)]
pub fn build_pyramid<'a>(
    base: &'a GrayImage,
    params: &PyramidParams,
    buffers: &'a mut PyramidBuffers,
) -> Pyramid<'a> {
    if params.num_levels == 0 || base.width() < params.min_size || base.height() < params.min_size {
        return Pyramid { levels: Vec::new() };
    }

    #[derive(Clone, Copy)]
    enum LevelSource {
        Base,
        Buffer(usize),
    }

    let mut sources: Vec<(LevelSource, f32)> = Vec::with_capacity(params.num_levels as usize);
    sources.push((LevelSource::Base, 1.0));

    let mut current_src = LevelSource::Base;
    let mut current_w = base.width();
    let mut current_h = base.height();
    let mut scale = 1.0f32;

    for level_idx in 1..params.num_levels {
        let w2 = current_w / 2;
        let h2 = current_h / 2;

        if w2 == 0 || h2 == 0 || w2 < params.min_size || h2 < params.min_size {
            break;
        }

        let buf_idx = (level_idx - 1) as usize;
        buffers.ensure_level_shape(buf_idx, w2, h2);

        let (src_img, dst): (&GrayImage, &mut GrayImage) = match current_src {
            LevelSource::Base => (base, &mut buffers.levels[buf_idx]),
            LevelSource::Buffer(src_idx) => {
                debug_assert!(src_idx < buf_idx);
                let (head, tail) = buffers.levels.split_at_mut(buf_idx);
                (&head[src_idx], &mut tail[0])
            }
        };

        downsample_2x_box(src_img, dst);

        scale *= 0.5;
        current_src = LevelSource::Buffer(buf_idx);
        current_w = w2;
        current_h = h2;
        sources.push((current_src, scale));
    }

    let mut levels = Vec::with_capacity(sources.len());
    for (source, lvl_scale) in sources {
        let img = match source {
            LevelSource::Base => base,
            LevelSource::Buffer(idx) => &buffers.levels[idx],
        };
        levels.push(PyramidLevel {
            img,
            scale: lvl_scale,
        });
    }

    Pyramid { levels }
}

/// Fast 2× downsample with a 2×2 box filter into a pre-allocated destination.
///
/// Uses a SIMD specialization when the `simd` feature is enabled.
#[inline]
fn downsample_2x_box(src: &GrayImage, dst: &mut GrayImage) {
    #[cfg(all(feature = "rayon", feature = "simd"))]
    return downsample_2x_box_parallel_simd(src, dst);

    #[cfg(all(feature = "rayon", not(feature = "simd")))]
    return downsample_2x_box_parallel_scalar(src, dst);

    #[cfg(all(not(feature = "rayon"), feature = "simd"))]
    return downsample_2x_box_simd(src, dst);

    #[cfg(all(not(feature = "rayon"), not(feature = "simd")))]
    return downsample_2x_box_scalar(src, dst);
}

#[inline]
#[cfg(all(not(feature = "simd"), not(feature = "rayon")))]
fn downsample_2x_box_scalar(src: &GrayImage, dst: &mut GrayImage) {
    debug_assert_eq!(src.width() / 2, dst.width());
    debug_assert_eq!(src.height() / 2, dst.height());

    let src_w = src.width() as usize;
    let dst_w = dst.width() as usize;
    let dst_h = dst.height() as usize;

    let src_pixels = src.as_raw();
    let dst_pixels = dst.as_mut();

    for y in 0..dst_h {
        let row0 = (y * 2) * src_w;
        let row1 = row0 + src_w;

        downsample_row_scalar(
            &src_pixels[row0..row0 + src_w],
            &src_pixels[row1..row1 + src_w],
            &mut dst_pixels[y * dst_w..(y + 1) * dst_w],
        );
    }
}

#[cfg(all(not(feature = "rayon"), feature = "simd"))]
fn downsample_2x_box_simd(src: &GrayImage, dst: &mut GrayImage) {
    debug_assert_eq!(src.width() / 2, dst.width());
    debug_assert_eq!(src.height() / 2, dst.height());

    let (src_w, src_h) = src.dimensions();
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;

    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();

    let src_stride = src_w as usize;
    let dst_stride = dst_w as usize;

    for y_out in 0..dst_h as usize {
        let y0 = 2 * y_out;
        let y1 = y0 + 1;

        let row0 = &src_buf[y0 * src_stride..(y0 + 1) * src_stride];
        let row1 = &src_buf[y1 * src_stride..(y1 + 1) * src_stride];

        let dst_row = &mut dst_buf[y_out * dst_stride..(y_out + 1) * dst_stride];

        downsample_row_simd(row0, row1, dst_row);
    }
}

#[cfg(all(feature = "rayon", not(feature = "simd")))]
fn downsample_2x_box_parallel_scalar(src: &GrayImage, dst: &mut GrayImage) {
    use rayon::prelude::*;

    debug_assert_eq!(src.width() / 2, dst.width());
    debug_assert_eq!(src.height() / 2, dst.height());

    let src_w = src.width();
    let dst_w = src_w / 2;

    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();

    let src_stride = src_w as usize;
    let dst_stride = dst_w as usize;

    dst_buf
        .par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(y_out, dst_row)| {
            let y0 = 2 * y_out;
            let y1 = y0 + 1;

            let row0 = &src_buf[y0 * src_stride..(y0 + 1) * src_stride];
            let row1 = &src_buf[y1 * src_stride..(y1 + 1) * src_stride];

            downsample_row_scalar(row0, row1, dst_row);
        });
}

#[cfg(all(feature = "rayon", feature = "simd"))]
fn downsample_2x_box_parallel_simd(src: &GrayImage, dst: &mut GrayImage) {
    use rayon::prelude::*;

    debug_assert_eq!(src.width() / 2, dst.width());
    debug_assert_eq!(src.height() / 2, dst.height());

    let src_w = src.width();
    let dst_w = src_w / 2;

    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();

    let src_stride = src_w as usize;
    let dst_stride = dst_w as usize;

    dst_buf
        .par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(y_out, dst_row)| {
            let y0 = 2 * y_out;
            let y1 = y0 + 1;

            let row0 = &src_buf[y0 * src_stride..(y0 + 1) * src_stride];
            let row1 = &src_buf[y1 * src_stride..(y1 + 1) * src_stride];

            downsample_row_simd(row0, row1, dst_row);
        });
}

#[inline]
#[cfg_attr(feature = "simd", allow(dead_code))]
fn downsample_row_scalar(row0: &[u8], row1: &[u8], dst_row: &mut [u8]) {
    let dst_w = dst_row.len();

    for (x, item) in dst_row.iter_mut().enumerate().take(dst_w) {
        let sx = x * 2;
        let p00 = row0[sx] as u16;
        let p01 = row0[sx + 1] as u16;
        let p10 = row1[sx] as u16;
        let p11 = row1[sx + 1] as u16;
        let sum = p00 + p01 + p10 + p11;
        *item = ((sum + 2) >> 2) as u8;
    }
}

#[cfg(feature = "simd")]
fn downsample_row_simd(row0: &[u8], row1: &[u8], dst_row: &mut [u8]) {
    use std::simd::num::SimdUint;
    use std::simd::{u16x16, u8x16};

    const LANES: usize = 16;
    let mut x_out = 0usize;

    while x_out + LANES <= dst_row.len() {
        let mut p00 = [0u8; LANES];
        let mut p01 = [0u8; LANES];
        let mut p10 = [0u8; LANES];
        let mut p11 = [0u8; LANES];

        for lane in 0..LANES {
            let x = x_out + lane;
            let sx = 2 * x;
            p00[lane] = row0[sx];
            p01[lane] = row0[sx + 1];
            p10[lane] = row1[sx];
            p11[lane] = row1[sx + 1];
        }

        let a0 = u8x16::from_array(p00);
        let a1 = u8x16::from_array(p01);
        let b0 = u8x16::from_array(p10);
        let b1 = u8x16::from_array(p11);

        let sum: u16x16 = a0.cast::<u16>() + a1.cast::<u16>() + b0.cast::<u16>() + b1.cast::<u16>();

        let avg = (sum + u16x16::splat(2)) >> 2;
        let out: u8x16 = avg.cast();

        out.copy_to_slice(&mut dst_row[x_out..x_out + LANES]);
        x_out += LANES;
    }

    for x in x_out..dst_row.len() {
        let sx = 2 * x;
        let p00 = row0[sx] as u16;
        let p01 = row0[sx + 1] as u16;
        let p10 = row1[sx] as u16;
        let p11 = row1[sx + 1] as u16;
        let sum = p00 + p01 + p10 + p11;
        dst_row[x] = ((sum + 2) >> 2) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn blank(w: u32, h: u32) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([0u8]))
    }

    #[test]
    fn build_pyramid_respects_min_size_and_levels() {
        let base = blank(64, 64);
        let params = PyramidParams {
            num_levels: 4,
            min_size: 8,
        };
        let mut buffers = PyramidBuffers::with_capacity(params.num_levels);
        buffers.prepare_for_image(&base, &params);
        let pyramid = build_pyramid(&base, &params, &mut buffers);
        let dims: Vec<_> = pyramid
            .levels
            .iter()
            .map(|lvl| (lvl.img.width(), lvl.img.height(), lvl.scale))
            .collect();
        assert_eq!(
            dims,
            vec![(64, 64, 1.0), (32, 32, 0.5), (16, 16, 0.25), (8, 8, 0.125)]
        );
    }

    #[test]
    fn build_pyramid_stops_when_reaching_min_size() {
        let base = blank(20, 20);
        let params = PyramidParams {
            num_levels: 5,
            min_size: 6,
        };
        let mut buffers = PyramidBuffers::new();
        buffers.prepare_for_image(&base, &params);
        let pyramid = build_pyramid(&base, &params, &mut buffers);
        // 20 -> 10 -> 5 (stop at min_size)
        let dims: Vec<_> = pyramid
            .levels
            .iter()
            .map(|lvl| (lvl.img.width(), lvl.img.height()))
            .collect();
        assert_eq!(dims, vec![(20, 20), (10, 10)]);
    }

    #[test]
    fn prepare_reuses_buffers_when_shape_matches() {
        let base = blank(512, 512);
        let params = PyramidParams::default();
        let mut buffers = PyramidBuffers::new();
        let first = buffers.prepare_for_image(&base, &params);
        let ptrs: Vec<*const u8> = buffers.levels.iter().map(|l| l.as_ptr()).collect();
        let second = buffers.prepare_for_image(&base, &params);
        let ptrs_after: Vec<*const u8> = buffers.levels.iter().map(|l| l.as_ptr()).collect();
        assert_eq!(first, 2); // 512 -> 256 -> 128 with default params
        assert_eq!(second, first);
        assert_eq!(
            ptrs, ptrs_after,
            "buffers should be reused when shapes match"
        );
    }

    /// Slow, test-only reference implementation of 2× box downsampling.
    fn reference_downsample_2x_box(src: &GrayImage) -> GrayImage {
        let (w, h) = src.dimensions();
        assert!(w % 2 == 0 && h % 2 == 0);
        let dst_w = w / 2;
        let dst_h = h / 2;

        let mut dst = GrayImage::new(dst_w, dst_h);
        let src_w = w as usize;
        let dst_w_usize = dst_w as usize;

        let src_pixels = src.as_raw();
        let dst_pixels = dst.as_mut();

        for y in 0..dst_h as usize {
            let row0 = (y * 2) * src_w;
            let row1 = row0 + src_w;
            for x in 0..dst_w_usize {
                let sx = 2 * x;
                let p00 = src_pixels[row0 + sx] as u16;
                let p01 = src_pixels[row0 + sx + 1] as u16;
                let p10 = src_pixels[row1 + sx] as u16;
                let p11 = src_pixels[row1 + sx + 1] as u16;
                let sum = p00 + p01 + p10 + p11;
                dst_pixels[y * dst_w_usize + x] = ((sum + 2) >> 2) as u8;
            }
        }

        dst
    }

    fn pattern_image(w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // Simple but non-trivial pattern so averaging changes values.
                let v = ((x as u16 * 3 + y as u16 * 5) % 251) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }
        img
    }

    #[test]
    fn downsample_2x_box_matches_reference_on_aligned_size() {
        let src = pattern_image(64, 48);
        let expected = reference_downsample_2x_box(&src);

        let mut dst = GrayImage::new(32, 24);
        super::downsample_2x_box(&src, &mut dst);

        assert_eq!(
            expected.as_raw(),
            dst.as_raw(),
            "downsample_2x_box should match reference implementation"
        );
    }

    #[test]
    fn downsample_2x_box_matches_reference_with_simd_tail() {
        // Width chosen so SIMD handles the first chunk and the tail is scalar.
        let src = pattern_image(38, 34); // dst: 19 x 17
        let expected = reference_downsample_2x_box(&src);

        let mut dst = GrayImage::new(19, 17);
        super::downsample_2x_box(&src, &mut dst);

        assert_eq!(
            expected.as_raw(),
            dst.as_raw(),
            "downsample_2x_box should match reference on non-multiple-of-lanes width"
        );
    }
}
