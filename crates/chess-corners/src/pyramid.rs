//! Simple image pyramid utilities used by the multiscale corner finder.
//!
//! The API is allocation-friendly: construct a [`PyramidBuffers`] once, then
//! reuse it to build pyramids for successive frames without re-allocating
//! intermediate levels. When both the `par_pyramid` and `simd` features are
//! enabled, the 2× box downsample uses portable SIMD for higher throughput.

use chess_corners_core::ImageView;
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Owned grayscale image buffer (u8).
#[derive(Clone, Debug)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width.saturating_mul(height)],
        }
    }

    pub fn as_view(&self) -> ImageView<'_> {
        ImageView::from_u8_slice(self.width, self.height, &self.data).unwrap()
    }
}

/// Reusable backing storage for pyramid construction.
///
/// Typically you construct a [`PyramidBuffers`] once (for example with
/// [`PyramidBuffers::with_capacity`]) and reuse it across frames by
/// passing a mutable reference into higher-level helpers such as
/// [`crate::find_chess_corners_buff`]. The internal level buffers are
/// resized on demand to match the requested pyramid shape.
pub struct PyramidBuffers {
    levels: Vec<ImageBuffer>,
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

    fn ensure_level_shape(&mut self, idx: usize, w: usize, h: usize) {
        if idx >= self.levels.len() {
            self.levels.resize_with(idx + 1, || ImageBuffer::new(w, h));
        }

        let level = &mut self.levels[idx];
        if level.width != w || level.height != h {
            *level = ImageBuffer::new(w, h);
        }
    }
}

/// A single pyramid level. The `scale` is relative to the base image.
pub struct PyramidLevel<'a> {
    pub img: ImageView<'a>,
    pub scale: f32, // relative to base (e.g. 1.0, 0.5, 0.25, ...)
}

/// A top-down pyramid where `levels[0]` is the base (full resolution).
pub struct Pyramid<'a> {
    pub levels: Vec<PyramidLevel<'a>>, // levels[0] is base
}

/// Parameters controlling pyramid generation.
#[derive(Clone, Debug)]
pub struct PyramidParams {
    /// Maximum number of levels (including the base).
    pub num_levels: u8,
    /// Stop building when either dimension falls below this value.
    pub min_size: usize,
}

impl Default for PyramidParams {
    fn default() -> Self {
        Self {
            num_levels: 1,
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
        level = "info",
        skip(base, params, buffers),
        fields(levels = params.num_levels, min_size = params.min_size)
    )
)]
pub fn build_pyramid<'a>(
    base: ImageView<'a>,
    params: &PyramidParams,
    buffers: &'a mut PyramidBuffers,
) -> Pyramid<'a> {
    if params.num_levels == 0 || base.width < params.min_size || base.height < params.min_size {
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
    let mut current_w = base.width;
    let mut current_h = base.height;
    let mut scale = 1.0f32;

    for level_idx in 1..params.num_levels {
        let w2 = current_w / 2;
        let h2 = current_h / 2;

        if w2 == 0 || h2 == 0 || w2 < params.min_size || h2 < params.min_size {
            break;
        }

        let buf_idx = (level_idx - 1) as usize;
        buffers.ensure_level_shape(buf_idx, w2, h2);

        let (src_img, dst): (ImageView<'_>, &mut ImageBuffer) = match current_src {
            LevelSource::Base => (base, &mut buffers.levels[buf_idx]),
            LevelSource::Buffer(src_idx) => {
                debug_assert!(src_idx < buf_idx);
                let (head, tail) = buffers.levels.split_at_mut(buf_idx);
                (head[src_idx].as_view(), &mut tail[0])
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
            LevelSource::Buffer(idx) => buffers.levels[idx].as_view(),
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
/// Uses SIMD and/or `rayon` specializations when the `par_pyramid`
/// feature is enabled alongside the relevant flags.
#[inline]
fn downsample_2x_box(src: ImageView<'_>, dst: &mut ImageBuffer) {
    #[cfg(all(feature = "par_pyramid", feature = "rayon", feature = "simd"))]
    return downsample_2x_box_parallel_simd(src, dst);

    #[cfg(all(feature = "par_pyramid", feature = "rayon", not(feature = "simd")))]
    return downsample_2x_box_parallel_scalar(src, dst);

    #[cfg(all(feature = "par_pyramid", not(feature = "rayon"), feature = "simd"))]
    return downsample_2x_box_simd(src, dst);

    #[cfg(all(feature = "par_pyramid", not(feature = "rayon"), not(feature = "simd")))]
    return downsample_2x_box_scalar(src, dst);

    #[cfg(not(feature = "par_pyramid"))]
    return downsample_2x_box_scalar(src, dst);
}

#[inline]
#[cfg_attr(
    all(feature = "par_pyramid", any(feature = "rayon", feature = "simd")),
    allow(dead_code)
)]
fn downsample_2x_box_scalar(src: ImageView<'_>, dst: &mut ImageBuffer) {
    debug_assert_eq!(src.width / 2, dst.width);
    debug_assert_eq!(src.height / 2, dst.height);

    let src_w = src.width;
    let dst_w = dst.width;
    let dst_h = dst.height;

    for y in 0..dst_h {
        let row0 = (y * 2) * src_w;
        let row1 = row0 + src_w;

        downsample_row_scalar(
            &src.data[row0..row0 + src_w],
            &src.data[row1..row1 + src_w],
            &mut dst.data[y * dst_w..(y + 1) * dst_w],
        );
    }
}

#[cfg(all(feature = "par_pyramid", not(feature = "rayon"), feature = "simd"))]
fn downsample_2x_box_simd(src: ImageView<'_>, dst: &mut ImageBuffer) {
    debug_assert_eq!(src.width / 2, dst.width);
    debug_assert_eq!(src.height / 2, dst.height);

    let src_w = src.width;
    let dst_w = dst.width;
    let dst_h = dst.height;

    for y_out in 0..dst_h {
        let y0 = 2 * y_out;
        let y1 = y0 + 1;

        let row0 = &src.data[y0 * src_w..(y0 + 1) * src_w];
        let row1 = &src.data[y1 * src_w..(y1 + 1) * src_w];

        let dst_row = &mut dst.data[y_out * dst_w..(y_out + 1) * dst_w];

        downsample_row_simd(row0, row1, dst_row);
    }
}

#[cfg(all(feature = "par_pyramid", feature = "rayon", not(feature = "simd")))]
fn downsample_2x_box_parallel_scalar(src: ImageView<'_>, dst: &mut ImageBuffer) {
    use rayon::prelude::*;

    debug_assert_eq!(src.width / 2, dst.width);
    debug_assert_eq!(src.height / 2, dst.height);

    let src_w = src.width;
    let dst_w = dst.width;

    dst.data
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y_out, dst_row)| {
            let y0 = 2 * y_out;
            let y1 = y0 + 1;

            let row0 = &src.data[y0 * src_w..(y0 + 1) * src_w];
            let row1 = &src.data[y1 * src_w..(y1 + 1) * src_w];

            downsample_row_scalar(row0, row1, dst_row);
        });
}

#[cfg(all(feature = "par_pyramid", feature = "rayon", feature = "simd"))]
fn downsample_2x_box_parallel_simd(src: ImageView<'_>, dst: &mut ImageBuffer) {
    use rayon::prelude::*;

    debug_assert_eq!(src.width / 2, dst.width);
    debug_assert_eq!(src.height / 2, dst.height);

    let src_w = src.width;
    let dst_w = dst.width;

    dst.data
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y_out, dst_row)| {
            let y0 = 2 * y_out;
            let y1 = y0 + 1;

            let row0 = &src.data[y0 * src_w..(y0 + 1) * src_w];
            let row1 = &src.data[y1 * src_w..(y1 + 1) * src_w];

            downsample_row_simd(row0, row1, dst_row);
        });
}

#[inline]
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

#[cfg(all(feature = "par_pyramid", feature = "simd"))]
fn downsample_row_simd(row0: &[u8], row1: &[u8], dst_row: &mut [u8]) {
    use std::ops::Shr;
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

        let p00v = u8x16::from_array(p00).cast::<u16>();
        let p01v = u8x16::from_array(p01).cast::<u16>();
        let p10v = u8x16::from_array(p10).cast::<u16>();
        let p11v = u8x16::from_array(p11).cast::<u16>();

        let sum = p00v + p01v + p10v + p11v;
        let avg = (sum + u16x16::splat(2)).shr(2);
        let out = avg.cast::<u8>();

        dst_row[x_out..x_out + LANES].copy_from_slice(out.as_array());
        x_out += LANES;
    }

    // Tail
    if x_out < dst_row.len() {
        downsample_row_scalar(row0, row1, &mut dst_row[x_out..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_downsample(src: &ImageBuffer) -> ImageBuffer {
        let mut dst = ImageBuffer::new(src.width / 2, src.height / 2);
        downsample_2x_box_scalar(src.as_view(), &mut dst);
        dst
    }

    #[cfg(feature = "image")]
    fn gray_to_buffer(img: &image::GrayImage) -> ImageBuffer {
        ImageBuffer {
            width: img.width() as usize,
            height: img.height() as usize,
            data: img.as_raw().clone(),
        }
    }

    #[cfg(feature = "image")]
    fn make_checker(w: u32, h: u32, a: u8, b: u8) -> image::GrayImage {
        image::GrayImage::from_fn(w, h, |x, y| {
            if (x + y) % 2 == 0 {
                image::Luma([a])
            } else {
                image::Luma([b])
            }
        })
    }

    #[test]
    fn downsample_matches_reference() {
        let mut src = ImageBuffer::new(8, 8);
        for (i, p) in src.data.iter_mut().enumerate() {
            *p = (i % 251) as u8;
        }
        let mut dst = ImageBuffer::new(4, 4);
        downsample_2x_box(src.as_view(), &mut dst);
        let expected = reference_downsample(&src);
        assert_eq!(dst.data, expected.data);
    }

    #[cfg(feature = "image")]
    #[test]
    fn downsample_matches_reference_on_checker() {
        let img = make_checker(17, 15, 0, 255);
        let src = gray_to_buffer(&img);
        let mut dst = ImageBuffer::new(src.width / 2, src.height / 2);
        downsample_2x_box(src.as_view(), &mut dst);
        let expected = reference_downsample(&src);
        assert_eq!(dst.data, expected.data);
    }
}
