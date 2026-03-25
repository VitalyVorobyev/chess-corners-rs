#![cfg_attr(all(feature = "simd", feature = "par_pyramid"), feature(portable_simd))]
//! Minimal image pyramid using 2x box-filter downsampling.
//!
//! This crate provides a simple, efficient image pyramid for u8 grayscale
//! images. Each level is produced by a 2x2 box-filter downsample (averaging
//! four pixels into one). It is designed for real-time pipelines where you
//! need coarse-to-fine processing without pulling in a full image processing
//! library.
//!
//! # Key features
//!
//! - **Reusable buffers**: construct a [`PyramidBuffers`] once and reuse it
//!   across frames to avoid repeated allocations.
//! - **Optional parallelism**: enable the `rayon` and `par_pyramid` features
//!   for parallel row processing.
//! - **Optional SIMD**: enable the `simd` and `par_pyramid` features for
//!   portable SIMD acceleration (requires nightly Rust).
//!
//! # Example
//!
//! ```
//! use box_image_pyramid::{ImageView, PyramidParams, PyramidBuffers, build_pyramid};
//!
//! let pixels = vec![128u8; 256 * 256];
//! let base = ImageView::new(256, 256, &pixels).unwrap();
//!
//! let params = PyramidParams { num_levels: 3, min_size: 32 };
//! let mut buffers = PyramidBuffers::new();
//! let pyramid = build_pyramid(base, &params, &mut buffers);
//!
//! assert_eq!(pyramid.levels.len(), 3);
//! assert_eq!(pyramid.levels[0].img.width, 256);
//! assert_eq!(pyramid.levels[1].img.width, 128);
//! assert_eq!(pyramid.levels[2].img.width, 64);
//! ```

mod imageview;
mod pyramid;

pub use imageview::{ImageBuffer, ImageView};
pub use pyramid::{build_pyramid, Pyramid, PyramidBuffers, PyramidLevel, PyramidParams};
