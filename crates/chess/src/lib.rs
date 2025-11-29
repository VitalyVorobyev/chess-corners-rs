#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Ergonomic wrappers over `chess-core` that accept `image::GrayImage` inputs.
//!
//! This crate is organized into a few focused modules:
//! - [`image`] – single-scale helpers on `image::GrayImage`.
//! - [`multiscale`] – pyramid-based multiscale and coarse-to-fine detection.
//! - [`pyramid`] – reusable buffers and downsampling for image pyramids.
//! - [`logger`] – a simple `log` implementation used by examples.

pub mod image;
pub mod logger;
pub mod multiscale;
pub mod pyramid;

// Re-export core types for convenience.
pub use chess_core::*;

// High-level helpers on `image::GrayImage`.
pub use crate::image::{chess_response_image, find_corners_image, find_corners_image_trace};

// Multiscale/coarse-to-fine API.
pub use crate::multiscale::{
    find_corners_coarse_to_fine_image, find_corners_coarse_to_fine_image_trace,
    find_corners_multiscale_image, CoarseToFineParams, CoarseToFineResult, MultiscaleCorner,
};

// Pyramid utilities are re-exported from the crate root for ergonomic access.
pub use crate::pyramid::{build_pyramid, Pyramid, PyramidBuffers, PyramidLevel, PyramidParams};
