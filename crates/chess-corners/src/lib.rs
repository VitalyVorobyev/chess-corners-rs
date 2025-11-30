#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Ergonomic wrappers over `chess-corners-core` with optional `image` helpers.
//!
//! This crate is organized into a few focused modules:
//! - [`multiscale`] – unified single/multiscale corner finder.
//! - [`pyramid`] – minimal u8 buffers and downsampling for pyramids.
//! - optional `image` helpers for `image::GrayImage`.

pub mod multiscale;
pub mod pyramid;

// Re-export a focused subset of core types for convenience. Consumers that
// need lower-level primitives (rings, raw response functions, etc.) are
// encouraged to depend on `chess-corners-core` directly.
pub use chess_corners_core::{ChessParams, CornerDescriptor, ResponseMap};

// High-level helpers on `image::GrayImage`.
#[cfg(feature = "image")]
pub mod image;
#[cfg(feature = "image")]
pub use image::find_chess_corners_image;

// Multiscale/coarse-to-fine API.
pub use crate::multiscale::{
    find_chess_corners, CoarseToFineParams, CoarseToFineResult,
};

// Pyramid utilities are re-exported from the crate root for ergonomic access.
pub use crate::pyramid::{build_pyramid, Pyramid, PyramidBuffers, PyramidLevel, PyramidParams};
pub use crate::pyramid::{ImageBuffer, ImageView};
