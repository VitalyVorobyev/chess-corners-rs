//! ChESS detector pipeline.
//!
//! Implements the corner detector described in Bennett & Lasenby,
//! *ChESS: A Fast and Accurate Chessboard Corner Detector* (CVIU 2014):
//!
//! 1. [`response::chess_response_u8`] computes a dense response map
//!    from a 16-sample ring around each pixel.
//! 2. [`detect::detect_corners_from_response`] thresholds the map,
//!    runs non-maximum suppression, and rejects isolated peaks.
//! 3. The detector candidate is then handed to a refiner from
//!    [`crate::refine`] and an orientation fit from
//!    [`crate::orientation`] to produce a final
//!    [`crate::CornerDescriptor`].

pub mod detect;
pub mod response;
pub mod ring;
