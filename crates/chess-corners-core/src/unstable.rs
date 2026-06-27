//! Implementation primitives exposed for benchmarks, experiments, and
//! advanced use.
//!
//! The items re-exported here are the detector's internal scaffolding:
//! ring offset tables, the scalar reference response path, sub-stage
//! detection entry points, and the ChESS parameter / refiner-selector
//! translation types (`ChessParams`, `RefinerKind`) that the
//! `chess-corners` facade maps its configuration onto. They are made
//! reachable so that benchmarks, accuracy experiments, and advanced
//! callers can drive the pipeline at a finer granularity than the
//! curated crate-root surface allows.
//!
//! # Stability
//!
//! Nothing in this module carries a semver guarantee. Any item here
//! may change signature, change behavior, or disappear in any release,
//! including a patch release. Code that needs a stable contract should
//! use the curated crate-root exports
//! ([`chess_response_u8`](crate::chess_response_u8),
//! [`find_corners_u8`](crate::find_corners_u8),
//! [`describe_corners`](crate::describe_corners), and the refiner and
//! orientation types) instead.

/// Low-level ChESS detection parameters consumed by the response and
/// detection stages. The `chess-corners` facade translates its detector
/// configuration into this type internally.
pub use crate::params::ChessParams;

/// Serializable selector for the built-in subpixel refiner variants.
/// Construct a runtime [`Refiner`](crate::Refiner) from one via
/// [`Refiner::from_kind`](crate::Refiner::from_kind).
pub use crate::refine::RefinerKind;

/// Ring offset table for the canonical radius-5 sampling ring.
pub use crate::detect::chess::ring::RING5;

/// Ring offset table for the radius-10 sampling ring.
pub use crate::detect::chess::ring::RING10;

/// Select the static ring offset table for a sampling radius.
pub use crate::detect::chess::ring::ring_offsets;

/// Static 16-sample ring offset table addressed by radius.
pub use crate::detect::chess::ring::RingOffsets;

/// Scalar reference implementation of the ChESS response (no SIMD,
/// no parallelism); useful for golden-value comparisons.
pub use crate::detect::chess::response::chess_response_u8_scalar;

/// Compute the ChESS response over a sub-rectangle of an 8-bit image.
pub use crate::detect::chess::response::chess_response_u8_patch;

/// Detect raw corner candidates from a response map without applying a
/// subpixel refiner.
pub use crate::detect::chess::detect::detect_peaks_from_response;

/// Detect raw corner candidates from a response map with an explicit
/// refinement search radius.
pub use crate::detect::chess::detect::detect_peaks_from_response_with_refine_radius;

/// Detect corners directly from an 8-bit image with an explicit refiner.
pub use crate::detect::chess::detect::find_corners_u8_with_refiner;

/// Refine an existing set of corner candidates against an image.
pub use crate::detect::chess::detect::refine_corners_on_image;

/// Largest supported integer image-upsample factor for the Radon
/// detector.
pub use crate::detect::radon::response::MAX_IMAGE_UPSAMPLE;
