//! Implementation primitives exposed for benchmarks, experiments, and
//! advanced use.
//!
//! The items re-exported here are the detector's internal scaffolding:
//! ring offset tables, the Radon angular basis, the scalar reference
//! response path, sub-stage detection entry points, and similar
//! low-level building blocks. They are made reachable so that
//! benchmarks, accuracy experiments, and advanced callers can drive
//! the pipeline at a finer granularity than the curated crate-root
//! surface allows.
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

/// Number of Radon projection angles in the localized Radon basis.
pub use crate::detect::radon::primitives::ANGLES;

/// Cosine components of the Radon projection directions.
pub use crate::detect::radon::primitives::DIR_COS;

/// Sine components of the Radon projection directions.
pub use crate::detect::radon::primitives::DIR_SIN;

/// Three-point subpixel peak fit along one axis.
pub use crate::detect::radon::primitives::fit_peak_frac;

/// In-place separable box blur over a response buffer.
pub use crate::detect::radon::primitives::box_blur_inplace;

/// Summed-area-table element type used by the Radon response path.
pub use crate::detect::radon::response::SatElem;

/// Largest supported integer image-upsample factor for the Radon
/// detector.
pub use crate::detect::radon::response::MAX_IMAGE_UPSAMPLE;
