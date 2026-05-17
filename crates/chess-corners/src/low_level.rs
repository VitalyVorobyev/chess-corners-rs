//! Deliberate escape hatch for callers hand-composing the detection
//! pipeline.
//!
//! The supported way to detect corners is [`Detector`](crate::Detector)
//! plus [`DetectorConfig`](crate::DetectorConfig): one `detect` call
//! drives the response, detection, refinement, and orientation stages
//! behind a stable contract. Most users never need this module.
//!
//! This module exists for the minority of callers who want to drive the
//! individual pipeline stages themselves — typically
//! response -> detect -> describe — so they can interleave custom logic
//! between stages, reuse intermediate buffers in an unusual way, or
//! build a detector variant the facade does not offer. It re-exports a
//! curated set of stage functions, parameter structs, and scratch
//! buffers sufficient to assemble such a pipeline without taking a
//! direct dependency on `chess-corners-core`.
//!
//! # Stability
//!
//! This channel carries a **weaker stability promise** than the facade
//! root ([`Detector`](crate::Detector),
//! [`DetectorConfig`](crate::DetectorConfig), and the result/config
//! types). The individual pipeline stages are implementation surface:
//! their signatures, parameter structs, and buffer layouts may change
//! as the detector internals evolve, even when the
//! [`Detector`](crate::Detector) contract does not. Use the facade root
//! unless you specifically need stage-level control.

/// Compute the dense ChESS response map for an 8-bit grayscale image —
/// the first stage of a hand-composed pipeline.
pub use chess_corners_core::chess_response_u8;

/// Sub-rectangle of an image, used to scope a response computation.
pub use chess_corners_core::Roi;

/// Detect raw corner candidates from a dense response map.
pub use chess_corners_core::detect_corners_from_response;

/// Detect raw corner candidates from a dense response map, applying a
/// subpixel refiner to each.
pub use chess_corners_core::detect_corners_from_response_with_refiner;

/// A raw corner candidate produced by the detection stage.
pub use chess_corners_core::Corner;

/// Lift raw [`Corner`] detections into oriented
/// [`CornerDescriptor`](chess_corners_core::CornerDescriptor) values.
pub use chess_corners_core::describe_corners;

/// The two-axis orientation fit produced by the orientation stage,
/// returned by [`fit_axes_at_point`](chess_corners_core::fit_axes_at_point).
pub use chess_corners_core::AxisFitResult;

/// Low-level ChESS detection parameters consumed by the response and
/// detection stages.
pub use chess_corners_core::ChessParams;

/// Low-level Radon detection parameters.
pub use chess_corners_core::RadonDetectorParams;

/// Borrowed view over an 8-bit grayscale image accepted by the stage
/// functions.
pub use chess_corners_core::ImageView;

/// Reusable scratch buffers for the Radon detection stage.
pub use chess_corners_core::RadonBuffers;

/// Pluggable subpixel-refiner trait implemented by the built-in
/// refiners.
pub use chess_corners_core::CornerRefiner;

/// Configured subpixel refiner passed to
/// [`detect_corners_from_response_with_refiner`].
pub use chess_corners_core::Refiner;

/// Selector for the built-in subpixel refiner variants.
pub use chess_corners_core::RefinerKind;

/// Outcome of a single subpixel-refinement attempt.
pub use chess_corners_core::RefineResult;

/// Status flag carried by a [`RefineResult`].
pub use chess_corners_core::RefineStatus;

/// Coarse-to-fine multiscale parameters for a hand-composed pyramid
/// pipeline.
pub use crate::multiscale::CoarseToFineParams;

/// Integer bilinear upscaling of an 8-bit grayscale buffer — an
/// optional pre-pipeline stage for low-resolution inputs.
pub use crate::upscale::upscale_bilinear_u8;

/// Rescale descriptor coordinates from a working resolution back to the
/// original input pixel frame.
pub use crate::upscale::rescale_descriptors_to_input;

/// Reusable scratch buffers for the bilinear upscaling stage.
pub use crate::upscale::UpscaleBuffers;

/// Owned 8-bit image buffer used by the pyramid stage.
pub use box_image_pyramid::ImageBuffer;

/// Reusable scratch buffers for the image pyramid.
pub use box_image_pyramid::PyramidBuffers;

/// Parameters controlling pyramid construction.
pub use box_image_pyramid::PyramidParams;
