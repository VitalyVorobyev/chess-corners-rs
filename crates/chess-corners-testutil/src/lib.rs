//! Shared synthetic-image fixtures for the chess-corners-rs workspace.
//!
//! This is a `publish = false`, dependency-free dev-utility crate: the
//! single home for the deterministic synthetic boards, blur, and noise
//! helpers that the workspace's tests, benches, and examples would
//! otherwise copy-paste across crates (SOLID-01 / SOLID-03). Pull it in as
//! a path `dev-dependency`; it never enters a published crate's dependency
//! tree.
//!
//! Every generator is fully deterministic — the only randomness is the
//! explicitly seeded noise helpers — so detector outputs and the
//! committed perf baseline stay reproducible across machines and feature
//! builds.
//!
//! ## Board generators
//!
//! - [`synth_chessboard`] — hard 40/215 step board (the response/NMS
//!   timing fixtures).
//! - [`synth_chessboard_soft`] / [`synth_chessboard_warped`] — anti-aliased
//!   and affine-skewed variants used by the orientation/descriptor fits.
//! - [`aa_chessboard`] — anti-aliased board with caller-chosen cell size,
//!   subpixel `offset`, and `dark`/`bright` levels; the refiner-accuracy
//!   and Radon-vs-ChESS fixtures.
//! - [`optical_chessboard`] — a board on any [`Lattice`] (rotated, sheared)
//!   formed as a camera forms it: Gaussian lens blur, then exact
//!   integration over each pixel. Corners sit exactly on the lattice; the
//!   fixture for refiner accuracy and covariance calibration.
//! - [`lattice_area_chessboard`] / [`area_chessboard`] — the same lattice
//!   with an ideally sharp lens (exact pixel area of the hard pattern).
//!   Blurring its output with [`gaussian_blur`] is *not* a camera: it keeps
//!   the aliasing of the sharp edges.
//!
//! ## Degradations
//!
//! - [`gaussian_blur`] — separable Gaussian, 3σ radius, edge-clamped.
//! - [`add_gaussian_noise`] — seeded additive Gaussian noise.
//! - [`add_signal_dependent_noise`] — seeded Gaussian noise of variance
//!   `a + b·I` (read plus shot noise).

#![forbid(unsafe_code)]

mod boards;
mod lattice;
mod noise;

pub use boards::{
    aa_chessboard, expected_corner_count, rotated_chessboard, synth_chessboard,
    synth_chessboard_soft, synth_chessboard_warped, ORIENT_CORNER, ORIENT_DIM,
};
pub use lattice::{area_chessboard, lattice_area_chessboard, optical_chessboard, Lattice};
pub use noise::{add_gaussian_noise, add_signal_dependent_noise, gaussian_blur};
