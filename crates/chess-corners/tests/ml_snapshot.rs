//! Numerical-regression snapshot for the `ChessRefiner::Ml` pipeline.
//!
//! ## What this guards
//!
//! `ChessRefiner::Ml` has two distinct call paths through the facade: the
//! plain single-scale detector loop, and the coarse-to-fine multiscale
//! traversal (seeds detected on a coarser pyramid level, refined into the
//! base image — see `crates/chess-corners/src/multiscale.rs`). REVIEW.md
//! C11 flagged that the multiscale/ML combination had zero automated
//! coverage: its bit-stability during the C5 traversal refactor was proven
//! by a manual one-off harness across the real test images, not by a test
//! that runs on every `cargo test`. This file pins the **corner count** of
//! both `Ml` presets on a small deterministic fixture so a future
//! refactor that silently changes the set of corners produced (wrong seed
//! carried across levels, a merge-radius regression, an ROI off-by-one)
//! moves the count and fails CI.
//!
//! ## Why count-only, not positions
//!
//! `tests/snapshot_regression.rs` pins ChESS/CenterOfMass corner *counts*
//! rather than positions, because subpixel coordinates are f32-valued and
//! depend on operations (FMA contraction, SIMD width, rounding mode) that
//! differ observably between CPU microarchitectures — a bit-for-bit pin
//! captured on one CI runner fails on another even though the algorithm is
//! unchanged. The same risk applies here, one level down: `tract`'s CPU
//! matmul kernels are exactly this class of platform-sensitive FP code.
//! Per that file's own note, exact-position pinning "produced false
//! positives on every non-Apple-Silicon CI runner" the one time it was
//! tried, so this test mirrors the count-only convention rather than
//! attempting position tolerances. Per-corner positions are still dumped
//! to NDJSON on failure for local inspection (see `CornerSnap`), matching
//! `snapshot_regression.rs`'s `ACTUAL_SUBDIR` convention.
//!
//! ## Fixture sizing
//!
//! `DetectorConfig::chess_multiscale()`'s pyramid uses `min_size = 128`
//! (`MultiscaleConfig::pyramid_default`): a level is only added while both
//! dimensions of the *next* downsample stay `>= 128`. A 128 px base (the
//! canonical "8×8 checkerboard" fixture used elsewhere in this crate's
//! doctests) therefore collapses to a single-level pyramid — it would not
//! exercise the coarse-to-fine seed path this test exists to guard. `DIM`
//! is set to 256 so the pyramid builds a genuine second, coarser level
//! (256 -> 128, then 128 -> 64 stops below `min_size`).
//!
//! The fixture also can't reuse `chess_corners_testutil::synth_chessboard`
//! as-is: at `DIM = 256` its auto-computed cell size is 10 px, which halves
//! to 5 px at the coarse pyramid level — at or below the default ring
//! radius (5 px), so the coarse-level ChESS response degenerates and the
//! whole multiscale traversal returns **zero** corners (confirmed by a
//! throwaway probe before writing this fixture). [`aa_chessboard`] is used
//! instead with an explicit 32 px cell (16 px at the coarse level, well
//! above the ring radius) and a `(16.0, 16.0)` offset so every grid corner
//! sits strictly inside the frame, clear of the ML refiner's 21×21 default
//! patch margin.
//!
//! ## Manifest format
//!
//! `tests/snapshots/ml_manifest.txt` is a per-preset line:
//!
//! ```text
//! ml_single_scale count=36
//! ```
//!
//! Refresh after an intentional algorithm change:
//!
//! ```ignore
//! UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --features ml-refiner \
//!     --test ml_snapshot
//! ```
#![cfg(feature = "ml-refiner")]

use std::collections::BTreeMap;

use chess_corners::{ChessRefiner, CornerDescriptor, Detector, DetectorConfig};
use chess_corners_testutil::aa_chessboard;
use serde::Serialize;

#[path = "common/snapshot.rs"]
mod snapshot;
use snapshot::{check_or_update_manifest, ndjson, pinned_pool};

const MANIFEST_FILE: &str = "ml_manifest.txt";
const MANIFEST_HEADER: &str = "# Per-preset corner count for the ChessRefiner::Ml pipelines.
# Refresh with UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --features ml-refiner --test ml_snapshot
";

/// Base image side. See the module doc "Fixture sizing" note: must be
/// `>= 256` for `chess_multiscale()`'s default pyramid to build a second
/// level against `min_size = 128`.
const DIM: usize = 256;
/// Cell size and grid offset for the [`aa_chessboard`] fixture. See the
/// module doc "Fixture sizing" note for why these specific values.
const CELL: usize = 32;
const OFFSET: (f32, f32) = (16.0, 16.0);
const DARK: u8 = 30;
const BRIGHT: u8 = 230;

struct Preset {
    name: &'static str,
    build: fn() -> DetectorConfig,
}

fn ml_single_scale() -> DetectorConfig {
    DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::Ml)
}

fn ml_multiscale() -> DetectorConfig {
    DetectorConfig::chess_multiscale().with_chess(|c| c.refiner = ChessRefiner::Ml)
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "ml_single_scale",
        build: ml_single_scale,
    },
    Preset {
        name: "ml_multiscale",
        build: ml_multiscale,
    },
];

/// One corner record. Carried only for the failure-inspection dump
/// (the manifest itself stores only counts). Mirrors
/// `snapshot_regression.rs`'s `CornerSnap`.
#[derive(Serialize)]
struct CornerSnap {
    x: f64,
    y: f64,
    response: f64,
}

impl CornerSnap {
    fn from_descriptor(c: &CornerDescriptor) -> Self {
        Self {
            x: c.x as f64,
            y: c.y as f64,
            response: c.response as f64,
        }
    }
}

/// Render the detector output for one preset on the shared fixture.
/// Returns `(count, ndjson_for_local_debugging)`.
fn render_snapshot(img: &[u8], preset: &Preset) -> (usize, String) {
    let cfg = (preset.build)();
    let mut detector = Detector::new(cfg).expect("build detector");
    let mut corners = pinned_pool().install(|| {
        detector
            .detect_u8(img, DIM as u32, DIM as u32)
            .expect("detect")
    });
    (
        corners.len(),
        ndjson(&mut corners, CornerSnap::from_descriptor),
    )
}

/// Single test that exercises both `Ml` presets in one pass. Skipped
/// entirely (not failing) when the `ml-refiner` feature is off — see the
/// `#![cfg(feature = "ml-refiner")]` file-level gate above.
#[test]
fn ml_snapshot() {
    let img = aa_chessboard(DIM, CELL, OFFSET, DARK, BRIGHT);
    let mut current = BTreeMap::new();
    let mut actuals = Vec::new();
    for preset in PRESETS {
        let (count, dump) = render_snapshot(&img, preset);
        current.insert(preset.name.to_string(), count);
        actuals.push((preset.name.to_string(), dump));
    }
    check_or_update_manifest(MANIFEST_FILE, MANIFEST_HEADER, &current, &actuals);
}
