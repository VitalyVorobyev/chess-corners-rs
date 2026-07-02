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
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use chess_corners::{ChessRefiner, CornerDescriptor, Detector, DetectorConfig};
use chess_corners_testutil::aa_chessboard;
use rayon::ThreadPool;
use serde::Serialize;

const SNAPSHOT_DIR: &str = "tests/snapshots";
const MANIFEST_FILE: &str = "ml_manifest.txt";
const ACTUAL_SUBDIR: &str = "actual";

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

/// One-thread rayon pool, lazily initialised. Pinning the pool removes
/// thread-count-dependent reduction order (e.g. under `--all-features`,
/// which enables the facade's own `rayon` response path) so this test
/// only fails on a genuine algorithm change, not on thread-pool sizing.
/// See `snapshot_regression.rs::pinned_pool` for the original rationale.
fn pinned_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build single-threaded rayon pool")
    })
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

    corners.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then_with(|| a.y.partial_cmp(&b.y).unwrap())
    });

    let mut out = format!("# count={}\n", corners.len());
    for c in &corners {
        let snap = CornerSnap::from_descriptor(c);
        out.push_str(&serde_json::to_string(&snap).expect("serialize corner"));
        out.push('\n');
    }
    (corners.len(), out)
}

/// Parse the manifest file into a `preset name → count` map.
fn parse_manifest(text: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Format: "<key> count=<N>"
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("").to_string();
        let count_part = parts.next().unwrap_or("");
        let count: usize = count_part
            .strip_prefix("count=")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("manifest line {}: bad count: {line}", lineno + 1));
        out.insert(key, count);
    }
    out
}

/// Format the manifest map back to text. Sorted by key for stable commits.
fn format_manifest(entries: &BTreeMap<String, usize>) -> String {
    let mut out = String::from("# Per-preset corner count for the ChessRefiner::Ml pipelines.\n");
    out.push_str("# Refresh with UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --features ml-refiner --test ml_snapshot\n");
    for (key, count) in entries {
        out.push_str(&format!("{key} count={count}\n"));
    }
    out
}

fn check_or_update_manifest() {
    let manifest_path = PathBuf::from(SNAPSHOT_DIR).join(MANIFEST_FILE);
    let img = aa_chessboard(DIM, CELL, OFFSET, DARK, BRIGHT);

    let mut current: BTreeMap<String, usize> = BTreeMap::new();
    let mut actuals: Vec<(String, String)> = Vec::new();

    for preset in PRESETS {
        let (count, ndjson) = render_snapshot(&img, preset);
        current.insert(preset.name.to_string(), count);
        actuals.push((preset.name.to_string(), ndjson));
    }

    if env::var_os("UPDATE_SNAPSHOTS").is_some() {
        fs::create_dir_all(SNAPSHOT_DIR).expect("create snapshot dir");
        fs::write(&manifest_path, format_manifest(&current))
            .unwrap_or_else(|e| panic!("write {}: {e}", manifest_path.display()));
        return;
    }

    let expected_text = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "missing baseline {} ({e}); run with UPDATE_SNAPSHOTS=1 to seed it",
            manifest_path.display()
        )
    });
    let expected = parse_manifest(&expected_text);

    let mut mismatches: Vec<String> = Vec::new();
    for (key, cur_count) in &current {
        match expected.get(key) {
            Some(exp_count) if exp_count == cur_count => {}
            Some(exp_count) => mismatches.push(format!(
                "{key}: expected count={exp_count}, got count={cur_count}"
            )),
            None => mismatches.push(format!(
                "{key}: missing from manifest (current count={cur_count})"
            )),
        }
    }
    for key in expected.keys() {
        if !current.contains_key(key) {
            mismatches.push(format!(
                "{key}: in manifest but not produced by current code"
            ));
        }
    }

    if !mismatches.is_empty() {
        let actual_dir = PathBuf::from(SNAPSHOT_DIR).join(ACTUAL_SUBDIR);
        let _ = fs::create_dir_all(&actual_dir);
        for (key, ndjson) in &actuals {
            let path = actual_dir.join(format!("{key}.json"));
            let _ = fs::write(&path, ndjson);
        }
        panic!(
            "ml snapshot regression failed:\n  {}\n\nActual NDJSON dumps for inspection: {}\nIf intended, refresh with UPDATE_SNAPSHOTS=1.",
            mismatches.join("\n  "),
            actual_dir.display(),
        );
    }
}

/// Single test that exercises both `Ml` presets in one pass. Skipped
/// entirely (not failing) when the `ml-refiner` feature is off — see the
/// `#![cfg(feature = "ml-refiner")]` file-level gate above.
#[test]
fn ml_snapshot() {
    check_or_update_manifest();
}
