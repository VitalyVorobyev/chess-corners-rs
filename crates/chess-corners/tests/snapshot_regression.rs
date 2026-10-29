//! Numerical-regression snapshot for the public detector API.
//!
//! Goal: pin the **corner count** of the two ChESS presets
//! (`single_scale`, `multiscale`) on the three public test images.
//! A multi-step refactor of the detection pipeline that accidentally
//! changes the set of corners produced will move the count; an
//! invariance-preserving refactor will not. This is the cheapest
//! signal that survives the cross-platform reality below.
//!
//! ## Why count-only, not bit-for-bit
//!
//! Subpixel corner positions are f32-valued and depend on operations
//! (FMA contraction, denormals, rounding modes) that differ
//! observably between CPU microarchitectures. A bit-for-bit hash
//! pinned on macOS-arm64 will fail on linux-x86 even though the
//! algorithm is unchanged — every corner's `(x, y)` shifts by ~1 ulp
//! and the hash flips. We tried that path; it produced false
//! positives on every non-Apple-Silicon CI runner.
//!
//! Counts, by contrast, are integer-valued and stable as long as the
//! response threshold isn't crossed differently by FP rounding. For
//! ChESS that holds on the three test images at the default
//! threshold; the test would catch any genuine algorithm change
//! (different filtering, different ring, different NMS radius)
//! because such a change moves counts by ≥ one corner per image.
//!
//! The Radon presets (`radon`, `radon_multiscale`) are intentionally
//! excluded: their `(max - min)²` response sits near the
//! relative `0.01` cutoff for many pixels, so
//! sub-ulp FP differences DO cross the threshold and shift counts
//! by ±0.5% across CPUs. The Radon path is covered end-to-end by
//! `radon_pipeline.rs` (facade) and `radon_vs_chess.rs` (core).
//!
//! ## Manifest format
//!
//! `tests/snapshots/manifest.txt` is a per-`(image, preset)` line:
//!
//! ```text
//! mid-single_scale count=1199
//! ```
//!
//! Refresh after an intentional algorithm change:
//!
//! ```ignore
//! UPDATE_SNAPSHOTS=1 cargo test -p chess-corners \
//!     --test snapshot_regression --all-features
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use chess_corners::{CornerDescriptor, Detector, DetectorConfig};
use image::ImageReader;
use serde::Serialize;

#[path = "common/snapshot.rs"]
mod snapshot;
use snapshot::{check_or_update_manifest, ndjson, pinned_pool};

const MANIFEST_FILE: &str = "manifest.txt";
const MANIFEST_HEADER: &str = "# Per-case corner count for the ChESS presets.
# Refresh with UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --test snapshot_regression --all-features
";

const IMAGES: &[&str] = &["small", "mid", "large"];

#[derive(Clone)]
struct Preset {
    name: &'static str,
    build: fn() -> DetectorConfig,
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "single_scale",
        build: DetectorConfig::chess,
    },
    Preset {
        name: "multiscale",
        build: DetectorConfig::chess_multiscale,
    },
];

/// One corner record. Carried only for the failure-inspection dump
/// (the manifest itself stores only counts).
#[derive(Serialize)]
struct CornerSnap {
    x: f64,
    y: f64,
    response: f64,
    ax0_angle: f64,
    ax0_fit_std_error: f64,
    ax1_angle: f64,
    ax1_fit_std_error: f64,
}

impl CornerSnap {
    fn from_descriptor(c: &CornerDescriptor) -> Self {
        let axes = c.axes.expect("orientation enabled (default config)");
        Self {
            x: c.x as f64,
            y: c.y as f64,
            response: c.response as f64,
            ax0_angle: axes[0].angle as f64,
            ax0_fit_std_error: axes[0].fit_std_error as f64,
            ax1_angle: axes[1].angle as f64,
            ax1_fit_std_error: axes[1].fit_std_error as f64,
        }
    }
}

/// Render the detector output for one `(image, preset)` cell on the pinned
/// one-thread pool. Returns `(count, ndjson_for_local_debugging)`. Positions
/// still differ across CPU microarchitectures at the last bit; the counts
/// (the thing we check) stay stable.
fn render_snapshot(image: &str, preset: &Preset) -> (usize, String) {
    let img_path = PathBuf::from("../../testimages").join(format!("{image}.png"));
    let img = ImageReader::open(&img_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", img_path.display()))
        .decode()
        .expect("decode")
        .to_luma8();

    let cfg = (preset.build)();
    let mut detector = Detector::new(cfg).expect("build detector");
    let mut corners = pinned_pool().install(|| detector.detect(&img).expect("detect"));
    (
        corners.len(),
        ndjson(&mut corners, CornerSnap::from_descriptor),
    )
}

/// Single test that exercises every `(image, preset)` cell in one
/// pass. Aggregating keeps the failure message coherent and the
/// manifest read cheap.
#[test]
fn snapshot_regression() {
    let mut current = BTreeMap::new();
    let mut actuals = Vec::new();
    for image in IMAGES {
        for preset in PRESETS {
            let (count, dump) = render_snapshot(image, preset);
            let key = format!("{image}-{}", preset.name);
            current.insert(key.clone(), count);
            actuals.push((key, dump));
        }
    }
    check_or_update_manifest(MANIFEST_FILE, MANIFEST_HEADER, &current, &actuals);
}
