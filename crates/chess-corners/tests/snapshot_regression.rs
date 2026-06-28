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
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use chess_corners::{CornerDescriptor, Detector, DetectorConfig};
use image::ImageReader;
use rayon::ThreadPool;
use serde::Serialize;

const SNAPSHOT_DIR: &str = "tests/snapshots";
const MANIFEST_FILE: &str = "manifest.txt";
const ACTUAL_SUBDIR: &str = "actual";

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
    ax0_sigma: f64,
    ax1_angle: f64,
    ax1_sigma: f64,
}

impl CornerSnap {
    fn from_descriptor(c: &CornerDescriptor) -> Self {
        Self {
            x: c.x as f64,
            y: c.y as f64,
            response: c.response as f64,
            ax0_angle: c.axes[0].angle as f64,
            ax0_sigma: c.axes[0].sigma as f64,
            ax1_angle: c.axes[1].angle as f64,
            ax1_sigma: c.axes[1].sigma as f64,
        }
    }
}

/// One-thread rayon pool, lazily initialised. Pinning the pool
/// removes thread-count-dependent reduction order; the test still
/// fails across CPU microarchitectures because FP results differ
/// at the last bit, but the counts (the thing we actually check)
/// stay stable.
fn pinned_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build single-threaded rayon pool")
    })
}

fn manifest_key(image: &str, preset: &str) -> String {
    format!("{image}-{preset}")
}

/// Render the detector output for one `(image, preset)` cell.
/// Returns `(count, ndjson_for_local_debugging)`.
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

/// Parse the manifest file into a `key → count` map.
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

/// Format the manifest map back to text. Sorted by key for stable
/// commits.
fn format_manifest(entries: &BTreeMap<String, usize>) -> String {
    let mut out = String::from("# Per-case corner count for the ChESS presets.\n");
    out.push_str("# Refresh with UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --test snapshot_regression --all-features\n");
    for (key, count) in entries {
        out.push_str(&format!("{key} count={count}\n"));
    }
    out
}

fn check_or_update_manifest() {
    let manifest_path = PathBuf::from(SNAPSHOT_DIR).join(MANIFEST_FILE);

    let mut current: BTreeMap<String, usize> = BTreeMap::new();
    let mut actuals: Vec<(String, String)> = Vec::new();

    for image in IMAGES {
        for preset in PRESETS {
            let (count, ndjson) = render_snapshot(image, preset);
            let key = manifest_key(image, preset.name);
            current.insert(key.clone(), count);
            actuals.push((key, ndjson));
        }
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
            "snapshot regression failed:\n  {}\n\nActual NDJSON dumps for inspection: {}\nIf intended, refresh with UPDATE_SNAPSHOTS=1.",
            mismatches.join("\n  "),
            actual_dir.display(),
        );
    }
}

/// Single test that exercises every `(image, preset)` cell in one
/// pass. Aggregating keeps the failure message coherent and the
/// manifest read cheap.
#[test]
fn snapshot_regression() {
    check_or_update_manifest();
}

#[allow(dead_code)]
fn _coverage_anchor() {
    // Touches the public IMAGES list so removing an image without a
    // corresponding test removal would be a compile-time tell.
    let _ = IMAGES;
}
