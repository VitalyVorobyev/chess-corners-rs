//! Numerical-regression snapshot for the public detector API.
//!
//! Goal: pin the corner-descriptor output of the four documented
//! presets (`single_scale`, `multiscale`, `radon`, `radon_multiscale`)
//! on the three public test images, so multi-step refactors of the
//! detection pipeline cannot silently drift the user-visible numbers.
//!
//! Run with:
//!
//! ```ignore
//! cargo test -p chess-corners --test snapshot_regression --all-features
//! ```
//!
//! To refresh the committed baselines (after an intentional change):
//!
//! ```ignore
//! UPDATE_SNAPSHOTS=1 cargo test -p chess-corners \
//!     --test snapshot_regression --all-features
//! ```
//!
//! The committed baseline is a single small manifest file
//! `tests/snapshots/manifest.txt` with one line per `(image, preset)`:
//!
//! ```text
//! mid-single_scale count=359 hash=0123456789abcdef
//! ```
//!
//! The hash is FNV-1a-64 over the sorted NDJSON corner dump (one line
//! per corner, `(x, y, response, contrast, fit_rms, axes[0..2])`)
//! prefixed with a `# count=N` header. Hashing makes the baseline
//! committable in ~1 KB instead of ~13 MB of raw JSON, at the cost of
//! a less-readable diff on failure. To inspect the actual output when
//! a test fails, look at `tests/snapshots/actual/<image>-<preset>.json`,
//! which the failing test writes for local debugging (gitignored).
//!
//! Cross-platform determinism: FNV-1a is byte-for-byte the same on any
//! platform; the corner ordering is stable (sort by `(x, y)` with
//! `partial_cmp`); the underlying detector math is f32-deterministic.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use chess_corners::{CornerDescriptor, Detector, DetectorConfig};
use image::ImageReader;
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
        build: DetectorConfig::single_scale,
    },
    Preset {
        name: "multiscale",
        build: DetectorConfig::multiscale,
    },
    Preset {
        name: "radon",
        build: DetectorConfig::radon,
    },
    Preset {
        name: "radon_multiscale",
        build: DetectorConfig::radon_multiscale,
    },
];

/// One corner record in the snapshot. Float fields are stored as f64
/// to give serde_json full f32-round-trip precision, so the JSON is a
/// deterministic representation of the bit pattern.
#[derive(Serialize)]
struct CornerSnap {
    x: f64,
    y: f64,
    response: f64,
    contrast: f64,
    fit_rms: f64,
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
            contrast: c.contrast as f64,
            fit_rms: c.fit_rms as f64,
            ax0_angle: c.axes[0].angle as f64,
            ax0_sigma: c.axes[0].sigma as f64,
            ax1_angle: c.axes[1].angle as f64,
            ax1_sigma: c.axes[1].sigma as f64,
        }
    }
}

/// FNV-1a 64-bit hash. Platform-independent, deterministic, tiny.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn manifest_key(image: &str, preset: &str) -> String {
    format!("{image}-{preset}")
}

/// Render the detector output for one `(image, preset)` cell into the
/// canonical NDJSON form. Returns `(count, ndjson_text)`.
fn render_snapshot(image: &str, preset: &Preset) -> (usize, String) {
    let img_path = PathBuf::from("../../testimages").join(format!("{image}.png"));
    let img = ImageReader::open(&img_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", img_path.display()))
        .decode()
        .expect("decode")
        .to_luma8();

    let cfg = (preset.build)();
    let mut detector = Detector::new(cfg).expect("build detector");
    let mut corners = detector.detect(&img).expect("detect");

    // Stable order across runs: sort by (x, y) — both are subpixel f32
    // so use total ordering. We do not expect ties on real imagery.
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

/// Parse the manifest file into a `key → (count, hash)` map.
fn parse_manifest(text: &str) -> BTreeMap<String, (usize, u64)> {
    let mut out = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Format: "<key> count=<N> hash=<hex>"
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("").to_string();
        let count_part = parts.next().unwrap_or("");
        let hash_part = parts.next().unwrap_or("");
        let count: usize = count_part
            .strip_prefix("count=")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("manifest line {}: bad count: {line}", lineno + 1));
        let hash: u64 = hash_part
            .strip_prefix("hash=")
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or_else(|| panic!("manifest line {}: bad hash: {line}", lineno + 1));
        out.insert(key, (count, hash));
    }
    out
}

/// Format the manifest map back to text. Sorted by key for stable
/// commits.
fn format_manifest(entries: &BTreeMap<String, (usize, u64)>) -> String {
    let mut out =
        String::from("# Per-case (count, fnv1a-64 hash) of the canonical NDJSON corner dump.\n");
    out.push_str("# Refresh with UPDATE_SNAPSHOTS=1 cargo test -p chess-corners --test snapshot_regression --all-features\n");
    for (key, (count, hash)) in entries {
        out.push_str(&format!("{key} count={count} hash={hash:016x}\n"));
    }
    out
}

fn check_or_update_manifest() {
    let manifest_path = PathBuf::from(SNAPSHOT_DIR).join(MANIFEST_FILE);

    let mut current: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut actuals: Vec<(String, String)> = Vec::new();

    for image in IMAGES {
        for preset in PRESETS {
            let (count, ndjson) = render_snapshot(image, preset);
            let hash = fnv1a_64(ndjson.as_bytes());
            let key = manifest_key(image, preset.name);
            current.insert(key.clone(), (count, hash));
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
    for (key, (cur_count, cur_hash)) in &current {
        match expected.get(key) {
            Some((exp_count, exp_hash)) if exp_count == cur_count && exp_hash == cur_hash => {}
            Some((exp_count, exp_hash)) => mismatches.push(format!(
                "{key}: expected count={exp_count} hash={exp_hash:016x}, got count={cur_count} hash={cur_hash:016x}"
            )),
            None => mismatches.push(format!(
                "{key}: missing from manifest (current count={cur_count}, hash={cur_hash:016x})"
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
/// pass. Aggregating into one test keeps the manifest read + parse
/// cheap and surfaces every drift in one panic message rather than
/// splitting failures across 12 test outputs.
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
