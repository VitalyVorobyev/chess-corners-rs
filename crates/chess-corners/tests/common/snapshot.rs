//! Count-manifest snapshot plumbing shared by `tests/snapshot_regression.rs`
//! and `tests/ml_snapshot.rs`, plus the single-thread rayon pool those and
//! `tests/orientation_slot_parity.rs` detect under.
//!
//! Included via `#[path = "common/snapshot.rs"] mod snapshot;` (kept out of
//! `tests/`'s auto-discovered targets by living in a subdirectory).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use chess_corners::CornerDescriptor;
use rayon::ThreadPool;
use serde::Serialize;

const SNAPSHOT_DIR: &str = "tests/snapshots";
const ACTUAL_SUBDIR: &str = "actual";

/// One-thread rayon pool, lazily initialised. Pinning the pool removes
/// thread-count-dependent reduction order (e.g. under `--all-features`, which
/// enables the facade's own `rayon` response path), so a snapshot only moves
/// on a genuine algorithm change, not on thread-pool sizing.
pub fn pinned_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build single-threaded rayon pool")
    })
}

/// Sort `corners` by `(x, y)` and render them as NDJSON, one `snap(corner)`
/// record per line under a `# count=N` header: the failure-inspection dump.
pub fn ndjson<S: Serialize>(
    corners: &mut [CornerDescriptor],
    snap: impl Fn(&CornerDescriptor) -> S,
) -> String {
    corners.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then_with(|| a.y.partial_cmp(&b.y).unwrap())
    });
    let mut out = format!("# count={}\n", corners.len());
    for c in corners.iter() {
        out.push_str(&serde_json::to_string(&snap(c)).expect("serialize corner"));
        out.push('\n');
    }
    out
}

/// Check `current` (key → corner count) against `tests/snapshots/<manifest>`,
/// or rewrite the manifest (`header` followed by sorted `<key> count=<N>`
/// lines) when `UPDATE_SNAPSHOTS` is set. On a mismatch, writes each key's
/// NDJSON from `actuals` to `tests/snapshots/actual/<key>.json` and panics.
pub fn check_or_update_manifest(
    manifest: &str,
    header: &str,
    current: &BTreeMap<String, usize>,
    actuals: &[(String, String)],
) {
    let manifest_path = PathBuf::from(SNAPSHOT_DIR).join(manifest);

    if env::var_os("UPDATE_SNAPSHOTS").is_some() {
        let mut text = header.to_string();
        for (key, count) in current {
            text.push_str(&format!("{key} count={count}\n"));
        }
        fs::create_dir_all(SNAPSHOT_DIR).expect("create snapshot dir");
        fs::write(&manifest_path, text)
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
    for (key, cur_count) in current {
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
        for (key, ndjson) in actuals {
            let _ = fs::write(actual_dir.join(format!("{key}.json")), ndjson);
        }
        panic!(
            "{manifest} snapshot regression failed:\n  {}\n\nActual NDJSON dumps for inspection: {}\nIf intended, refresh with UPDATE_SNAPSHOTS=1.",
            mismatches.join("\n  "),
            actual_dir.display(),
        );
    }
}

/// Parse `<key> count=<N>` lines (blank and `#` lines skipped).
fn parse_manifest(text: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("").to_string();
        let count: usize = parts
            .next()
            .and_then(|s| s.strip_prefix("count="))
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("manifest line {}: bad count: {line}", lineno + 1));
        out.insert(key, count);
    }
    out
}
