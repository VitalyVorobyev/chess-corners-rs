//! Axis-slot consistency regression for [`OrientationMethod`].
//!
//! ## What this guards
//!
//! For a planar chessboard the two local lattice directions a corner
//! reports as `axes[0]` / `axes[1]` must be ordered **globally
//! consistently** across the board. Every `OrientationMethod` emits
//! axes under the same convention (`axes[0].angle ∈ [0, π)`, the CCW arc
//! `axes[0] → axes[1]` is a *dark* sector), so two methods run on the
//! same image — whose corner **positions** are identical, since detection
//! runs before orientation — must agree on which of a corner's two lines
//! is `axes[0]`.
//!
//! `RingFit` is the consistent reference (its dark-sector pick comes from
//! the ring's gradient phase). This test asserts `DiskFit` agrees with it:
//! across position-matched corners, the fraction whose `axes[0]` is the
//! *other* line ("swapped") must stay small. A coherent antipodal-sector
//! inversion in `DiskFit` would push that fraction up and collapse the
//! board's Canonical/Swapped split — the failure this test exists to catch.
//!
//! ## Only disk-path corners carry signal
//!
//! On a full frame the descriptor stage runs the (expensive) full-disk
//! estimator on at most the strongest `FULL_DISK_MAX_FULL_IMAGE_CORNERS`
//! candidates and leaves the rest on the `RingFit` fallback (see
//! `chess-corners-core/src/orientation/descriptor.rs`). Fallback corners
//! are bit-identical to the `RingFit` run by construction, so including
//! them in the denominator would dilute the swap fraction to the point
//! where even a *total* inversion of every disk-path corner stays under
//! any small all-corner threshold. The swap fraction is therefore computed
//! **only over corners where `DiskFit` actually produced a result** (its
//! axes differ from `RingFit`'s) — exactly the set where an antipodal-sector
//! inversion can manifest. The test also asserts that set is non-trivially
//! large, so it retains power to fail.
//!
//! The per-method global split (≈50/50 for a consistent fitter) is also
//! computed and printed for diagnosis; run with `-- --nocapture` to see it.
#![cfg(feature = "image")]

use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::OnceLock;

use chess_corners::{CornerDescriptor, Detector, DetectorConfig, OrientationMethod};
use image::ImageReader;
use rayon::ThreadPool;

/// One-thread rayon pool so the corner set is reduction-order stable
/// regardless of host thread count (same rationale as the snapshot test).
fn pinned_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build single-threaded rayon pool")
    })
}

/// Undirected angular distance between two line angles, folded into
/// `[0, π/2]`. Lines are equivalent mod π, so this is the natural metric
/// for "do these two `axes[*].angle` values point along the same line".
fn line_delta(a: f32, b: f32) -> f32 {
    ((a - b + PI * 0.5).rem_euclid(PI) - PI * 0.5).abs()
}

fn detect(image: &str, method: OrientationMethod) -> Vec<CornerDescriptor> {
    let img_path = PathBuf::from("../../testimages").join(format!("{image}.png"));
    let img = ImageReader::open(&img_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", img_path.display()))
        .decode()
        .expect("decode")
        .to_luma8();
    let cfg = DetectorConfig::chess().with_orientation_method(method);
    let mut detector = Detector::new(cfg).expect("build detector");
    pinned_pool().install(|| detector.detect(&img).expect("detect"))
}

/// Match DiskFit corners to RingFit corners by nearest position (≤ 0.5 px)
/// and, among **disk-path** corners (DiskFit result differs from RingFit's)
/// where both methods recovered the same line *pair* (nearest line within
/// `SET_TOL`), report the fraction where DiskFit's `axes[0]` is the *other*
/// line than RingFit's `axes[0]`.
struct SwapStats {
    /// Corners where DiskFit's axes differ from RingFit's (disk path ran
    /// and was accepted) — the only corners that can carry an inversion.
    disk_path: usize,
    /// Disk-path corners where both methods agree on the line *set* (so a
    /// disagreement is a slot swap, not a genuine line-direction change).
    counted: usize,
    /// Disk-path, line-set-agreeing corners whose `axes[0]` is swapped.
    swapped: usize,
}

impl SwapStats {
    fn fraction(&self) -> f32 {
        if self.counted == 0 {
            0.0
        } else {
            self.swapped as f32 / self.counted as f32
        }
    }
}

const SET_TOL: f32 = 15.0_f32 * (PI / 180.0); // 15° — line-pair agreement gate
const MATCH_PX: f32 = 0.5; // position-match radius
/// A corner is "disk-path" when any axis differs from the RingFit run by
/// more than this. Fallback corners are bit-identical (Δ = 0); even the
/// finest disk refinement step (0.25°) moves an angle far past this floor.
const INFLUENCE_TOL: f32 = 1.0e-3;

fn swap_stats(ring: &[CornerDescriptor], disk: &[CornerDescriptor]) -> SwapStats {
    let mut disk_path = 0usize;
    let mut counted = 0usize;
    let mut swapped = 0usize;
    for d in disk {
        // Nearest RingFit corner by position (detection is identical, so
        // this is effectively an index match, but we stay robust to order).
        let Some(r) = ring
            .iter()
            .map(|r| {
                let dx = r.x - d.x;
                let dy = r.y - d.y;
                (r, dx * dx + dy * dy)
            })
            .filter(|&(_, d2)| d2 <= MATCH_PX * MATCH_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(r, _)| r)
        else {
            continue;
        };

        // Both runs use a concrete orientation method, so axes are present.
        let da = d.axes.expect("orientation enabled");
        let ra = r.axes.expect("orientation enabled");

        // Skip corners DiskFit left on the RingFit fallback: identical to
        // the RingFit run, so they carry no slot-ordering signal and only
        // dilute the fraction.
        let differs = line_delta(da[0].angle, ra[0].angle) > INFLUENCE_TOL
            || line_delta(da[1].angle, ra[1].angle) > INFLUENCE_TOL;
        if !differs {
            continue;
        }
        disk_path += 1;

        let to_r0 = line_delta(da[0].angle, ra[0].angle);
        let to_r1 = line_delta(da[0].angle, ra[1].angle);
        // Only judge corners where the two methods agree on the line set.
        if to_r0.min(to_r1) > SET_TOL {
            continue;
        }
        counted += 1;
        if to_r1 < to_r0 {
            swapped += 1;
        }
    }
    SwapStats {
        disk_path,
        counted,
        swapped,
    }
}

/// Global Canonical/Swapped split: cluster `axes[0].angle` (a line angle,
/// period π) into the two grid directions and return the minority fraction.
/// A consistent fitter on a chessboard alternates parity across neighbours
/// ⇒ ≈0.5; a collapsed fitter ⇒ ≪0.5.
///
/// Doubling folds the π period to 2π; the two grid directions (~90° apart)
/// become ~antipodal, so the *quadrupled*-angle mean recovers the cluster
/// axis in one closed-form pass (no RNG, fully deterministic).
fn minority_split(corners: &[CornerDescriptor]) -> (f32, usize, usize) {
    if corners.is_empty() {
        return (0.0, 0, 0);
    }
    let (mut sc, mut ss) = (0.0f32, 0.0f32);
    for c in corners {
        let a = c.axes.expect("orientation enabled")[0].angle;
        sc += (4.0 * a).cos();
        ss += (4.0 * a).sin();
    }
    let axis = ss.atan2(sc) * 0.5; // cluster axis in doubled-angle space
    let (mut n0, mut n1) = (0usize, 0usize);
    for c in corners {
        let beta = 2.0 * c.axes.expect("orientation enabled")[0].angle;
        let off = (beta - axis + PI).rem_euclid(2.0 * PI) - PI; // (-π, π]
        if off.abs() < PI * 0.5 {
            n0 += 1;
        } else {
            n1 += 1;
        }
    }
    let minority = n0.min(n1) as f32 / (n0 + n1) as f32;
    (minority, n0, n1)
}

/// Maximum tolerated swapped fraction *over disk-path corners*. A
/// consistent DiskFit sits at 0; a coherent antipodal inversion drives it
/// to ≈1.0 (every accepted disk fit flips). 0.10 is a wide guard band that
/// an inversion cannot satisfy while tolerating a handful of genuinely
/// ambiguous near-parallel corners.
const MAX_SWAPPED_FRACTION: f32 = 0.10;

/// Minimum disk-path corners required for the assertion to have power. Far
/// below the per-frame disk cap (`FULL_DISK_MAX_FULL_IMAGE_CORNERS = 80`),
/// so a healthy run clears it comfortably while a future change that stops
/// running the disk path at all is caught as a vacuous test instead of a
/// silent pass.
const MIN_DISK_PATH_CORNERS: usize = 20;

fn check_image(image: &str) {
    let ring = detect(image, OrientationMethod::RingFit);
    let disk = detect(image, OrientationMethod::DiskFit);
    assert!(!ring.is_empty(), "{image}: no corners detected (RingFit)");
    assert_eq!(
        ring.len(),
        disk.len(),
        "{image}: corner count differs between methods ({} ring vs {} disk); \
         detection should be orientation-independent",
        ring.len(),
        disk.len(),
    );

    let stats = swap_stats(&ring, &disk);
    let (ring_split, rn0, rn1) = minority_split(&ring);
    let (disk_split, dn0, dn1) = minority_split(&disk);

    println!(
        "[{image}] corners={} | disk-path={} | DiskFit vs RingFit slot swap: \
         {}/{} = {:.3} | global split  RingFit {rn0}/{rn1} (minority {ring_split:.3})  \
         DiskFit {dn0}/{dn1} (minority {disk_split:.3})",
        ring.len(),
        stats.disk_path,
        stats.swapped,
        stats.counted,
        stats.fraction(),
    );

    assert!(
        stats.counted >= MIN_DISK_PATH_CORNERS,
        "{image}: only {} comparable disk-path corners (need ≥ {}); the test \
         would be vacuous — DiskFit may have stopped taking the disk path.",
        stats.counted,
        MIN_DISK_PATH_CORNERS,
    );
    assert!(
        stats.fraction() < MAX_SWAPPED_FRACTION,
        "{image}: DiskFit axis-slot ordering disagrees with RingFit on \
         {}/{} ({:.1}%) of disk-path corners (limit {:.0}%). DiskFit is \
         picking the wrong antipodal dark sector, collapsing the global \
         axes[0]/axes[1] ordering (DiskFit global minority {:.3} vs RingFit {:.3}).",
        stats.swapped,
        stats.counted,
        stats.fraction() * 100.0,
        MAX_SWAPPED_FRACTION * 100.0,
        disk_split,
        ring_split,
    );
}

/// Clean, well-lit board — the image the original defect was reported on.
#[test]
fn diskfit_slot_ordering_matches_ringfit_mid() {
    check_image("mid");
}

/// Higher-resolution board with more perspective ⇒ more corners take the
/// full disk path (past the lazy-disk gate), exercising the antipodal pick
/// on the subset most prone to the inversion.
#[test]
fn diskfit_slot_ordering_matches_ringfit_large() {
    check_image("large");
}
