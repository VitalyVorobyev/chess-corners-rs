//! Full-disk two-axis orientation estimator.
//!
//! Models a chessboard corner as two crossing transition lines through
//! the detected center and recovers their angles from all pixels in a
//! support disk around that center. The intensity model is
//! `I(p) = μ + A · tanh(d₀(p)/w) · tanh(d₁(p)/w)`, where `d_i(p)` is
//! the signed perpendicular distance from pixel `p` to the i-th line
//! and `w` is a discrete edge width.
//!
//! Generates candidate line directions from a Sobel gradient-direction
//! histogram, the ring-fit seed angles plus small offsets, and a sparse
//! global grid; prunes to the top pairs by edge alignment; sweeps a
//! small fixed set of edge widths with a closed-form OLS solve for
//! amplitude; and refines the strongest survivors via local angular
//! grid search. The disk result only displaces the ring fit when
//! residual margin, edge alignment, and axis-separation checks all
//! confirm it; otherwise the ring fit is returned unchanged.
//!
//! See `book/src/part-03-chess-detector.md` for the step-by-step
//! algorithm description.
//!
//! ## Module layout
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`geometry`] | Angular arithmetic: `line_delta`, `wrap_pi`, `valid_pair`, … |
//! | [`data`] | `DiskData`, `extract_disk`, `sobel_at` |
//! | [`candidates`] | `CandidateSet`, `histogram_candidates`, pair pool |
//! | [`score`] | `Fit`, `score_pair`, `score_pair_cached`, `edge_pair_score`, … |
//! | [`search`] | `best_disk_fit`, `refine_fit`, `insert_top` |

mod candidates;
mod data;
mod geometry;
mod score;
mod search;

use super::ring_fit;
use data::extract_disk;
use geometry::{line_separation, pair_disagreement};
use score::{
    sigma_for_fit, ACCEPT_REL_MARGIN, ACCEPT_REL_RATIO, INFLATED_SIGMA_RAD, MIN_CONTRAST, MIN_CORR,
    WEAK_DISAGREE_RAD,
};
use search::best_disk_fit;

/// Lazy-disk gate threshold on the legacy fit's `rms / max(amp, 1.0)`.
/// Clean chessboard corners settle near `rel_rms ≈ 0.01` while the
/// confused regimes the disk fit was added for (extreme skew, blurry
/// edges, low contrast) sit at 0.1–0.4. 0.04 is comfortably below the
/// `ACCEPT_REL_MARGIN = 0.03` margin that the disk fit must clear to
/// take over, so any corner short-circuited here would have failed the
/// margin test even after running the full disk fit. The headline
/// extreme-skew cell (30°/150°) collapses the legacy fit to ~21° error
/// and inflates `rel_rms` well past 0.1; that case still pays the full
/// disk cost.
const LAZY_DISK_REL_RMS_MAX: f32 = 0.04;

/// Lazy-disk gate band on `line_separation(theta1, theta2)` in degrees.
/// The disk-sector benchmark concentrates the useful disk-fit cases
/// outside this band: extreme skew and mis-converged baselines that snap
/// to near-parallel axes. [70°, 110°] keeps clean orthogonal corners
/// inside the lazy path while leaving margin for suspect corners with
/// mis-aligned legacy axes.
const LAZY_DISK_SEP_DEG_MIN: f32 = 70.0;
const LAZY_DISK_SEP_DEG_MAX: f32 = 110.0;

/// Full-disk estimator used by the public image-side dispatcher.
///
/// Returned axes follow the same canonical convention as
/// [`crate::orientation::OrientationMethod::RingFit`]: `axes[0].angle ∈
/// [0, π)`, `axes[1].angle ∈ (axes[0].angle, axes[0].angle + π)`, and
/// the CCW arc `(axes[0], axes[1])` is a *dark* sector. The
/// per-candidate scorers in [`score::score_pair`] and
/// [`score::score_pair_cached`] route their raw `(theta, theta', amp)`
/// through `ring_fit::canonicalize`, the single source of truth for the
/// dark-sector / antipodal-pair selection rule. The OLS amplitude `A =
/// dot / denom` carries the polarity information: a candidate pair
/// rotated 90° relative to the true lines yields `A < 0`, which
/// `canonicalize` resolves into the matching `axes[0]` slot of the
/// ring fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit(
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> ring_fit::TwoAxisFit {
    let fallback = ring_fit::fit_ring(samples, ring_phi);

    // Lazy-disk gate: most chessboard corners are well-fit by the
    // legacy 16-ring path. Skip the ~131 µs disk fit when both signals
    // agree the legacy result is already a confident orthogonal-ish
    // corner. Suspect corners (extreme skew, blurry edges, low
    // contrast) keep the high rel_rms or off-band separation that puts
    // them through the full disk path unchanged.
    if lazy_disk_skip(&fallback) {
        return fallback;
    }

    let Some(data) = extract_disk(img, w, h, cx, cy, radius) else {
        return fallback;
    };
    let Some(fit) = best_disk_fit(&data, fallback.theta1, fallback.theta2) else {
        return fallback;
    };

    let base_rel = fallback.rms.max(0.0) / fallback.amp.max(1.0);
    let disagreement = pair_disagreement(fit.theta1, fit.theta2, fallback.theta1, fallback.theta2);
    let sep = line_separation(fit.theta1, fit.theta2);

    let accept_by_margin = fit.rel_rms <= base_rel - ACCEPT_REL_MARGIN;
    let accept_by_ratio = fit.rel_rms <= base_rel * ACCEPT_REL_RATIO;
    let strong_nonorthogonal = sep < 55.0_f32.to_radians();
    let sharp_orthogonal = sep >= 75.0_f32.to_radians() && fit.width <= 0.70;
    let edge_disagreement =
        disagreement > WEAK_DISAGREE_RAD && fit.edge_score >= 0.18 && fit.rel_rms <= 0.45;
    let near_orthogonal_blur = sep >= 65.0_f32.to_radians()
        && fit.width >= 2.80
        && disagreement <= WEAK_DISAGREE_RAD
        && fit.rel_rms > base_rel - 0.08;
    let residual_accept = (accept_by_margin || accept_by_ratio) && !near_orthogonal_blur;

    let accepted = fit.amp >= MIN_CONTRAST
        && fit.corr >= MIN_CORR
        && fit.edge_score >= 0.035
        && fit.rel_rms.is_finite()
        && (residual_accept || edge_disagreement || strong_nonorthogonal || sharp_orthogonal);

    if accepted {
        let sigma = sigma_for_fit(&fit);
        ring_fit::TwoAxisFit {
            amp: fit.amp,
            theta1: fit.theta1,
            theta2: fit.theta2,
            sigma_theta1: sigma,
            sigma_theta2: sigma,
            rms: fit.rms,
        }
    } else if disagreement > WEAK_DISAGREE_RAD {
        ring_fit::TwoAxisFit {
            sigma_theta1: fallback.sigma_theta1.max(INFLATED_SIGMA_RAD),
            sigma_theta2: fallback.sigma_theta2.max(INFLATED_SIGMA_RAD),
            ..fallback
        }
    } else {
        fallback
    }
}

/// Lazy-disk gate predicate. Returns `true` when the legacy fallback
/// already explains the ring well enough that the full disk fit is
/// statistically guaranteed not to displace it via the
/// `ACCEPT_REL_MARGIN`/`ACCEPT_REL_RATIO` route.
///
/// Both signals are derived purely from `fallback`; no per-corner
/// allocation is introduced. The matched-rationale constants live next
/// to the other module thresholds.
fn lazy_disk_skip(fallback: &ring_fit::TwoAxisFit) -> bool {
    if !fallback.amp.is_finite() || !fallback.rms.is_finite() {
        return false;
    }
    let rel_rms = fallback.rms / fallback.amp.max(1.0);
    if rel_rms >= LAZY_DISK_REL_RMS_MAX {
        return false;
    }
    let sep_deg = line_separation(fallback.theta1, fallback.theta2).to_degrees();
    (LAZY_DISK_SEP_DEG_MIN..=LAZY_DISK_SEP_DEG_MAX).contains(&sep_deg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::chess::ring::ring_offsets;
    use crate::orientation::descriptor::{ring_angles, sample_ring};

    fn synthetic_corner(size: usize, theta0: f32, theta1: f32, width: f32) -> Vec<u8> {
        let cx = (size / 2) as f32;
        let cy = cx;
        let (s0, c0) = theta0.sin_cos();
        let (s1, c1) = theta1.sin_cos();
        let mut img = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d0 = -s0 * dx + c0 * dy;
                let d1 = -s1 * dx + c1 * dy;
                let q = (d0 / width).tanh() * (d1 / width).tanh();
                let val = 128.0 + 80.0 * q;
                img[y * size + x] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    fn pair_err(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
        pair_disagreement(a0, a1, b0, b1)
    }

    #[test]
    fn recovers_projective_disk_axes() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 20.0_f32.to_radians();
        let target1 = 50.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 0.7);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        assert!(
            pair_err(fit.theta1, fit.theta2, target0, target1) < 3.0_f32.to_radians(),
            "fit={fit:?}"
        );
        assert!(fit.sigma_theta1.is_finite());
        assert!(fit.sigma_theta2.is_finite());
    }

    #[test]
    fn falls_back_near_border() {
        let size = 41usize;
        let img = synthetic_corner(size, 0.2, 1.2, 1.0);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, 4.0, 4.0, ring);
        let fallback = ring_fit::fit_ring(&samples, &phi);
        let fit = fit(&img, size, size, 4.0, 4.0, 5, &samples, &phi);

        assert!((fit.theta1 - fallback.theta1).abs() < 1e-6);
        assert!((fit.theta2 - fallback.theta2).abs() < 1e-6);
        assert!((fit.rms - fallback.rms).abs() < 1e-6);
    }

    /// Regression fixture from the disk-sector benchmark: extreme
    /// axis-skew (30°/150° crossing — 30° between lines) must recover
    /// within a few degrees. The legacy fit collapsed to ≈21° error on
    /// this case when the test was added; the disk fit must clear ≤5°.
    #[test]
    fn recovers_extreme_skew_30_150() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 30.0_f32.to_radians();
        let target1 = 150.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 0.7);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        let err = pair_err(fit.theta1, fit.theta2, target0, target1);
        assert!(
            err < 5.0_f32.to_radians(),
            "extreme-skew err {}° (fit={fit:?})",
            err.to_degrees()
        );
    }

    /// Sharp-edge orthogonal corner. Width 0.35 mimics the
    /// "blur=0" cell from the bench, where the legacy fit
    /// inflates to ~4.3° RMSE.
    #[test]
    fn recovers_sharp_orthogonal() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 25.0_f32.to_radians();
        let target1 = 115.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 0.35);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        let err = pair_err(fit.theta1, fit.theta2, target0, target1);
        assert!(
            err < 3.0_f32.to_radians(),
            "sharp-orthogonal err {}° (fit={fit:?})",
            err.to_degrees()
        );
    }

    /// Lazy-disk gate must short-circuit a clean orthogonal corner so
    /// `fit()` returns the legacy fallback bit-exactly. The ~131 µs
    /// disk fit only justifies its cost on suspect corners; a synthetic
    /// 90°-separation, mid-width corner is the canonical "easy" case.
    #[test]
    fn lazy_gate_short_circuits_clean_orthogonal() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 0.0_f32.to_radians();
        let target1 = 90.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 1.0);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);
        let fallback = ring_fit::fit_ring(&samples, &phi);

        // Pre-condition: this fallback is well within the gate band.
        assert!(
            lazy_disk_skip(&fallback),
            "synthetic clean orthogonal must trip the gate; fallback={fallback:?}"
        );

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        assert_eq!(fit.theta1.to_bits(), fallback.theta1.to_bits());
        assert_eq!(fit.theta2.to_bits(), fallback.theta2.to_bits());
        assert_eq!(fit.amp.to_bits(), fallback.amp.to_bits());
        assert_eq!(fit.sigma_theta1.to_bits(), fallback.sigma_theta1.to_bits());
        assert_eq!(fit.sigma_theta2.to_bits(), fallback.sigma_theta2.to_bits());
        assert_eq!(fit.rms.to_bits(), fallback.rms.to_bits());
    }

    /// Lazy-disk gate must NOT short-circuit the headline extreme-skew
    /// case. The legacy fallback collapses to ~21° error on 30°/150°,
    /// so the disk path must still run; we detect that by checking that
    /// at least one returned axis differs measurably from the fallback.
    #[test]
    fn lazy_gate_does_not_skip_extreme_skew() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 30.0_f32.to_radians();
        let target1 = 150.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 0.7);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);
        let fallback = ring_fit::fit_ring(&samples, &phi);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        let dt1 = geometry::line_delta(fit.theta1, fallback.theta1);
        let dt2 = geometry::line_delta(fit.theta2, fallback.theta2);
        let one_deg = 1.0_f32.to_radians();
        assert!(
            dt1 > one_deg || dt2 > one_deg,
            "disk fit appears to have been skipped on extreme skew: \
             fallback={fallback:?}, fit={fit:?}"
        );
    }

    /// Medium projective skew with broader edge width. Pair pruning
    /// must keep the strong-evidence pair on the seed list even when
    /// the histogram only resolves one of the two lines clearly.
    #[test]
    fn recovers_medium_skew_60_140() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 60.0_f32.to_radians();
        let target1 = 140.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 1.4);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        let err = pair_err(fit.theta1, fit.theta2, target0, target1);
        assert!(
            err < 3.0_f32.to_radians(),
            "medium-skew err {}° (fit={fit:?})",
            err.to_degrees()
        );
    }
}
