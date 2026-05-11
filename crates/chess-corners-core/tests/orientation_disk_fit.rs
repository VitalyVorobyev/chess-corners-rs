//! Integration tests for [`OrientationMethod::DiskFit`] exercising the
//! public dispatch entry point [`fit_axes_at_point`].
//!
//! These tests mirror the in-module unit tests in `disk_sector.rs` (which drive
//! `disk_sector::fit` directly) but go through the public API to confirm the
//! dispatch wiring and fallback paths are correct end-to-end.

use chess_corners_core::orientation::{fit_axes_at_point, OrientationMethod};
use core::f32::consts::PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic chessboard corner image of `size × size` pixels.
/// The intensity model is `128 + 80·tanh(d0/width)·tanh(d1/width)` where
/// `d0`, `d1` are signed perpendicular distances to the two crossing lines.
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

/// Undirected angular distance between two line angles (both in `[0, π)`).
fn line_delta(a: f32, b: f32) -> f32 {
    ((a - b + PI * 0.5).rem_euclid(PI) - PI * 0.5).abs()
}

/// Max angular error between two axis pairs — tests both direct and swapped.
fn pair_err(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    let direct = line_delta(a0, b0).max(line_delta(a1, b1));
    let swapped = line_delta(a0, b1).max(line_delta(a1, b0));
    direct.min(swapped)
}

// ---------------------------------------------------------------------------
// F8-1: Extreme-skew via public dispatch
// ---------------------------------------------------------------------------

/// Drive `fit_axes_at_point` with `DiskFit` on a 30°/150° projective
/// corner. The ring fit has higher error here; `DiskFit` must recover
/// both axes within 5°.
#[test]
fn disk_fit_recovers_extreme_skew_30_150() {
    let size = 41usize;
    let cx = 20.0f32;
    let cy = 20.0f32;
    let target0 = 30.0_f32.to_radians();
    let target1 = 150.0_f32.to_radians();
    let img = synthetic_corner(size, target0, target1, 0.7);

    let fit = fit_axes_at_point(&img, size, size, cx, cy, 5, OrientationMethod::DiskFit);
    let err = pair_err(fit.theta1, fit.theta2, target0, target1);
    assert!(
        err < 5.0_f32.to_radians(),
        "extreme-skew err {}° (fit theta1={:.2}°, theta2={:.2}°)",
        err.to_degrees(),
        fit.theta1.to_degrees(),
        fit.theta2.to_degrees(),
    );
    assert!(fit.sigma_theta1.is_finite());
    assert!(fit.sigma_theta2.is_finite());
}

// ---------------------------------------------------------------------------
// F8-2: Lazy-gate short-circuit on clean orthogonal corner
// ---------------------------------------------------------------------------

/// A clean orthogonal corner (0°/90°, width 1.0) should trip the lazy-disk
/// gate and return bit-identical output to [`OrientationMethod::RingFit`].
///
/// Both calls use `fit_axes_at_point` on the same image at the same point,
/// so they sample the ring identically. The lazy gate fires because the
/// clean orthogonal corner produces RMS well below `LAZY_DISK_REL_RMS_MAX`.
#[test]
fn disk_fit_lazy_gate_matches_ring_fit_on_clean_orthogonal() {
    let size = 41usize;
    let cx = 20.0f32;
    let cy = 20.0f32;
    // 0°/90° is the canonical clean orthogonal case used by the inline gate test.
    let theta0 = 0.0_f32.to_radians();
    let theta1 = 90.0_f32.to_radians();
    let img = synthetic_corner(size, theta0, theta1, 1.0);

    let lut = fit_axes_at_point(&img, size, size, cx, cy, 5, OrientationMethod::RingFit);
    let disk = fit_axes_at_point(&img, size, size, cx, cy, 5, OrientationMethod::DiskFit);

    // The lazy gate fires and returns the ring-fit fallback bit-identically.
    assert_eq!(
        disk.theta1.to_bits(),
        lut.theta1.to_bits(),
        "lazy gate: theta1 mismatch (disk={}, lut={})",
        disk.theta1,
        lut.theta1
    );
    assert_eq!(
        disk.theta2.to_bits(),
        lut.theta2.to_bits(),
        "lazy gate: theta2 mismatch"
    );
    assert_eq!(
        disk.amp.to_bits(),
        lut.amp.to_bits(),
        "lazy gate: amp mismatch"
    );
    assert_eq!(
        disk.sigma_theta1.to_bits(),
        lut.sigma_theta1.to_bits(),
        "lazy gate: sigma_theta1 mismatch"
    );
    assert_eq!(
        disk.sigma_theta2.to_bits(),
        lut.sigma_theta2.to_bits(),
        "lazy gate: sigma_theta2 mismatch"
    );
}

// ---------------------------------------------------------------------------
// F8-3: Border fallback
// ---------------------------------------------------------------------------

/// Smallest signed angular distance between `a` and `b` modulo π,
/// returned as an unsigned magnitude in `[0, π/2]`.
fn line_err_mod_pi(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(PI);
    d.min(PI - d)
}

// ---------------------------------------------------------------------------
// F8-4: Canonical convention parity between RingFit and DiskFit.
//
// `OrientationMethod::DiskFit` and `OrientationMethod::RingFit` MUST place
// `axes[0]` and `axes[1]` on the same canonical branch. Downstream
// consumers (e.g. calibration-target BFS edge gates that compare slot
// parity between cardinal grid neighbours) require this invariant: if
// the two methods disagreed by an antipodal flip on the same corner,
// the BFS would reject the corner spuriously.
//
// The test walks a representative sweep of corner orientations covering
// the full directional cycle (axis-0 from 0° to 170° in 10° steps; axis
// separation in {60°, 90°, 120°}) and asserts:
//
// 1. Both methods recover the same unordered axis set (mod π) within a
//    coarse 0.20 rad tolerance — guards against either method failing
//    catastrophically.
// 2. `axes[0]` (mod π) agrees between methods within 0.10 rad — guards
//    against the antipodal flip the regression test was added to catch.
//
// History: a previous canonicalisation in
// `orientation/ring_fit/solver.rs::canonicalize` rotated `θ₁` by `π`
// without flipping the amplitude when the input pair arrived in
// descending order, silently re-labelling the dark sector and producing
// `axes[0]/axes[1]` swaps relative to RingFit. The fix routes the
// short-arc selection through a model-preserving swap.
// ---------------------------------------------------------------------------

#[test]
fn canonical_axes_match_across_methods() {
    let size = 41usize;
    let cx = 20.0f32;
    let cy = 20.0f32;
    let radius = 5u32;
    let width = 0.7f32;

    // Axis-0 sweeps the full directional cycle (mod π); a few separations
    // probe the off-orthogonal regime where the disk fit is most active.
    let mut violations: Vec<String> = Vec::new();
    let mut total = 0usize;
    for sep_deg in [60.0_f32, 90.0, 120.0] {
        for axis0_deg in (0..180).step_by(10) {
            let theta0 = (axis0_deg as f32).to_radians();
            let theta1 = theta0 + sep_deg.to_radians();
            let img = synthetic_corner(size, theta0, theta1, width);

            let ring =
                fit_axes_at_point(&img, size, size, cx, cy, radius, OrientationMethod::RingFit);
            let disk =
                fit_axes_at_point(&img, size, size, cx, cy, radius, OrientationMethod::DiskFit);

            // (1) Same unordered axis set (mod π).
            let set_err = pair_err(disk.theta1, disk.theta2, ring.theta1, ring.theta2);
            if set_err > 0.20 {
                violations.push(format!(
                    "set: axis0={axis0_deg}° sep={sep_deg}° \
                     ring=({:.3},{:.3}) disk=({:.3},{:.3}) err={:.3} rad",
                    ring.theta1, ring.theta2, disk.theta1, disk.theta2, set_err
                ));
                total += 1;
                continue;
            }

            // (2) axes[0] agrees mod π — the slot-parity invariant.
            let axis0_err = line_err_mod_pi(ring.theta1, disk.theta1);
            if axis0_err > 0.10 {
                violations.push(format!(
                    "axis0 flip: axis0={axis0_deg}° sep={sep_deg}° \
                     ring.theta1={:.3} disk.theta1={:.3} err={:.3} rad",
                    ring.theta1, disk.theta1, axis0_err
                ));
            }
            total += 1;
        }
    }

    assert!(
        violations.is_empty(),
        "{} of {} corners had RingFit/DiskFit canonical convention \
         disagreements:\n{}",
        violations.len(),
        total,
        violations.join("\n"),
    );
}

/// A corner placed near the image border (4, 4) prevents disk extraction
/// (the support disk clips outside the image). `DiskFit` must fall
/// back to `RingFit` output bit-identically.
///
/// Both calls use `fit_axes_at_point` on the same image at the same point,
/// so the ring samples are identical. The disk fallback is triggered because
/// the border position causes `extract_disk` to return `None`.
#[test]
fn disk_fit_border_falls_back_to_ring_fit() {
    let size = 41usize;
    // Near the top-left corner: disk extraction will fail.
    let cx = 4.0f32;
    let cy = 4.0f32;
    let img = synthetic_corner(size, 0.2, 1.2, 1.0);

    let lut = fit_axes_at_point(&img, size, size, cx, cy, 5, OrientationMethod::RingFit);
    let disk = fit_axes_at_point(&img, size, size, cx, cy, 5, OrientationMethod::DiskFit);

    // Border → extract_disk returns None → falls back to ring-fit bit-identically.
    assert_eq!(
        disk.theta1.to_bits(),
        lut.theta1.to_bits(),
        "border fallback: theta1 mismatch"
    );
    assert_eq!(
        disk.theta2.to_bits(),
        lut.theta2.to_bits(),
        "border fallback: theta2 mismatch"
    );
    assert_eq!(
        disk.rms.to_bits(),
        lut.rms.to_bits(),
        "border fallback: rms mismatch"
    );
}
