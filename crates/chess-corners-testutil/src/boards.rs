//! Deterministic synthetic chessboard generators.
//!
//! Three board flavours share the same cell geometry and intensity levels,
//! so the same corner coordinate lands on a real 4-quadrant crossing in
//! each:
//!
//! - [`synth_chessboard`] — hard 40/215 steps. The tanh ring/disk model
//!   cannot represent an instantaneous step, so even a perfect corner fits
//!   with a large relative residual (`rel_rms ≈ 0.47`). This is the worst
//!   case for the orientation fits.
//! - [`synth_chessboard_soft`] — anti-aliased (8× supersampled + mild box
//!   blur). The graded edge matches the tanh model, so a clean corner has a
//!   *low* `rel_rms`, exercising RingFit's fast 2nd-harmonic seed path and
//!   DiskFit's lazy-gate short-circuit.
//! - [`synth_chessboard_warped`] — the soft board under a fixed affine skew
//!   so the corner's axis separation is far from 90°. This is the regime
//!   DiskFit's full disk estimator targets.
//!
//! [`aa_chessboard`] is a separate anti-aliased generator that exposes the
//! cell size, a subpixel `offset`, and the `dark`/`bright` levels — used by
//! the refiner-accuracy sweeps and the Radon-vs-ChESS hostile fixtures.

/// Dark-cell intensity (gray levels) shared by the parameter-free boards.
const DARK: u8 = 40;
/// Bright-cell intensity (gray levels) shared by the parameter-free boards.
const BRIGHT: u8 = 215;

/// Supersampling factor for the anti-aliased generators. Each output pixel
/// averages a `SUPER × SUPER` grid of hard-pattern sub-samples, softening
/// every step into a graded edge.
const SUPER: usize = 8;

/// Box-blur passes applied to the anti-aliased boards after supersampling.
/// Supersampling alone leaves a ~1 px edge, which is *sharper* than the ring
/// model's effective edge width (`r/β = 1.25 px` at radius 5, i.e. a 10–90%
/// rise of ~2.7 px). Two 3×3 box passes widen the data edge into that sweet
/// spot, dropping the clean-corner `rel_rms` to ~0.01 — well below DiskFit's
/// 0.04 lazy gate and RingFit's 0.12 robust trigger — so the soft fixture
/// reliably reaches the fast/lazy paths. Empirically chosen (PERF-12 probe).
const SOFT_BLUR_PASSES: u32 = 2;

/// Affine shear applied by [`synth_chessboard_warped`]. The vertical edge
/// family is mapped to direction `(SHEAR, 1)`; at `SHEAR = 0.6` the corner's
/// two axes sit at 0° and `atan2(1, 0.6) ≈ 59°`, an axis separation ~59°
/// (measured ~61° after the soft blur) — well outside DiskFit's [70°, 90°]
/// lazy-gate band, so the full disk estimator runs. The margin is kept
/// generous so the full-disk path stays reachable independent of the exact
/// [`SOFT_BLUR_PASSES`] setting.
const WARP_SHEAR: f32 = 0.6;

/// Image side and corner used by the single-corner orientation fixtures.
/// The soft and warped boards are calibrated so this corner lands on a real
/// 4-quadrant crossing; the warp's fixed row is `ORIENT_CORNER.1`, keeping
/// the corner stationary while the axes skew.
pub const ORIENT_DIM: usize = 256;

/// Benchmark corner coordinate (a 90° checkerboard crossing on the hard and
/// soft boards, a ~63° crossing on the warped board).
pub const ORIENT_CORNER: (f32, f32) = (130.0, 130.0);

/// Cell size in pixels: ~25 cells across the shorter image dimension (floor
/// 8), matching a typical calibration target in frame. Shared by every
/// parameter-free generator so corner coordinates coincide across boards.
#[inline]
fn cell_size(w: usize, h: usize) -> i32 {
    (h.min(w) / 25).max(8) as i32
}

/// Hard checkerboard cell parity: `true` selects the dark level.
#[inline]
fn is_dark_cell(cx: i32, cy: i32) -> bool {
    (cx + cy).rem_euclid(2) == 0
}

/// Deterministic hard-edge chessboard image in row-major 8-bit grayscale.
///
/// Corner density is roughly 25 cells across the shorter image dimension,
/// which matches a typical calibration target in frame. The pattern is fully
/// deterministic so response timings are comparable across machines and
/// across feature builds.
pub fn synth_chessboard(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = (x as i32) / cell;
            let cy = (y as i32) / cell;
            out[y * w + x] = if is_dark_cell(cx, cy) { DARK } else { BRIGHT };
        }
    }
    out
}

/// Soft-edge (anti-aliased) chessboard.
///
/// Same geometry and levels as [`synth_chessboard`], but each pixel is the
/// average of an `SUPER × SUPER` grid of hard-pattern sub-samples followed by
/// two `3×3` box passes. The resulting graded edge matches the tanh intensity
/// model, so a clean corner fits with a small relative residual.
pub fn synth_chessboard_soft(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    render_soft(w, h, cell, SOFT_BLUR_PASSES, |x, y| (x, y))
}

/// Soft-edge chessboard under a fixed affine skew.
///
/// Built like [`synth_chessboard_soft`] but each sub-sample's lookup
/// coordinate is run through the inverse of a horizontal shear pinned to the
/// row `ORIENT_CORNER.1`. The corner at [`ORIENT_CORNER`] stays put while its
/// axes skew to a ~63° separation, the projective-skew regime the full-disk
/// estimator targets. Calibrated for the [`ORIENT_DIM`] orientation fixtures.
pub fn synth_chessboard_warped(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    let py = ORIENT_CORNER.1;
    // Forward edges: (1,0) -> (1,0) and (0,1) -> (SHEAR,1). The inverse map
    // used to resample the source pattern is x' = x - SHEAR·(y - py), y' = y;
    // the row y = py is fixed, so the corner does not move.
    render_soft(w, h, cell, SOFT_BLUR_PASSES, move |x, y| {
        (x - WARP_SHEAR * (y - py), y)
    })
}

/// Render an anti-aliased chessboard. `map` transforms an output pixel
/// coordinate into the board coordinate looked up against the hard cell
/// parity: the identity yields an axis-aligned soft board, an inverse shear
/// yields a skewed one. Anti-aliasing comes from `SUPER × SUPER` box
/// supersampling; `blur_passes` 3×3 box passes then widen the edge toward the
/// ring/disk model. Fully deterministic — no RNG.
fn render_soft<F>(w: usize, h: usize, cell: i32, blur_passes: u32, map: F) -> Vec<u8>
where
    F: Fn(f32, f32) -> (f32, f32),
{
    let inv = 1.0 / (SUPER * SUPER) as f32;
    let cellf = cell as f32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for sy in 0..SUPER {
                let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                for sx in 0..SUPER {
                    let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                    let (bx, by) = map(xf, yf);
                    let cx = (bx / cellf).floor() as i32;
                    let cy = (by / cellf).floor() as i32;
                    acc += if is_dark_cell(cx, cy) {
                        DARK as f32
                    } else {
                        BRIGHT as f32
                    };
                }
            }
            out[y * w + x] = (acc * inv).round().clamp(0.0, 255.0) as u8;
        }
    }
    for _ in 0..blur_passes {
        out = box_blur_3x3(&out, w, h);
    }
    out
}

/// One 3×3 box blur pass with edge clamping. Mirrors the mild camera-PSF
/// smoothing used by the Radon-peak refiner's synthetic fixtures.
fn box_blur_3x3(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for ky in -1i32..=1 {
                let sy = (y as i32 + ky).clamp(0, h as i32 - 1) as usize;
                for kx in -1i32..=1 {
                    let sx = (x as i32 + kx).clamp(0, w as i32 - 1) as usize;
                    acc += src[sy * w + sx] as u32;
                }
            }
            out[y * w + x] = (acc / 9) as u8;
        }
    }
    out
}

/// Anti-aliased `size × size` chessboard with caller-chosen cell size,
/// subpixel `offset`, and `dark`/`bright` intensity levels.
///
/// Each pixel averages an `SUPER × SUPER` grid of hard-pattern sub-samples,
/// so the corner at `(offset.0 + k·cell, offset.1 + m·cell)` carries a graded
/// edge without a separate blur pass. The cell parity is taken modulo 2 with
/// `rem_euclid`, so a negative `offset` (which pushes lookup cells below
/// zero) keeps the correct checker pattern.
pub fn aa_chessboard(
    size: usize,
    cell: usize,
    offset: (f32, f32),
    dark: u8,
    bright: u8,
) -> Vec<u8> {
    let (ox, oy) = offset;
    let c = cell as f32;
    let dark_f = dark as f32;
    let bright_f = bright as f32;
    let inv_super2 = 1.0 / (SUPER * SUPER) as f32;
    let mut img = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0f32;
            for sy in 0..SUPER {
                let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                let cy = ((yf - oy) / c).floor() as i32;
                for sx in 0..SUPER {
                    let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                    let cx = ((xf - ox) / c).floor() as i32;
                    let dark_cell = (cx + cy).rem_euclid(2) == 0;
                    acc += if dark_cell { dark_f } else { bright_f };
                }
            }
            img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

/// Exact pixel-area chessboard, axis-aligned or sheared, with exact corner
/// positions.
///
/// The board's lattice axes are `e1 = (1, 0)` and
/// `e2 = (cos axis_angle, sin axis_angle)` (image frame, y down; `axis_angle`
/// in radians, in `(0, π)`), so every corner sits at
/// `offset + cell·(k·e1 + m·e2)` for integers `k`, `m`. `axis_angle = π/2`
/// gives an axis-aligned board; any other angle shears the vertical edge
/// family. Cell parity matches [`aa_chessboard`].
///
/// Each pixel is the exact mean of the hard pattern over its unit square
/// (pixel centres on integer coordinates, the same convention as
/// [`aa_chessboard`]), i.e. an ideal box-filtering sensor, rounded to `u8`.
/// Unlike supersampling, the edge position carries no sub-sample
/// quantization, so a refiner's error against the lattice corner is its own
/// bias plus the final 8-bit rounding.
///
/// # Panics
///
/// If `axis_angle` is not in `(0, π)`.
pub fn area_chessboard(
    size: usize,
    cell: usize,
    offset: (f32, f32),
    axis_angle: f32,
    dark: u8,
    bright: u8,
) -> Vec<u8> {
    let angle = f64::from(axis_angle);
    assert!(
        angle > 0.0 && angle < core::f64::consts::PI,
        "axis_angle {axis_angle} outside (0, π)"
    );
    let c = cell as f64;
    let (ox, oy) = (f64::from(offset.0), f64::from(offset.1));
    let (sin, cos) = angle.sin_cos();
    // Lattice coordinates: u2 = (y − oy)/(c·sin), u1 = (x − ox − (y − oy)·cot)/c.
    let cot = cos / sin;
    let u2 = |y: f64| (y - oy) / (c * sin);
    let u1 = |x: f64, y: f64| (x - ox - (y - oy) * cot) / c;
    let mid = 0.5 * (f64::from(dark) + f64::from(bright));
    let half = 0.5 * (f64::from(bright) - f64::from(dark));

    let mut img = vec![0u8; size * size];
    let mut breaks = Vec::new();
    for py in 0..size {
        let (y0, y1) = (py as f64 - 0.5, py as f64 + 0.5);
        for px in 0..size {
            let (x0, x1) = (px as f64 - 0.5, px as f64 + 0.5);
            // ∫ s(u1) dx across the row is c·[S(u1(x1, y)) − S(u1(x0, y))]: linear
            // in y between the rows where u1 at either x-bound is an integer;
            // s(u2) is constant between the rows where u2 is an integer. Split
            // [y0, y1] at all of them; the midpoint rule is then exact.
            breaks.clear();
            breaks.extend([y0, y1]);
            push_integer_crossings(&mut breaks, y0, y1, u2);
            if cot.abs() > 1e-12 {
                push_integer_crossings(&mut breaks, y0, y1, |y| u1(x0, y));
                push_integer_crossings(&mut breaks, y0, y1, |y| u1(x1, y));
            }
            breaks.sort_by(f64::total_cmp);
            let mut avg = 0.0;
            for seg in breaks.windows(2) {
                let ym = 0.5 * (seg[0] + seg[1]);
                let row = c * (tri(u1(x1, ym)) - tri(u1(x0, ym)));
                avg += square(u2(ym)) * row * (seg[1] - seg[0]);
            }
            // Dark where s(u1)·s(u2) = +1.
            img[py * size + px] = (mid - half * avg).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

/// Square wave: `+1` where `floor(t)` is even, `−1` where it is odd.
#[inline]
fn square(t: f64) -> f64 {
    if t.floor().rem_euclid(2.0) == 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Antiderivative of [`square`] from 0: a triangle wave of period 2.
#[inline]
fn tri(t: f64) -> f64 {
    let r = t.rem_euclid(2.0);
    if r < 1.0 {
        r
    } else {
        2.0 - r
    }
}

/// Push every `y` in `(y0, y1)` where the affine `f(y)` takes an integer
/// value.
fn push_integer_crossings(out: &mut Vec<f64>, y0: f64, y1: f64, f: impl Fn(f64) -> f64) {
    let (f0, f1) = (f(y0), f(y1));
    if f0 == f1 {
        return;
    }
    let (lo, hi) = (f0.min(f1), f0.max(f1));
    let mut k = lo.ceil();
    while k <= hi {
        let y = y0 + (k - f0) / (f1 - f0) * (y1 - y0);
        if y > y0 && y < y1 {
            out.push(y);
        }
        k += 1.0;
    }
}

/// Count the ground-truth corners visible in a `size × size` image given a
/// chessboard with `cell`-pixel cells and `offset` origin. A "visible" corner
/// sits on an `(offset.0 + k·cell, offset.1 + m·cell)` junction strictly
/// inside a `border`-wide interior.
pub fn expected_corner_count(size: usize, cell: usize, offset: (f32, f32), border: usize) -> usize {
    let lo = border as f32;
    let hi = (size - border) as f32;
    let (ox, oy) = offset;
    let c = cell as f32;
    let mut n = 0;
    let mut k = -((size as f32) / c).ceil() as i32;
    while (k as f32) * c + ox < hi {
        let kx = ox + (k as f32) * c;
        if kx >= lo && kx <= hi {
            let mut m = -((size as f32) / c).ceil() as i32;
            while (m as f32) * c + oy < hi {
                let my = oy + (m as f32) * c;
                if my >= lo && my <= hi {
                    n += 1;
                }
                m += 1;
            }
        }
        k += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hard board's cell parity uses `rem_euclid(2)`, but the inline
    /// copies it replaces used `& 1`. Both agree because the hard board's
    /// cell indices `x / cell`, `y / cell` are always non-negative — lock
    /// that equivalence so a future edit can't silently diverge from the
    /// fixtures the perf baseline was captured on.
    #[test]
    fn hard_board_matches_bitand_parity() {
        for &(w, h) in &[(64usize, 48usize), (256, 256), (129, 97)] {
            let got = synth_chessboard(w, h);
            let cell = (h.min(w) / 25).max(8) as i32;
            let mut want = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let cx = (x as i32) / cell;
                    let cy = (y as i32) / cell;
                    want[y * w + x] = if (cx + cy) & 1 == 0 { 40 } else { 215 };
                }
            }
            assert_eq!(got, want, "hard board {w}x{h} drifted from &1 parity");
        }
    }

    /// Brute-force pixel mean over a `n × n` sub-sample grid, the reference
    /// for [`area_chessboard`] (converges to the exact area as `n` grows).
    fn supersampled_lattice(
        size: usize,
        cell: usize,
        offset: (f32, f32),
        angle: f64,
        n: usize,
    ) -> Vec<f64> {
        let c = cell as f64;
        let (sin, cos) = angle.sin_cos();
        let mut out = vec![0.0; size * size];
        for py in 0..size {
            for px in 0..size {
                let mut acc = 0.0;
                for sy in 0..n {
                    let y = py as f64 + (sy as f64 + 0.5) / n as f64 - 0.5 - offset.1 as f64;
                    let v = (y / (c * sin)).floor();
                    for sx in 0..n {
                        let x = px as f64 + (sx as f64 + 0.5) / n as f64 - 0.5 - offset.0 as f64;
                        let u = ((x - y * cos / sin) / c).floor();
                        acc += if (u + v).rem_euclid(2.0) == 0.0 {
                            40.0
                        } else {
                            200.0
                        };
                    }
                }
                out[py * size + px] = acc / (n * n) as f64;
            }
        }
        out
    }

    /// Axis-aligned and sheared boards agree with a 256×256-sub-sample
    /// brute force to within the final 8-bit rounding (0.5) plus the brute
    /// force's own edge error (under 1/512 px of a 160-level step, 0.31). An
    /// edge misplaced by 0.01 px would show as a 1.6-level error.
    #[test]
    fn area_board_matches_brute_force() {
        const SIZE: usize = 14;
        for (angle, offset) in [
            (core::f64::consts::FRAC_PI_2, (6.3f32, 7.55f32)),
            (60f64.to_radians(), (6.71, 6.2)),
            (115f64.to_radians(), (7.05, 6.9)),
        ] {
            let got = area_chessboard(SIZE, 5, offset, angle as f32, 40, 200);
            let want = supersampled_lattice(SIZE, 5, offset, angle, 256);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - w).abs() <= 0.5 + 0.32,
                    "angle {angle}: pixel {i} = {g}, brute force {w}"
                );
            }
        }
    }

    /// A chessboard is point-symmetric about each corner, so with the corner
    /// on a pixel centre the rendered image is exactly symmetric about it:
    /// the true corner is where the lattice says it is.
    #[test]
    fn area_board_is_point_symmetric_about_corner() {
        const SIZE: usize = 21;
        let img = area_chessboard(SIZE, 6, (10.0, 10.0), 60f32.to_radians(), 40, 200);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let mirror = (SIZE - 1 - y) * SIZE + (SIZE - 1 - x);
                assert_eq!(img[y * SIZE + x], img[mirror], "pixel ({x}, {y})");
            }
        }
        // Not trivially symmetric: the board has both levels.
        assert!(img.contains(&40) && img.contains(&200));
    }

    /// The 3-argument anti-aliased fixtures the benches/tests used hard-coded
    /// 30/230 levels; assert the 5-argument [`aa_chessboard`] reproduces them.
    #[test]
    fn aa_board_30_230_matches_three_arg_form() {
        const SUPER3: usize = 8;
        let three_arg = |size: usize, cell: usize, offset: (f32, f32)| -> Vec<u8> {
            let (ox, oy) = offset;
            let c = cell as f32;
            let inv = 1.0 / (SUPER3 * SUPER3) as f32;
            let mut img = vec![0u8; size * size];
            for y in 0..size {
                for x in 0..size {
                    let mut acc = 0.0f32;
                    for sy in 0..SUPER3 {
                        let yf = y as f32 + (sy as f32 + 0.5) / SUPER3 as f32 - 0.5;
                        let cy = ((yf - oy) / c).floor() as i32;
                        for sx in 0..SUPER3 {
                            let xf = x as f32 + (sx as f32 + 0.5) / SUPER3 as f32 - 0.5;
                            let cx = ((xf - ox) / c).floor() as i32;
                            let dark = (cx + cy).rem_euclid(2) == 0;
                            acc += if dark { 30.0 } else { 230.0 };
                        }
                    }
                    img[y * size + x] = (acc * inv).round().clamp(0.0, 255.0) as u8;
                }
            }
            img
        };
        let off = (12.0 / 2.0 + 0.31, 12.0 / 2.0 + 0.47);
        assert_eq!(aa_chessboard(64, 12, off, 30, 230), three_arg(64, 12, off));
    }
}
