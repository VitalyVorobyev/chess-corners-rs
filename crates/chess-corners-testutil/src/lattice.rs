//! Chessboards with exactly known corners, rendered by exact integration.
//!
//! A [`Lattice`] places the corners; two image formations render it:
//!
//! - [`lattice_area_chessboard`] (and [`area_chessboard`]) — each pixel is
//!   the mean of the hard pattern over its unit square: an ideally sharp lens
//!   on a box-filtering sensor. Oblique sharp edges alias in this formation;
//!   blurring its output afterwards keeps that aliasing.
//! - [`optical_chessboard`] — the pattern is blurred by a Gaussian lens PSF
//!   and *then* integrated over each pixel, the order a camera forms an image
//!   in. Use it wherever a fixture stands for a real camera.

/// Chessboard lattice in image coordinates (px; pixel centres on integers,
/// y down).
///
/// Corners sit at `origin + cell·(k·e1 + m·e2)` for integers `k`, `m`, with
/// `e1 = (cos rotation, sin rotation)` and
/// `e2 = (cos(rotation + axis_angle), sin(rotation + axis_angle))` (radians).
/// `axis_angle = π/2` is a square board; other values shear it. Cell parity
/// matches [`aa_chessboard`](crate::aa_chessboard) for an unrotated square
/// board.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    /// Cell side in px.
    pub cell: f32,
    /// Position of one corner.
    pub origin: (f32, f32),
    /// Angle of `e1` from the image x axis, radians.
    pub rotation: f32,
    /// Angle from `e1` to `e2`, radians, in `(0, π)`.
    pub axis_angle: f32,
}

impl Lattice {
    /// Square, unrotated board with `cell`-px cells and a corner at `origin`.
    pub fn new(cell: f32, origin: (f32, f32)) -> Self {
        Self {
            cell,
            origin,
            rotation: 0.0,
            axis_angle: core::f32::consts::FRAC_PI_2,
        }
    }

    /// The same board rotated by `rotation` radians.
    pub fn rotated(self, rotation: f32) -> Self {
        Self { rotation, ..self }
    }

    /// The same board with `axis_angle` radians between its axes.
    pub fn with_axis_angle(self, axis_angle: f32) -> Self {
        Self { axis_angle, ..self }
    }

    /// The corner nearest to `(x, y)`.
    pub fn nearest_corner(&self, x: f32, y: f32) -> (f32, f32) {
        let g = Geometry::new(self);
        let (dx, dy) = (f64::from(x) - g.origin[0], f64::from(y) - g.origin[1]);
        let k = (g.grad[0][0] * dx + g.grad[0][1] * dy).round();
        let m = (g.grad[1][0] * dx + g.grad[1][1] * dy).round();
        let (cx, cy) = g.corner(k, m);
        (cx as f32, cy as f32)
    }
}

/// [`lattice_area_chessboard`] of an unrotated board whose axes are
/// `axis_angle` radians apart, with a corner at `offset`: axis-aligned for
/// `axis_angle = π/2`, otherwise with the vertical edge family sheared.
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
    let lattice = Lattice::new(cell as f32, offset).with_axis_angle(axis_angle);
    lattice_area_chessboard(size, &lattice, dark, bright)
}

/// `size × size` chessboard whose pixels are the exact mean of the hard
/// pattern over their unit square (an ideally sharp lens on a box-filtering
/// sensor), rounded to `u8`; `dark` where the cell parity is even.
///
/// The edge positions carry no sub-sample quantization, so a refiner's error
/// against [`Lattice`] corners is its own error plus the 8-bit rounding.
///
/// # Panics
///
/// If `lattice.axis_angle` is not in `(0, π)`.
pub fn lattice_area_chessboard(size: usize, lattice: &Lattice, dark: u8, bright: u8) -> Vec<u8> {
    to_levels(&area_pattern(size, lattice), dark, bright)
}

/// `size × size` chessboard formed as a camera forms it: the hard pattern
/// blurred by an isotropic Gaussian lens PSF of standard deviation
/// `psf_sigma` px, then integrated over each pixel's unit square, rounded to
/// `u8`. `psf_sigma = 0` is [`lattice_area_chessboard`].
///
/// Both integrals are evaluated exactly up to a numerical quadrature whose
/// error is far below the 8-bit rounding (each pixel is the pattern weighted
/// by the PSF-blurred pixel aperture; the integral across each image row is
/// closed-form, the one down the rows is Gauss–Legendre on the pieces where
/// it is smooth). The PSF is truncated at 5σ.
///
/// # Panics
///
/// If `lattice.axis_angle` is not in `(0, π)` or `psf_sigma` is negative or
/// not finite.
pub fn optical_chessboard(
    size: usize,
    lattice: &Lattice,
    psf_sigma: f32,
    dark: u8,
    bright: u8,
) -> Vec<u8> {
    to_levels(&optical_pattern(size, lattice, psf_sigma), dark, bright)
}

/// Pattern means in `[-1, 1]` (+1 on dark cells) to `u8` levels.
fn to_levels(pattern: &[f64], dark: u8, bright: u8) -> Vec<u8> {
    let mid = 0.5 * (f64::from(dark) + f64::from(bright));
    let half = 0.5 * (f64::from(bright) - f64::from(dark));
    pattern
        .iter()
        .map(|&v| (mid - half * v).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Per-pixel mean of `s(u1)·s(u2)` over the unit square.
fn area_pattern(size: usize, lattice: &Lattice) -> Vec<f64> {
    let g = Geometry::new(lattice);
    let mut scratch = (Vec::new(), Vec::new());
    let mut out = Vec::with_capacity(size * size);
    for py in 0..size {
        for px in 0..size {
            out.push(g.pixel_mean(px as f64, py as f64, &mut scratch));
        }
    }
    out
}

/// Per-pixel `∫ H(p − n)·s(u1)s(u2) dp` with the PSF-blurred aperture `H`.
fn optical_pattern(size: usize, lattice: &Lattice, psf_sigma: f32) -> Vec<f64> {
    assert!(
        psf_sigma.is_finite() && psf_sigma >= 0.0,
        "psf_sigma {psf_sigma} must be finite and non-negative"
    );
    if psf_sigma == 0.0 {
        return area_pattern(size, lattice);
    }
    let g = Geometry::new(lattice);
    let aperture = Aperture::new(f64::from(psf_sigma));
    let mut scratch = (Vec::new(), Vec::new());
    let mut out = Vec::with_capacity(size * size);
    for py in 0..size {
        for px in 0..size {
            out.push(g.pixel_response(&aperture, px as f64, py as f64, &mut scratch));
        }
    }
    out
}

/// The lattice as `s(u1)·s(u2)`, `uᵢ = gradᵢ · (p − origin)`, with
/// [`square`] waves `s`.
struct Geometry {
    origin: [f64; 2],
    /// `∇uᵢ`: rows of `(cell·[e1 e2])⁻¹`.
    grad: [[f64; 2]; 2],
    /// `cell·e1`, `cell·e2`.
    step: [[f64; 2]; 2],
}

impl Geometry {
    fn new(l: &Lattice) -> Self {
        let alpha = f64::from(l.axis_angle);
        assert!(
            alpha > 0.0 && alpha < core::f64::consts::PI,
            "axis_angle {} outside (0, π)",
            l.axis_angle
        );
        let c = f64::from(l.cell);
        let r = f64::from(l.rotation);
        let e1 = [r.cos(), r.sin()];
        let e2 = [(r + alpha).cos(), (r + alpha).sin()];
        // det [e1 e2] = sin α.
        let det = c * alpha.sin();
        Self {
            origin: [f64::from(l.origin.0), f64::from(l.origin.1)],
            grad: [[e2[1] / det, -e2[0] / det], [-e1[1] / det, e1[0] / det]],
            step: [e1.map(|v| c * v), e2.map(|v| c * v)],
        }
    }

    #[inline]
    fn u(&self, i: usize, x: f64, y: f64) -> f64 {
        self.grad[i][0] * (x - self.origin[0]) + self.grad[i][1] * (y - self.origin[1])
    }

    #[inline]
    fn corner(&self, k: f64, m: f64) -> (f64, f64) {
        (
            self.origin[0] + k * self.step[0][0] + m * self.step[1][0],
            self.origin[1] + k * self.step[0][1] + m * self.step[1][1],
        )
    }

    #[inline]
    fn pattern(&self, x: f64, y: f64) -> f64 {
        square(self.u(0, x, y)) * square(self.u(1, x, y))
    }

    /// Rows `y` in `(y0, y1)` where the integral across `[x0, x1]` stops
    /// being smooth: an edge crosses `x = x0` or `x = x1`, or a lattice
    /// corner lies inside the rectangle. Includes `y0` and `y1`; sorted.
    fn row_breaks(&self, x0: f64, x1: f64, y0: f64, y1: f64, ys: &mut Vec<f64>) {
        ys.clear();
        ys.extend([y0, y1]);
        for i in 0..2 {
            for xb in [x0, x1] {
                push_integer_crossings(ys, y0, y1, |y| self.u(i, xb, y));
            }
        }
        let us = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];
        let range = |i: usize| {
            let v = us.map(|[x, y]| self.u(i, x, y));
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min).ceil();
            let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max).floor();
            (lo as i64, hi as i64)
        };
        let ((k0, k1), (m0, m1)) = (range(0), range(1));
        for k in k0..=k1 {
            for m in m0..=m1 {
                let (x, y) = self.corner(k as f64, m as f64);
                if x > x0 && x < x1 && y > y0 && y < y1 {
                    ys.push(y);
                }
            }
        }
        ys.sort_by(f64::total_cmp);
    }

    /// Edge crossings of row `y` in `(x0, x1)`, with `x0` and `x1`; sorted.
    fn column_breaks(&self, x0: f64, x1: f64, y: f64, xs: &mut Vec<f64>) {
        xs.clear();
        xs.extend([x0, x1]);
        for i in 0..2 {
            push_integer_crossings(xs, x0, x1, |x| self.u(i, x, y));
        }
        xs.sort_by(f64::total_cmp);
    }

    /// Mean of the pattern over the unit square centred on `(cx, cy)`.
    ///
    /// The integral across a row is exact (the pattern is piecewise constant
    /// between the row's edge crossings) and linear in `y` between the
    /// [`row_breaks`](Self::row_breaks), so the midpoint rule over `y` is
    /// exact too.
    fn pixel_mean(&self, cx: f64, cy: f64, scratch: &mut (Vec<f64>, Vec<f64>)) -> f64 {
        let (ys, xs) = scratch;
        let (x0, x1) = (cx - 0.5, cx + 0.5);
        self.row_breaks(x0, x1, cy - 0.5, cy + 0.5, ys);
        let mut sum = 0.0;
        for seg in ys.windows(2) {
            let ym = 0.5 * (seg[0] + seg[1]);
            self.column_breaks(x0, x1, ym, xs);
            let row: f64 = xs
                .windows(2)
                .map(|s| self.pattern(0.5 * (s[0] + s[1]), ym) * (s[1] - s[0]))
                .sum();
            sum += row * (seg[1] - seg[0]);
        }
        sum
    }

    /// `∫∫ h(x − cx)·h(y − cy)·pattern(x, y) dx dy` for the blurred aperture
    /// `h` of `ap`.
    ///
    /// Across a row the pattern is piecewise constant, so the `x` integral
    /// is a sum of differences of `h`'s antiderivative. Down the rows that
    /// integral is smooth between the [`row_breaks`](Self::row_breaks), and
    /// so is `h`; each piece is split to at most `σ/2` and integrated by
    /// 6-point Gauss–Legendre.
    fn pixel_response(
        &self,
        ap: &Aperture,
        cx: f64,
        cy: f64,
        scratch: &mut (Vec<f64>, Vec<f64>),
    ) -> f64 {
        let (ys, xs) = scratch;
        let (x0, x1) = (cx - ap.reach, cx + ap.reach);
        self.row_breaks(x0, x1, cy - ap.reach, cy + ap.reach, ys);
        let mut sum = 0.0;
        for seg in ys.windows(2) {
            let n = ((seg[1] - seg[0]) / ap.piece).ceil().max(1.0);
            let len = (seg[1] - seg[0]) / n;
            for p in 0..n as usize {
                let mid = seg[0] + (p as f64 + 0.5) * len;
                for (node, weight) in GAUSS_LEGENDRE_6 {
                    let y = mid + 0.5 * len * node;
                    self.column_breaks(x0, x1, y, xs);
                    let mut row = 0.0;
                    let mut prev = ap.cumulative(xs[0] - cx);
                    for s in xs.windows(2) {
                        let next = ap.cumulative(s[1] - cx);
                        row += self.pattern(0.5 * (s[0] + s[1]), y) * (next - prev);
                        prev = next;
                    }
                    sum += 0.5 * len * weight * ap.density(y - cy) * row;
                }
            }
        }
        sum
    }
}

/// 6-point Gauss–Legendre nodes and weights on `[-1, 1]`.
const GAUSS_LEGENDRE_6: [(f64, f64); 6] = [
    (-0.932_469_514_203_152, 0.171_324_492_379_170),
    (-0.661_209_386_466_265, 0.360_761_573_048_139),
    (-0.238_619_186_083_197, 0.467_913_934_572_691),
    (0.238_619_186_083_197, 0.467_913_934_572_691),
    (0.661_209_386_466_265, 0.360_761_573_048_139),
    (0.932_469_514_203_152, 0.171_324_492_379_170),
];

/// One axis of a unit pixel aperture blurred by a Gaussian PSF:
/// `h(t) = Φ((t + ½)/σ) − Φ((t − ½)/σ)`, truncated at `|t| ≤ ½ + 5σ`.
struct Aperture {
    sigma: f64,
    reach: f64,
    /// Longest piece of a row-integral interval given to one Gauss–Legendre
    /// rule.
    piece: f64,
}

impl Aperture {
    fn new(sigma: f64) -> Self {
        Self {
            sigma,
            reach: 0.5 + 5.0 * sigma,
            piece: 0.5 * sigma,
        }
    }

    /// `h(t)`.
    #[inline]
    fn density(&self, t: f64) -> f64 {
        normal_cdf((t + 0.5) / self.sigma) - normal_cdf((t - 0.5) / self.sigma)
    }

    /// `∫_{−∞}^{t} h`, from `∫ Φ(τ) dτ = τΦ(τ) + φ(τ)`.
    #[inline]
    fn cumulative(&self, t: f64) -> f64 {
        let psi = |tau: f64| tau * normal_cdf(tau) + normal_pdf(tau);
        self.sigma * (psi((t + 0.5) / self.sigma) - psi((t - 0.5) / self.sigma))
    }
}

#[inline]
fn normal_pdf(t: f64) -> f64 {
    (-0.5 * t * t).exp() / (2.0 * core::f64::consts::PI).sqrt()
}

#[inline]
fn normal_cdf(t: f64) -> f64 {
    0.5 * (1.0 + erf(t / core::f64::consts::SQRT_2))
}

/// Error function, absolute error below 1.5·10⁻⁷ (Abramowitz & Stegun
/// 7.1.26); exactly odd.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    y.copysign(x)
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

/// Push every `t` in `(t0, t1)` where the affine `f(t)` takes an integer
/// value.
fn push_integer_crossings(out: &mut Vec<f64>, t0: f64, t1: f64, f: impl Fn(f64) -> f64) {
    let (f0, f1) = (f(t0), f(t1));
    if f0 == f1 {
        return;
    }
    let (lo, hi) = (f0.min(f1), f0.max(f1));
    let mut k = lo.ceil();
    while k <= hi {
        let t = t0 + (k - f0) / (f1 - f0) * (t1 - t0);
        if t > t0 && t < t1 {
            out.push(t);
        }
        k += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEG: f32 = core::f32::consts::PI / 180.0;

    /// Brute-force pixel mean over a `n × n` sub-sample grid, the reference
    /// for [`area_pattern`] (converges to the exact area as `n` grows).
    fn supersampled_area(size: usize, lattice: &Lattice, n: usize) -> Vec<f64> {
        let g = Geometry::new(lattice);
        let mut out = vec![0.0; size * size];
        for py in 0..size {
            for px in 0..size {
                let mut acc = 0.0;
                for sy in 0..n {
                    let y = py as f64 + (sy as f64 + 0.5) / n as f64 - 0.5;
                    for sx in 0..n {
                        let x = px as f64 + (sx as f64 + 0.5) / n as f64 - 0.5;
                        acc += g.pattern(x, y);
                    }
                }
                out[py * size + px] = acc / (n * n) as f64;
            }
        }
        out
    }

    /// Axis-aligned, sheared and rotated boards agree with a
    /// 256×256-sub-sample brute force to within the final 8-bit rounding
    /// (0.5) plus the brute force's own edge error (under 1/512 px of a
    /// 160-level step, 0.31). An edge misplaced by 0.01 px would show as a
    /// 1.6-level error.
    #[test]
    fn area_board_matches_brute_force() {
        const SIZE: usize = 14;
        for lattice in [
            Lattice::new(5.0, (6.3, 7.55)),
            Lattice::new(5.0, (6.71, 6.2)).with_axis_angle(60.0 * DEG),
            Lattice::new(5.0, (7.05, 6.9)).with_axis_angle(115.0 * DEG),
            Lattice::new(5.0, (6.37, 6.81)).rotated(8.0 * DEG),
            Lattice::new(5.0, (7.12, 6.44)).rotated(45.0 * DEG),
            Lattice::new(5.0, (6.58, 7.23))
                .rotated(-23.0 * DEG)
                .with_axis_angle(70.0 * DEG),
        ] {
            let got = lattice_area_chessboard(SIZE, &lattice, 40, 200);
            let want = supersampled_area(SIZE, &lattice, 256);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let w = 120.0 - 80.0 * w;
                assert!(
                    (f64::from(g) - w).abs() <= 0.5 + 0.32,
                    "{lattice:?}: pixel {i} = {g}, brute force {w}"
                );
            }
        }
    }

    /// A chessboard is point-symmetric about each corner, so with the corner
    /// on a pixel centre the rendered image is symmetric about it, in both
    /// formations: the true corner is where the lattice says it is.
    #[test]
    fn boards_are_point_symmetric_about_corner() {
        const SIZE: usize = 21;
        for lattice in [
            Lattice::new(6.0, (10.0, 10.0)).with_axis_angle(60.0 * DEG),
            Lattice::new(6.0, (10.0, 10.0)).rotated(8.0 * DEG),
        ] {
            let area = area_pattern(SIZE, &lattice);
            let optical = optical_pattern(SIZE, &lattice, 0.7);
            for img in [&area, &optical] {
                for k in 0..SIZE * SIZE {
                    let mirror = SIZE * SIZE - 1 - k;
                    assert!((img[k] - img[mirror]).abs() < 1e-9, "{lattice:?} pixel {k}");
                }
            }
            // Not trivially symmetric: the board has both levels.
            assert!(area.contains(&1.0) && area.contains(&-1.0));
        }
    }

    /// On an unrotated square board the pattern and the blurred aperture are
    /// both separable, so each pixel is the product of two 1-D integrals
    /// `∫ h(t − n)·s((t − o)/c) dt`, each a sum over the edges. The 2-D
    /// quadrature reproduces them.
    #[test]
    fn optical_board_matches_separable_form() {
        const SIZE: usize = 16;
        let (cell, origin, sigma) = (5.0f64, (7.3f64, 8.62f64), 0.8f64);
        let ap = Aperture::new(sigma);
        let one_d = |n: f64, o: f64| {
            // Edges at o + k·c; s = +1 on [o, o + c).
            let (lo, hi) = (n - ap.reach, n + ap.reach);
            let mut xs = vec![lo, hi];
            push_integer_crossings(&mut xs, lo, hi, |t| (t - o) / cell);
            xs.sort_by(f64::total_cmp);
            xs.windows(2)
                .map(|s| {
                    let m = 0.5 * (s[0] + s[1]);
                    square((m - o) / cell) * (ap.cumulative(s[1] - n) - ap.cumulative(s[0] - n))
                })
                .sum::<f64>()
        };
        let lattice = Lattice::new(cell as f32, (origin.0 as f32, origin.1 as f32));
        let got = optical_pattern(SIZE, &lattice, sigma as f32);
        let o = (f64::from(lattice.origin.0), f64::from(lattice.origin.1));
        for py in 0..SIZE {
            for px in 0..SIZE {
                let want = one_d(px as f64, o.0) * one_d(py as f64, o.1);
                let g = got[py * SIZE + px];
                assert!((g - want).abs() < 1e-6, "pixel ({px}, {py}): {g} vs {want}");
            }
        }
    }

    /// Rotated and sheared boards against a brute force: the blurred
    /// aperture `h(x)·h(y)` integrated against the pattern on a fine
    /// midpoint grid of 1/128 px (an edge cut by the grid moves its weight by
    /// at most half a step, so the brute force is good to a few 10⁻³).
    #[test]
    fn optical_board_matches_brute_force() {
        let sigma = 0.6f64;
        let ap = Aperture::new(sigma);
        let n = 128usize;
        for lattice in [
            Lattice::new(5.0, (3.31, 3.77)).rotated(8.0 * DEG),
            Lattice::new(5.0, (3.62, 3.18))
                .rotated(30.0 * DEG)
                .with_axis_angle(65.0 * DEG),
        ] {
            let g = Geometry::new(&lattice);
            let got = optical_pattern(7, &lattice, sigma as f32);
            for (px, py) in [(3usize, 3usize), (4, 2), (2, 5)] {
                let (cx, cy) = (px as f64, py as f64);
                let steps = (2.0 * ap.reach * n as f64).ceil() as usize;
                let dt = 2.0 * ap.reach / steps as f64;
                let w: Vec<f64> = (0..steps)
                    .map(|i| ap.density(-ap.reach + (i as f64 + 0.5) * dt) * dt)
                    .collect();
                let mut want = 0.0;
                for (j, wy) in w.iter().enumerate() {
                    let y = cy - ap.reach + (j as f64 + 0.5) * dt;
                    for (i, wx) in w.iter().enumerate() {
                        let x = cx - ap.reach + (i as f64 + 0.5) * dt;
                        want += wx * wy * g.pattern(x, y);
                    }
                }
                let got = got[py * 7 + px];
                assert!(
                    (got - want).abs() < 4e-3,
                    "{lattice:?} pixel ({px}, {py}): {got} vs brute force {want}"
                );
            }
        }
    }

    /// The Gauss–Legendre pieces are short enough: four times shorter ones
    /// change no pixel by more than 10⁻⁷ of the contrast, from a sharp PSF
    /// to a wide one.
    #[test]
    fn optical_quadrature_has_converged() {
        let lattice = Lattice::new(6.0, (5.37, 5.81))
            .rotated(11.0 * DEG)
            .with_axis_angle(70.0 * DEG);
        let g = Geometry::new(&lattice);
        let mut scratch = (Vec::new(), Vec::new());
        for sigma in [0.3, 0.8, 2.5] {
            let ap = Aperture::new(sigma);
            let fine = Aperture {
                piece: 0.25 * ap.piece,
                ..Aperture::new(sigma)
            };
            for (px, py) in [(5.0, 5.0), (6.0, 4.0), (3.0, 7.0), (8.0, 8.0)] {
                let a = g.pixel_response(&ap, px, py, &mut scratch);
                let b = g.pixel_response(&fine, px, py, &mut scratch);
                assert!((a - b).abs() < 2e-7, "σ {sigma} ({px}, {py}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn nearest_corner_finds_the_lattice_corner() {
        let l = Lattice::new(8.0, (20.3, 19.6))
            .rotated(12.0 * DEG)
            .with_axis_angle(75.0 * DEG);
        let g = Geometry::new(&l);
        for (k, m) in [(0.0, 0.0), (2.0, -1.0), (-3.0, 4.0)] {
            let (x, y) = g.corner(k, m);
            let (nx, ny) = l.nearest_corner(x as f32 + 0.9, y as f32 - 1.1);
            assert!((f64::from(nx) - x).abs() < 1e-4 && (f64::from(ny) - y).abs() < 1e-4);
        }
    }
}
