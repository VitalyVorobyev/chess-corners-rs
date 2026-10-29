//! Test-only evaluation of the position covariance of every localizer:
//! Monte Carlo scatter against the propagated noise term, the whitened
//! error variance factor σ̂₀² on camera-formed synthetic boards, and the
//! per-corner records the systematic models are calibrated from.
//!
//! Boards come from two independent generators: the exact lens-blur
//! renderer `optical_chessboard`, rendered here, and the supersampled
//! generator of `tools/refiner_robustness/synth.py`, whose noise-free
//! irradiance images `tools/refiner_robustness/export_uq24.py` writes to a
//! directory (`UQ24_SYNTH_DIR`); tests that need them skip when it is unset.

use crate::detect::radon::{detect_peaks_from_radon, radon_response_u8, RadonBuffers};
use crate::uncertainty::{edge_width, PixelTile, ROUNDING_VAR};
use crate::{
    chess_response_u8, detect_peaks_from_response_with_refine_radius, CenterOfMassConfig,
    ChessParams, Corner, CornerRefiner, DenseDetector, ImageView, PositionCovariance,
    RadonDetector, RadonDetectorParams, RefineContext, RefineStatus, Refiner, RefinerKind,
    SaddlePointConfig,
};
use chess_corners_testutil::{optical_chessboard, Lattice};
use std::path::PathBuf;

/// A localizer as the facade composes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Localizer {
    CenterOfMass,
    SaddlePoint,
    RadonPeak,
    RadonSaddle,
}

impl Localizer {
    pub(crate) const ALL: [Localizer; 4] = [
        Localizer::CenterOfMass,
        Localizer::SaddlePoint,
        Localizer::RadonPeak,
        Localizer::RadonSaddle,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Localizer::CenterOfMass => "CoM",
            Localizer::SaddlePoint => "SaddlePoint",
            Localizer::RadonPeak => "RadonPeak",
            Localizer::RadonSaddle => "RadonSaddle",
        }
    }
}

/// One localized corner with the observables of its systematic model.
#[derive(Clone, Debug)]
pub(crate) struct Det {
    pub corner: Corner,
    /// Centre of the localizer's fit window (input px).
    pub centre: [f32; 2],
    /// [`edge_width`] at the window.
    pub width: Option<f32>,
}

impl Det {
    pub(crate) fn xy(&self) -> [f32; 2] {
        [self.corner.x, self.corner.y]
    }
}

fn chess_params(kind: RefinerKind) -> ChessParams {
    ChessParams {
        // The facade's ChESS preset.
        threshold: 30.0,
        refiner: kind,
        orientation_method: None,
        ..ChessParams::default()
    }
}

/// Localize every corner of `img` with `loc`, reporting a covariance when
/// `covariance` is set.
pub(crate) fn localize(
    img: &[u8],
    w: usize,
    h: usize,
    loc: Localizer,
    covariance: bool,
) -> Vec<Det> {
    let view = ImageView::from_u8_slice(w, h, img).unwrap();
    let chess_kind = match loc {
        Localizer::CenterOfMass => Some(RefinerKind::CenterOfMass(CenterOfMassConfig::default())),
        Localizer::SaddlePoint => Some(RefinerKind::SaddlePoint(SaddlePointConfig::default())),
        _ => None,
    };
    if let Some(kind) = chess_kind {
        let params = chess_params(kind.clone());
        let resp = chess_response_u8(img, w, h, &params);
        let mut refiner = Refiner::from_kind(kind).with_covariance(covariance);
        let peaks = detect_peaks_from_response_with_refine_radius(&resp, &params, refiner.radius());
        let ctx = RefineContext::new(Some(view), Some(&resp));
        let mut out = Vec::new();
        let mut tile = PixelTile::default();
        for p in peaks {
            let res = refiner.refine([p.x, p.y], ctx);
            if res.status != RefineStatus::Accepted {
                continue;
            }
            let mut corner = Corner::new(res.x, res.y, p.strength);
            corner.model = res.model;
            let centre = if loc == Localizer::SaddlePoint {
                [res.x.round(), res.y.round()]
            } else {
                [p.x.round(), p.y.round()]
            };
            let width = edge_width(&view, centre.map(|v| v as i32), &mut tile);
            out.push(Det {
                corner,
                centre,
                width,
            });
        }
        return out;
    }
    let params = RadonDetectorParams::default();
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(img, w, h, &params, &mut buffers);
    let mut peaks = detect_peaks_from_radon(&resp, &params);
    if loc == Localizer::RadonSaddle {
        let mut sp = Refiner::from_kind(RefinerKind::SaddlePoint(SaddlePointConfig::default()))
            .with_covariance(covariance);
        peaks = RadonDetector.refine_peaks_on_image(peaks, view, &resp, &mut sp);
    }
    if covariance {
        RadonDetector.attach_peak_covariance(&mut peaks, view, &params, &mut buffers);
    }
    let mut tile = PixelTile::default();
    let up = params.image_upsample_clamped() as f32;
    peaks
        .into_iter()
        .map(|corner| {
            let centre = if corner.at_peak_fit {
                [corner.x, corner.y].map(|v| (v * up).round() / up)
            } else {
                [corner.x.round(), corner.y.round()]
            };
            let width = edge_width(
                &view,
                [corner.x.round() as i32, corner.y.round() as i32],
                &mut tile,
            );
            Det {
                corner,
                centre,
                width,
            }
        })
        .collect()
}

/// A synthetic board with exactly known corners.
pub(crate) trait Scene {
    fn size(&self) -> (usize, usize);
    /// Noise-free 8-bit image.
    fn image(&self) -> &[u8];
    /// The true corner nearest `xy`, if one is within `radius` px.
    fn truth(&self, xy: [f32; 2], radius: f32) -> Option<[f32; 2]>;
}

/// An `optical_chessboard`.
pub(crate) struct OpticalScene {
    pub side: usize,
    pub lattice: Lattice,
    pub img: Vec<u8>,
}

impl OpticalScene {
    pub(crate) fn new(side: usize, lattice: Lattice, blur: f32) -> Self {
        let img = optical_chessboard(side, &lattice, blur, 40, 200);
        Self { side, lattice, img }
    }
}

impl Scene for OpticalScene {
    fn size(&self) -> (usize, usize) {
        (self.side, self.side)
    }
    fn image(&self) -> &[u8] {
        &self.img
    }
    fn truth(&self, xy: [f32; 2], radius: f32) -> Option<[f32; 2]> {
        let (tx, ty) = self.lattice.nearest_corner(xy[0], xy[1]);
        ((tx - xy[0]).hypot(ty - xy[1]) <= radius).then_some([tx, ty])
    }
}

/// A `synth.py` irradiance image exported by `export_uq24.py`: rounded to
/// 8 bits here, truth corners from its CSV.
pub(crate) struct ExportedScene {
    pub name: String,
    pub w: usize,
    pub h: usize,
    pub img: Vec<u8>,
    pub corners: Vec<[f32; 2]>,
}

impl Scene for ExportedScene {
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    fn image(&self) -> &[u8] {
        &self.img
    }
    fn truth(&self, xy: [f32; 2], radius: f32) -> Option<[f32; 2]> {
        self.corners
            .iter()
            .copied()
            .map(|c| (c, (c[0] - xy[0]).hypot(c[1] - xy[1])))
            .filter(|(_, d)| *d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _)| c)
    }
}

/// The exported `synth.py` scenes in `UQ24_SYNTH_DIR`, or `None` when it is
/// unset. Each scene is `<name>.f32` (little-endian irradiance, row-major),
/// `<name>.json` (`{"w", "h"}`) and `<name>.csv` (`x,y` truth corners).
pub(crate) fn exported_scenes(filter: impl Fn(&str) -> bool) -> Option<Vec<ExportedScene>> {
    let dir = PathBuf::from(std::env::var_os("UQ24_SYNTH_DIR")?);
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "f32").then(|| p.file_stem()?.to_str().map(String::from))?
        })
        .filter(|n| filter(n))
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).ok()?)
                .ok()?;
        let w = meta["w"].as_u64()? as usize;
        let h = meta["h"].as_u64()? as usize;
        let raw = std::fs::read(dir.join(format!("{name}.f32"))).ok()?;
        let img = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b).round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<_>>();
        assert_eq!(img.len(), w * h, "{name}");
        let corners = std::fs::read_to_string(dir.join(format!("{name}.csv")))
            .ok()?
            .lines()
            .filter_map(|l| {
                let mut it = l.split(',').map(|v| v.trim().parse::<f32>());
                Some([it.next()?.ok()?, it.next()?.ok()?])
            })
            .collect();
        out.push(ExportedScene {
            name,
            w,
            h,
            img,
            corners,
        });
    }
    Some(out)
}

/// Directory for local evaluation output (`bench_results/uq24`, gitignored),
/// or `UQ24_OUT`.
pub(crate) fn out_dir() -> PathBuf {
    let dir = std::env::var_os("UQ24_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench_results/uq24")
        });
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Gaussian noise of σ, then 8-bit rounding, on a copy of `clean`.
pub(crate) fn noisy(clean: &[u8], sigma: f32, seed: u64) -> Vec<u8> {
    let mut img = clean.to_vec();
    chess_corners_testutil::add_gaussian_noise(&mut img, sigma, seed);
    img
}

/// `eᵀ Σ⁻¹ e`.
pub(crate) fn mahalanobis(e: [f64; 2], c: PositionCovariance) -> f64 {
    let (a, b, d) = (f64::from(c.xx), f64::from(c.xy), f64::from(c.yy));
    let det = a * d - b * b;
    (d * e[0] * e[0] - 2.0 * b * e[0] * e[1] + a * e[1] * e[1]) / det
}

/// Corners at least `margin` px inside the image.
pub(crate) fn interior(xy: [f32; 2], size: (usize, usize), margin: f32) -> bool {
    xy[0] >= margin
        && xy[1] >= margin
        && xy[0] <= size.0 as f32 - 1.0 - margin
        && xy[1] <= size.1 as f32 - 1.0 - margin
}

/// Monte Carlo noise scatter of one scene and localizer: over `draws`
/// noisy copies, each truth corner's estimates about their own mean,
/// whitened by the predicted noise term at `σ² + 1/12`, the total noise
/// of the 8-bit image (with any
/// second-order growth). Returns the
/// std of the whitened components (1 when the noise term is right) and the
/// number of corner-draws.
///
/// Estimates of the ChESS single-shot refiners are grouped by truth corner
/// *and* fit-window centre: noise can move their window by a pixel (the
/// seed peak changes), which moves the estimate by the window-offset pull —
/// a systematic-model effect, not first-order noise scatter. The other
/// localizers are continuous across their window changes (the Radon fit's
/// peak pixel, the re-centred fits), and grouping them by window would
/// truncate their scatter.
pub(crate) fn mc_z_std(
    scene: &dyn Scene,
    loc: Localizer,
    sigma: f32,
    draws: u64,
    seed: u64,
) -> (f64, usize) {
    use std::collections::HashMap;
    let (w, h) = scene.size();

    // (truth corner, window centre) -> (estimates, predicted covariances)
    let mut per: HashMap<[u32; 4], Vec<([f64; 2], PositionCovariance)>> = HashMap::new();
    for d in 0..draws {
        let img = noisy(scene.image(), sigma, seed * 1_000 + d + 1);
        for det in localize(&img, w, h, loc, true) {
            let Some(model) = det.corner.model else {
                continue;
            };
            // The noise term at the draw's total 8-bit noise (Gaussian plus
            // rounding): the σ production takes, so `noise_at`'s clamp is idle.
            let c = model.noise_at((sigma * sigma + ROUNDING_VAR).sqrt());
            if !interior(det.xy(), (w, h), 12.0) {
                continue;
            }
            let Some(t) = scene.truth(det.xy(), 0.75) else {
                continue;
            };
            let by_window = loc == Localizer::CenterOfMass;
            let centre = if by_window { det.centre } else { [0.0; 2] };
            let key = [t[0], t[1], centre[0], centre[1]].map(f32::to_bits);
            per.entry(key)
                .or_default()
                .push((det.xy().map(f64::from), c));
        }
    }
    let (mut q, mut dof) = (0.0f64, 0.0f64);
    let mut n = 0usize;
    for est in per.values() {
        if est.len() < 8 {
            continue;
        }
        let m = est.len() as f64;
        let mean = [0, 1].map(|a| est.iter().map(|e| e.0[a]).sum::<f64>() / m);
        for (xy, c) in est {
            q += mahalanobis([xy[0] - mean[0], xy[1] - mean[1]], *c);
        }
        dof += 2.0 * (m - 1.0);
        n += est.len();
    }
    ((q / dof).sqrt(), n)
}

/// σ̂₀² of one scene and localizer: `mean eᵀ Σ⁻¹ e / 2` over the corners
/// of `draws` noisy copies, with `Σ` the reported total covariance for the
/// noise `sigma_used(img)` (the true σ, or an estimate). Returns
/// `(σ̂₀², corners, fraction with a covariance)`.
pub(crate) fn sigma0_sq(
    scene: &dyn Scene,
    loc: Localizer,
    sigma: f32,
    draws: u64,
    seed: u64,
    sigma_used: &dyn Fn(&[u8]) -> Option<f32>,
) -> (f64, usize, f64) {
    let (w, h) = scene.size();
    let (mut q, mut n, mut total) = (0.0f64, 0usize, 0usize);
    for d in 0..draws {
        let img = noisy(scene.image(), sigma, seed * 1_000 + d + 1);
        let s = sigma_used(&img);
        for det in localize(&img, w, h, loc, true) {
            if !interior(det.xy(), (w, h), 12.0) {
                continue;
            }
            let Some(t) = scene.truth(det.xy(), 0.75) else {
                continue;
            };
            total += 1;
            let Some(cov) = s.and_then(|s| det.corner.position_covariance(s)) else {
                continue;
            };
            let e = [
                f64::from(det.corner.x - t[0]),
                f64::from(det.corner.y - t[1]),
            ];
            q += mahalanobis(e, cov);
            n += 1;
        }
    }
    (q / (2.0 * n as f64), n, n as f64 / total.max(1) as f64)
}

/// Deterministic uniform `[0, 1)` source for board phases.
pub(crate) struct Phases(u64);

impl Phases {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub(crate) fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

const DEG: f32 = core::f32::consts::PI / 180.0;

/// The board of one grid condition: `cell` px, rotated `rot` degrees,
/// sheared to `axis` degrees between its axes, lens blur σ `blur`, with a
/// corner at a pseudo-random phase.
pub(crate) fn optical(
    side: usize,
    cell: f32,
    rot: f32,
    axis: f32,
    blur: f32,
    seed: u64,
) -> OpticalScene {
    let mut ph = Phases::new(seed);
    let c = side as f32 / 2.0;
    let lattice = Lattice::new(cell, (c + ph.next() - 0.5, c + ph.next() - 0.5))
        .rotated(rot * DEG)
        .with_axis_angle(axis * DEG);
    OpticalScene::new(side, lattice, blur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// ±25% about the measured σ̂₀² (cell 10, 8°, blur 1, σ 2): 0.88, 0.08,
    /// 0.18, 0.08. Below 1 by design: each localizer's systematic term
    /// carries a floor measured on real images, larger than the error of
    /// these ideal synthetic corners. The center of mass's error depends on
    /// the board's rotation, which its model does not see (high at 8°).
    const BANDS_A: [(f64, f64); 4] = [
        (0.662, 1.1),
        (0.0585, 0.0975),
        (0.136, 0.226),
        (0.0585, 0.0975),
    ];
    /// ±25% about the measured σ̂₀² (cell 12, 30°, blur 2, σ 4): 0.22, 0.54,
    /// 1.22, 0.53. The Radon peak fit is the least conservative on
    /// rotated, blurred boards, where noise moves its peak between pixels.
    const BANDS_B: [(f64, f64); 4] = [
        (0.164, 0.274),
        (0.407, 0.679),
        (0.918, 1.53),
        (0.399, 0.665),
    ];

    /// Monte Carlo: the propagated noise term predicts the scatter of every
    /// localizer (whitened std within 0.9–1.1), on the lens-blur renderer,
    /// where the first-order model holds for all of them (the Radon peak
    /// fit's flat-topped response at heavy blur and 45° is excluded; see
    /// `mc_sweep`).
    #[test]
    fn noise_term_predicts_scatter() {
        for (cell, rot, blur, sigma) in [(10.0f32, 8.0f32, 1.0f32, 3.0f32), (12.0, 22.0, 0.7, 6.0)]
        {
            let scene = optical(64, cell, rot, 90.0, blur, 7);
            for loc in Localizer::ALL {
                let (z, n) = mc_z_std(&scene, loc, sigma, 40, 1);
                assert!(
                    n > 150,
                    "{} cell {cell} rot {rot}: only {n} corner-draws",
                    loc.name()
                );
                assert!(
                    (0.9..=1.1).contains(&z),
                    "{} cell {cell} rot {rot} blur {blur} σ {sigma}: whitened std {z:.3}",
                    loc.name()
                );
            }
        }
    }

    /// σ̂₀² regression bands per localizer on a fixed subset of the grid
    /// (noise σ supplied). The bands bracket the measured value on these
    /// deterministic boards; the full grid is `sigma0_sweep`.
    #[test]
    fn sigma0_regression_bands() {
        // (cell, rotation, blur, noise σ, [CoM, SaddlePoint, RadonPeak, RadonSaddle] bands)
        type Band = (f64, f64);
        let cases: [(f32, f32, f32, f32, [Band; 4]); 2] = [
            (10.0, 8.0, 1.0, 2.0, BANDS_A),
            (12.0, 30.0, 2.0, 4.0, BANDS_B),
        ];
        let mut outside = Vec::new();
        for (cell, rot, blur, sigma, bands) in cases {
            let scene = optical(80, cell, rot, 90.0, blur, 9);
            for (loc, band) in Localizer::ALL.into_iter().zip(bands) {
                let (s0, n, frac) = sigma0_sq(&scene, loc, sigma, 6, 3, &|_| Some(sigma));
                println!(
                    "{} cell {cell} rot {rot} blur {blur} σ {sigma}: σ̂₀² {s0:.3} n {n} cov {frac:.2}",
                    loc.name()
                );
                assert!(
                    n > 100 && frac > 0.95,
                    "{}: n {n}, covariance on {frac:.2}",
                    loc.name()
                );
                if !(band.0..=band.1).contains(&s0) {
                    outside.push(format!(
                        "{} cell {cell} rot {rot} blur {blur} σ {sigma}: σ̂₀² {s0:.3} outside {band:?}",
                        loc.name()
                    ));
                }
            }
        }
        assert!(outside.is_empty(), "{outside:#?}");
    }

    /// Full Monte Carlo sweep on both generators (the exported `synth.py`
    /// scenes when `UQ24_SYNTH_DIR` is set). A measurement:
    /// `cargo test -p chess-corners-core --release mc_sweep -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mc_sweep() {
        let mut out = std::fs::File::create(out_dir().join("mc.csv")).unwrap();
        writeln!(out, "generator,condition,localizer,sigma,z_std,n").unwrap();
        println!("generator  condition                      localizer     σ    z_std    n");
        for &cell in &[8.0f32, 12.0] {
            for &rot in &[0.0f32, 8.0, 45.0] {
                for &blur in &[0.5f32, 1.5, 2.5] {
                    let scene = optical(96, cell, rot, 90.0, blur, 3);
                    for &sigma in &[1.0f32, 2.0, 8.0] {
                        for loc in Localizer::ALL {
                            let (z, n) = mc_z_std(&scene, loc, sigma, 64, 2);
                            let cond = format!("cell{cell}_rot{rot}_blur{blur}");
                            println!(
                                "optical    {cond:30} {:12} {sigma:4} {z:7.3} {n:6}",
                                loc.name()
                            );
                            writeln!(out, "optical,{cond},{},{sigma},{z},{n}", loc.name()).unwrap();
                        }
                    }
                }
            }
        }
        if let Some(scenes) = exported_scenes(|n| n.starts_with("mc_")) {
            for scene in &scenes {
                for &sigma in &[1.0f32, 2.0, 8.0] {
                    for loc in Localizer::ALL {
                        let (z, n) = mc_z_std(scene, loc, sigma, 32, 2);
                        println!(
                            "synth      {:30} {:12} {sigma:4} {z:7.3} {n:6}",
                            scene.name,
                            loc.name()
                        );
                        writeln!(out, "synth,{},{},{sigma},{z},{n}", scene.name, loc.name())
                            .unwrap();
                    }
                }
            }
        }
    }

    /// σ̂₀² of every localizer over the full grid of both generators, with
    /// the noise σ supplied and estimated per image. A measurement:
    /// `UQ24_SYNTH_DIR=… cargo test -p chess-corners-core --release
    /// sigma0_sweep -- --ignored --nocapture`; writes
    /// `bench_results/uq24/sigma0.csv`.
    #[test]
    #[ignore]
    fn sigma0_sweep() {
        let mut out = std::fs::File::create(out_dir().join("sigma0.csv")).unwrap();
        writeln!(
            out,
            "generator,condition,localizer,sigma,s0_true,s0_est,n,cov_frac"
        )
        .unwrap();
        let est = |img: &[u8], w: usize, h: usize| crate::estimate_noise_sigma_u8(img, w, h);
        let mut run = |gen: &str, cond: &str, scene: &dyn Scene, draws: u64| {
            let (w, h) = scene.size();
            for &sigma in &[1.0f32, 2.0, 8.0] {
                for loc in Localizer::ALL {
                    let (s_true, n, frac) =
                        sigma0_sq(scene, loc, sigma, draws, 5, &|_| Some(sigma));
                    let (s_est, _, _) =
                        sigma0_sq(scene, loc, sigma, draws, 5, &|img| est(img, w, h));
                    println!(
                        "{gen:8} {cond:36} {:12} σ {sigma:3}: s0 {s_true:6.2} (σ est {s_est:6.2}) n {n:5} cov {frac:.2}",
                        loc.name()
                    );
                    writeln!(
                        out,
                        "{gen},{cond},{},{sigma},{s_true},{s_est},{n},{frac}",
                        loc.name()
                    )
                    .unwrap();
                }
            }
        };
        for &cell in &[8.0f32, 12.0] {
            for &rot in &[0.0f32, 8.0, 45.0] {
                for &blur in &[0.5f32, 1.5, 2.5] {
                    let scene = optical(128, cell, rot, 90.0, blur, 11);
                    run(
                        "optical",
                        &format!("cell{cell}_rot{rot}_blur{blur}"),
                        &scene,
                        6,
                    );
                }
            }
        }
        if let Some(scenes) = exported_scenes(|n| n.starts_with("s0_")) {
            for scene in &scenes {
                run("synth", &scene.name.clone(), scene, 6);
            }
        }
    }

    /// Header of the per-corner covariance records of [`write_records`].
    const RECORD_HEADER: &str =
        "generator,condition,localizer,sigma,draw,x,y,ex,ey,nxx,nxy,nyy,sxx,sxy,syy,ux,uy,width,grid";

    /// One CSV row per corner `img` yields under every localizer: the
    /// position, its error against `scene`'s truth (NaN without a scene),
    /// the noise term at `sigma` (with any growth), the shipped systematic
    /// term, the offset from the fit-window centre, the edge width and the
    /// window grid spacing — everything needed to re-score the covariance
    /// under another systematic model offline.
    fn write_records(
        out: &mut impl Write,
        tag: [&str; 2],
        img: &[u8],
        (w, h): (usize, usize),
        sigma: f32,
        draw: u64,
        scene: Option<&dyn Scene>,
    ) {
        let up = RadonDetectorParams::default().image_upsample_clamped() as f32;
        for loc in Localizer::ALL {
            let grid = if matches!(loc, Localizer::RadonPeak | Localizer::RadonSaddle) {
                1.0 / up
            } else {
                1.0
            };
            for det in localize(img, w, h, loc, true) {
                let Some(model) = det.corner.model else {
                    continue;
                };
                let mut e = [f32::NAN; 2];
                if let Some(scene) = scene {
                    if !interior(det.xy(), (w, h), 12.0) {
                        continue;
                    }
                    let Some(t) = scene.truth(det.xy(), 0.75) else {
                        continue;
                    };
                    e = [det.corner.x - t[0], det.corner.y - t[1]];
                }
                let (n, s) = (model.noise_at(sigma), model.systematic);
                writeln!(
                    out,
                    "{},{},{},{sigma},{draw},{},{},{},{},{},{},{},{},{},{},{},{},{},{grid}",
                    tag[0],
                    tag[1],
                    loc.name(),
                    det.corner.x,
                    det.corner.y,
                    e[0],
                    e[1],
                    n.xx,
                    n.xy,
                    n.yy,
                    s.xx,
                    s.xy,
                    s.yy,
                    det.corner.x - det.centre[0],
                    det.corner.y - det.centre[1],
                    det.width.unwrap_or(f32::NAN),
                )
                .unwrap();
            }
        }
    }

    /// Per-corner covariance records on the `sigma0_sweep` grid (same
    /// boards, noise draws and seeds), for scoring alternative systematic
    /// models offline (`tools/refiner_robustness/real_validation/
    /// ablation.py`): `UQ24_SYNTH_DIR=… cargo test -p chess-corners-core
    /// --release dump_ablation_synth -- --ignored`; writes
    /// `bench_results/uq24/ablation_synth.csv`.
    #[test]
    #[ignore]
    fn dump_ablation_synth() {
        let file = std::fs::File::create(out_dir().join("ablation_synth.csv")).unwrap();
        let mut out = std::io::BufWriter::new(file);
        writeln!(out, "{RECORD_HEADER}").unwrap();
        let mut run = |gen: &str, cond: &str, scene: &dyn Scene, draws: u64| {
            let (w, h) = scene.size();
            for &sigma in &[1.0f32, 2.0, 8.0] {
                // `sigma0_sq`'s draws: seed 5.
                for d in 0..draws {
                    let img = noisy(scene.image(), sigma, 5 * 1_000 + d + 1);
                    write_records(&mut out, [gen, cond], &img, (w, h), sigma, d, Some(scene));
                }
            }
        };
        for &cell in &[8.0f32, 12.0] {
            for &rot in &[0.0f32, 8.0, 45.0] {
                for &blur in &[0.5f32, 1.5, 2.5] {
                    let scene = optical(128, cell, rot, 90.0, blur, 11);
                    run(
                        "optical",
                        &format!("cell{cell}_rot{rot}_blur{blur}"),
                        &scene,
                        6,
                    );
                }
            }
        }
        if let Some(scenes) = exported_scenes(|n| n.starts_with("s0_")) {
            for scene in &scenes {
                run("synth", &scene.name.clone(), scene, 6);
            }
        }
    }

    /// Per-corner covariance records of real 8-bit images: every
    /// `<name>.u8` (row-major) with `<name>.json` (`{"w", "h"}`) in
    /// `UQ36_REAL_DIR`, noise σ estimated per image as the detector does.
    /// Writes `$UQ24_OUT/ablation_real.csv` (keep it outside the
    /// repository for private data). `cargo test -p chess-corners-core
    /// --release dump_ablation_real -- --ignored`.
    #[test]
    #[ignore]
    fn dump_ablation_real() {
        let Some(dir) = std::env::var_os("UQ36_REAL_DIR").map(PathBuf::from) else {
            return;
        };
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension()? == "u8").then(|| p.file_stem()?.to_str().map(String::from))?
            })
            .collect();
        names.sort();
        let file = std::fs::File::create(out_dir().join("ablation_real.csv")).unwrap();
        let mut out = std::io::BufWriter::new(file);
        writeln!(out, "{RECORD_HEADER}").unwrap();
        for name in names {
            let meta: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            let (w, h) = (
                meta["w"].as_u64().unwrap() as usize,
                meta["h"].as_u64().unwrap() as usize,
            );
            let img = std::fs::read(dir.join(format!("{name}.u8"))).unwrap();
            assert_eq!(img.len(), w * h, "{name}");
            let Some(sigma) = crate::estimate_noise_sigma_u8(&img, w, h) else {
                continue;
            };
            write_records(&mut out, ["real", &name], &img, (w, h), sigma, 0, None);
        }
    }

    /// Noise-free per-corner records for calibrating the systematic models:
    /// `cargo test -p chess-corners-core --release dump_systematic -- --ignored`.
    /// Writes `bench_results/uq24/systematic.csv` with, per localized
    /// corner, the condition, the window centre, the estimate, the truth
    /// and the edge width.
    #[test]
    #[ignore]
    fn dump_systematic() {
        let mut out = std::fs::File::create(out_dir().join("systematic.csv")).unwrap();
        writeln!(
            out,
            "generator,condition,cell,rot,axis,blur,localizer,cx,cy,x,y,tx,ty,width"
        )
        .unwrap();
        let mut write = |gen: &str, cond: &str, meta: [f32; 4], scene: &dyn Scene| {
            let (w, h) = scene.size();
            for loc in Localizer::ALL {
                for det in localize(scene.image(), w, h, loc, true) {
                    if !interior(det.xy(), (w, h), 12.0) {
                        continue;
                    }
                    let Some(t) = scene.truth(det.xy(), 0.75) else {
                        continue;
                    };
                    writeln!(
                        out,
                        "{gen},{cond},{},{},{},{},{},{},{},{},{},{},{},{}",
                        meta[0],
                        meta[1],
                        meta[2],
                        meta[3],
                        loc.name(),
                        det.centre[0],
                        det.centre[1],
                        det.corner.x,
                        det.corner.y,
                        t[0],
                        t[1],
                        det.width.unwrap_or(f32::NAN),
                    )
                    .unwrap();
                }
            }
        };
        let mut seed = 0u64;
        for &cell in &[6.0f32, 8.0, 12.0] {
            for &(rot, axis) in &[
                (0.0f32, 90.0f32),
                (8.0, 90.0),
                (22.0, 90.0),
                (45.0, 90.0),
                (10.0, 60.0),
            ] {
                for &blur in &[0.3f32, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0] {
                    // Axis-aligned boards share one phase per image: several.
                    let images = if rot == 0.0 { 6 } else { 2 };
                    for _ in 0..images {
                        seed += 1;
                        let scene = optical(128, cell, rot, axis, blur, seed);
                        let cond = format!("cell{cell}_rot{rot}_axis{axis}_blur{blur}_s{seed}");
                        write("optical", &cond, [cell, rot, axis, blur], &scene);
                    }
                }
            }
        }
        if let Some(scenes) = exported_scenes(|n| n.starts_with("cal_")) {
            for scene in &scenes {
                write("synth", &scene.name, [f32::NAN; 4], scene);
            }
        }
    }
}
