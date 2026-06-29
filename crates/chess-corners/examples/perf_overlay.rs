//! Per-stage timing matrix + detection overlays for the public
//! performance page.
//!
//! For each public `testimages/` sample this measures a **matrix** of
//! detector configurations, all built through the facade
//! [`DetectorConfig`] builders so the numbers match a real
//! `Detector::detect` run:
//!
//! - **ChESS**: refiner ∈ {`center_of_mass`, `forstner`, `saddle_point`}
//!   × orientation ∈ {`ring_fit`, `disk_fit`, `off`} = 9 configs.
//! - **Radon**: no pluggable refiner (built-in Gaussian peak fit)
//!   × orientation ∈ {`ring_fit`, `disk_fit`, `off`} = 3 configs.
//!
//! Every config is decomposed into the same four stages, each timed in
//! isolation so the orientation share is directly comparable:
//!
//! 1. **response**    — dense response map
//!    ([`chess_response_u8`] / [`radon_response_u8`]).
//! 2. **detection**   — threshold + NMS + cluster filter peak extraction
//!    ([`detect_peaks_from_response_with_refine_radius`] /
//!    [`detect_peaks_from_radon`]). For Radon this stage also runs the
//!    3-point Gaussian response-map peak fit — Radon's only subpixel step.
//! 3. **refinement**  — image-domain subpixel refinement
//!    ([`refine_corners_on_image`]). The ChESS strategy runs the
//!    configured refiner here; the Radon strategy has no pluggable
//!    image-domain refiner — its subpixel step is the 3-point Gaussian
//!    peak fit inside detection — so the Radon refinement stage measures
//!    as `0`.
//! 4. **orientation** — two-axis orientation fit / descriptor assembly
//!    ([`describe_corners`]). `off` configs skip the per-corner fit, so
//!    the orientation stage collapses to descriptor assembly (~0 ms).
//!
//! This is exactly the stage order the single-scale facade `Detector`
//! runs (`Chess`/`Radon` strategy + a near-zero-cost duplicate merge
//! between refinement and orientation), so the per-stage p50s and final
//! corner counts match a `DetectorConfig` detection on the same image.
//!
//! Each stage is timed with `--warmup` discarded iterations followed by
//! `--repeats` measured iterations; the reported figure is the p50. Per-
//! iteration input setup (cloning the owned corner vectors the
//! refinement/orientation stages consume) happens outside the timed
//! region.
//!
//! It also writes one detection-overlay PNG per (image × detector) at the
//! detector's default refiner + `ring_fit` into `--overlay-dir` (the
//! orientation method and refiner do not visibly move corners), as
//! `<base>__chess.png` / `<base>__radon.png`. Previews wider than 1024 px
//! are downscaled while detection always runs on the full image.
//!
//! Output: one JSON document (to `--out`, else stdout) consumed by
//! `scripts/gen_perf_data.py`.
//!
//! Usage:
//! ```sh
//! cargo run --release -q -p chess-corners --example perf_overlay \
//!     --features simd -- \
//!     --repeats 60 --warmup 8 \
//!     --overlay-dir .github/pages/performance/img \
//!     --out raw.json
//! ```

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chess_corners::low_level::{to_chess_params, to_radon_detector_params};
use chess_corners::{ChessRefiner, DetectorConfig, OrientationMethod};
use chess_corners_core::unstable::{
    detect_peaks_from_response_with_refine_radius, refine_corners_on_image, ChessParams,
};
use chess_corners_core::{
    chess_response_u8, describe_corners, detect_peaks_from_radon, merge_corners_simple,
    radon_response_u8, CornerDescriptor, CornerRefiner, ImageView, RadonBuffers, Refiner,
};
use image::{imageops::FilterType, ImageReader, Rgb, RgbImage};
use serde_json::json;

/// Public report images (`base`, relative path). `base` is the overlay
/// filename stem and the merge key (via the file basename). These are the
/// ONLY public sample images in the repository.
const IMAGES: &[(&str, &str)] = &[
    ("small", "testimages/small.png"),
    ("mid", "testimages/mid.png"),
    ("large", "testimages/large.png"),
];

/// Orientation variants measured for every detector/refiner pair.
const ORIENTATIONS: [&str; 3] = ["ring_fit", "disk_fit", "off"];

/// Config ids whose descriptors are rendered as the representative
/// per-detector overlays (detector default refiner + `ring_fit`).
const CHESS_OVERLAY_ID: &str = "chess__center_of_mass__ring_fit";
const RADON_OVERLAY_ID: &str = "radon__gaussian_peak__ring_fit";

/// Previews wider than this are downscaled for the committed PNG; the
/// detector still runs on the full-resolution image.
const MAX_PREVIEW_WIDTH: u32 = 1024;

fn p50_ms(mut samples: Vec<u128>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1e6
}

/// Time a stage that borrows its inputs (no owned setup per iteration).
fn time_stage<F: FnMut()>(warmup: usize, reps: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos());
    }
    p50_ms(samples)
}

/// Time a stage that consumes an owned input. A fresh clone is made
/// before each timed call so the clone cost is excluded.
fn time_stage_owned<I: Clone, F: FnMut(I)>(warmup: usize, reps: usize, proto: &I, mut f: F) -> f64 {
    for _ in 0..warmup {
        f(proto.clone());
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let input = proto.clone();
        let t = Instant::now();
        f(input);
        samples.push(t.elapsed().as_nanos());
    }
    p50_ms(samples)
}

struct StageTimes {
    response: f64,
    detection: f64,
    refinement: f64,
    orientation: f64,
}

/// One measured detector configuration.
struct ConfigResult {
    id: String,
    detector: &'static str,
    refiner: &'static str,
    orientation: &'static str,
    corner_count: usize,
    stages: StageTimes,
    total_ms: f64,
    throughput_mpix_s: f64,
}

struct ImageResult {
    label: String,
    file: String,
    width: usize,
    height: usize,
    chess_overlay: String,
    radon_overlay: String,
    configs: Vec<ConfigResult>,
}

/// A detector config to measure, with its display labels and merge id.
struct ConfigSpec {
    id: String,
    detector: &'static str,
    refiner: &'static str,
    orientation: &'static str,
    cfg: DetectorConfig,
}

/// Apply the requested orientation variant to a base config.
fn with_orientation(cfg: DetectorConfig, orientation: &str) -> DetectorConfig {
    match orientation {
        "disk_fit" => cfg.with_orientation_method(OrientationMethod::DiskFit),
        "off" => cfg.without_orientation(),
        // "ring_fit" (and any unexpected value) → explicit RingFit.
        _ => cfg.with_orientation_method(OrientationMethod::RingFit),
    }
}

/// Build the 9 ChESS + 3 Radon configs measured per image.
fn build_specs() -> Vec<ConfigSpec> {
    let mut specs = Vec::with_capacity(12);

    let chess_refiners: [(&str, ChessRefiner); 3] = [
        ("center_of_mass", ChessRefiner::center_of_mass()),
        ("forstner", ChessRefiner::forstner()),
        ("saddle_point", ChessRefiner::saddle_point()),
    ];
    for (refiner, kind) in chess_refiners {
        for orientation in ORIENTATIONS {
            let cfg = with_orientation(
                DetectorConfig::chess().with_chess(|c| c.refiner = kind),
                orientation,
            );
            specs.push(ConfigSpec {
                id: format!("chess__{refiner}__{orientation}"),
                detector: "chess",
                refiner,
                orientation,
                cfg,
            });
        }
    }

    // Radon's subpixel step is its built-in Gaussian peak fit; it has no
    // pluggable refiner, so the refiner axis is fixed.
    for orientation in ORIENTATIONS {
        let cfg = with_orientation(DetectorConfig::radon(), orientation);
        specs.push(ConfigSpec {
            id: format!("radon__gaussian_peak__{orientation}"),
            detector: "radon",
            refiner: "gaussian_peak",
            orientation,
            cfg,
        });
    }

    specs
}

/// Faithful single-scale ChESS stage decomposition (matches the facade
/// `Detector` for `DetectionStrategy::Chess`).
fn measure_chess(
    cfg: &DetectorConfig,
    data: &[u8],
    w: usize,
    h: usize,
    view: ImageView<'_>,
    warmup: usize,
    reps: usize,
) -> (Vec<CornerDescriptor>, StageTimes) {
    let params = to_chess_params(cfg);
    let method = cfg.orientation_method;
    let desc_radius = params.ring_radius();
    let merge_radius = cfg.merge_radius;

    let mut refiner = Refiner::from_kind(params.refiner.clone());
    let refine_border = refiner.radius();

    // Canonical artifacts threaded between stages and used for the final
    // descriptor count / overlay.
    let resp = chess_response_u8(data, w, h, &params);
    let peaks = detect_peaks_from_response_with_refine_radius(&resp, &params, refine_border);
    let refined = refine_corners_on_image(peaks.clone(), Some(view), Some(&resp), &mut refiner);
    let mut refined_for_merge = refined.clone();
    let merged = merge_corners_simple(&mut refined_for_merge, merge_radius);
    let descriptors = describe_corners(data, w, h, desc_radius, merged.clone(), method);

    let t_response = time_stage(warmup, reps, || {
        black_box(chess_response_u8(data, w, h, &params));
    });
    let t_detection = time_stage(warmup, reps, || {
        black_box(detect_peaks_from_response_with_refine_radius(
            &resp,
            &params,
            refine_border,
        ));
    });
    let t_refinement = time_stage_owned(warmup, reps, &peaks, |p| {
        black_box(refine_corners_on_image(
            p,
            Some(view),
            Some(&resp),
            &mut refiner,
        ));
    });
    let t_orientation = time_stage_owned(warmup, reps, &merged, |m| {
        black_box(describe_corners(data, w, h, desc_radius, m, method));
    });

    (
        descriptors,
        StageTimes {
            response: t_response,
            detection: t_detection,
            refinement: t_refinement,
            orientation: t_orientation,
        },
    )
}

/// Faithful single-scale Radon stage decomposition (matches the facade
/// `Detector` for `DetectionStrategy::Radon`).
///
/// The Radon strategy has no pluggable image-domain refiner, so the
/// refinement stage is reported as `0`. Radon's subpixel step is the
/// 3-point Gaussian fit inside `detect_peaks_from_radon`, already
/// accounted for in `detection`.
fn measure_radon(
    cfg: &DetectorConfig,
    data: &[u8],
    w: usize,
    h: usize,
    warmup: usize,
    reps: usize,
) -> (Vec<CornerDescriptor>, StageTimes) {
    let params = to_radon_detector_params(cfg);
    let method = cfg.orientation_method;
    // The facade samples descriptors at the canonical r=5 ring for the
    // Radon strategy (it has no descriptor-ring knob of its own).
    let desc_radius = ChessParams::default().ring_radius();
    let merge_radius = cfg.merge_radius;

    // Canonical artifacts. `resp` borrows `buffers` for the rest of the
    // function; the response-timing loop below uses a separate buffer set.
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(data, w, h, &params, &mut buffers);
    let peaks = detect_peaks_from_radon(&resp, &params);
    // Facade no-op refinement: forward the detector peaks unchanged.
    let mut corners = peaks.clone();
    let merged = merge_corners_simple(&mut corners, merge_radius);
    let descriptors = describe_corners(data, w, h, desc_radius, merged.clone(), method);

    let mut timing_buffers = RadonBuffers::new();
    let t_response = time_stage(warmup, reps, || {
        black_box(radon_response_u8(data, w, h, &params, &mut timing_buffers));
    });
    let t_detection = time_stage(warmup, reps, || {
        black_box(detect_peaks_from_radon(&resp, &params));
    });
    // Radon's facade refinement is a no-op (see fn doc) — no measurable cost.
    let t_refinement = 0.0;
    let t_orientation = time_stage_owned(warmup, reps, &merged, |m| {
        black_box(describe_corners(data, w, h, desc_radius, m, method));
    });

    (
        descriptors,
        StageTimes {
            response: t_response,
            detection: t_detection,
            refinement: t_refinement,
            orientation: t_orientation,
        },
    )
}

fn measure_image(
    base: &str,
    file: &str,
    overlay_dir: &Path,
    warmup: usize,
    reps: usize,
) -> Result<ImageResult, Box<dyn std::error::Error>> {
    let gray = ImageReader::open(file)?.decode()?.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let data = gray.as_raw().as_slice();
    let view = ImageView::from_u8_slice(w, h, data).ok_or("invalid image view")?;

    let specs = build_specs();
    let mut configs = Vec::with_capacity(specs.len());
    let mut chess_overlay: Option<Vec<CornerDescriptor>> = None;
    let mut radon_overlay: Option<Vec<CornerDescriptor>> = None;

    for spec in &specs {
        let (descriptors, stages) = if spec.detector == "chess" {
            measure_chess(&spec.cfg, data, w, h, view, warmup, reps)
        } else {
            measure_radon(&spec.cfg, data, w, h, warmup, reps)
        };
        let corner_count = descriptors.len();
        if spec.id == CHESS_OVERLAY_ID {
            chess_overlay = Some(descriptors.clone());
        } else if spec.id == RADON_OVERLAY_ID {
            radon_overlay = Some(descriptors.clone());
        }

        let total_ms = stages.response + stages.detection + stages.refinement + stages.orientation;
        let throughput_mpix_s = if total_ms > 0.0 {
            (w * h) as f64 / (total_ms * 1000.0)
        } else {
            0.0
        };
        eprintln!(
            "  {:<34} {:>5} corners  {:.3} ms",
            spec.id, corner_count, total_ms
        );
        configs.push(ConfigResult {
            id: spec.id.clone(),
            detector: spec.detector,
            refiner: spec.refiner,
            orientation: spec.orientation,
            corner_count,
            stages,
            total_ms,
            throughput_mpix_s,
        });
    }

    // Representative overlays: one per detector at its default refiner +
    // ring_fit. The orientation method and refiner do not visibly move
    // corners, so a single overlay per detector is faithful.
    let chess_desc = chess_overlay.unwrap_or_default();
    let radon_desc = radon_overlay.unwrap_or_default();
    write_overlay(
        &gray,
        &chess_desc,
        &overlay_dir.join(format!("{base}__chess.png")),
    )?;
    write_overlay(
        &gray,
        &radon_desc,
        &overlay_dir.join(format!("{base}__radon.png")),
    )?;

    Ok(ImageResult {
        label: base.to_string(),
        file: file.to_string(),
        width: w,
        height: h,
        chess_overlay: format!("./img/{base}__chess.png"),
        radon_overlay: format!("./img/{base}__radon.png"),
        configs,
    })
}

/// Render detected corners as small crosses on a dimmed copy of the
/// grayscale input, downscaling previews wider than [`MAX_PREVIEW_WIDTH`].
fn write_overlay(
    gray: &image::GrayImage,
    corners: &[CornerDescriptor],
    out_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (gray.width(), gray.height());
    let scale = if w > MAX_PREVIEW_WIDTH {
        MAX_PREVIEW_WIDTH as f32 / w as f32
    } else {
        1.0
    };
    let (tw, th) = (
        (w as f32 * scale).round() as u32,
        (h as f32 * scale).round() as u32,
    );

    // Dim the background so markers stand out on both black and white squares.
    let base = if scale < 1.0 {
        image::imageops::resize(gray, tw, th, FilterType::Triangle)
    } else {
        gray.clone()
    };
    let mut rgb = RgbImage::new(tw, th);
    for (dst, src) in rgb.pixels_mut().zip(base.pixels()) {
        let v = (src[0] as f32 * 0.6 + 18.0).round().clamp(0.0, 255.0) as u8;
        *dst = Rgb([v, v, v]);
    }

    // Bright green crosses (arm = 1 → 3 px span) scale-mapped to the preview.
    let mark = Rgb([0x3d, 0xdc, 0x84]);
    let arm: i32 = 1;
    for c in corners {
        let cx = (c.x * scale).round() as i32;
        let cy = (c.y * scale).round() as i32;
        for d in -arm..=arm {
            put(&mut rgb, cx + d, cy, mark);
            put(&mut rgb, cx, cy + d, mark);
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    rgb.save(out_path)?;
    Ok(())
}

#[inline]
fn put(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, color);
    }
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
fn round2(x: f64) -> f64 {
    (x * 1e2).round() / 1e2
}

fn emit_json(results: &[ImageResult]) -> String {
    let images: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let configs: Vec<serde_json::Value> = r
                .configs
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "detector": c.detector,
                        "refiner": c.refiner,
                        "orientation": c.orientation,
                        "corner_count": c.corner_count,
                        "stages": {
                            "response": round4(c.stages.response),
                            "detection": round4(c.stages.detection),
                            "refinement": round4(c.stages.refinement),
                            "orientation": round4(c.stages.orientation),
                        },
                        "total_ms": round4(c.total_ms),
                        "throughput_mpix_s": round2(c.throughput_mpix_s),
                    })
                })
                .collect();
            json!({
                "label": r.label,
                "file": r.file,
                "width": r.width,
                "height": r.height,
                "overlays": { "chess": r.chess_overlay, "radon": r.radon_overlay },
                "configs": configs,
            })
        })
        .collect();
    let doc = json!({ "images": images });
    let mut s = serde_json::to_string_pretty(&doc).expect("serialize perf json");
    s.push('\n');
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut repeats = 60usize;
    let mut warmup = 8usize;
    let mut overlay_dir = PathBuf::from(".github/pages/performance/img");
    let mut out: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repeats" => repeats = args.next().and_then(|v| v.parse().ok()).unwrap_or(repeats),
            "--warmup" => warmup = args.next().and_then(|v| v.parse().ok()).unwrap_or(warmup),
            "--overlay-dir" => overlay_dir = args.next().map(PathBuf::from).unwrap_or(overlay_dir),
            "--out" => out = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    std::fs::create_dir_all(&overlay_dir)?;

    let mut results = Vec::new();
    for (base, file) in IMAGES {
        eprintln!("perf_overlay: measuring {base} ({file})");
        results.push(measure_image(base, file, &overlay_dir, warmup, repeats)?);
    }

    let json = emit_json(&results);
    match out {
        Some(path) => {
            std::fs::write(&path, &json)?;
            eprintln!("perf_overlay: wrote {}", path.display());
        }
        None => print!("{json}"),
    }
    Ok(())
}
