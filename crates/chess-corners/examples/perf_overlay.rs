//! Per-stage timing + detection overlays for the public performance page.
//!
//! Drives the single-scale ChESS pipeline on the public `testimages/`
//! samples, decomposed into its four stages so each can be timed in
//! isolation:
//!
//! 1. **response**    — [`chess_response_u8`] dense ChESS response map.
//! 2. **detection**   — threshold + NMS + cluster-filter peak extraction
//!    ([`detect_peaks_from_response_with_refine_radius`]).
//! 3. **refinement**  — image-domain subpixel refinement
//!    ([`refine_corners_on_image`]).
//! 4. **orientation** — two-axis orientation fit / descriptor assembly
//!    ([`describe_corners`]).
//!
//! This is exactly the stage order the single-scale facade `Detector`
//! runs (`ChessDetector` + a near-zero-cost duplicate merge between
//! refinement and orientation), so the per-stage p50s and the final
//! corner count match a `DetectorConfig::chess()` detection on the same
//! image.
//!
//! Each stage is timed with `--warmup` discarded iterations followed by
//! `--repeats` measured iterations; the reported figure is the p50. Per-
//! iteration input setup (cloning the owned corner vectors the
//! refinement/orientation stages consume) happens outside the timed
//! region.
//!
//! It also writes a detection-overlay PNG per image (corner markers on a
//! dimmed copy of the input) into `--overlay-dir`, downscaling previews
//! wider than 1024 px while always detecting on the full-resolution
//! image.
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

use chess_corners::low_level::to_chess_params;
use chess_corners::DetectorConfig;
use chess_corners_core::unstable::{
    detect_peaks_from_response_with_refine_radius, refine_corners_on_image,
};
use chess_corners_core::{
    chess_response_u8, describe_corners, merge_corners_simple, CornerDescriptor, CornerRefiner,
    ImageView, Refiner,
};
use image::{imageops::FilterType, ImageReader, Rgb, RgbImage};

/// Public report images (label, relative path). These are the ONLY
/// public sample images in the repository.
const IMAGES: &[(&str, &str)] = &[
    ("small", "testimages/small.png"),
    ("mid", "testimages/mid.png"),
    ("large", "testimages/large.png"),
];

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

struct ImageResult {
    label: String,
    file: String,
    img: String,
    width: usize,
    height: usize,
    corner_count: usize,
    total_ms: f64,
    throughput_mpix_s: f64,
    stages: StageTimes,
}

fn measure(
    label: &str,
    file: &str,
    overlay_dir: &Path,
    warmup: usize,
    reps: usize,
) -> Result<ImageResult, Box<dyn std::error::Error>> {
    let gray = ImageReader::open(file)?.decode()?.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let data = gray.as_raw().as_slice();
    let view = ImageView::from_u8_slice(w, h, data).ok_or("invalid image view")?;

    // Faithful single-scale ChESS parameters (matches DetectorConfig::chess()).
    let cfg = DetectorConfig::chess();
    let params = to_chess_params(&cfg);
    let method = params.orientation_method;
    let desc_radius = params.descriptor_ring_radius();
    let merge_radius = cfg.merge_radius;

    let mut refiner = Refiner::from_kind(params.refiner.clone());
    let refine_border = refiner.radius();

    // Canonical artifacts threaded between stages and used for the
    // overlay + final count.
    let resp = chess_response_u8(data, w, h, &params);
    let peaks = detect_peaks_from_response_with_refine_radius(&resp, &params, refine_border);
    let refined = refine_corners_on_image(peaks.clone(), Some(view), Some(&resp), &mut refiner);
    let mut refined_for_merge = refined.clone();
    let merged = merge_corners_simple(&mut refined_for_merge, merge_radius);
    let descriptors = describe_corners(data, w, h, desc_radius, merged.clone(), method);

    // ---- per-stage timing ----
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

    let total_ms = t_response + t_detection + t_refinement + t_orientation;
    let throughput_mpix_s = if total_ms > 0.0 {
        (w * h) as f64 / (total_ms * 1000.0)
    } else {
        0.0
    };

    // ---- overlay PNG ----
    let img_rel = format!("./img/{label}.png");
    let out_path = overlay_dir.join(format!("{label}.png"));
    write_overlay(&gray, &descriptors, &out_path)?;

    Ok(ImageResult {
        label: label.to_string(),
        file: file.to_string(),
        img: img_rel,
        width: w,
        height: h,
        corner_count: descriptors.len(),
        total_ms,
        throughput_mpix_s,
        stages: StageTimes {
            response: t_response,
            detection: t_detection,
            refinement: t_refinement,
            orientation: t_orientation,
        },
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

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn emit_json(results: &[ImageResult]) -> String {
    let mut out = String::from("{\n  \"images\": [\n");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            concat!(
                "    {{\n",
                "      \"label\": \"{label}\",\n",
                "      \"file\": \"{file}\",\n",
                "      \"img\": \"{img}\",\n",
                "      \"width\": {width},\n",
                "      \"height\": {height},\n",
                "      \"corner_count\": {corner_count},\n",
                "      \"total_ms\": {total:.4},\n",
                "      \"throughput_mpix_s\": {tput:.2},\n",
                "      \"stages\": {{\n",
                "        \"response\": {resp:.4},\n",
                "        \"detection\": {det:.4},\n",
                "        \"refinement\": {refi:.4},\n",
                "        \"orientation\": {ori:.4}\n",
                "      }}\n",
                "    }}"
            ),
            label = json_escape(&r.label),
            file = json_escape(&r.file),
            img = json_escape(&r.img),
            width = r.width,
            height = r.height,
            corner_count = r.corner_count,
            total = r.total_ms,
            tput = r.throughput_mpix_s,
            resp = r.stages.response,
            det = r.stages.detection,
            refi = r.stages.refinement,
            ori = r.stages.orientation,
        ));
    }
    out.push_str("\n  ]\n}\n");
    out
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
    for (label, file) in IMAGES {
        eprintln!("perf_overlay: measuring {label} ({file})");
        results.push(measure(label, file, &overlay_dir, warmup, repeats)?);
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
