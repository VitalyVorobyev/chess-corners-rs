//! Ad-hoc timing comparison for all `OrientationMethod` variants on a
//! single synthetic image. Used for a quick relative-cost sketch — not
//! a Criterion-grade measurement.
//!
//! Run with `cargo run --release -p chess-corners-core --example
//! orientation_methods_bench`.

use chess_corners_core::orientation::OrientationMethod;
use chess_corners_core::{detect::Corner, orientation::describe_corners};
use std::hint::black_box;
use std::time::Instant;

fn synth_image(w: usize, h: usize) -> Vec<u8> {
    let period = 24i32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let rx = ((x as i32) / period) & 1;
            let ry = ((y as i32) / period) & 1;
            out[y * w + x] = if rx ^ ry == 0 { 40 } else { 215 };
        }
    }
    out
}

fn synth_corners(n: usize, w: usize, h: usize) -> Vec<Corner> {
    let mut v = Vec::with_capacity(n);
    let margin = 12;
    let cols = ((n as f32).sqrt() as usize).max(1);
    let rows = n.div_ceil(cols);
    let x_stride = ((w - 2 * margin) / cols).max(1);
    let y_stride = ((h - 2 * margin) / rows).max(1);
    for j in 0..rows {
        for i in 0..cols {
            if v.len() >= n {
                break;
            }
            v.push(Corner::new(
                (margin + i * x_stride) as f32 + 0.25,
                (margin + j * y_stride) as f32 + 0.25,
                1.0,
            ));
        }
    }
    v
}

fn time_method(
    label: &str,
    img: &[u8],
    w: usize,
    h: usize,
    base_corners: &[Corner],
    method: OrientationMethod,
    iters: usize,
) {
    for _ in 0..3 {
        let corners = base_corners.to_vec();
        let d = describe_corners(img, w, h, 5, corners, method);
        black_box(d);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let corners = base_corners.to_vec();
        let t0 = Instant::now();
        let d = describe_corners(img, w, h, 5, corners, method);
        black_box(d);
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let p10 = samples[samples.len() / 10];
    let p90 = samples[samples.len() * 9 / 10];
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "  {label:>26}: median {median:8.1} us  p10 {p10:8.1}  p90 {p90:8.1}  mean {mean:8.1}",
    );
}

fn main() {
    const W: usize = 640;
    const H: usize = 480;
    let img = synth_image(W, H);
    let iters = 50;

    for n in [64usize, 256, 1024] {
        let corners = synth_corners(n, W, H);
        println!("== n_corners = {n} (image {W}x{H}, ring=5) ==");
        time_method(
            "RingFit",
            &img,
            W,
            H,
            &corners,
            OrientationMethod::RingFit,
            iters,
        );
        time_method(
            "DiskFit",
            &img,
            W,
            H,
            &corners,
            OrientationMethod::DiskFit,
            iters,
        );
        println!();
    }
}
