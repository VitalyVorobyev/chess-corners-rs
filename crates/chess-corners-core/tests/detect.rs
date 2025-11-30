use chess_corners_core::detect::{detect_corners_from_response, find_corners_u8};
#[cfg(feature = "simd")]
use chess_corners_core::response::chess_response_u8_scalar;
use chess_corners_core::response::{self, chess_response_u8, chess_response_u8_patch};

use chess_corners_core::ring::{ring_offsets, RING10, RING5};
use chess_corners_core::{ChessParams, ResponseMap};

fn idx(w: usize, x: usize, y: usize) -> usize {
    y * w + x
}

#[test]
fn ring_offsets_switch_with_radius() {
    assert_eq!(ring_offsets(5), &RING5);
    assert_eq!(ring_offsets(10), &RING10);
    // Any unknown radius currently falls back to the canonical r=5 offsets.
    assert_eq!(ring_offsets(3), &RING5);
}

#[test]
fn response_on_uniform_image_is_zero() {
    let params = ChessParams::default();
    let w = 16usize;
    let h = 16usize;
    let img = vec![7u8; w * h];

    let resp = chess_response_u8(&img, w, h, &params);
    assert_eq!(resp.w, w);
    assert_eq!(resp.h, h);
    assert!(resp.data.iter().all(|v| v.abs() < 1e-6));
}

#[cfg(feature = "simd")]
#[test]
fn simd_matches_scalar_reasonably() {
    let params = ChessParams::default();
    let img = image::GrayImage::from_fn(256, 256, |x, y| image::Luma([(x ^ y) as u8]));
    let w = img.width() as usize;
    let h = img.height() as usize;

    let ref_map = chess_response_u8_scalar(img.as_raw(), w, h, &params);
    let simd_map = chess_response_u8(img.as_raw(), w, h, &params);

    let eps = 1e-3_f32;
    for (a, b) in ref_map.data.iter().zip(simd_map.data.iter()) {
        assert!((a - b).abs() <= eps, "diff: {a} vs {b}");
    }
}

#[cfg(all(feature = "simd", feature = "rayon"))]
#[test]
fn simd_parallel_matches_scalar() {
    let params = ChessParams::default();
    let img = image::GrayImage::from_fn(192, 192, |x, y| {
        image::Luma([(x.wrapping_mul(7) ^ y) as u8])
    });
    let w = img.width() as usize;
    let h = img.height() as usize;

    let ref_map = chess_response_u8_scalar(img.as_raw(), w, h, &params);
    let simd_map = chess_response_u8(img.as_raw(), w, h, &params);

    let eps = 1e-3_f32;
    for (a, b) in ref_map.data.iter().zip(simd_map.data.iter()) {
        assert!((a - b).abs() <= eps, "diff: {a} vs {b}");
    }
}

#[test]
fn response_matches_manual_ring_layout() {
    let params = ChessParams::default();
    let w = 11usize;
    let h = 11usize;
    let cx = 5usize;
    let cy = 5usize;
    let mut img = vec![0u8; w * h];

    // Populate the 16 ring samples with the sequence 0..15.
    for (i, (dx, dy)) in RING5.iter().enumerate() {
        let x = (cx as i32 + dx) as usize;
        let y = (cy as i32 + dy) as usize;
        img[idx(w, x, y)] = i as u8;
    }

    // Fill the 5-pixel cross used in the local mean with distinct values.
    for (dx, dy, v) in [
        (0, 0, 10u8),
        (0, -1, 20u8),
        (0, 1, 30u8),
        (1, 0, 40u8),
        (-1, 0, 50u8),
    ] {
        let x = (cx as i32 + dx) as usize;
        let y = (cy as i32 + dy) as usize;
        img[idx(w, x, y)] = v;
    }

    let resp = chess_response_u8(&img, w, h, &params);
    let center = resp.at(cx, cy);

    // Expected value computed from the ring/cross assignments above.
    let expected = -392.0_f32;
    assert!(
        (center - expected).abs() < 1e-3,
        "expected center response {expected}, got {center}"
    );

    for (i, v) in resp.data.iter().enumerate() {
        if i == idx(w, cx, cy) {
            continue;
        }
        assert!(
            v.abs() < 1e-6,
            "non-center response should stay zero (idx={i}, val={v})"
        );
    }
}

#[test]
fn detect_corners_respects_threshold_and_cluster_size() {
    let w = 21usize;
    let h = 21usize;
    let cx = 10usize;
    let cy = 10usize;
    let mut data = vec![0.0f32; w * h];
    data[idx(w, cx, cy)] = 10.0;
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let x = (cx as i32 + dx) as usize;
        let y = (cy as i32 + dy) as usize;
        data[idx(w, x, y)] = 4.0;
    }
    let resp = ResponseMap { w, h, data };
    let params = ChessParams {
        threshold_abs: Some(6.0),
        ..Default::default()
    };

    let corners = detect_corners_from_response(&resp, &params);
    assert_eq!(corners.len(), 1);

    let c = &corners[0];
    assert!((c.xy[0] - cx as f32).abs() < 0.2);
    assert!((c.xy[1] - cy as f32).abs() < 0.2);
    assert!((c.strength - 10.0).abs() < f32::EPSILON);
}

#[test]
fn detect_corners_rejects_maps_without_margin() {
    let params = ChessParams::default();
    let resp = ResponseMap {
        w: 8,
        h: 8,
        data: vec![1.0; 64],
    };

    let corners = detect_corners_from_response(&resp, &params);
    assert!(corners.is_empty());
}

#[test]
fn tracing_path_reports_elapsed_times() {
    let params = ChessParams::default();
    let w = 24usize;
    let h = 24usize;
    let img = vec![0u8; w * h];

    let corners = find_corners_u8(&img, w, h, &params);
    assert!(corners.is_empty());
}

#[test]
fn patch_response_matches_full_map_slice() {
    let params = ChessParams::default();
    let img = image::GrayImage::from_fn(64, 48, |x, y| image::Luma([(x * 7 + y * 13) as u8]));
    let w = img.width() as usize;
    let h = img.height() as usize;

    let full = chess_response_u8(img.as_raw(), w, h, &params);

    let roi = response::Roi {
        x0: 5,
        y0: 7,
        x1: 37,
        y1: 29,
    };
    let patch = chess_response_u8_patch(img.as_raw(), w, h, &params, roi);

    assert_eq!(patch.w, roi.x1 - roi.x0);
    assert_eq!(patch.h, roi.y1 - roi.y0);

    for py in 0..patch.h {
        for px in 0..patch.w {
            let gx = roi.x0 + px;
            let gy = roi.y0 + py;
            let full_val = full.at(gx, gy);
            let patch_val = patch.at(px, py);
            assert!(
                (full_val - patch_val).abs() <= 1e-3,
                "mismatch at ({gx},{gy}) -> ({px},{py}): {full_val} vs {patch_val}"
            );
        }
    }
}
