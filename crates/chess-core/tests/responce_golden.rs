use chess_core::detect::{detect_corners_from_response, find_corners_u8_with_trace};
use chess_core::response::chess_response_u8;
use chess_core::ring::{ring_offsets, RING10, RING5};
use chess_core::{ChessParams, ResponseMap};

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
    assert_eq!(c.scale, 0);
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

    let res = find_corners_u8_with_trace(&img, w, h, &params);
    assert!(res.resp_ms >= 0.0);
    assert!(res.detect_ms >= 0.0);
    assert!(res.corners.is_empty());
}
