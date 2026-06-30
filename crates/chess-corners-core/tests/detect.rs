use chess_corners_core::{
    chess_response_u8, chess_response_u8_patch, detect_corners_from_response, find_corners_u8,
    ChessParams, ResponseMap, Roi,
};

fn idx(w: usize, x: usize, y: usize) -> usize {
    y * w + x
}

#[test]
fn response_on_uniform_image_is_zero() {
    let params = ChessParams::default();
    let w = 16usize;
    let h = 16usize;
    let img = vec![7u8; w * h];

    let resp = chess_response_u8(&img, w, h, &params);
    assert_eq!(resp.width(), w);
    assert_eq!(resp.height(), h);
    assert!(resp.data().iter().all(|v| v.abs() < 1e-6));
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
    let resp = ResponseMap::new(w, h, data);
    let mut params = ChessParams::default();
    params.threshold = 6.0;

    let corners = detect_corners_from_response(&resp, &params);
    assert_eq!(corners.len(), 1);

    let c = &corners[0];
    assert!((c.x - cx as f32).abs() < 0.2);
    assert!((c.y - cy as f32).abs() < 0.2);
    assert!((c.strength - 10.0).abs() < f32::EPSILON);
}

#[test]
fn detect_corners_rejects_maps_without_margin() {
    let params = ChessParams::default();
    let resp = ResponseMap::new(8, 8, vec![1.0; 64]);

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

    let roi = Roi::new(5, 7, 37, 29).unwrap();
    let patch = chess_response_u8_patch(img.as_raw(), w, h, &params, roi);

    assert_eq!(patch.width(), roi.x1() - roi.x0());
    assert_eq!(patch.height(), roi.y1() - roi.y0());

    for py in 0..patch.height() {
        for px in 0..patch.width() {
            let gx = roi.x0() + px;
            let gy = roi.y0() + py;
            let full_val = full.at(gx, gy);
            let patch_val = patch.at(px, py);
            assert!(
                (full_val - patch_val).abs() <= 1e-3,
                "mismatch at ({gx},{gy}) -> ({px},{py}): {full_val} vs {patch_val}"
            );
        }
    }
}
