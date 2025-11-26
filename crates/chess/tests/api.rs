use chess::{chess_response_image, find_corners_image, find_corners_image_trace, ChessParams};
use chess_core::response::chess_response_u8;
use image::GrayImage;

fn make_gradient_image(w: u32, h: u32) -> GrayImage {
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push(((x + y) % 255) as u8);
        }
    }
    GrayImage::from_vec(w, h, data).expect("gradient image")
}

#[test]
fn response_helper_matches_core_kernel() {
    let params = ChessParams::default();
    let w = 12u32;
    let h = 12u32;
    let img = make_gradient_image(w, h);

    let helper_resp = chess_response_image(&img, &params);
    let core_resp = chess_response_u8(img.as_raw(), w as usize, h as usize, &params);

    assert_eq!(helper_resp.w, core_resp.w);
    assert_eq!(helper_resp.h, core_resp.h);
    assert_eq!(helper_resp.data, core_resp.data);
}

#[test]
fn corner_helpers_keep_behavior_in_sync() {
    let params = ChessParams::default();
    let w = 32u32;
    let h = 32u32;
    let img = GrayImage::from_pixel(w, h, image::Luma([0u8]));

    let plain = find_corners_image(&img, &params);
    let traced = find_corners_image_trace(&img, &params);

    assert_eq!(plain.len(), traced.corners.len());
    assert!(traced.resp_ms >= 0.0);
    assert!(traced.detect_ms >= 0.0);
}
