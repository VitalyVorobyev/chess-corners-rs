use crate::{ChessParams, ResponseMap};
use crate::ring::ring_offsets;

pub fn chess_response_u8(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    let r = params.radius as i32;
    let ring = ring_offsets(params.radius);

    let mut data = vec![0.0f32; w*h];

    // only evaluate where full ring fits
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    for y in y0..y1 {
        for x in x0..x1 {
            let resp = chess_response_at_u8(img, w, x as i32, y as i32, ring);
            data[y*w + x] = resp;
        }
    }

    ResponseMap { w, h, data }
}

#[inline]
fn chess_response_at_u8(
    img: &[u8], w: usize, x: i32, y: i32,
    ring: &[(i32,i32);16]
) -> f32 {
    // gather ring samples into i32
    let mut s = [0i32; 16];
    for k in 0..16 {
        let (dx, dy) = ring[k];
        let xx = (x + dx) as usize;
        let yy = (y + dy) as usize;
        s[k] = img[yy*w + xx] as i32;
    }

    // SR
    let mut sr = 0i32;
    for k in 0..4 {
        let a = s[k] + s[k+8];
        let b = s[k+4] + s[k+12];
        sr += (a - b).abs();
    }

    // DR
    let mut dr = 0i32;
    for k in 0..8 {
        dr += (s[k] - s[k+8]).abs();
    }

    // neighbor mean
    let sum_ring: i32 = s.iter().sum();
    let mu_n = sum_ring as f32 / 16.0;

    // local mean (5 px cross)
    let c  = img[(y as usize)*w + (x as usize)] as f32;
    let n  = img[((y-1) as usize)*w + (x as usize)] as f32;
    let s0 = img[((y+1) as usize)*w + (x as usize)] as f32;
    let e  = img[(y as usize)*w + ((x+1) as usize)] as f32;
    let w0 = img[(y as usize)*w + ((x-1) as usize)] as f32;
    let mu_l = (c + n + s0 + e + w0) / 5.0;

    let mr = (mu_n - mu_l).abs();

    (sr as f32) - (dr as f32) - 16.0*mr
}
