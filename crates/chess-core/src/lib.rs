#![cfg_attr(not(feature="std"), no_std)]

pub mod ring;
pub mod response;

#[derive(Clone, Debug)]
pub struct ChessParams {
    pub radius: u32,         // canonical: 5 (or 10 for heavy blur) :contentReference[oaicite:2]{index=2}
    pub threshold_rel: f32,  // used in later phase; keep here for API stability
}

impl Default for ChessParams {
    fn default() -> Self {
        Self {
            radius: 5,
            threshold_rel: 0.015, // paper suggests ~1.5% of max response :contentReference[oaicite:3]{index=3}
        }
    }
}

/// Response map in row-major layout
#[derive(Clone, Debug)]
pub struct ResponseMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl ResponseMap {
    #[inline] pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y*self.w + x]
    }
}
