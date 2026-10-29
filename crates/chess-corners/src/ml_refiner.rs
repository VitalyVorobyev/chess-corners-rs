//! ML subpixel refiner integration (feature-gated).

use chess_corners_core::{Corner, ImageView};
use chess_corners_ml::{MlModel, ModelSource};
use log::{info, warn};
use std::path::PathBuf;
#[cfg(feature = "tracing")]
use std::time::{Duration, Instant};
#[cfg(feature = "tracing")]
use tracing::info_span;

/// Configuration for the ML subpixel refiner.
#[derive(Clone, Debug)]
pub(crate) struct MlRefinerParams {
    pub model_path: Option<PathBuf>,
    pub patch_size: u32,
    pub batch_size: u32,
}

impl Default for MlRefinerParams {
    fn default() -> Self {
        Self {
            model_path: None,
            patch_size: 21,
            batch_size: 64,
        }
    }
}

/// The loaded model and its reusable batch buffers.
pub(crate) struct MlRefinerState {
    model: Option<MlModel>,
    patch_size: usize,
    patch_area: usize,
    batch_size: usize,
    buffer: Vec<f32>,
    indices: Vec<usize>,
}

impl MlRefinerState {
    pub(crate) fn new(params: &MlRefinerParams) -> Self {
        let patch_size = params.patch_size.max(1) as usize;
        let patch_area = patch_size * patch_size;
        let batch_size = params.batch_size.max(1) as usize;
        Self {
            model: load_model(params, patch_size),
            patch_size,
            patch_area,
            batch_size,
            buffer: vec![0.0f32; batch_size * patch_area],
            indices: Vec::with_capacity(batch_size),
        }
    }
}

impl MlRefinerState {
    /// Half-width of the model's patch: the margin a refined corner needs.
    pub(crate) fn patch_radius(&self) -> i32 {
        (self.patch_size as i32 - 1) / 2
    }
}

/// Refine every corner of `corners` on `image` with the model, in order.
///
/// The model predicts each corner's offset from a patch centred on it. A
/// corner it cannot refine — its patch leaves the image, inference fails,
/// or the model is unavailable — is returned unchanged, so a detector's own
/// subpixel peak (and the state that says so) survives.
pub(crate) fn refine_corners(
    state: &mut MlRefinerState,
    mut corners: Vec<Corner>,
    image: ImageView<'_>,
) -> Vec<Corner> {
    if corners.is_empty() {
        return corners;
    }

    let mut stats = MlRefineStats {
        total: corners.len(),
        ..Default::default()
    };

    let Some(model) = state.model.as_ref() else {
        warn!("ML model unavailable; keeping every candidate");
        info!("ml refiner: total={} kept=all (no model)", stats.total);
        return corners;
    };

    #[cfg(feature = "tracing")]
    let ml_span = info_span!(
        "ml_refiner",
        candidates = stats.total,
        patch_size = state.patch_size,
        batch_size = state.batch_size
    );
    #[cfg(feature = "tracing")]
    let _ml_guard = ml_span.enter();
    #[cfg(feature = "tracing")]
    let total_start = Instant::now();
    #[cfg(feature = "tracing")]
    let mut timing = BatchTiming::default();

    state.indices.clear();
    for idx in 0..corners.len() {
        let corner = &corners[idx];
        let offset = state.indices.len() * state.patch_area;
        let patch_slice = &mut state.buffer[offset..offset + state.patch_area];
        if extract_patch_u8_to_f32(image, corner.x, corner.y, state.patch_size, patch_slice)
            .is_none()
        {
            stats.oob += 1;
            continue;
        }

        stats.extracted += 1;
        state.indices.push(idx);
        if state.indices.len() == state.batch_size {
            run_batch(
                model,
                &state.buffer[..state.indices.len() * state.patch_area],
                &state.indices,
                &mut corners,
                &mut stats,
                #[cfg(feature = "tracing")]
                &mut timing,
            );
            state.indices.clear();
        }
    }
    if !state.indices.is_empty() {
        run_batch(
            model,
            &state.buffer[..state.indices.len() * state.patch_area],
            &state.indices,
            &mut corners,
            &mut stats,
            #[cfg(feature = "tracing")]
            &mut timing,
        );
        state.indices.clear();
    }

    #[cfg(feature = "tracing")]
    {
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let infer_ms = timing.infer.as_secs_f64() * 1000.0;
        tracing::info!(
            target: "chess_corners::ml",
            total_ms,
            infer_ms,
            infer_batches = timing.batches,
            candidates = stats.total,
            extracted = stats.extracted,
            oob = stats.oob,
            inferred = stats.inferred,
            applied = stats.applied,
            output = corners.len(),
            "ml refiner timing"
        );
    }
    info!(
        "ml refiner: total={} extracted={} oob={} inferred={} applied={} output={}",
        stats.total,
        stats.extracted,
        stats.oob,
        stats.inferred,
        stats.applied,
        corners.len()
    );
    corners
}

fn load_model(params: &MlRefinerParams, patch_size: usize) -> Option<MlModel> {
    let source = match &params.model_path {
        Some(path) => ModelSource::Path(path.clone()),
        None => ModelSource::EmbeddedDefault,
    };

    match MlModel::load(source) {
        Ok(model) => {
            if model.patch_size() != patch_size {
                warn!(
                    "ML model patch size {} does not match config {}; keeping candidates",
                    model.patch_size(),
                    patch_size
                );
                None
            } else {
                Some(model)
            }
        }
        Err(err) => {
            warn!("failed to load ML model: {err}");
            None
        }
    }
}

/// Inference time and batch count, for the tracing summary.
#[cfg(feature = "tracing")]
#[derive(Default)]
struct BatchTiming {
    infer: Duration,
    batches: usize,
}

/// Run the model on the patches in `patches` (one per index of `indices`)
/// and move each corner `indices` names by its predicted offset. A corner
/// without a prediction keeps its position.
fn run_batch(
    model: &MlModel,
    patches: &[f32],
    indices: &[usize],
    corners: &mut [Corner],
    stats: &mut MlRefineStats,
    #[cfg(feature = "tracing")] timing: &mut BatchTiming,
) {
    #[cfg(feature = "tracing")]
    let infer_start = Instant::now();
    let preds_result = model.infer_batch(patches, indices.len());
    #[cfg(feature = "tracing")]
    {
        timing.infer += infer_start.elapsed();
        timing.batches += 1;
    }
    let preds = match preds_result {
        Ok(preds) => preds,
        Err(err) => {
            warn!(
                "ML inference failed: {err}; keeping {} candidates",
                indices.len()
            );
            stats.infer_fail += indices.len();
            return;
        }
    };

    let used = preds.len().min(indices.len());
    stats.inferred += used;
    for (&idx, pred) in indices.iter().zip(&preds) {
        let corner = &mut corners[idx];
        stats.applied += 1;
        *corner = Corner::new(corner.x + pred[0], corner.y + pred[1], corner.strength);
    }
    if preds.len() < indices.len() {
        warn!(
            "ML output length {} shorter than batch size {}; keeping the rest",
            preds.len(),
            indices.len()
        );
        stats.infer_fail += indices.len() - preds.len();
    }
}

#[derive(Default)]
struct MlRefineStats {
    total: usize,
    extracted: usize,
    oob: usize,
    inferred: usize,
    applied: usize,
    infer_fail: usize,
}

pub(crate) fn extract_patch_u8_to_f32(
    view: ImageView<'_>,
    x: f32,
    y: f32,
    patch_size: usize,
    out: &mut [f32],
) -> Option<()> {
    if patch_size == 0 {
        return None;
    }
    let width = view.width() as i32;
    let height = view.height() as i32;
    if width <= 0 || height <= 0 {
        return None;
    }
    let patch_area = patch_size * patch_size;
    if out.len() < patch_area {
        return None;
    }

    let half = (patch_size as f32 - 1.0) * 0.5;
    let [ox, oy] = view.origin();
    let origin_x = ox as f32;
    let origin_y = oy as f32;

    let min_x = x - half + origin_x;
    let max_x = x + half + origin_x;
    let min_y = y - half + origin_y;
    let max_y = y + half + origin_y;

    let max_x_allowed = (width - 1) as f32;
    let max_y_allowed = (height - 1) as f32;
    if min_x < 0.0 || min_y < 0.0 || max_x > max_x_allowed || max_y > max_y_allowed {
        return None;
    }

    let mut idx = 0;
    for iy in 0..patch_size {
        let v = iy as f32 - half;
        let gy = y + v + origin_y;
        let y0 = gy.floor() as i32;
        let y1 = (y0 + 1).min(height - 1);
        let wy = gy - y0 as f32;
        let row0 = (y0 as usize) * view.width();
        let row1 = (y1 as usize) * view.width();

        for ix in 0..patch_size {
            let u = ix as f32 - half;
            let gx = x + u + origin_x;
            let x0 = gx.floor() as i32;
            let x1 = (x0 + 1).min(width - 1);
            let wx = gx - x0 as f32;

            let p00 = view.data()[row0 + x0 as usize] as f32;
            let p10 = view.data()[row0 + x1 as usize] as f32;
            let p01 = view.data()[row1 + x0 as usize] as f32;
            let p11 = view.data()[row1 + x1 as usize] as f32;

            let w00 = (1.0 - wx) * (1.0 - wy);
            let w10 = wx * (1.0 - wy);
            let w01 = (1.0 - wx) * wy;
            let w11 = wx * wy;

            out[idx] = (p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11) / 255.0;
            idx += 1;
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_patch_center_identity() {
        let width = 5;
        let height = 5;
        let mut img = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                img[y * width + x] = (y * 10 + x) as u8;
            }
        }
        let view = ImageView::from_u8_slice(width, height, &img).unwrap();
        let mut out = vec![0.0f32; 9];
        let ok = extract_patch_u8_to_f32(view, 2.0, 2.0, 3, &mut out).is_some();
        assert!(ok);

        let expected = img[width + 1] as f32 / 255.0;
        assert!((out[0] - expected).abs() < 1e-6);

        let expected_center = img[2 * width + 2] as f32 / 255.0;
        assert!((out[4] - expected_center).abs() < 1e-6);
    }

    #[test]
    fn extract_patch_oob_reject() {
        let width = 5;
        let height = 5;
        let img = vec![0u8; width * height];
        let view = ImageView::from_u8_slice(width, height, &img).unwrap();
        let mut out = vec![0.0f32; 9];
        assert!(extract_patch_u8_to_f32(view, 0.0, 0.0, 3, &mut out).is_none());
    }

    #[test]
    fn missing_model_keeps_every_candidate() {
        let (w, h) = (32, 32);
        let img = vec![128u8; w * h];
        let view = ImageView::from_u8_slice(w, h, &img).unwrap();
        let params = MlRefinerParams {
            model_path: Some(PathBuf::from("missing.onnx")),
            patch_size: 3,
            ..Default::default()
        };
        let mut state = MlRefinerState::new(&params);
        let corners = vec![Corner::new(16.25, 15.5, 10.0), Corner::new(8.0, 9.0, 4.0)];
        let out = refine_corners(&mut state, corners.clone(), view);
        assert_eq!(out.len(), corners.len());
        for (a, b) in corners.iter().zip(&out) {
            assert_eq!((a.x, a.y, a.strength), (b.x, b.y, b.strength));
        }
    }
}
