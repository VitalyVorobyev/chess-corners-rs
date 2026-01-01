//! ML subpixel refiner integration (feature-gated).

use crate::{ImageView, RefineResult, RefineStatus, Refiner, RefinerKind};
use chess_corners_core::descriptor::Corner;
use chess_corners_core::detect::detect_corners_from_response_with_refiner;
use chess_corners_core::{ChessParams, CornerRefiner, RefineContext, ResponseMap};
use chess_corners_ml::{MlModel, ModelSource};
use log::{info, warn};
use std::path::PathBuf;
#[cfg(feature = "tracing")]
use std::time::{Duration, Instant};
#[cfg(feature = "tracing")]
use tracing::info_span;

/// ML refiner fallback behavior when inference is unavailable.
#[derive(Clone, Debug)]
pub(crate) enum MlFallback {
    /// Keep the original candidate without refinement.
    KeepCandidate,
    /// Use the classic refiner configured in `ChessParams::refiner`.
    UseClassicRefiner,
    /// Drop the candidate entirely.
    Reject,
}

/// Configuration for the ML subpixel refiner.
#[derive(Clone, Debug)]
pub(crate) struct MlRefinerParams {
    pub model_path: Option<PathBuf>,
    pub patch_size: u32,
    pub batch_size: u32,
    pub fallback: MlFallback,
}

impl Default for MlRefinerParams {
    fn default() -> Self {
        // Keep fallback variants reachable for internal use without exposing them in the public API.
        let _ = (MlFallback::UseClassicRefiner, MlFallback::Reject);
        Self {
            model_path: None,
            patch_size: 21,
            batch_size: 64,
            fallback: MlFallback::KeepCandidate,
        }
    }
}

pub(crate) struct MlRefinerState {
    params: MlRefinerParams,
    model: Option<MlModel>,
    patch_size: usize,
    patch_area: usize,
    batch_size: usize,
    buffer: Vec<f32>,
    indices: Vec<usize>,
    fallback_refiner: Option<Refiner>,
}

impl MlRefinerState {
    pub(crate) fn new(params: &MlRefinerParams, fallback_kind: &RefinerKind) -> Self {
        let patch_size = params.patch_size.max(1) as usize;
        let patch_area = patch_size * patch_size;
        let batch_size = params.batch_size.max(1) as usize;
        let buffer = vec![0.0f32; batch_size * patch_area];
        let indices = Vec::with_capacity(batch_size);
        let fallback_refiner = match params.fallback {
            MlFallback::UseClassicRefiner => Some(Refiner::from_kind(fallback_kind.clone())),
            _ => None,
        };
        let model = load_model(params, patch_size);

        Self {
            params: params.clone(),
            model,
            patch_size,
            patch_area,
            batch_size,
            buffer,
            indices,
            fallback_refiner,
        }
    }
}

pub(crate) fn patch_radius(params: &MlRefinerParams) -> i32 {
    let size = params.patch_size.max(1) as i32;
    (size - 1) / 2
}

pub(crate) fn detect_corners_with_ml(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    state: &mut MlRefinerState,
) -> Vec<Corner> {
    let mut noop = NoopRefiner::new(patch_radius(&state.params));
    let candidates = detect_corners_from_response_with_refiner(resp, params, image, &mut noop);

    if candidates.is_empty() {
        return candidates;
    }

    let mut stats = MlRefineStats {
        total: candidates.len(),
        ..Default::default()
    };

    let image = match image {
        Some(view) => view,
        None => {
            warn!("ML refiner requires an image; falling back for all candidates");
            info!("ml refiner: total={} fallback=all (no image)", stats.total);
            return apply_fallbacks(
                resp,
                None,
                &state.params,
                &mut state.fallback_refiner,
                candidates,
            );
        }
    };

    let model = match state.model.as_ref() {
        Some(model) => model,
        None => {
            warn!("ML model unavailable; falling back for all candidates");
            info!("ml refiner: total={} fallback=all (no model)", stats.total);
            return apply_fallbacks(
                resp,
                Some(image),
                &state.params,
                &mut state.fallback_refiner,
                candidates,
            );
        }
    };

    if model.patch_size() != state.patch_size {
        warn!(
            "ML patch size mismatch (model {}, config {}); falling back",
            model.patch_size(),
            state.patch_size
        );
        info!(
            "ml refiner: total={} fallback=all (patch size mismatch)",
            stats.total
        );
        return apply_fallbacks(
            resp,
            Some(image),
            &state.params,
            &mut state.fallback_refiner,
            candidates,
        );
    }

    let ctx = RefineContext {
        image: Some(image),
        response: Some(resp),
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
    let mut infer_time = Duration::ZERO;
    #[cfg(feature = "tracing")]
    let mut infer_batches = 0usize;

    let mut results: Vec<Option<Corner>> = vec![None; candidates.len()];
    state.indices.clear();

    for (idx, corner) in candidates.iter().enumerate() {
        let offset = state.indices.len() * state.patch_area;
        let patch_slice = &mut state.buffer[offset..offset + state.patch_area];
        if extract_patch_u8_to_f32(
            image,
            corner.xy[0],
            corner.xy[1],
            state.patch_size,
            patch_slice,
        )
        .is_none()
        {
            stats.oob += 1;
            results[idx] = apply_fallback(corner, &state.params, &ctx, &mut state.fallback_refiner);
            continue;
        }

        stats.extracted += 1;
        state.indices.push(idx);
        if state.indices.len() == state.batch_size {
            let input = BatchInput {
                model,
                patch_size: state.patch_size,
                buffer: &state.buffer,
                candidates: &candidates,
                params: &state.params,
                ctx: &ctx,
                #[cfg(feature = "tracing")]
                infer_time: &mut infer_time,
                #[cfg(feature = "tracing")]
                infer_batches: &mut infer_batches,
            };
            run_batch(
                input,
                state.indices.len(),
                &state.indices,
                &mut state.fallback_refiner,
                &mut results,
                &mut stats,
            );
            state.indices.clear();
        }
    }

    if !state.indices.is_empty() {
        let input = BatchInput {
            model,
            patch_size: state.patch_size,
            buffer: &state.buffer,
            candidates: &candidates,
            params: &state.params,
            ctx: &ctx,
            #[cfg(feature = "tracing")]
            infer_time: &mut infer_time,
            #[cfg(feature = "tracing")]
            infer_batches: &mut infer_batches,
        };
        run_batch(
            input,
            state.indices.len(),
            &state.indices,
            &mut state.fallback_refiner,
            &mut results,
            &mut stats,
        );
        state.indices.clear();
    }

    let mut out = Vec::with_capacity(candidates.len());
    for (corner, refined) in candidates.into_iter().zip(results.into_iter()) {
        if let Some(mut c) = refined {
            c.strength = corner.strength;
            out.push(c);
        }
    }
    #[cfg(feature = "tracing")]
    {
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let infer_ms = infer_time.as_secs_f64() * 1000.0;
        tracing::info!(
            target: "chess_corners::ml",
            total_ms,
            infer_ms,
            infer_batches,
            candidates = stats.total,
            extracted = stats.extracted,
            oob = stats.oob,
            inferred = stats.inferred,
            applied = stats.applied,
            output = out.len(),
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
        out.len()
    );
    out
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
                    "ML model patch size {} does not match config {}; using fallback",
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

struct BatchInput<'a> {
    model: &'a MlModel,
    patch_size: usize,
    buffer: &'a [f32],
    candidates: &'a [Corner],
    params: &'a MlRefinerParams,
    ctx: &'a RefineContext<'a>,
    #[cfg(feature = "tracing")]
    infer_time: &'a mut Duration,
    #[cfg(feature = "tracing")]
    infer_batches: &'a mut usize,
}

fn run_batch(
    input: BatchInput<'_>,
    batch_count: usize,
    indices: &[usize],
    fallback_refiner: &mut Option<Refiner>,
    results: &mut [Option<Corner>],
    stats: &mut MlRefineStats,
) {
    let patch_area = input.patch_size * input.patch_size;
    let end = batch_count * patch_area;
    #[cfg(feature = "tracing")]
    let infer_start = Instant::now();
    let preds_result = input.model.infer_batch(&input.buffer[..end], batch_count);
    #[cfg(feature = "tracing")]
    {
        *input.infer_time += infer_start.elapsed();
        *input.infer_batches += 1;
    }
    let preds = match preds_result {
        Ok(preds) => preds,
        Err(err) => {
            warn!("ML inference failed: {err}");
            stats.infer_fail += indices.len();
            for &idx in indices {
                results[idx] = apply_fallback(
                    &input.candidates[idx],
                    input.params,
                    input.ctx,
                    fallback_refiner,
                );
            }
            return;
        }
    };

    let used = preds.len().min(indices.len());
    stats.inferred += used;
    for (slot, pred) in preds.iter().take(used).enumerate() {
        let idx = indices[slot];
        let corner = &input.candidates[idx];
        let dx = pred[0];
        let dy = pred[1];

        stats.applied += 1;
        results[idx] = Some(Corner {
            xy: [corner.xy[0] + dx, corner.xy[1] + dy],
            strength: corner.strength,
        });
    }
    if preds.len() < indices.len() {
        warn!(
            "ML output length {} shorter than batch size {}; falling back",
            preds.len(),
            indices.len()
        );
        stats.infer_fail += indices.len() - preds.len();
        for &idx in &indices[preds.len()..] {
            results[idx] = apply_fallback(
                &input.candidates[idx],
                input.params,
                input.ctx,
                fallback_refiner,
            );
        }
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

fn apply_fallbacks(
    resp: &ResponseMap,
    image: Option<ImageView<'_>>,
    params: &MlRefinerParams,
    fallback_refiner: &mut Option<Refiner>,
    candidates: Vec<Corner>,
) -> Vec<Corner> {
    let ctx = RefineContext {
        image,
        response: Some(resp),
    };
    let mut out = Vec::with_capacity(candidates.len());
    for corner in candidates {
        if let Some(refined) = apply_fallback(&corner, params, &ctx, fallback_refiner) {
            out.push(refined);
        }
    }
    out
}

fn apply_fallback(
    corner: &Corner,
    params: &MlRefinerParams,
    ctx: &RefineContext<'_>,
    fallback_refiner: &mut Option<Refiner>,
) -> Option<Corner> {
    match params.fallback {
        MlFallback::KeepCandidate => Some(Corner {
            xy: corner.xy,
            strength: corner.strength,
        }),
        MlFallback::Reject => None,
        MlFallback::UseClassicRefiner => {
            let refiner = fallback_refiner.as_mut()?;
            let res = refiner.refine(corner.xy, *ctx);
            if matches!(res.status, RefineStatus::Accepted) {
                Some(Corner {
                    xy: res.xy,
                    strength: corner.strength,
                })
            } else {
                None
            }
        }
    }
}

struct NoopRefiner {
    radius: i32,
}

impl NoopRefiner {
    fn new(radius: i32) -> Self {
        Self { radius }
    }
}

impl CornerRefiner for NoopRefiner {
    fn radius(&self) -> i32 {
        self.radius
    }

    fn refine(&mut self, seed_xy: [f32; 2], _ctx: RefineContext<'_>) -> RefineResult {
        RefineResult::accepted(seed_xy, 0.0)
    }
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
    let width = view.width as i32;
    let height = view.height as i32;
    if width <= 0 || height <= 0 {
        return None;
    }
    let patch_area = patch_size * patch_size;
    if out.len() < patch_area {
        return None;
    }

    let half = (patch_size as f32 - 1.0) * 0.5;
    let origin_x = view.origin[0] as f32;
    let origin_y = view.origin[1] as f32;

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
        let row0 = (y0 as usize) * view.width;
        let row1 = (y1 as usize) * view.width;

        for ix in 0..patch_size {
            let u = ix as f32 - half;
            let gx = x + u + origin_x;
            let x0 = gx.floor() as i32;
            let x1 = (x0 + 1).min(width - 1);
            let wx = gx - x0 as f32;

            let p00 = view.data[row0 + x0 as usize] as f32;
            let p10 = view.data[row0 + x1 as usize] as f32;
            let p01 = view.data[row1 + x0 as usize] as f32;
            let p11 = view.data[row1 + x1 as usize] as f32;

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
    fn ml_fallback_respects_refiner_config() {
        let w = 32;
        let h = 32;
        let mut resp = ResponseMap {
            w,
            h,
            data: vec![0.0f32; w * h],
        };
        let idx = |x: usize, y: usize| y * w + x;
        resp.data[idx(16, 16)] = 10.0;
        resp.data[idx(16, 17)] = 1.0;
        resp.data[idx(17, 16)] = 1.0;
        resp.data[idx(18, 16)] = 5.0;

        let params = ChessParams {
            refiner: RefinerKind::CenterOfMass(crate::CenterOfMassConfig { radius: 1 }),
            ..Default::default()
        };

        let ml_params = MlRefinerParams {
            model_path: Some(PathBuf::from("missing.onnx")),
            patch_size: 3,
            fallback: MlFallback::UseClassicRefiner,
            ..Default::default()
        };

        let mut state = MlRefinerState::new(&ml_params, &params.refiner);
        let corners = detect_corners_with_ml(&resp, &params, None, &mut state);
        assert_eq!(corners.len(), 1);

        let expected_x = (16.0 * 10.0 + 16.0 + 17.0) / 12.0;
        let expected_y = (16.0 * 10.0 + 17.0 + 16.0) / 12.0;
        let c = corners[0].xy;
        assert!((c[0] - expected_x).abs() < 1e-4);
        assert!((c[1] - expected_y).abs() < 1e-4);
    }
}
