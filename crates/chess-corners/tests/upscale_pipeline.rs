//! Integration tests for the optional pre-pipeline upscaling stage.

use chess_corners::{find_chess_corners_u8, ChessConfig, ChessError, UpscaleConfig};

/// Render a synthetic quadrant-corner tile of the given size.
///
/// Half-plane split at (size/2, size/2). in_top ^ in_left chooses
/// bright. The ChESS corner sits at `(size as f32 / 2.0 - 0.5,
/// size as f32 / 2.0 - 0.5)` (the pixel junction).
fn quadrant_corner(size: u32, dark: u8, bright: u8) -> Vec<u8> {
    let s = size as usize;
    let mut out = vec![dark; s * s];
    let mid = s / 2;
    for y in 0..s {
        for x in 0..s {
            let in_top = y < mid;
            let in_left = x < mid;
            if in_top ^ in_left {
                out[y * s + x] = bright;
            }
        }
    }
    out
}

fn low_res_cfg() -> ChessConfig {
    let mut cfg = ChessConfig::default();
    cfg.threshold_value = 0.01;
    // Small images — drop the pyramid-min-size gate so the single-scale
    // path actually runs.
    cfg.pyramid_min_size = 16;
    cfg
}

#[test]
fn upscale_disabled_is_passthrough() {
    let size = 32u32;
    let img = quadrant_corner(size, 20, 220);
    let mut cfg = low_res_cfg();
    let baseline = find_chess_corners_u8(&img, size, size, &cfg).unwrap();
    cfg.upscale = UpscaleConfig::disabled();
    let passthrough = find_chess_corners_u8(&img, size, size, &cfg).unwrap();
    assert_eq!(baseline.len(), passthrough.len());
    for (a, b) in baseline.iter().zip(passthrough.iter()) {
        assert!((a.x - b.x).abs() < 1e-5 && (a.y - b.y).abs() < 1e-5);
    }
}

#[test]
fn upscale_recovers_corner_in_low_res_tile() {
    // A 14x14 tile has 11x11 usable response region (r=5 ring leaves
    // a 5 px margin on each side). The corner at (6.5, 6.5) is right
    // at the edge of the usable region, detection is flaky. Upscaling
    // by 2x turns the 14x14 tile into 28x28 — ample margin for the
    // ring to support the corner.
    let size = 14u32;
    let img = quadrant_corner(size, 20, 220);

    let mut cfg = low_res_cfg();
    cfg.upscale = UpscaleConfig::fixed(2);

    let corners = find_chess_corners_u8(&img, size, size, &cfg).unwrap();
    assert!(
        !corners.is_empty(),
        "expected upscale pipeline to find the low-res corner"
    );
    let best = corners
        .iter()
        .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
        .unwrap();
    let expected = (size as f32) / 2.0 - 0.5;
    assert!(
        (best.x - expected).abs() < 1.0 && (best.y - expected).abs() < 1.0,
        "corner at ({}, {}) not within 1px of expected ({}, {})",
        best.x,
        best.y,
        expected,
        expected
    );
}

#[test]
fn upscale_returns_input_frame_coordinates() {
    // Detection on upscaled input must report coordinates in the
    // original input pixel frame. Compare a disabled run against an
    // upscaled run on a size where both succeed — the corner
    // locations should agree to within interpolation noise.
    let size = 32u32;
    let img = quadrant_corner(size, 20, 220);

    let cfg_off = low_res_cfg();
    let mut cfg_on = cfg_off.clone();
    cfg_on.upscale = UpscaleConfig::fixed(2);

    let corners_off = find_chess_corners_u8(&img, size, size, &cfg_off).unwrap();
    let corners_on = find_chess_corners_u8(&img, size, size, &cfg_on).unwrap();
    assert!(!corners_off.is_empty());
    assert!(!corners_on.is_empty());

    let best_off = corners_off
        .iter()
        .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
        .unwrap();
    let best_on = corners_on
        .iter()
        .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
        .unwrap();
    // Tight tolerance: the inverse half-pixel-center mapping in
    // `rescale_descriptors_to_input` must put the upscaled corner in
    // the same input-pixel frame as the disabled run. A naive `x /= k`
    // biases by (k − 1) / (2k) ≈ 0.25 px at k = 2, which this
    // assertion will reject.
    assert!(
        (best_off.x - best_on.x).abs() < 0.1 && (best_off.y - best_on.y).abs() < 0.1,
        "upscaled corner ({}, {}) differs from disabled ({}, {})",
        best_on.x,
        best_on.y,
        best_off.x,
        best_off.y
    );
}

#[test]
fn upscale_fixed_factor_one_returns_err() {
    // `UpscaleMode::Fixed` with factor 1 is an invariant violation:
    // `UpscaleConfig::validate()` rejects it, and the detection
    // entrypoint must return Err instead of silently treating it as
    // disabled.
    let size = 32u32;
    let img = quadrant_corner(size, 20, 220);
    let mut cfg = low_res_cfg();
    cfg.upscale = UpscaleConfig::fixed(1);
    let result = find_chess_corners_u8(&img, size, size, &cfg);
    assert!(
        matches!(result, Err(ChessError::Upscale(_))),
        "expected Err(ChessError::Upscale) for factor=1, got {:?}",
        result
    );
}

#[test]
fn upscale_fixed_factor_zero_returns_err() {
    let size = 32u32;
    let img = quadrant_corner(size, 20, 220);
    let mut cfg = low_res_cfg();
    cfg.upscale = UpscaleConfig::fixed(0);
    let result = find_chess_corners_u8(&img, size, size, &cfg);
    assert!(
        matches!(result, Err(ChessError::Upscale(_))),
        "expected Err(ChessError::Upscale) for factor=0, got {:?}",
        result
    );
}
