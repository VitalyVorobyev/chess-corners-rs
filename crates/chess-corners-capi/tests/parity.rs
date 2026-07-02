//! Marshalling parity test.
//!
//! Runs the C ABI entry point against the Rust facade on the same synthetic
//! checkerboard and asserts that the returned corners agree, guarding the
//! `cc_config` ↔ `DetectorConfig` and `CornerDescriptor` → `cc_corner`
//! conversions. Beyond the plain presets, each newly-exposed flat knob is
//! exercised with a non-default value against the equivalent
//! `DetectorConfig`, so the flat → facade lowering is proven end-to-end.

use chess_corners::{ChessRing, Detector, DetectorConfig, PeakFitMode, UpscaleConfig};
use chess_corners_capi::{
    cc_config, cc_config_chess, cc_config_radon, cc_detect_u8, cc_result, cc_result_free,
    cc_status, CC_CHESS_RING_BROAD, CC_PEAK_FIT_PARABOLIC,
};

/// 128×128 black/white checkerboard of 16-pixel squares.
fn synthetic_checkerboard() -> (Vec<u8>, u32, u32) {
    let (w, h) = (128u32, 128u32);
    let mut img = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            if ((x / 16) + (y / 16)) % 2 == 0 {
                img[(y * w + x) as usize] = 255;
            }
        }
    }
    (img, w, h)
}

/// Detect through the Rust facade with `reference`, detect through the C ABI
/// with `cfg`, and assert the two corner sets agree exactly. Returns the
/// number of corners so callers can assert the case is non-vacuous.
fn assert_ffi_matches_facade(cfg: &cc_config, reference: DetectorConfig) -> usize {
    let (img, w, h) = synthetic_checkerboard();

    // Reference via the Rust facade.
    let mut detector = Detector::new(reference).expect("valid config");
    let reference = detector.detect_u8(&img, w, h).expect("detection succeeds");

    // C ABI path.
    let mut out = cc_result {
        corners: std::ptr::null_mut(),
        len: 0,
    };
    // SAFETY: `img` is valid for `w * h` bytes; `cfg` and `out` are valid.
    let status = unsafe { cc_detect_u8(img.as_ptr(), w, h, cfg, &mut out) };
    assert_eq!(status, cc_status::CC_OK);
    assert_eq!(out.len, reference.len(), "corner counts must match");
    if out.len > 0 {
        assert!(!out.corners.is_null());
    }

    // SAFETY: on `CC_OK` the library wrote `out.len` valid corners.
    let ffi = unsafe { std::slice::from_raw_parts(out.corners, out.len) };
    for (a, b) in ffi.iter().zip(reference.iter()) {
        assert!((a.x - b.x).abs() < 1e-4, "x mismatch: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < 1e-4, "y mismatch: {} vs {}", a.y, b.y);
        assert!(
            (a.response - b.response).abs() < 1e-4,
            "response mismatch: {} vs {}",
            a.response,
            b.response
        );
        match b.axes {
            Some(b_axes) => {
                assert_eq!(a.has_orientation, 1, "ffi corner should carry orientation");
                for (a_axis, b_axis) in a.axes.iter().zip(b_axes.iter()) {
                    assert!((a_axis.angle - b_axis.angle).abs() < 1e-4);
                    assert!((a_axis.sigma - b_axis.sigma).abs() < 1e-4);
                }
            }
            None => assert_eq!(a.has_orientation, 0, "orientation should be skipped"),
        }
    }

    let len = out.len;

    // SAFETY: `out` was written by `cc_detect_u8` and not yet freed.
    unsafe { cc_result_free(&mut out) };
    assert!(out.corners.is_null());
    assert_eq!(out.len, 0);

    // Free is idempotent / null-safe.
    // SAFETY: freeing an already-freed (now-null) result is a no-op.
    unsafe { cc_result_free(&mut out) };

    len
}

#[test]
fn ffi_chess_preset_matches_reference() {
    let len = assert_ffi_matches_facade(&cc_config_chess(), DetectorConfig::chess());
    assert!(len > 0, "fixture should yield corners");
}

/// Exercises the ChESS-side new flat knobs: `chess_ring` (Broad),
/// `upscale_factor` (2), and `merge_radius` (non-default) — each compared
/// corner-by-corner against the equivalent `DetectorConfig`.
#[test]
fn ffi_chess_new_fields_match_reference() {
    let mut cfg = cc_config_chess();
    cfg.chess_ring = CC_CHESS_RING_BROAD;
    cfg.upscale_factor = 2;
    cfg.merge_radius = 5.0;

    let reference = DetectorConfig::chess()
        .with_chess(|c| c.ring = ChessRing::Broad)
        .with_upscale(UpscaleConfig::fixed(2))
        .with_merge_radius(5.0);

    let len = assert_ffi_matches_facade(&cfg, reference);
    assert!(
        len > 0,
        "broad-ring upscaled detection should yield corners"
    );
}

/// Exercises the Radon-side new flat knobs: `ray_radius`, `image_upsample`,
/// `response_blur_radius`, and `peak_fit` (Parabolic) — each compared
/// corner-by-corner against the equivalent `DetectorConfig`.
#[test]
fn ffi_radon_new_fields_match_reference() {
    let mut cfg = cc_config_radon();
    cfg.ray_radius = 6;
    cfg.image_upsample = 1;
    cfg.response_blur_radius = 0;
    cfg.peak_fit = CC_PEAK_FIT_PARABOLIC;

    let reference = DetectorConfig::radon().with_radon(|r| {
        r.ray_radius = 6;
        r.image_upsample = 1;
        r.response_blur_radius = 0;
        r.peak_fit = PeakFitMode::Parabolic;
    });

    let len = assert_ffi_matches_facade(&cfg, reference);
    assert!(len > 0, "radon detection should yield corners");
}
