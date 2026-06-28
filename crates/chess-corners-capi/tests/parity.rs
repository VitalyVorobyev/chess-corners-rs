//! Marshalling parity test.
//!
//! Runs the C ABI entry point against the Rust facade on the same synthetic
//! checkerboard and asserts that the returned corners agree, guarding the
//! `cc_config` ↔ `DetectorConfig` and `CornerDescriptor` → `cc_corner`
//! conversions.

use chess_corners::{Detector, DetectorConfig};
use chess_corners_capi::{cc_config_chess, cc_detect_u8, cc_result, cc_result_free, cc_status};

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

#[test]
fn ffi_detect_matches_reference() {
    let (img, w, h) = synthetic_checkerboard();

    // Reference via the Rust facade.
    let mut detector = Detector::new(DetectorConfig::chess()).expect("valid config");
    let reference = detector.detect_u8(&img, w, h).expect("detection succeeds");
    assert!(!reference.is_empty(), "fixture should yield corners");

    // C ABI path.
    let cfg = cc_config_chess();
    let mut out = cc_result {
        corners: std::ptr::null_mut(),
        len: 0,
    };
    // SAFETY: `img` is valid for `w * h` bytes; `cfg` and `out` are valid.
    let status = unsafe { cc_detect_u8(img.as_ptr(), w, h, &cfg, &mut out) };
    assert_eq!(status, cc_status::CC_OK);
    assert_eq!(out.len, reference.len(), "corner counts must match");
    assert!(!out.corners.is_null());

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
        for k in 0..2 {
            assert!((a.axes[k].angle - b.axes[k].angle).abs() < 1e-4);
            assert!((a.axes[k].sigma - b.axes[k].sigma).abs() < 1e-4);
        }
    }

    // SAFETY: `out` was written by `cc_detect_u8` and not yet freed.
    unsafe { cc_result_free(&mut out) };
    assert!(out.corners.is_null());
    assert_eq!(out.len, 0);

    // Free is idempotent / null-safe.
    // SAFETY: freeing an already-freed (now-null) result is a no-op.
    unsafe { cc_result_free(&mut out) };
}
