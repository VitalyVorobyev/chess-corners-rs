//! Stable C ABI for the `chess_corners` detector facade.
//!
//! This crate is a thin marshalling layer; it owns no detection logic. It
//! exposes a flat, C-fillable configuration struct, a single self-contained
//! detect entry point, and an explicit allocate/free pair for the returned
//! corner array.
//!
//! # Ownership
//!
//! `cc_detect_u8` allocates the corner array; the caller must release it
//! with `cc_result_free`. The library frees exactly what it allocated, so
//! the corner array never crosses an allocator boundary. Pointers returned
//! by `cc_status_str` are static and must not be freed.
//!
//! # Reentrancy and panic-safety
//!
//! There is no global mutable state, so every call is self-contained and
//! reentrant. Every entry point that runs detection traps panics at the
//! boundary and reports `cc_status::CC_ERR_PANIC` rather than unwinding
//! across FFI (which would be undefined behaviour).
//!
//! # ABI version
//!
//! `cc_abi_version` returns a manually bumped integer that a C/C++ header
//! can check against its own expectation to detect a header/library
//! mismatch.

#![allow(non_camel_case_types)]
#![deny(unsafe_op_in_unsafe_fn)]

mod convert;

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use chess_corners::{Detector, DetectorConfig};

// ─── Enum-tag typedefs ──────────────────────────────────────────────────
//
// These tags are written by C, so they are plain `u32` typedefs with named
// constants rather than Rust enums: loading an out-of-range Rust enum
// discriminant from C-filled memory would be undefined behaviour. The
// conversion layer validates each tag and rejects unknown values.

/// Detector strategy tag stored in `cc_config::strategy`.
pub type cc_strategy_t = u32;
/// ChESS kernel detector.
pub const CC_STRATEGY_CHESS: cc_strategy_t = 0;
/// Whole-image Radon detector.
pub const CC_STRATEGY_RADON: cc_strategy_t = 1;

/// Subpixel-refiner tag stored in `cc_config::refiner`.
///
/// Refiner-specific tuning is not exposed over the flat ABI; the selected
/// refiner runs with its library defaults.
pub type cc_refiner_t = u32;
/// Center-of-mass refiner. Valid for both strategies.
pub const CC_REFINER_CENTER_OF_MASS: cc_refiner_t = 0;
/// Förstner refiner. Valid for the ChESS strategy only.
pub const CC_REFINER_FORSTNER: cc_refiner_t = 1;
/// Saddle-point refiner. Valid for the ChESS strategy only.
pub const CC_REFINER_SADDLE_POINT: cc_refiner_t = 2;
/// Radon-peak refiner. Valid for the Radon strategy only.
pub const CC_REFINER_RADON_PEAK: cc_refiner_t = 3;

/// Orientation-fit tag stored in `cc_config::orientation_method`.
pub type cc_orientation_method_t = u32;
/// 16-sample ring Gauss-Newton fit (the default).
pub const CC_ORIENTATION_RING_FIT: cc_orientation_method_t = 0;
/// Full-disk crossing-line estimator with a ring-fit fallback.
pub const CC_ORIENTATION_DISK_FIT: cc_orientation_method_t = 1;

// ─── Result types (library writes, C reads) ─────────────────────────────

/// One local grid-axis direction with its 1σ angular uncertainty.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cc_axis {
    /// Axis direction in radians.
    pub angle: f32,
    /// 1σ angular uncertainty in radians.
    pub sigma: f32,
}

/// One detected chessboard corner in full-resolution image pixels.
///
/// The two axes follow the joint polarity convention of the Rust
/// `CornerDescriptor`: `axes[0].angle` lies in `[0, π)` and `axes[1].angle`
/// in `(axes[0].angle, axes[0].angle + π)`, with the CCW arc between them
/// covering a dark sector of the corner.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cc_corner {
    /// Subpixel x position.
    pub x: f32,
    /// Subpixel y position.
    pub y: f32,
    /// Raw, unnormalized detector response at the peak.
    pub response: f32,
    /// The two local grid-axis directions.
    pub axes: [cc_axis; 2],
}

/// Owned array of detected corners returned by `cc_detect_u8`.
///
/// The library allocates `corners`; release it with `cc_result_free`.
#[repr(C)]
#[derive(Debug)]
pub struct cc_result {
    /// Pointer to `len` corners. May be null when `len == 0`.
    pub corners: *mut cc_corner,
    /// Number of corners pointed to by `corners`.
    pub len: usize,
}

// ─── Flat configuration (C writes) ──────────────────────────────────────

/// Flat, C-fillable detector configuration.
///
/// Construct one with a preset (`cc_config_default`, `cc_config_chess`, …)
/// and tweak the exposed fields. Knobs not represented here (per-strategy
/// ring/ray geometry, refiner tuning, pre-detection upscaling, cross-level
/// merge radius) fall back to the selected strategy preset's defaults.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cc_config {
    /// One of the `CC_STRATEGY_*` constants.
    pub strategy: cc_strategy_t,
    /// Acceptance threshold. ChESS reads it as an absolute response floor;
    /// Radon as a fraction of the per-frame maximum.
    pub threshold: f32,
    /// Non-maximum-suppression half-radius in working-resolution pixels.
    pub nms_radius: u32,
    /// Minimum positive-response neighbours required to accept a candidate.
    pub min_cluster_size: u32,
    /// One of the `CC_REFINER_*` constants. Must be valid for `strategy`.
    pub refiner: cc_refiner_t,
    /// One of the `CC_ORIENTATION_*` constants.
    pub orientation_method: cc_orientation_method_t,
    /// `0` runs single-scale detection; any non-zero value enables the
    /// default three-level coarse-to-fine pyramid.
    pub multiscale: u32,
}

// ─── Status codes (library returns) ─────────────────────────────────────

/// Status returned by fallible entry points.
///
/// This is the only Rust enum in the ABI: it is produced by the library and
/// read by C, so an invalid discriminant can never be loaded from C-filled
/// memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cc_status {
    /// Success.
    CC_OK = 0,
    /// A required pointer argument was null.
    CC_ERR_NULL_POINTER,
    /// The image dimensions were invalid (zero, or `width * height`
    /// overflows `size_t`).
    CC_ERR_DIMENSION_MISMATCH,
    /// The upscale configuration was invalid.
    CC_ERR_UPSCALE,
    /// The configuration was invalid (e.g. an enum tag was out of range, or
    /// a refiner not valid for the selected strategy).
    CC_ERR_INVALID_CONFIG,
    /// A panic was trapped at the FFI boundary.
    CC_ERR_PANIC,
}

// ─── Configuration presets ──────────────────────────────────────────────

/// Default configuration (single-scale ChESS).
#[no_mangle]
pub extern "C" fn cc_config_default() -> cc_config {
    config_preset(DetectorConfig::default)
}

/// Single-scale ChESS preset.
#[no_mangle]
pub extern "C" fn cc_config_chess() -> cc_config {
    config_preset(DetectorConfig::chess)
}

/// Three-level coarse-to-fine ChESS preset.
#[no_mangle]
pub extern "C" fn cc_config_chess_multiscale() -> cc_config {
    config_preset(DetectorConfig::chess_multiscale)
}

/// Single-scale Radon preset.
#[no_mangle]
pub extern "C" fn cc_config_radon() -> cc_config {
    config_preset(DetectorConfig::radon)
}

/// Three-level coarse-to-fine Radon preset.
#[no_mangle]
pub extern "C" fn cc_config_radon_multiscale() -> cc_config {
    config_preset(DetectorConfig::radon_multiscale)
}

/// Build a preset and flatten it, trapping the (in practice impossible)
/// panic so a preset constructor never unwinds across FFI.
fn config_preset(make: fn() -> DetectorConfig) -> cc_config {
    catch_unwind(AssertUnwindSafe(|| convert::flatten(&make())))
        .unwrap_or_else(|_| convert::zeroed_config())
}

// ─── Detection ──────────────────────────────────────────────────────────

/// Detect corners in an 8-bit, row-major grayscale image.
///
/// On `CC_OK`, `*out` owns a heap-allocated corner array that the caller
/// must release with `cc_result_free`. On any error status `*out` is left
/// untouched.
///
/// # Safety
///
/// - `pixels` must point to at least `width * height` readable bytes.
/// - `cfg` must point to a readable `cc_config`.
/// - `out` must point to a writable `cc_result`.
#[no_mangle]
pub unsafe extern "C" fn cc_detect_u8(
    pixels: *const u8,
    width: u32,
    height: u32,
    cfg: *const cc_config,
    out: *mut cc_result,
) -> cc_status {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        detect_impl(pixels, width, height, cfg, out)
    })) {
        Ok(status) => status,
        Err(_) => cc_status::CC_ERR_PANIC,
    }
}

unsafe fn detect_impl(
    pixels: *const u8,
    width: u32,
    height: u32,
    cfg: *const cc_config,
    out: *mut cc_result,
) -> cc_status {
    if pixels.is_null() || cfg.is_null() || out.is_null() {
        return cc_status::CC_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return cc_status::CC_ERR_DIMENSION_MISMATCH;
    }

    // SAFETY: `cfg` is non-null (checked above) and valid per the contract.
    let cfg = unsafe { &*cfg };
    let detector_cfg = match convert::to_detector_config(cfg) {
        Ok(c) => c,
        Err(status) => return status,
    };

    let Some(len) = (width as usize).checked_mul(height as usize) else {
        return cc_status::CC_ERR_DIMENSION_MISMATCH;
    };

    // SAFETY: the caller guarantees `pixels` is valid for `width * height`
    // bytes, which equals `len`.
    let pixels = unsafe { slice::from_raw_parts(pixels, len) };

    let mut detector = match Detector::new(detector_cfg) {
        Ok(d) => d,
        Err(e) => return convert::map_error(&e),
    };
    let corners = match detector.detect_u8(pixels, width, height) {
        Ok(c) => c,
        Err(e) => return convert::map_error(&e),
    };

    // The lib allocates the array (as a `Box<[cc_corner]>`) so the matching
    // `cc_result_free` can release it with the same allocator.
    let boxed: Box<[cc_corner]> = corners.iter().map(convert::corner_to_ffi).collect();
    let out_len = boxed.len();
    let corners_ptr = Box::into_raw(boxed) as *mut cc_corner;

    // SAFETY: `out` is non-null (checked above) and writable per the contract.
    unsafe {
        *out = cc_result {
            corners: corners_ptr,
            len: out_len,
        };
    }
    cc_status::CC_OK
}

/// Release a `cc_result` previously written by `cc_detect_u8`.
///
/// Null-safe and idempotent: after the call `r->corners` is null and
/// `r->len` is zero.
///
/// # Safety
///
/// `r` must be null, or point to a `cc_result` previously written by
/// `cc_detect_u8` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn cc_result_free(r: *mut cc_result) {
    // Dropping a `Box<[cc_corner]>` of plain data cannot panic, but trap
    // anyway so this function can never unwind across FFI.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { free_impl(r) }));
}

unsafe fn free_impl(r: *mut cc_result) {
    if r.is_null() {
        return;
    }
    // SAFETY: `r` is non-null (checked above) and valid per the contract.
    let result = unsafe { &mut *r };
    if !result.corners.is_null() && result.len != 0 {
        let fat = ptr::slice_from_raw_parts_mut(result.corners, result.len);
        // SAFETY: `corners`/`len` describe a `Box<[cc_corner]>` produced by
        // `cc_detect_u8`; reconstruct that exact box and drop it.
        drop(unsafe { Box::from_raw(fat) });
    }
    result.corners = ptr::null_mut();
    result.len = 0;
}

// ─── Introspection ──────────────────────────────────────────────────────

/// Human-readable, static description of a status code.
///
/// The returned pointer is a static, NUL-terminated string and must not be
/// freed. `status` must be a value previously returned by this library.
#[no_mangle]
pub extern "C" fn cc_status_str(status: cc_status) -> *const c_char {
    let bytes: &[u8] = match status {
        cc_status::CC_OK => b"ok\0",
        cc_status::CC_ERR_NULL_POINTER => b"null pointer argument\0",
        cc_status::CC_ERR_DIMENSION_MISMATCH => b"invalid image dimensions\0",
        cc_status::CC_ERR_UPSCALE => b"invalid upscale configuration\0",
        cc_status::CC_ERR_INVALID_CONFIG => b"invalid configuration\0",
        cc_status::CC_ERR_PANIC => b"internal panic trapped at the FFI boundary\0",
    };
    bytes.as_ptr().cast()
}

/// ABI version of this library. Bumped manually on any breaking ABI change.
#[no_mangle]
pub extern "C" fn cc_abi_version() -> u32 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each preset must round-trip flatten → `to_detector_config` back to the
    /// exact facade config it was built from.
    fn assert_round_trip(make: fn() -> DetectorConfig) {
        let flat = config_preset(make);
        let restored = convert::to_detector_config(&flat).expect("preset is valid");
        assert_eq!(restored, make());
    }

    #[test]
    fn default_preset_round_trips() {
        assert_round_trip(DetectorConfig::default);
    }

    #[test]
    fn chess_preset_round_trips() {
        assert_round_trip(DetectorConfig::chess);
    }

    #[test]
    fn chess_multiscale_preset_round_trips() {
        assert_round_trip(DetectorConfig::chess_multiscale);
    }

    #[test]
    fn radon_preset_round_trips() {
        assert_round_trip(DetectorConfig::radon);
    }

    #[test]
    fn radon_multiscale_preset_round_trips() {
        assert_round_trip(DetectorConfig::radon_multiscale);
    }

    #[test]
    fn out_of_range_tags_are_rejected() {
        let mut cfg = cc_config_chess();
        cfg.strategy = 99;
        assert_eq!(
            convert::to_detector_config(&cfg),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn cross_strategy_refiner_is_rejected() {
        // RadonPeak is not a valid ChESS refiner.
        let mut cfg = cc_config_chess();
        cfg.refiner = CC_REFINER_RADON_PEAK;
        assert_eq!(
            convert::to_detector_config(&cfg),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn null_arguments_are_rejected() {
        let mut out = cc_result {
            corners: ptr::null_mut(),
            len: 0,
        };
        let cfg = cc_config_chess();
        // SAFETY: passing a null pixel pointer is exactly the contract
        // violation the null check guards against.
        let status = unsafe { cc_detect_u8(ptr::null(), 8, 8, &cfg, &mut out) };
        assert_eq!(status, cc_status::CC_ERR_NULL_POINTER);
    }

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(cc_abi_version(), 2);
    }
}
