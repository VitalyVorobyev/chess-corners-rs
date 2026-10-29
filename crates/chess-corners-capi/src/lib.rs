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

// Justified: the C ABI surface uses C-style `cc_*` type names (structs,
// enums, typedefs) to match the generated header and the wider C/C++
// naming convention consumers expect, rather than Rust's `CamelCase`.
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
/// refiner runs with its library defaults. A tag not valid for the
/// selected strategy is rejected with `CC_ERR_INVALID_CONFIG`.
pub type cc_refiner_t = u32;
/// Center-of-mass refiner. ChESS strategy only (the ChESS default).
pub const CC_REFINER_CENTER_OF_MASS: cc_refiner_t = 0;
/// Saddle-point refiner. Valid for both strategies. Under Radon a corner
/// whose refinement does not converge keeps its peak-fit position and
/// covariance.
pub const CC_REFINER_SADDLE_POINT: cc_refiner_t = 1;
/// Radon's response-map peak fit, with no image refinement. Radon strategy
/// only (the Radon default).
pub const CC_REFINER_PEAK_FIT: cc_refiner_t = 2;

/// Orientation-fit tag stored in `cc_config::orientation_method`.
pub type cc_orientation_method_t = u32;
/// 16-sample ring Gauss-Newton fit (the default).
pub const CC_ORIENTATION_RING_FIT: cc_orientation_method_t = 0;
/// Full-disk crossing-line estimator with a ring-fit fallback.
pub const CC_ORIENTATION_DISK_FIT: cc_orientation_method_t = 1;
/// Skip the per-corner orientation fit. Detected corners then have
/// `cc_corner::has_orientation == 0` and a zeroed `axes` array.
pub const CC_ORIENTATION_NONE: cc_orientation_method_t = 2;

/// Position-uncertainty tag stored in `cc_config::uncertainty`.
pub type cc_uncertainty_t = u32;
/// No position covariance and no noise estimate: every corner has
/// `has_position_covariance == 0`. The default: every preset
/// (`cc_config_default()` and the strategy presets) and a zero-initialised
/// `cc_config` select it. `has_noise_sigma` and `noise_sigma` are not read.
pub const CC_UNCERTAINTY_OFF: cc_uncertainty_t = 0;
/// Every corner whose localizer supports it reports a position covariance:
/// image noise propagated through the localizer plus its systematic error.
/// That error includes a per-localizer floor measured on real images,
/// usually the largest part of the covariance there; weighting by it has not
/// been shown to beat uniform weighting on real data. The noise level comes
/// from `cc_config::has_noise_sigma` / `cc_config::noise_sigma`. Positions
/// are identical to `CC_UNCERTAINTY_OFF`.
pub const CC_UNCERTAINTY_PROPAGATED: cc_uncertainty_t = 1;

/// ChESS sampling-ring tag stored in `cc_config::chess_ring`.
///
/// Applies to the ChESS strategy only; ignored when `strategy` is
/// `CC_STRATEGY_RADON`.
pub type cc_chess_ring_t = u32;
/// Paper-default radius-5 ring (16 samples).
pub const CC_CHESS_RING_CANONICAL: cc_chess_ring_t = 0;
/// Radius-10 ring: a larger support window that samples farther from
/// the candidate centre.
pub const CC_CHESS_RING_BROAD: cc_chess_ring_t = 1;

/// Radon subpixel peak-fit tag stored in `cc_config::peak_fit`.
///
/// Applies to the Radon strategy only; ignored when `strategy` is
/// `CC_STRATEGY_CHESS`.
pub type cc_peak_fit_t = u32;
/// Parabolic fit on the raw response values.
pub const CC_PEAK_FIT_PARABOLIC: cc_peak_fit_t = 0;
/// Parabolic fit on `log(response)` (Gaussian peak). The Radon default.
pub const CC_PEAK_FIT_GAUSSIAN: cc_peak_fit_t = 1;

// ─── Result types (library writes, C reads) ─────────────────────────────

/// One local grid-axis direction with a fit-quality score.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cc_axis {
    /// Axis direction in radians.
    pub angle: f32,
    /// Fit-quality score in radians: larger means a less reliable axis. Not a
    /// calibrated 1σ; it ignores the noise level and `cc_uncertainty_t`.
    /// Use it for ranking and gating.
    pub sigma: f32,
}

/// Symmetric 2×2 covariance of a corner position, in px².
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cc_covariance {
    /// Variance of x.
    pub xx: f32,
    /// Covariance of x and y.
    pub xy: f32,
    /// Variance of y.
    pub yy: f32,
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
    /// The two local grid-axis directions. Valid only when
    /// `has_orientation` is `1`; zeroed otherwise.
    pub axes: [cc_axis; 2],
    /// Covariance of `(x, y)` in px²: first-order scatter due to white
    /// pixel noise plus the localizer's systematic error (the Rust
    /// `Corner::position_covariance` documents the model). Valid only
    /// when `has_position_covariance` is `1`; zeroed otherwise.
    pub position_covariance: cc_covariance,
    /// `1` when `axes` holds a fitted orientation, `0` when the
    /// orientation fit was skipped (`CC_ORIENTATION_NONE`) and `axes`
    /// is zeroed.
    pub has_orientation: u8,
    /// `1` when `position_covariance` is set. Under
    /// `CC_UNCERTAINTY_PROPAGATED` every localizer reports one (a ChESS
    /// saddle-point refinement only when it converged); none does under
    /// `CC_UNCERTAINTY_OFF` or when the input is upscaled
    /// (`upscale_factor` other than `0`).
    pub has_position_covariance: u8,
}

/// Owned array of detected corners returned by `cc_detect_u8`, with the
/// image noise level behind their position covariances.
///
/// The library allocates `corners`; release it with `cc_result_free`.
#[repr(C)]
#[derive(Debug)]
pub struct cc_result {
    /// Pointer to `len` corners. May be null when `len == 0`.
    pub corners: *mut cc_corner,
    /// Number of corners pointed to by `corners`.
    pub len: usize,
    /// `1` when `noise_sigma` is set: at least one corner has a position
    /// covariance. `0` under `CC_UNCERTAINTY_OFF`, on upscaled input, or
    /// when the noise estimate declined.
    pub has_noise_sigma: u32,
    /// Image noise σ in gray levels the corners' position covariances were
    /// computed with: `cc_config::noise_sigma` when supplied, otherwise the
    /// per-image estimate (what `cc_estimate_noise_sigma_u8` returns for the
    /// same image). `0` when `has_noise_sigma` is `0`.
    pub noise_sigma: f32,
}

// ─── Flat configuration (C writes) ──────────────────────────────────────

/// Flat, C-fillable detector configuration.
///
/// Construct one with a preset (`cc_config_default`, `cc_config_chess`, …)
/// and tweak the exposed fields. Every `DetectorConfig` knob is
/// representable here except two that fall back to library defaults:
/// refiner-specific tuning (only the refiner *kind* is exposed) and the
/// multiscale pyramid detail (level count, minimum size, refinement
/// radius) behind the on/off `multiscale` switch.
///
/// Fields are grouped by applicability: `chess_ring` applies to the
/// ChESS strategy only, while `ray_radius` / `image_upsample` /
/// `response_blur_radius` / `peak_fit` apply to the Radon strategy only.
/// Fields that do not apply to the active `strategy` are ignored.
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
    /// strategy's preset coarse-to-fine pyramid (three levels for ChESS,
    /// two for Radon).
    pub multiscale: u32,
    /// Cross-level duplicate-suppression radius in base-image pixels.
    /// After coarse-level seeds are refined into the base image, refined
    /// positions within this radius are merged into one output corner.
    pub merge_radius: f32,
    /// Pre-pipeline integer upscale factor. `0` disables upscaling (the
    /// explicit off-state); `2`, `3`, or `4` upscale by that factor before
    /// detection. Any other value (including `1`) is rejected with
    /// `CC_ERR_UPSCALE`. Output coordinates are rescaled back to the input
    /// pixel frame.
    pub upscale_factor: u32,
    /// One of the `CC_CHESS_RING_*` constants. Applies to the ChESS
    /// strategy only; ignored for Radon.
    pub chess_ring: cc_chess_ring_t,
    /// Radon ray half-length in working-resolution pixels (ray has
    /// `2·ray_radius + 1` samples). Applies to the Radon strategy only;
    /// ignored for ChESS. Clamped to at least `1` by the core detector.
    pub ray_radius: u32,
    /// Radon image-level supersampling factor applied before ray
    /// integration (`1` or `2`). Applies to the Radon strategy only;
    /// ignored for ChESS. Values `≥ 3` are clamped to `2`.
    pub image_upsample: u32,
    /// Half-size of the box blur applied to the Radon response map (`0`
    /// disables it). Applies to the Radon strategy only; ignored for ChESS.
    pub response_blur_radius: u32,
    /// One of the `CC_PEAK_FIT_*` constants. Applies to the Radon strategy
    /// only; ignored for ChESS.
    pub peak_fit: cc_peak_fit_t,
    /// The noise level of `CC_UNCERTAINTY_PROPAGATED`, which turns each
    /// corner's position covariance into px²: `0` estimates it from each
    /// image; any non-zero value uses `noise_sigma`. Read only under
    /// `CC_UNCERTAINTY_PROPAGATED`.
    pub has_noise_sigma: u32,
    /// Standard deviation of the image's pixel noise in gray levels: the
    /// total noise of the 8-bit image, rounding included, as
    /// `cc_estimate_noise_sigma_u8` measures it (finite, `>= 0`; otherwise
    /// `CC_ERR_INVALID_CONFIG`). Read only under `CC_UNCERTAINTY_PROPAGATED`
    /// when `has_noise_sigma` is non-zero.
    pub noise_sigma: f32,
    /// One of the `CC_UNCERTAINTY_*` constants (any other value is
    /// `CC_ERR_INVALID_CONFIG`). The presets and a zero-initialised config
    /// select `CC_UNCERTAINTY_OFF`.
    pub uncertainty: cc_uncertainty_t,
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

/// Default configuration (single-scale ChESS). Its `uncertainty` is
/// `CC_UNCERTAINTY_OFF`, like every preset. Start from this or a strategy
/// preset rather than a zero-initialised `cc_config`, whose threshold and
/// NMS fields are zero.
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

/// Two-level coarse-to-fine Radon preset.
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
    // SAFETY: `AssertUnwindSafe` is sound here because a caught panic
    // discards the closure's captured state (`pixels`/`cfg`/`out` are
    // `Copy` pointers, not shared mutable references) — there is no
    // partially-mutated state left observable afterward. `detect_impl`'s
    // own pointer preconditions are documented on this function's
    // `# Safety` section and re-checked (null/dimension) before any
    // dereference, so unwinding out of this boundary and reporting
    // `CC_ERR_PANIC` is the only externally-visible effect of a panic.
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
    let noise_sigma = detector.last_noise_sigma();

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
            has_noise_sigma: u32::from(noise_sigma.is_some()),
            noise_sigma: noise_sigma.unwrap_or(0.0),
        };
    }
    cc_status::CC_OK
}

/// Release a `cc_result` previously written by `cc_detect_u8`.
///
/// Null-safe and idempotent: after the call `r->corners` is null and
/// `r->len`, `r->has_noise_sigma` and `r->noise_sigma` are zero.
///
/// # Safety
///
/// `r` must be null, or point to a `cc_result` previously written by
/// `cc_detect_u8` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn cc_result_free(r: *mut cc_result) {
    // Dropping a `Box<[cc_corner]>` of plain data cannot panic, but trap
    // anyway so this function can never unwind across FFI.
    //
    // SAFETY: `AssertUnwindSafe` is sound because `free_impl` only
    // consumes the raw pointer `r` (itself `Copy`) and never leaves
    // shared mutable state half-updated before a possible panic point —
    // it either fully nulls out `*r` or (on an unexpected panic from the
    // infallible `Box::from_raw`/`drop`) the caller never touches `*r`
    // again per this function's `# Safety` contract. `free_impl`'s own
    // pointer preconditions are documented above and re-checked
    // (null-safety) before any dereference.
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
    result.has_noise_sigma = 0;
    result.noise_sigma = 0.0;
}

// ─── Noise estimate ─────────────────────────────────────────────────────

/// Estimate the standard deviation of the pixel noise of an 8-bit,
/// row-major grayscale image, in gray levels: the per-image estimate
/// `cc_detect_u8` uses to express position covariances when
/// `cc_config::has_noise_sigma` is `0`.
///
/// On `CC_OK`, `*out_has_sigma` is `1` and `*out_sigma` holds the estimate,
/// or `*out_has_sigma` is `0` (and `*out_sigma` is `0`) when the image is
/// smaller than 4×4 or has too few structure-free areas for an estimate.
/// On any error status the outputs are left untouched.
///
/// # Safety
///
/// - `pixels` must point to at least `width * height` readable bytes.
/// - `out_sigma` and `out_has_sigma` must point to writable values.
#[no_mangle]
pub unsafe extern "C" fn cc_estimate_noise_sigma_u8(
    pixels: *const u8,
    width: u32,
    height: u32,
    out_sigma: *mut f32,
    out_has_sigma: *mut u32,
) -> cc_status {
    // SAFETY: as in `cc_detect_u8` — the closure captures only `Copy`
    // pointers, re-checked for null before any dereference, so a caught
    // panic leaves no partially-mutated state observable.
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        noise_impl(pixels, width, height, out_sigma, out_has_sigma)
    })) {
        Ok(status) => status,
        Err(_) => cc_status::CC_ERR_PANIC,
    }
}

unsafe fn noise_impl(
    pixels: *const u8,
    width: u32,
    height: u32,
    out_sigma: *mut f32,
    out_has_sigma: *mut u32,
) -> cc_status {
    if pixels.is_null() || out_sigma.is_null() || out_has_sigma.is_null() {
        return cc_status::CC_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return cc_status::CC_ERR_DIMENSION_MISMATCH;
    }
    let Some(len) = (width as usize).checked_mul(height as usize) else {
        return cc_status::CC_ERR_DIMENSION_MISMATCH;
    };
    // SAFETY: the caller guarantees `pixels` is valid for `width * height`
    // bytes, which equals `len`.
    let pixels = unsafe { slice::from_raw_parts(pixels, len) };
    let sigma = match chess_corners::estimate_noise_sigma_u8(pixels, width, height) {
        Ok(s) => s,
        Err(e) => return convert::map_error(&e),
    };
    // SAFETY: both outputs are non-null (checked above) and writable per
    // the contract.
    unsafe {
        *out_sigma = sigma.unwrap_or(0.0);
        *out_has_sigma = u32::from(sigma.is_some());
    }
    cc_status::CC_OK
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
    8
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
    fn out_of_range_refiner_is_rejected() {
        let mut cfg = cc_config_chess();
        cfg.refiner = 99;
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
            has_noise_sigma: 0,
            noise_sigma: 0.0,
        };
        let cfg = cc_config_chess();
        // SAFETY: passing a null pixel pointer is exactly the contract
        // violation the null check guards against.
        let status = unsafe { cc_detect_u8(ptr::null(), 8, 8, &cfg, &mut out) };
        assert_eq!(status, cc_status::CC_ERR_NULL_POINTER);
    }

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(cc_abi_version(), 8);
    }

    #[test]
    fn out_of_range_chess_ring_is_rejected() {
        let mut cfg = cc_config_chess();
        cfg.chess_ring = 99;
        assert_eq!(
            convert::to_detector_config(&cfg),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn out_of_range_peak_fit_is_rejected() {
        let mut cfg = cc_config_radon();
        cfg.peak_fit = 99;
        assert_eq!(
            convert::to_detector_config(&cfg),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn invalid_upscale_factor_is_rejected() {
        // 0 is the explicit off-state (valid); 1 and 5 are invalid factors.
        for bad in [1u32, 5, 8] {
            let mut cfg = cc_config_chess();
            cfg.upscale_factor = bad;
            assert_eq!(
                convert::to_detector_config(&cfg),
                Err(cc_status::CC_ERR_UPSCALE),
                "factor {bad} should be rejected"
            );
        }
    }

    /// A noisy 8 px-cell board, `BOARD × BOARD` pixels.
    const BOARD: u32 = 96;

    fn board() -> Vec<u8> {
        let size = BOARD;
        let mut img = vec![0u8; (size * size) as usize];
        let mut state = 7u32;
        for y in 0..size {
            for x in 0..size {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = (state >> 16) % 7;
                let bright = ((x / 8) + (y / 8)) % 2 == 0;
                img[(y * size + x) as usize] = if bright { 197 } else { 37 } + noise as u8;
            }
        }
        img
    }

    /// Detect on [`board`] through the C entry point and return the corners
    /// (freed before returning).
    fn detect_board(cfg: &cc_config) -> Result<Vec<cc_corner>, cc_status> {
        detect_board_frame(cfg).map(|(corners, _)| corners)
    }

    /// [`detect_board`] with the result's noise σ (`None` when unset).
    fn detect_board_frame(cfg: &cc_config) -> Result<(Vec<cc_corner>, Option<f32>), cc_status> {
        let (img, size) = (board(), BOARD);
        let mut out = cc_result {
            corners: ptr::null_mut(),
            len: 0,
            has_noise_sigma: 0,
            noise_sigma: 0.0,
        };
        // SAFETY: `img` holds `size * size` bytes; `cfg` and `out` are valid.
        let status = unsafe { cc_detect_u8(img.as_ptr(), size, size, cfg, &mut out) };
        if status != cc_status::CC_OK {
            return Err(status);
        }
        // SAFETY: on success `out` describes `len` initialised corners.
        let corners = unsafe { slice::from_raw_parts(out.corners, out.len) }.to_vec();
        let sigma = (out.has_noise_sigma == 1).then_some(out.noise_sigma);
        // SAFETY: `out` came from `cc_detect_u8` and is freed exactly once.
        unsafe { cc_result_free(&mut out) };
        assert_eq!((out.has_noise_sigma, out.noise_sigma), (0, 0.0));
        Ok((corners, sigma))
    }

    #[test]
    fn result_reports_the_noise_sigma() {
        let img = board();
        let (mut estimate, mut has) = (0.0f32, 0u32);
        // SAFETY: `img` holds `BOARD * BOARD` bytes; both outputs are writable.
        let status = unsafe {
            cc_estimate_noise_sigma_u8(img.as_ptr(), BOARD, BOARD, &mut estimate, &mut has)
        };
        assert_eq!((status, has), (cc_status::CC_OK, 1));
        for base in [cc_config_chess(), cc_config_radon()] {
            let mut cfg = base;
            cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
            let (_, sigma) = detect_board_frame(&cfg).expect("detection");
            assert_eq!(sigma, Some(estimate), "estimated per image");
            cfg.has_noise_sigma = 1;
            cfg.noise_sigma = 1.25;
            let (_, sigma) = detect_board_frame(&cfg).expect("detection");
            assert_eq!(sigma, Some(1.25), "supplied");
            // The noise level is not read under `CC_UNCERTAINTY_OFF`, not
            // even to validate it.
            cfg.uncertainty = CC_UNCERTAINTY_OFF;
            cfg.noise_sigma = -1.0;
            let (corners, sigma) = detect_board_frame(&cfg).expect("detection");
            assert!(!corners.is_empty());
            assert_eq!(sigma, None, "no covariance, no noise level");
        }
    }

    #[test]
    fn saddle_point_corners_carry_a_covariance() {
        let mut cfg = cc_config_chess();
        cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
        cfg.refiner = CC_REFINER_SADDLE_POINT;
        let corners = detect_board(&cfg).expect("detection");
        assert!(!corners.is_empty());
        assert!(corners.iter().any(|c| c.has_position_covariance == 1));
        for c in corners.iter().filter(|c| c.has_position_covariance == 1) {
            let p = c.position_covariance;
            assert!(p.xx > 0.0 && p.yy > 0.0 && p.xx * p.yy > p.xy * p.xy);
        }
        // The default refiner reports one too.
        cfg.refiner = cc_config_chess().refiner;
        let plain = detect_board(&cfg).expect("detection");
        assert!(plain.iter().any(|c| c.has_position_covariance == 1));
    }

    #[test]
    fn uncertainty_off_drops_covariance_not_positions() {
        for mut base in [cc_config_chess(), cc_config_radon()] {
            base.uncertainty = CC_UNCERTAINTY_PROPAGATED;
            let on = detect_board(&base).expect("detection");
            assert!(on.iter().any(|c| c.has_position_covariance == 1));
            let mut cfg = base;
            cfg.uncertainty = CC_UNCERTAINTY_OFF;
            let off = detect_board(&cfg).expect("detection");
            assert!(!off.is_empty());
            assert!(off.iter().all(|c| c.has_position_covariance == 0));
            assert_eq!(on.len(), off.len());
            for (a, b) in on.iter().zip(off.iter()) {
                assert_eq!((a.x, a.y), (b.x, b.y));
            }
        }
    }

    #[test]
    fn uncertainty_tag_is_validated_and_defaults_to_off() {
        for cfg in [
            cc_config_default(),
            cc_config_chess(),
            cc_config_chess_multiscale(),
            cc_config_radon(),
            cc_config_radon_multiscale(),
        ] {
            assert_eq!(cfg.uncertainty, CC_UNCERTAINTY_OFF);
        }
        let corners = detect_board(&cc_config_default()).expect("detection");
        assert!(!corners.is_empty());
        assert!(corners.iter().all(|c| c.has_position_covariance == 0));
        let mut cfg = cc_config_chess();
        let rs = convert::to_detector_config(&cfg).expect("valid");
        assert_eq!(rs.uncertainty, chess_corners::Uncertainty::Off);
        // Tag 0, the value of a zero-initialised config, is `Off`.
        assert_eq!(CC_UNCERTAINTY_OFF, 0);
        // `Propagated` carries its noise level both ways.
        use chess_corners::{NoiseLevel, Uncertainty};
        cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
        for (has, sigma, noise) in [
            (0, 0.0, NoiseLevel::Estimate),
            (1, 1.5, NoiseLevel::Known(1.5)),
        ] {
            cfg.has_noise_sigma = has;
            cfg.noise_sigma = sigma;
            let rs = convert::to_detector_config(&cfg).expect("valid");
            assert_eq!(rs.uncertainty, Uncertainty::Propagated { noise });
            let flat = convert::flatten(&rs);
            assert_eq!(
                (flat.uncertainty, flat.has_noise_sigma, flat.noise_sigma),
                (CC_UNCERTAINTY_PROPAGATED, has, sigma)
            );
        }
        cfg.uncertainty = 99;
        assert_eq!(
            convert::to_detector_config(&cfg),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn noise_sigma_is_read_only_under_propagated() {
        let mut cfg = cc_config_chess();
        assert_eq!(cfg.has_noise_sigma, 0);
        cfg.has_noise_sigma = 1;
        cfg.noise_sigma = -1.0;
        // Off: not read, so neither validated nor carried.
        let rs = convert::to_detector_config(&cfg).expect("valid");
        assert_eq!(rs.uncertainty, chess_corners::Uncertainty::Off);
        let back = convert::flatten(&rs);
        assert_eq!((back.has_noise_sigma, back.noise_sigma), (0, 0.0));
        assert!(detect_board(&cfg).is_ok());
        // Propagated: validated when the detector is built.
        cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
        assert_eq!(
            detect_board(&cfg).map(|c| c.len()),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn radon_refiner_tags_are_validated_and_round_trip() {
        let mut cfg = cc_config_radon();
        assert_eq!(cfg.refiner, CC_REFINER_PEAK_FIT);
        cfg.refiner = CC_REFINER_SADDLE_POINT;
        let rs = convert::to_detector_config(&cfg).expect("valid");
        assert_eq!(convert::flatten(&rs).refiner, CC_REFINER_SADDLE_POINT);
        for bad in [CC_REFINER_CENTER_OF_MASS, 99] {
            cfg.refiner = bad;
            assert_eq!(
                convert::to_detector_config(&cfg),
                Err(cc_status::CC_ERR_INVALID_CONFIG),
                "tag {bad} under Radon"
            );
        }
        let mut chess = cc_config_chess();
        chess.refiner = CC_REFINER_PEAK_FIT;
        assert_eq!(
            convert::to_detector_config(&chess),
            Err(cc_status::CC_ERR_INVALID_CONFIG)
        );
    }

    #[test]
    fn radon_saddle_point_corners_carry_a_covariance() {
        let mut cfg = cc_config_radon();
        cfg.uncertainty = CC_UNCERTAINTY_PROPAGATED;
        cfg.has_noise_sigma = 1;
        cfg.noise_sigma = 2.0;
        cfg.refiner = CC_REFINER_SADDLE_POINT;
        let corners = detect_board(&cfg).expect("detection");
        assert!(corners.iter().any(|c| c.has_position_covariance == 1));
        cfg.refiner = CC_REFINER_PEAK_FIT;
        let plain = detect_board(&cfg).expect("detection");
        assert!(!plain.is_empty());
        assert!(plain.iter().any(|c| c.has_position_covariance == 1));
    }

    #[test]
    fn noise_estimate_matches_the_facade() {
        let size = 96u32;
        let mut img = vec![0u8; (size * size) as usize];
        let mut state = 3u32;
        for y in 0..size {
            for x in 0..size {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let bright = ((x / 8) + (y / 8)) % 2 == 0;
                img[(y * size + x) as usize] =
                    if bright { 190 } else { 50 } + ((state >> 16) % 9) as u8;
            }
        }
        let (mut sigma, mut has) = (-1.0f32, 7u32);
        // SAFETY: `img` holds `size * size` bytes; the outputs are writable.
        let st =
            unsafe { cc_estimate_noise_sigma_u8(img.as_ptr(), size, size, &mut sigma, &mut has) };
        assert_eq!(st, cc_status::CC_OK);
        let expected = chess_corners::estimate_noise_sigma_u8(&img, size, size).unwrap();
        assert_eq!((has == 1).then_some(sigma), expected);
        assert!(expected.is_some());

        // SAFETY: a null pixel pointer is the violation the check guards.
        let st =
            unsafe { cc_estimate_noise_sigma_u8(ptr::null(), size, size, &mut sigma, &mut has) };
        assert_eq!(st, cc_status::CC_ERR_NULL_POINTER);
        // SAFETY: as above; zero dimensions are rejected before any read.
        let st = unsafe { cc_estimate_noise_sigma_u8(img.as_ptr(), 0, size, &mut sigma, &mut has) };
        assert_eq!(st, cc_status::CC_ERR_DIMENSION_MISMATCH);
    }

    #[test]
    fn zero_upscale_factor_is_disabled() {
        let mut cfg = cc_config_chess();
        cfg.upscale_factor = 0;
        assert!(convert::to_detector_config(&cfg).is_ok());
    }
}
