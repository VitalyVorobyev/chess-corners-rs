//! Public-API tests for the orientation stage.
//!
//! The white-box ring-fit parity / accuracy cases, which build synthetic
//! ring samples from the crate-private ring-offset geometry, live in
//! `src/orientation/api.rs`. This file keeps the cases that exercise the
//! public API alone:
//! - [`OrientationMethod`] defaults and serde keys.
//! - `describe_corners(.., None)` skips the fit while keeping positions.

use chess_corners_core::OrientationMethod;

#[test]
fn ring_fit_is_the_default() {
    assert_eq!(OrientationMethod::default(), OrientationMethod::RingFit);
}

#[test]
fn ring_fit_serde_key_is_ring_fit() {
    let s = serde_json::to_string(&OrientationMethod::RingFit).unwrap();
    assert_eq!(s, r#""ring_fit""#);
    let v: OrientationMethod = serde_json::from_str(r#""ring_fit""#).unwrap();
    assert_eq!(v, OrientationMethod::RingFit);
}

#[test]
fn disk_fit_serde_key_is_disk_fit() {
    let s = serde_json::to_string(&OrientationMethod::DiskFit).unwrap();
    assert_eq!(s, r#""disk_fit""#);
    let v: OrientationMethod = serde_json::from_str(r#""disk_fit""#).unwrap();
    assert_eq!(v, OrientationMethod::DiskFit);
}

// ---------------------------------------------------------------------------
// Optional orientation: `describe_corners(.., None)` skips the fit
// ---------------------------------------------------------------------------

#[test]
fn describe_corners_none_skips_orientation_keeps_positions() {
    use chess_corners_core::{describe_corners, Corner};

    // Synthetic checkerboard so the ring sampler has real support.
    let w = 48usize;
    let h = 48usize;
    let period = 12i32;
    let mut img = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let rx = ((x as i32) / period) & 1;
            let ry = ((y as i32) / period) & 1;
            img[y * w + x] = if rx ^ ry == 0 { 40 } else { 215 };
        }
    }

    let corners = vec![
        Corner::new(12.3, 12.7, 1.0),
        Corner::new(24.1, 24.2, 2.0),
        Corner::new(24.0, 12.0, 0.5),
    ];

    let oriented = describe_corners(
        &img,
        w,
        h,
        5,
        corners.clone(),
        Some(OrientationMethod::RingFit),
    );
    let skipped = describe_corners(&img, w, h, 5, corners.clone(), None);

    assert_eq!(oriented.len(), corners.len());
    assert_eq!(skipped.len(), corners.len());
    for (i, (o, s)) in oriented.iter().zip(skipped.iter()).enumerate() {
        // Position and response are bit-identical; only the fit is skipped.
        assert_eq!(o.x.to_bits(), s.x.to_bits(), "x mismatch at corner {i}");
        assert_eq!(o.y.to_bits(), s.y.to_bits(), "y mismatch at corner {i}");
        assert_eq!(
            o.response.to_bits(),
            s.response.to_bits(),
            "response mismatch at corner {i}"
        );
        assert!(o.axes.is_some(), "oriented corner {i} should carry axes");
        assert!(
            s.axes.is_none(),
            "skipped corner {i} must have axes == None"
        );
    }
}
