//! Typed `#[wasm_bindgen]` wrappers around `chess-corners` config
//! structs.
//!
//! ## Live nested edits
//!
//! Each wrapper stores its inner Rust value in a shared
//! `Rc<RefCell<T>>` cell, and compound wrappers (`DetectorConfig`,
//! `DetectionStrategy`, `ChessConfig`, `RadonConfig`,
//! `ChessRefiner`) hold `Rc` handles to their children's cells. A
//! getter returns a wrapper backed by the same cell as the parent,
//! so chained mutation propagates without a round-trip:
//!
//! ```js
//! const cfg = DetectorConfig.chessMultiscale();
//! cfg.strategy.chess.ring = ChessRing.Broad;             // works
//! cfg.strategy.chess.refiner.forstner.maxOffset = 2.0;   // works
//! cfg.detection.nmsRadius = 3;                           // works
//! cfg.multiscale = MultiscaleConfig.pyramid(4, 64, 3);   // works
//! ```
//!
//! Setters that take a nested wrapper (e.g. `cfg.strategy = newDs`)
//! reseat the parent's `Rc` to point at the new value's cell, so
//! future getter calls return wrappers backed by the new cell. Any
//! JS reference held to the *previous* nested wrapper still
//! observes the previous cell — matching natural attribute-
//! reassignment semantics in JS.
//!
//! ## Strategy discriminants
//!
//! [`DetectionStrategy`] is a tagged wrapper carrying both a
//! [`ChessConfig`] and a [`RadonConfig`] cell plus an internal
//! `kind` discriminant. Only the cell matching the active `kind` is
//! snapshotted into the Rust `RsDetectionStrategy`. Field edits to
//! the *inactive* variant are kept in their cell and become visible
//! again as soon as the discriminant flips back — this mirrors what
//! a JS developer expects when they pre-populate both branches
//! before choosing one.
//!
//! Single-threaded `Rc<RefCell<T>>` is sound on
//! `wasm32-unknown-unknown`; wasm-bindgen modules are not shared
//! across worker threads.
//!
//! Why a wrapper layer at all? The Rust source-of-truth structs live
//! in `chess-corners` / `chess-corners-core` and must not depend on
//! `wasm-bindgen` (per the workspace dependency rule in `AGENTS.md`).
//! These wrappers add the JS-facing attribute layer in the WASM
//! crate only.

use std::cell::RefCell;
use std::rc::Rc;

use chess_corners::{
    ChessRing as RsChessRing, OrientationMethod as RsOrientationMethod,
    PeakFitMode as RsPeakFitMode,
};
use wasm_bindgen::prelude::*;

pub(crate) mod detector;
pub(crate) mod multiscale;
pub(crate) mod refiners;
pub(crate) mod strategy;
pub(crate) mod upscale;

/// Shared mutable cell used by every wrapper. Single-threaded;
/// `wasm-bindgen` instances live entirely on one JS thread.
pub(crate) type Cell<T> = Rc<RefCell<T>>;

pub(crate) fn cell<T>(value: T) -> Cell<T> {
    Rc::new(RefCell::new(value))
}

// ---------------------------------------------------------------------------
// Plain enums
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Mirrors [`chess_corners::ChessRing`].
///
/// `Canonical` (= 0): paper-default radius-5 ring (16 samples).
/// `Broad` (= 1): radius-10 ring; useful for low-resolution / heavily
/// blurred imagery where the canonical ring under-samples.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChessRing {
    Canonical = 0,
    Broad = 1,
}

impl From<ChessRing> for RsChessRing {
    fn from(v: ChessRing) -> Self {
        match v {
            ChessRing::Canonical => RsChessRing::Canonical,
            ChessRing::Broad => RsChessRing::Broad,
        }
    }
}

impl From<RsChessRing> for ChessRing {
    fn from(v: RsChessRing) -> Self {
        match v {
            RsChessRing::Canonical => ChessRing::Canonical,
            RsChessRing::Broad => ChessRing::Broad,
            _ => ChessRing::Canonical,
        }
    }
}

/// Subpixel peak-fit mode (Radon). Mirrors
/// [`chess_corners::PeakFitMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeakFitMode {
    Parabolic = 0,
    Gaussian = 1,
}

impl From<PeakFitMode> for RsPeakFitMode {
    fn from(v: PeakFitMode) -> Self {
        match v {
            PeakFitMode::Parabolic => RsPeakFitMode::Parabolic,
            PeakFitMode::Gaussian => RsPeakFitMode::Gaussian,
        }
    }
}

impl From<RsPeakFitMode> for PeakFitMode {
    fn from(v: RsPeakFitMode) -> Self {
        match v {
            RsPeakFitMode::Parabolic => PeakFitMode::Parabolic,
            RsPeakFitMode::Gaussian => PeakFitMode::Gaussian,
            // The core enum is `#[non_exhaustive]`. New variants
            // would land here; map them to `Gaussian` (the default)
            // until the wrapper learns the new variant.
            _ => PeakFitMode::Gaussian,
        }
    }
}

/// Two-axis orientation-fit method. Mirrors [`chess_corners::OrientationMethod`].
///
/// `RingFit` *(default, = 0)* — fits the parametric two-axis chessboard
/// intensity model to the 16-sample ring via Gauss-Newton, with calibrated
/// per-axis 1σ uncertainties. Suitable for the full range of standard
/// chessboard images.
///
/// `DiskFit` *(= 1)* — full-disk crossing-line estimator. Samples all image
/// pixels in a disk around the corner center and fits two possibly
/// non-orthogonal axes. Use when corners are imaged under strong projective
/// warp. Falls back to `RingFit` on clean orthogonal corners.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrientationMethod {
    RingFit = 0,
    DiskFit = 1,
}

impl From<OrientationMethod> for RsOrientationMethod {
    fn from(v: OrientationMethod) -> Self {
        match v {
            OrientationMethod::RingFit => RsOrientationMethod::RingFit,
            OrientationMethod::DiskFit => RsOrientationMethod::DiskFit,
        }
    }
}

impl From<RsOrientationMethod> for OrientationMethod {
    fn from(v: RsOrientationMethod) -> Self {
        match v {
            RsOrientationMethod::RingFit => OrientationMethod::RingFit,
            RsOrientationMethod::DiskFit => OrientationMethod::DiskFit,
            // Any future variants map to the default.
            _ => OrientationMethod::RingFit,
        }
    }
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use detector::DetectorConfig;
pub use multiscale::MultiscaleConfig;
pub use refiners::{CenterOfMassConfig, ChessRefiner, ForstnerConfig, SaddlePointConfig};
pub use strategy::{ChessConfig, DetectionParams, DetectionStrategy, RadonConfig};
pub use upscale::UpscaleConfig;

#[cfg(test)]
mod tests {
    //! Native Rust tests of the shared-cell semantics. These run on
    //! `cargo test -p chess-corners-wasm` (host target) and don't
    //! need a JS runner. wasm-bindgen-test would be needed to
    //! exercise the JS-facing getter/setter mangling, but the cell
    //! plumbing is what matters for the live-edit guarantee and is
    //! identical on host and on wasm32.

    use super::*;
    use chess_corners::{
        ChessRing as RsChessRingCheck, DetectionStrategy as RsDetectionStrategy,
        MultiscaleConfig as RsMultiscaleConfig, OrientationMethod as RsOrientationMethod,
        UpscaleConfig as RsUpscaleConfig,
    };

    #[test]
    fn leaf_edits_propagate_through_chess_refiner_chain() {
        // Models the JS chain
        //   cfg.strategy.chess.refiner.forstner.maxOffset = 2.5
        // by walking the same getter chain in Rust and mutating the
        // leaf forstner cell. Every intermediate wrapper is a clone
        // that shares the underlying `Rc<RefCell<_>>` cells, so the
        // edit reaches the cfg's chess refiner cell.
        let cfg = DetectorConfig::new();
        // Flip kind to Forstner first (via set_refiner copy-into-cell).
        let mut chess = cfg.strategy().chess();
        chess.set_refiner(&ChessRefiner::forstner(&ForstnerConfig::new()));
        // Edit the leaf forstner cell through the chain.
        cfg.strategy()
            .chess()
            .refiner()
            .get_forstner()
            .set_max_offset(3.5);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        match c.refiner {
            chess_corners::ChessRefiner::Forstner(forstner) => {
                assert!((forstner.max_offset - 3.5).abs() < f32::EPSILON);
            }
            other => panic!("expected forstner refiner, got {other:?}"),
        }
    }

    #[test]
    fn chess_config_field_edits_propagate() {
        let cfg = DetectorConfig::single_scale();
        let mut chess = cfg.strategy().chess();
        chess.set_ring(ChessRing::Broad);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(s) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert_eq!(s.ring, RsChessRingCheck::Broad);
    }

    #[test]
    fn detection_field_edits_propagate() {
        // Shared NMS / clustering knobs live on `detection` and are
        // honoured regardless of the active strategy.
        let cfg = DetectorConfig::single_scale();
        let mut detection = cfg.detection();
        detection.set_nms_radius(7);
        detection.set_min_cluster_size(3);

        let snap = cfg.snapshot();
        assert_eq!(snap.detection.nms_radius, 7);
        assert_eq!(snap.detection.min_cluster_size, 3);
    }

    #[test]
    fn multiscale_pyramid_factory_round_trips_to_snapshot() {
        // The `levels`/`minSize`/`refinementRadius` getters return
        // `Result<_, JsValue>` and panic on host targets because
        // `JsValue` is wasm32-only. So we round-trip through the
        // snapshot, which exercises the same internal storage cells
        // without crossing the wasm-bindgen boundary.
        let mut cfg = DetectorConfig::single_scale();
        cfg.set_multiscale(&MultiscaleConfig::pyramid(4, 64, 5));
        assert_eq!(cfg.multiscale().kind(), "pyramid");

        let snap = cfg.snapshot();
        let RsMultiscaleConfig::Pyramid {
            levels,
            min_size,
            refinement_radius,
        } = snap.multiscale
        else {
            panic!("expected pyramid")
        };
        assert_eq!(levels, 4);
        assert_eq!(min_size, 64);
        assert_eq!(refinement_radius, 5);
    }

    #[test]
    fn multiscale_assigning_wrapper_swaps_to_single_scale() {
        let mut cfg = DetectorConfig::chess_multiscale();
        assert_eq!(cfg.multiscale().kind(), "pyramid");
        cfg.set_multiscale(&MultiscaleConfig::single_scale());
        assert_eq!(cfg.multiscale().kind(), "single_scale");

        let snap = cfg.snapshot();
        assert!(matches!(snap.multiscale, RsMultiscaleConfig::SingleScale));
    }

    #[test]
    fn radon_config_field_edits_propagate() {
        let cfg = DetectorConfig::radon();
        let mut radon = cfg.strategy().radon();
        radon.set_ray_radius(7);
        radon.set_image_upsample(2);
        let snap = cfg.snapshot();
        let RsDetectionStrategy::Radon(s) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert_eq!(s.ray_radius, 7);
        assert_eq!(s.image_upsample, 2);
    }

    #[test]
    fn threshold_numeric_round_trip() {
        let mut cfg = DetectorConfig::new();
        cfg.set_threshold(3.5);
        assert!((cfg.threshold() - 3.5).abs() < f32::EPSILON);
        let snap = cfg.snapshot();
        assert!((snap.threshold - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn cfg_threshold_propagates() {
        let mut cfg = DetectorConfig::new();
        cfg.set_threshold(0.15);
        let snap = cfg.snapshot();
        assert!((snap.threshold - 0.15).abs() < f32::EPSILON);

        cfg.set_threshold(0.25);
        let snap2 = cfg.snapshot();
        assert!((snap2.threshold - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn detection_strategy_switch_preserves_inactive_branch_state() {
        let mut ds = DetectionStrategy::default_chess();
        // Pre-populate the radon branch while chess is active.
        let mut radon = ds.radon();
        radon.set_ray_radius(11);
        radon.set_image_upsample(2);
        // Flip discriminant to radon — branch state must survive.
        ds.use_radon();
        assert_eq!(ds.kind(), "radon");
        match ds.snapshot() {
            RsDetectionStrategy::Radon(r) => assert_eq!(r.ray_radius, 11),
            other => panic!("expected radon, got {other:?}"),
        }
    }

    #[test]
    fn upscale_fixed_factor_round_trips() {
        // The `factor` getter/setter returns `Result<_, JsValue>` and
        // panics on host targets; round-trip through the snapshot.
        let mut cfg = DetectorConfig::new();
        cfg.set_upscale(&UpscaleConfig::fixed(3));
        assert_eq!(cfg.upscale().kind(), "fixed");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(3)));
    }

    #[test]
    fn upscale_disabled_round_trips() {
        let cfg = DetectorConfig::new();
        assert_eq!(cfg.upscale().kind(), "disabled");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Disabled));
    }

    #[test]
    fn upscale_assigning_wrapper_swaps_to_fixed() {
        let mut cfg = DetectorConfig::new();
        cfg.set_upscale(&UpscaleConfig::fixed(4));
        assert_eq!(cfg.upscale().kind(), "fixed");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(4)));
    }

    #[test]
    fn assigning_chess_refiner_copies_into_existing_cells() {
        // `cfg.strategy.chess.refiner = newRefiner` (modelled here as
        // `chess.set_refiner(&new_refiner)`) copies the new refiner's
        // payload contents into the cfg's chess refiner cells, so
        // future `cfg.strategy.chess.refiner.*` getter calls observe
        // the new state without any re-binding.
        let cfg = DetectorConfig::new();
        let mut chess = cfg.strategy().chess();
        let mut wrapped = SaddlePointConfig::new();
        wrapped.set_max_offset(3.5);
        let new_refiner = ChessRefiner::saddle_point(&wrapped);
        chess.set_refiner(&new_refiner);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        match c.refiner {
            chess_corners::ChessRefiner::SaddlePoint(s) => {
                assert!((s.max_offset - 3.5).abs() < f32::EPSILON);
            }
            other => panic!("expected saddle_point, got {other:?}"),
        }
    }

    #[test]
    fn snapshot_returns_independent_state() {
        let cfg = DetectorConfig::new();
        let mut cfg_mut = cfg;
        cfg_mut.set_threshold(0.1);
        let snap = cfg_mut.snapshot();
        // Replace the threshold after snapshotting — snapshot must not move.
        cfg_mut.set_threshold(0.9);
        assert!((snap.threshold - 0.1).abs() < 1e-6);
    }

    #[test]
    fn orientation_method_round_trips_all_variants() {
        let cases = [
            (OrientationMethod::RingFit, RsOrientationMethod::RingFit),
            (OrientationMethod::DiskFit, RsOrientationMethod::DiskFit),
        ];

        for (wasm_variant, rs_variant) in cases {
            let mut cfg = DetectorConfig::new();
            cfg.set_orientation_method(Some(wasm_variant));
            assert_eq!(cfg.orientation_method(), Some(wasm_variant));
            let snap = cfg.snapshot();
            assert_eq!(
                snap.orientation_method,
                Some(rs_variant),
                "snapshot mismatch for {wasm_variant:?}"
            );
        }
    }

    #[test]
    fn without_orientation_clears_method() {
        let cfg = DetectorConfig::chess().without_orientation();
        assert_eq!(cfg.orientation_method(), None);
        assert_eq!(cfg.snapshot().orientation_method, None);
    }

    #[test]
    fn radon_multiscale_preset_has_radon_strategy_and_multiscale() {
        let cfg = DetectorConfig::radon_multiscale();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Radon(_)),
            "radon_multiscale preset must use Radon strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::Pyramid { .. }),
            "radon_multiscale preset must enable pyramid multiscale"
        );
    }

    // ---- New 0.10.0 API ----

    #[test]
    fn chess_preset_is_single_scale() {
        let cfg = DetectorConfig::chess();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "chess() preset must use Chess strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::SingleScale),
            "chess() preset must be single-scale"
        );
    }

    #[test]
    fn chess_multiscale_preset_has_pyramid() {
        let cfg = DetectorConfig::chess_multiscale();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "chess_multiscale() preset must use Chess strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::Pyramid { .. }),
            "chess_multiscale() preset must enable pyramid multiscale"
        );
    }

    #[test]
    fn single_scale_delegates_to_chess() {
        // Deprecated shim must return an identical snapshot to chess().
        let snap_compat = DetectorConfig::single_scale().snapshot();
        let snap_new = DetectorConfig::chess().snapshot();
        // Both must be Chess/SingleScale.
        assert!(matches!(
            snap_compat.strategy,
            RsDetectionStrategy::Chess(_)
        ));
        assert!(matches!(
            snap_compat.multiscale,
            RsMultiscaleConfig::SingleScale
        ));
        // Threshold must match.
        assert_eq!(snap_compat.threshold, snap_new.threshold);
    }

    #[test]
    fn pyramid_default_matches_chess_multiscale_preset() {
        let ms = MultiscaleConfig::pyramid_default();
        assert_eq!(ms.kind(), "pyramid");
        // levels / min_size / refinement_radius must equal the facade defaults (3/128/3).
        let snap = ms.snapshot();
        assert!(
            matches!(
                snap,
                RsMultiscaleConfig::Pyramid {
                    levels: 3,
                    min_size: 128,
                    refinement_radius: 3,
                }
            ),
            "pyramid_default must equal (3, 128, 3); got {snap:?}"
        );
    }

    #[test]
    fn with_threshold_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let snap_before = cfg.snapshot();
        let before_val = snap_before.threshold;

        let cfg2 = cfg.with_threshold(0.12);
        let snap2 = cfg2.snapshot();
        assert!((snap2.threshold - 0.12).abs() < f32::EPSILON);

        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!((snap_orig.threshold - before_val).abs() < f32::EPSILON);
    }

    #[test]
    fn with_multiscale_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let ms = MultiscaleConfig::pyramid(4, 64, 5);
        let cfg2 = cfg.with_multiscale(&ms);
        let snap = cfg2.snapshot();
        assert!(
            matches!(
                snap.multiscale,
                RsMultiscaleConfig::Pyramid {
                    levels: 4,
                    min_size: 64,
                    refinement_radius: 5,
                }
            ),
            "with_multiscale must propagate to snapshot; got {snap:?}"
        );

        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!(matches!(
            snap_orig.multiscale,
            RsMultiscaleConfig::SingleScale
        ));
    }

    #[test]
    fn with_upscale_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let up = UpscaleConfig::fixed(2);
        let cfg2 = cfg.with_upscale(&up);
        let snap = cfg2.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(2)));
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!(matches!(snap_orig.upscale, RsUpscaleConfig::Disabled));
    }

    #[test]
    fn with_orientation_method_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let cfg2 = cfg.with_orientation_method(OrientationMethod::DiskFit);
        let snap = cfg2.snapshot();
        assert_eq!(snap.orientation_method, Some(RsOrientationMethod::DiskFit));
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert_eq!(
            snap_orig.orientation_method,
            Some(RsOrientationMethod::RingFit)
        );
    }

    #[test]
    fn with_merge_radius_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let cfg2 = cfg.with_merge_radius(7.5);
        let snap = cfg2.snapshot();
        assert!((snap.merge_radius - 7.5).abs() < f32::EPSILON);
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!((snap_orig.merge_radius - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_chess_refiner_builder_sets_forstner() {
        let cfg = DetectorConfig::chess();
        let mut forstner = ForstnerConfig::new();
        forstner.set_max_offset(4.0);
        let refiner = ChessRefiner::forstner(&forstner);
        let cfg2 = cfg.with_chess_refiner(&refiner);
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert!(
            matches!(c.refiner, chess_corners::ChessRefiner::Forstner(f) if (f.max_offset - 4.0).abs() < f32::EPSILON),
            "with_chess_refiner must set Forstner with max_offset=4.0"
        );
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        let RsDetectionStrategy::Chess(c_orig) = snap_orig.strategy else {
            panic!("expected chess strategy")
        };
        assert!(
            !matches!(c_orig.refiner, chess_corners::ChessRefiner::Forstner(_)),
            "original config must not have Forstner refiner"
        );
    }

    // The following tests use js_sys::Object / js_sys::Reflect which panic on
    // non-wasm32 targets ("cannot call wasm-bindgen imported functions on
    // non-wasm targets"). They are gated to wasm32 so `cargo test` on the
    // host continues to work. They run under `wasm-pack test` / wasm-bindgen-test.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_ring() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("ring"),
            &JsValue::from_f64(ChessRing::Broad as u8 as f64),
        )
        .unwrap();
        let cfg2 = cfg.with_chess(&opts).expect("with_chess must succeed");
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert_eq!(c.ring, chess_corners::ChessRing::Broad);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_detection_opts_set_shared_params() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("nmsRadius"),
            &JsValue::from_f64(7.0),
        )
        .unwrap();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("minClusterSize"),
            &JsValue::from_f64(3.0),
        )
        .unwrap();
        let cfg2 = cfg
            .with_detection(&opts)
            .expect("with_detection must succeed");
        let snap = cfg2.snapshot();
        assert_eq!(snap.detection.nms_radius, 7);
        assert_eq!(snap.detection.min_cluster_size, 3);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_unknown_key_returns_error() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("unknownField"),
            &JsValue::from_f64(1.0),
        )
        .unwrap();
        let err = cfg
            .with_chess(&opts)
            .expect_err("unknown key must produce an error");
        let msg = err.as_string().unwrap_or_default();
        assert!(
            msg.contains("unexpected option: 'unknownField'"),
            "error message must name the unknown key; got: {msg}"
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_radon_opts_ray_radius_and_image_upsample() {
        let cfg = DetectorConfig::radon();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("rayRadius"),
            &JsValue::from_f64(6.0),
        )
        .unwrap();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("imageUpsample"),
            &JsValue::from_f64(2.0),
        )
        .unwrap();
        let cfg2 = cfg.with_radon(&opts).expect("with_radon must succeed");
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Radon(r) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert_eq!(r.ray_radius, 6);
        assert_eq!(r.image_upsample, 2);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_radon_opts_unknown_key_returns_error() {
        let cfg = DetectorConfig::radon();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(&opts, &JsValue::from_str("badKey"), &JsValue::from_f64(1.0)).unwrap();
        let err = cfg
            .with_radon(&opts)
            .expect_err("unknown key must produce an error");
        let msg = err.as_string().unwrap_or_default();
        assert!(
            msg.contains("unexpected option: 'badKey'"),
            "error message must name the unknown key; got: {msg}"
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_on_radon_config_switches_strategy() {
        // Calling with_chess on a radon config must flip strategy to Chess.
        let cfg = DetectorConfig::radon();
        let snap_before = cfg.snapshot();
        assert!(matches!(
            snap_before.strategy,
            RsDetectionStrategy::Radon(_)
        ));

        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("ring"),
            &JsValue::from_f64(ChessRing::Broad as u8 as f64),
        )
        .unwrap();
        let cfg2 = cfg.with_chess(&opts).expect("with_chess must succeed");
        let snap = cfg2.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "strategy must flip to Chess"
        );
    }

    /// Pin the integer discriminants of every `#[wasm_bindgen]` numeric enum.
    ///
    /// wasm-bindgen exposes these values to JavaScript callers as plain
    /// numbers, so reordering or renumbering a variant is a breaking change
    /// for existing JS/TS consumers.  This test catches any accidental
    /// reordering at `cargo test` time, before the WASM package is published.
    #[test]
    fn enum_discriminants_are_pinned() {
        // ChessRing
        assert_eq!(ChessRing::Canonical as u32, 0);
        assert_eq!(ChessRing::Broad as u32, 1);

        // PeakFitMode
        assert_eq!(PeakFitMode::Parabolic as u32, 0);
        assert_eq!(PeakFitMode::Gaussian as u32, 1);

        // OrientationMethod
        assert_eq!(OrientationMethod::RingFit as u32, 0);
        assert_eq!(OrientationMethod::DiskFit as u32, 1);
    }
}
