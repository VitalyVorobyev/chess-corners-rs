//! `#[wasm_bindgen]` wrappers for `ChessConfig`, `RadonConfig`,
//! `DetectionParams`, and `DetectionStrategy`.

use std::rc::Rc;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessConfig as RsChessConfig,
    ChessRing as RsChessRing, DetectionParams as RsDetectionParams,
    DetectionStrategy as RsDetectionStrategy, ForstnerConfig as RsForstnerConfig,
    PeakFitMode as RsPeakFitMode, RadonConfig as RsRadonConfig,
    SaddlePointConfig as RsSaddlePointConfig,
};
use wasm_bindgen::prelude::*;

use super::refiners::{ChessRefiner, ChessRefinerKind};
use super::{cell, Cell, ChessRing, PeakFitMode};

// ---------------------------------------------------------------------------
// Strategy discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrategyKind {
    Chess,
    Radon,
}

impl StrategyKind {
    fn as_str(self) -> &'static str {
        match self {
            StrategyKind::Chess => "chess",
            StrategyKind::Radon => "radon",
        }
    }
}

// ---------------------------------------------------------------------------
// ChessConfig
// ---------------------------------------------------------------------------

/// ChESS-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::ChessConfig`].
///
/// All fields — including the refiner discriminant and payloads — are
/// stored as shared `Rc<RefCell<…>>` cells. The `refiner` getter
/// assembles a [`ChessRefiner`] wrapper out of clones of those cells,
/// so edits chained as `cfg.strategy.chess.refiner.forstner.maxOffset
/// = 2.5` propagate through the chess config without a round-trip.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessConfig {
    ring: Cell<RsChessRing>,
    refiner_kind: Cell<ChessRefinerKind>,
    refiner_center_of_mass: Cell<RsCenterOfMassConfig>,
    refiner_forstner: Cell<RsForstnerConfig>,
    refiner_saddle_point: Cell<RsSaddlePointConfig>,
}

#[wasm_bindgen]
impl ChessConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsChessConfig::default())
    }

    #[wasm_bindgen(getter)]
    pub fn ring(&self) -> ChessRing {
        (*self.ring.borrow()).into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_ring(&mut self, v: ChessRing) {
        *self.ring.borrow_mut() = v.into();
    }

    /// Subpixel refiner. Returns a wrapper that shares cells with this
    /// config; edits propagate without a round-trip.
    #[wasm_bindgen(getter)]
    pub fn refiner(&self) -> ChessRefiner {
        ChessRefiner {
            kind: Rc::clone(&self.refiner_kind),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
            forstner: Rc::clone(&self.refiner_forstner),
            saddle_point: Rc::clone(&self.refiner_saddle_point),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_refiner(&mut self, v: &ChessRefiner) {
        // Copy v's cell contents into this config's existing cells.
        // Cell-sharing observers (any clone of this ChessConfig
        // referencing the same Rc) see the update.
        *self.refiner_kind.borrow_mut() = *v.kind.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *v.center_of_mass.borrow();
        *self.refiner_forstner.borrow_mut() = *v.forstner.borrow();
        *self.refiner_saddle_point.borrow_mut() = *v.saddle_point.borrow();
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ChessConfig {
    fn from_value(value: RsChessConfig) -> Self {
        let refiner = ChessRefiner::from_value(value.refiner);
        Self {
            ring: cell(value.ring),
            refiner_kind: refiner.kind,
            refiner_center_of_mass: refiner.center_of_mass,
            refiner_forstner: refiner.forstner,
            refiner_saddle_point: refiner.saddle_point,
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`)
    /// see the update.
    fn copy_from(&self, other: &ChessConfig) {
        *self.ring.borrow_mut() = *other.ring.borrow();
        *self.refiner_kind.borrow_mut() = *other.refiner_kind.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *other.refiner_center_of_mass.borrow();
        *self.refiner_forstner.borrow_mut() = *other.refiner_forstner.borrow();
        *self.refiner_saddle_point.borrow_mut() = *other.refiner_saddle_point.borrow();
    }

    fn snapshot(&self) -> RsChessConfig {
        // Build a transient ChessRefiner wrapper sharing this config's
        // cells, then read its snapshot. The wrapper does no work on
        // construction (cell-handle clones only).
        let refiner_view = ChessRefiner {
            kind: Rc::clone(&self.refiner_kind),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
            forstner: Rc::clone(&self.refiner_forstner),
            saddle_point: Rc::clone(&self.refiner_saddle_point),
        };
        let mut s = RsChessConfig::default();
        s.ring = *self.ring.borrow();
        s.refiner = refiner_view.snapshot();
        s
    }
}

// ---------------------------------------------------------------------------
// RadonConfig
// ---------------------------------------------------------------------------

/// Radon-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::RadonConfig`]. All radii / counts are in
/// **working-resolution** pixels (i.e. after `imageUpsample`).
///
/// All fields are stored as shared `Rc<RefCell<…>>` cells, so nested
/// chained edits propagate through the radon config.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonConfig {
    ray_radius: Cell<u32>,
    image_upsample: Cell<u32>,
    response_blur_radius: Cell<u32>,
    peak_fit: Cell<RsPeakFitMode>,
}

#[wasm_bindgen]
impl RadonConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsRadonConfig::default())
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        *self.ray_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        *self.ray_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        *self.image_upsample.borrow()
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        *self.image_upsample.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        *self.response_blur_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        *self.response_blur_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        (*self.peak_fit.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        *self.peak_fit.borrow_mut() = v.into();
    }
}

impl Default for RadonConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonConfig {
    fn from_value(value: RsRadonConfig) -> Self {
        Self {
            ray_radius: cell(value.ray_radius),
            image_upsample: cell(value.image_upsample),
            response_blur_radius: cell(value.response_blur_radius),
            peak_fit: cell(value.peak_fit),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`)
    /// see the update.
    fn copy_from(&self, other: &RadonConfig) {
        *self.ray_radius.borrow_mut() = *other.ray_radius.borrow();
        *self.image_upsample.borrow_mut() = *other.image_upsample.borrow();
        *self.response_blur_radius.borrow_mut() = *other.response_blur_radius.borrow();
        *self.peak_fit.borrow_mut() = *other.peak_fit.borrow();
    }

    fn snapshot(&self) -> RsRadonConfig {
        let mut s = RsRadonConfig::default();
        s.ray_radius = *self.ray_radius.borrow();
        s.image_upsample = *self.image_upsample.borrow();
        s.response_blur_radius = *self.response_blur_radius.borrow();
        s.peak_fit = *self.peak_fit.borrow();
        s
    }
}

// ---------------------------------------------------------------------------
// DetectionParams (shared NMS / clustering knobs)
// ---------------------------------------------------------------------------

/// Shared non-maximum-suppression and peak-clustering thresholds.
/// Mirrors [`chess_corners::DetectionParams`]. Honoured by both the
/// ChESS and Radon strategies; lives on
/// [`DetectorConfig`](crate::DetectorConfig)'s `detection` field. All counts
/// are in working-resolution pixels.
///
/// Cell-backed so the `detection` getter returns a wrapper sharing the
/// parent's cells; edits chained as `cfg.detection.nmsRadius = 3`
/// propagate without a round-trip.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectionParams {
    nms_radius: Cell<u32>,
    min_cluster_size: Cell<u32>,
}

#[wasm_bindgen]
impl DetectionParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsDetectionParams::default())
    }

    #[wasm_bindgen(getter, js_name = nmsRadius)]
    pub fn nms_radius(&self) -> u32 {
        *self.nms_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = nmsRadius)]
    pub fn set_nms_radius(&mut self, v: u32) {
        *self.nms_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = minClusterSize)]
    pub fn min_cluster_size(&self) -> u32 {
        *self.min_cluster_size.borrow()
    }
    #[wasm_bindgen(setter, js_name = minClusterSize)]
    pub fn set_min_cluster_size(&mut self, v: u32) {
        *self.min_cluster_size.borrow_mut() = v;
    }
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self::new()
    }
}

impl DetectionParams {
    pub(crate) fn from_value(value: RsDetectionParams) -> Self {
        Self {
            nms_radius: cell(value.nms_radius),
            min_cluster_size: cell(value.min_cluster_size),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`) see
    /// the update.
    pub(crate) fn copy_from(&self, other: &DetectionParams) {
        *self.nms_radius.borrow_mut() = *other.nms_radius.borrow();
        *self.min_cluster_size.borrow_mut() = *other.min_cluster_size.borrow();
    }

    pub(crate) fn snapshot(&self) -> RsDetectionParams {
        let mut s = RsDetectionParams::default();
        s.nms_radius = *self.nms_radius.borrow();
        s.min_cluster_size = *self.min_cluster_size.borrow();
        s
    }
}

// ---------------------------------------------------------------------------
// DetectionStrategy (tagged wrapper, mirrors RsDetectionStrategy)
// ---------------------------------------------------------------------------

/// Top-level detector dispatch. Mirrors
/// [`chess_corners::DetectionStrategy`].
///
/// The wrapper carries both a [`ChessConfig`] and a [`RadonConfig`]
/// cell internally. The `kind` discriminant selects which one is
/// snapshotted into the Rust value. Field edits on the inactive
/// variant are preserved in its cell — they become live again when
/// the discriminant flips.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectionStrategy {
    kind: Cell<StrategyKind>,
    chess: ChessConfig,
    radon: RadonConfig,
}

#[wasm_bindgen]
impl DetectionStrategy {
    /// Construct a [`DetectionStrategy`] in the ChESS branch from a
    /// given [`ChessConfig`]. JS: `DetectionStrategy.fromChess(c)`.
    #[wasm_bindgen(js_name = fromChess)]
    pub fn from_chess(strategy: &ChessConfig) -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: strategy.clone(),
            radon: RadonConfig::default(),
        }
    }

    /// Construct a [`DetectionStrategy`] in the Radon branch from a
    /// given [`RadonConfig`]. JS: `DetectionStrategy.fromRadon(c)`.
    #[wasm_bindgen(js_name = fromRadon)]
    pub fn from_radon(strategy: &RadonConfig) -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessConfig::default(),
            radon: strategy.clone(),
        }
    }

    /// Default ChESS-branch strategy. Equivalent to
    /// `DetectionStrategy.fromChess(new ChessConfig())`.
    #[wasm_bindgen(js_name = defaultChess)]
    pub fn default_chess() -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: ChessConfig::default(),
            radon: RadonConfig::default(),
        }
    }

    /// Default Radon-branch strategy. Equivalent to
    /// `DetectionStrategy.fromRadon(new RadonConfig())`.
    #[wasm_bindgen(js_name = defaultRadon)]
    pub fn default_radon() -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessConfig::default(),
            radon: RadonConfig::default(),
        }
    }

    /// Discriminant tag: `"chess"` or `"radon"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// ChESS branch wrapper. Edits propagate even when the active kind
    /// is Radon (the cell is preserved for when the user flips back).
    #[wasm_bindgen(getter)]
    pub fn chess(&self) -> ChessConfig {
        self.chess.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_chess(&mut self, v: &ChessConfig) {
        // Copy v's contents into this strategy's chess cells so
        // cell-sharing observers see the update.
        self.chess.copy_from(v);
        // Setting a chess config implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Radon branch wrapper. Edits propagate even when the active kind
    /// is ChESS.
    #[wasm_bindgen(getter)]
    pub fn radon(&self) -> RadonConfig {
        self.radon.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_radon(&mut self, v: &RadonConfig) {
        self.radon.copy_from(v);
        // Setting a radon config implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Radon;
    }

    /// Switch the discriminant to ChESS without otherwise mutating
    /// the wrapper. The previously stored [`ChessConfig`] cell
    /// becomes the live source again.
    #[wasm_bindgen(js_name = useChess)]
    pub fn use_chess(&mut self) {
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Switch the discriminant to Radon without otherwise mutating
    /// the wrapper. The previously stored [`RadonConfig`] cell
    /// becomes the live source again.
    #[wasm_bindgen(js_name = useRadon)]
    pub fn use_radon(&mut self) {
        *self.kind.borrow_mut() = StrategyKind::Radon;
    }
}

impl Default for DetectionStrategy {
    fn default() -> Self {
        Self::default_chess()
    }
}

impl DetectionStrategy {
    pub(crate) fn from_value(value: RsDetectionStrategy) -> Self {
        match value {
            RsDetectionStrategy::Chess(c) => Self {
                kind: cell(StrategyKind::Chess),
                chess: ChessConfig::from_value(c),
                radon: RadonConfig::default(),
            },
            RsDetectionStrategy::Radon(r) => Self {
                kind: cell(StrategyKind::Radon),
                chess: ChessConfig::default(),
                radon: RadonConfig::from_value(r),
            },
            _ => Self::default_chess(),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// strategy's cells. Cell-sharing observers see the update.
    pub(crate) fn copy_from(&self, other: &DetectionStrategy) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        self.chess.copy_from(&other.chess);
        self.radon.copy_from(&other.radon);
    }

    pub(crate) fn snapshot(&self) -> RsDetectionStrategy {
        match *self.kind.borrow() {
            StrategyKind::Chess => RsDetectionStrategy::Chess(self.chess.snapshot()),
            StrategyKind::Radon => RsDetectionStrategy::Radon(self.radon.snapshot()),
        }
    }
}
