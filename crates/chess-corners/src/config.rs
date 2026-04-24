use box_image_pyramid::PyramidParams;
use chess_corners_core::{
    CenterOfMassConfig, ChessParams, ForstnerConfig, RadonPeakConfig, RefinerKind,
    SaddlePointConfig,
};
use serde::{Deserialize, Serialize};

use crate::multiscale::CoarseToFineParams;
use crate::upscale::UpscaleConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorMode {
    #[default]
    Canonical,
    Broad,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorMode {
    #[default]
    FollowDetector,
    Canonical,
    Broad,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdMode {
    #[default]
    Relative,
    Absolute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementMethod {
    #[default]
    CenterOfMass,
    Forstner,
    SaddlePoint,
    RadonPeak,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RefinerConfig {
    pub kind: RefinementMethod,
    pub center_of_mass: CenterOfMassConfig,
    pub forstner: ForstnerConfig,
    pub saddle_point: SaddlePointConfig,
    pub radon_peak: RadonPeakConfig,
}

impl RefinerConfig {
    pub fn center_of_mass() -> Self {
        Self {
            kind: RefinementMethod::CenterOfMass,
            ..Self::default()
        }
    }

    pub fn forstner() -> Self {
        Self {
            kind: RefinementMethod::Forstner,
            ..Self::default()
        }
    }

    pub fn saddle_point() -> Self {
        Self {
            kind: RefinementMethod::SaddlePoint,
            ..Self::default()
        }
    }

    pub fn radon_peak() -> Self {
        Self {
            kind: RefinementMethod::RadonPeak,
            ..Self::default()
        }
    }

    pub fn to_refiner_kind(&self) -> RefinerKind {
        match self.kind {
            RefinementMethod::CenterOfMass => RefinerKind::CenterOfMass(self.center_of_mass),
            RefinementMethod::Forstner => RefinerKind::Forstner(self.forstner),
            RefinementMethod::SaddlePoint => RefinerKind::SaddlePoint(self.saddle_point),
            RefinementMethod::RadonPeak => RefinerKind::RadonPeak(self.radon_peak),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessConfig {
    pub detector_mode: DetectorMode,
    pub descriptor_mode: DescriptorMode,
    pub threshold_mode: ThresholdMode,
    pub threshold_value: f32,
    pub nms_radius: u32,
    pub min_cluster_size: u32,
    pub refiner: RefinerConfig,
    pub pyramid_levels: u8,
    pub pyramid_min_size: usize,
    pub refinement_radius: u32,
    pub merge_radius: f32,
    /// Optional pre-pipeline integer upscaling. Disabled by default.
    pub upscale: UpscaleConfig,
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self {
            detector_mode: DetectorMode::default(),
            descriptor_mode: DescriptorMode::default(),
            // Paper's contract: any strictly positive ChESS response is
            // a corner candidate. Callers that want an adaptive
            // fraction-of-max threshold can opt into
            // `ThresholdMode::Relative` explicitly.
            threshold_mode: ThresholdMode::Absolute,
            threshold_value: 0.0,
            nms_radius: 2,
            min_cluster_size: 2,
            refiner: RefinerConfig::default(),
            pyramid_levels: 1,
            pyramid_min_size: 128,
            refinement_radius: 3,
            merge_radius: 3.0,
            upscale: UpscaleConfig::default(),
        }
    }
}

impl ChessConfig {
    pub fn single_scale() -> Self {
        Self::default()
    }

    pub fn multiscale() -> Self {
        Self {
            pyramid_levels: 3,
            pyramid_min_size: 128,
            ..Self::default()
        }
    }

    pub fn to_chess_params(&self) -> ChessParams {
        let mut params = ChessParams::default();
        params.use_radius10 = matches!(self.detector_mode, DetectorMode::Broad);
        params.descriptor_use_radius10 = match self.descriptor_mode {
            DescriptorMode::FollowDetector => None,
            DescriptorMode::Canonical => Some(false),
            DescriptorMode::Broad => Some(true),
        };
        match self.threshold_mode {
            ThresholdMode::Relative => {
                params.threshold_rel = self.threshold_value;
                params.threshold_abs = None;
            }
            ThresholdMode::Absolute => {
                params.threshold_abs = Some(self.threshold_value);
            }
        }
        params.nms_radius = self.nms_radius;
        params.min_cluster_size = self.min_cluster_size;
        params.refiner = self.refiner.to_refiner_kind();
        params
    }

    pub fn to_coarse_to_fine_params(&self) -> CoarseToFineParams {
        let mut cfg = CoarseToFineParams::default();
        let mut pyramid = PyramidParams::default();
        pyramid.num_levels = self.pyramid_levels;
        pyramid.min_size = self.pyramid_min_size;
        cfg.pyramid = pyramid;
        cfg.refinement_radius = self.refinement_radius;
        cfg.merge_radius = self.merge_radius;
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_accepts_any_positive_response() {
        let cfg = ChessConfig::default();
        let params = cfg.to_chess_params();
        let cf = cfg.to_coarse_to_fine_params();

        assert!(!params.use_radius10);
        assert_eq!(params.descriptor_use_radius10, None);
        // Paper's contract: accept strictly positive R.
        assert_eq!(cfg.threshold_mode, ThresholdMode::Absolute);
        assert_eq!(cfg.threshold_value, 0.0);
        assert_eq!(params.threshold_abs, Some(0.0));
        assert_eq!(params.nms_radius, 2);
        assert_eq!(params.min_cluster_size, 2);
        assert_eq!(
            params.refiner,
            RefinerKind::CenterOfMass(CenterOfMassConfig::default())
        );
        assert_eq!(cf.pyramid.num_levels, 1);
        assert_eq!(cf.pyramid.min_size, 128);
        assert_eq!(cf.refinement_radius, 3);
        assert_eq!(cf.merge_radius, 3.0);
    }

    #[test]
    fn absolute_threshold_maps_to_internal_params() {
        let cfg = ChessConfig {
            threshold_mode: ThresholdMode::Absolute,
            threshold_value: 7.5,
            ..ChessConfig::default()
        };

        let params = cfg.to_chess_params();
        assert_eq!(params.threshold_abs, Some(7.5));
        assert_eq!(params.threshold_rel, 0.2);
    }

    #[test]
    fn ring_and_refiner_modes_map_to_internal_params() {
        let cfg = ChessConfig {
            detector_mode: DetectorMode::Broad,
            descriptor_mode: DescriptorMode::Canonical,
            refiner: RefinerConfig {
                kind: RefinementMethod::Forstner,
                forstner: ForstnerConfig {
                    max_offset: 2.0,
                    ..ForstnerConfig::default()
                },
                ..RefinerConfig::default()
            },
            ..ChessConfig::default()
        };

        let params = cfg.to_chess_params();
        assert!(params.use_radius10);
        assert_eq!(params.descriptor_use_radius10, Some(false));
        assert_eq!(
            params.refiner,
            RefinerKind::Forstner(ForstnerConfig {
                max_offset: 2.0,
                ..ForstnerConfig::default()
            })
        );
    }

    #[test]
    fn multiscale_preset_has_expected_defaults() {
        let cfg = ChessConfig::multiscale();
        assert_eq!(cfg.pyramid_levels, 3);
        assert_eq!(cfg.pyramid_min_size, 128);
        assert_eq!(cfg.refinement_radius, 3);
        assert_eq!(cfg.merge_radius, 3.0);
    }
}
