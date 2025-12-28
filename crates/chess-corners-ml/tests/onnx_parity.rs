use chess_corners_ml::{MlModel, ModelSource};
use ndarray_npy::ReadNpyExt;
use std::io::Cursor;

const PATCHES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/fixtures/v2/patches.npy"
));
const TORCH_OUT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/fixtures/v2/torch_out.npy"
));
const META_JSON: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/fixtures/v2/meta.json"
));

#[test]
fn test_onnx_parity_with_fixtures() -> anyhow::Result<()> {
    let patches: ndarray::Array4<f32> = ndarray::Array4::read_npy(Cursor::new(PATCHES))?;
    let torch_out: ndarray::Array2<f32> = ndarray::Array2::read_npy(Cursor::new(TORCH_OUT))?;

    let batch = patches.shape()[0];
    let patches = patches
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("patch fixtures not contiguous"))?;
    let torch_out = torch_out
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("torch output fixtures not contiguous"))?;

    let model = MlModel::load(ModelSource::EmbeddedDefault)?;

    let meta: serde_json::Value = serde_json::from_slice(META_JSON)?;
    let patch_size = meta.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if patch_size > 0 {
        assert_eq!(model.patch_size(), patch_size);
    }

    let preds = model.infer_batch(patches, batch)?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..batch {
        for j in 0..3 {
            let torch_v = torch_out[i * 3 + j];
            let pred_v = preds[i][j];
            let diff = (pred_v - torch_v).abs();
            max_abs = max_abs.max(diff);
            let rel = diff / (torch_v.abs() + 1e-6);
            max_rel = max_rel.max(rel);
        }
    }

    assert!(max_abs < 2e-4, "max_abs {max_abs}");
    assert!(max_rel < 5e-4, "max_rel {max_rel}");

    let mut max_conf = 0.0f32;
    for i in 0..batch {
        let torch_logit = torch_out[i * 3 + 2];
        let pred_logit = preds[i][2];
        let torch_conf = 1.0 / (1.0 + (-torch_logit).exp());
        let pred_conf = 1.0 / (1.0 + (-pred_logit).exp());
        let diff = (pred_conf - torch_conf).abs();
        max_conf = max_conf.max(diff);
    }
    assert!(max_conf < 5e-4, "max_conf {max_conf}");

    Ok(())
}
