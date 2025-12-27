use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tract_onnx::prelude::tract_ndarray::{Array4, Ix2};
use tract_onnx::prelude::*;

#[derive(Clone, Debug)]
pub enum ModelSource {
    Path(PathBuf),
    EmbeddedDefault,
}

pub struct MlModel {
    model: TypedRunnableModel<TypedModel>,
    patch_size: usize,
    #[allow(dead_code)]
    // Keep SymbolScope alive for dynamic batch resolution.
    symbols: SymbolScope,
}

impl MlModel {
    pub fn load(source: ModelSource) -> Result<Self> {
        let (model_path, patch_size) = match source {
            ModelSource::Path(path) => {
                let patch_size =
                    patch_size_from_meta_path(&path).unwrap_or_else(default_patch_size);
                (path, patch_size)
            }
            ModelSource::EmbeddedDefault => {
                #[cfg(feature = "embed-model")]
                {
                    let patch_size = patch_size_from_meta_bytes(EMBED_META_JSON)
                        .unwrap_or_else(|_| default_patch_size());
                    let path = embedded_model_path()?;
                    (path, patch_size)
                }
                #[cfg(not(feature = "embed-model"))]
                {
                    return Err(anyhow!(
                        "embedded model support disabled; enable feature \"embed-model\""
                    ));
                }
            }
        };

        let mut model = tract_onnx::onnx()
            .model_for_path(&model_path)
            .with_context(|| format!("load ONNX model from {}", model_path.display()))?;
        let symbols = SymbolScope::default();
        let batch = symbols.sym("N");
        let shape = tvec!(
            batch.to_dim(),
            1.to_dim(),
            (patch_size as i64).to_dim(),
            (patch_size as i64).to_dim()
        );
        model
            .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), shape))
            .context("set ML refiner input fact")?;
        let model = model
            .into_optimized()
            .context("optimize ONNX model")?
            .into_runnable()
            .context("make ONNX model runnable")?;

        Ok(Self {
            model,
            patch_size,
            symbols,
        })
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn infer_batch(&self, patches: &[f32], batch: usize) -> Result<Vec<[f32; 3]>> {
        if batch == 0 {
            return Ok(Vec::new());
        }
        let patch_area = self.patch_size * self.patch_size;
        let expected = batch * patch_area;
        if patches.len() != expected {
            return Err(anyhow!(
                "expected {} floats (batch {} * patch {}x{}), got {}",
                expected,
                batch,
                self.patch_size,
                self.patch_size,
                patches.len()
            ));
        }

        let input = Array4::from_shape_vec(
            (batch, 1, self.patch_size, self.patch_size),
            patches.to_vec(),
        )
        .context("reshape input patches")?
        .into_tensor();
        let result = self
            .model
            .run(tvec!(input.into_tvalue()))
            .context("run ONNX inference")?;
        let output = result[0]
            .to_array_view::<f32>()
            .context("read ONNX output")?
            .into_dimensionality::<Ix2>()
            .context("reshape ONNX output")?;

        if output.ncols() != 3 {
            return Err(anyhow!(
                "expected output shape [N,3], got [N,{}]",
                output.ncols()
            ));
        }

        let mut out = Vec::with_capacity(batch);
        for row in output.outer_iter() {
            out.push([row[0], row[1], row[2]]);
        }
        Ok(out)
    }
}

fn patch_size_from_meta_bytes(bytes: &[u8]) -> Result<usize> {
    let meta: serde_json::Value =
        serde_json::from_slice(bytes).context("parse ML refiner meta.json")?;
    let size = meta
        .get("patch_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("meta.json missing patch_size"))?;
    Ok(size as usize)
}

fn patch_size_from_meta_path(path: &Path) -> Option<usize> {
    let meta_path = path.parent()?.join("fixtures").join("meta.json");
    let bytes = std::fs::read(meta_path).ok()?;
    patch_size_from_meta_bytes(&bytes).ok()
}

fn default_patch_size() -> usize {
    #[cfg(feature = "embed-model")]
    {
        patch_size_from_meta_bytes(EMBED_META_JSON).unwrap_or(21)
    }
    #[cfg(not(feature = "embed-model"))]
    {
        21
    }
}

#[cfg(feature = "embed-model")]
const EMBED_ONNX_NAME: &str = "chess_refiner_v2.onnx";
#[cfg(feature = "embed-model")]
const EMBED_ONNX_DATA_NAME: &str = "chess_refiner_v2.onnx.data";

#[cfg(feature = "embed-model")]
const EMBED_ONNX: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/chess_refiner_v2.onnx"
));
#[cfg(feature = "embed-model")]
const EMBED_ONNX_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/chess_refiner_v2.onnx.data"
));
#[cfg(feature = "embed-model")]
const EMBED_META_JSON: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ml/fixtures/v2/meta.json"
));

#[cfg(feature = "embed-model")]
fn embedded_model_path() -> Result<PathBuf> {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    if let Some(path) = PATH.get() {
        return Ok(path.clone());
    }

    let dir = std::env::temp_dir().join("chess_corners_ml");
    std::fs::create_dir_all(&dir).context("create ML model temp dir")?;
    let onnx_path = dir.join(EMBED_ONNX_NAME);
    let data_path = dir.join(EMBED_ONNX_DATA_NAME);
    std::fs::write(&onnx_path, EMBED_ONNX).context("write embedded ONNX model")?;
    std::fs::write(&data_path, EMBED_ONNX_DATA).context("write embedded ONNX data")?;
    let _ = PATH.set(onnx_path.clone());
    Ok(onnx_path)
}
