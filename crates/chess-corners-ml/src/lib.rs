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
const EMBED_ONNX_NAME: &str = "chess_refiner_v4.onnx";
#[cfg(feature = "embed-model")]
const EMBED_ONNX_DATA_NAME: &str = "chess_refiner_v4.onnx.data";

#[cfg(feature = "embed-model")]
const EMBED_ONNX: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/ml/chess_refiner_v4.onnx"
));
#[cfg(feature = "embed-model")]
const EMBED_ONNX_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/ml/chess_refiner_v4.onnx.data"
));
#[cfg(feature = "embed-model")]
const EMBED_META_JSON: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/ml/fixtures/v4/meta.json"
));

#[cfg(feature = "embed-model")]
fn embedded_model_path() -> Result<PathBuf> {
    // `OnceLock::get_or_init` serializes the writes across threads in
    // this process. Without it, parallel `#[test]` runs all entered
    // `write_if_changed`, the second `std::fs::write` truncated the
    // file to 0 bytes mid-rewrite, and a concurrent `tract_onnx`
    // model load saw an empty `.data` slice and panicked
    // (`range start index 768 out of range for slice of length 0`).
    //
    // For cross-process races (e.g. `cargo test -p A` and
    // `cargo test -p B` sharing `/tmp/chess_corners_ml/`), the
    // atomic write-then-rename in `write_if_changed` ensures the
    // file is either at its old contents or at its new contents,
    // never partially written.
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    let path = PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join("chess_corners_ml");
        std::fs::create_dir_all(&dir).expect("create ML model temp dir");
        let onnx_path = dir.join(EMBED_ONNX_NAME);
        let data_path = dir.join(EMBED_ONNX_DATA_NAME);
        // Write `.data` before `.onnx` so tract never sees an `.onnx`
        // that references a missing or partially-written `.data`.
        write_if_changed(&data_path, EMBED_ONNX_DATA).expect("write embedded ONNX data");
        write_if_changed(&onnx_path, EMBED_ONNX).expect("write embedded ONNX model");
        onnx_path
    });
    Ok(path.clone())
}

/// Write `data` to `path` only if the file doesn't already contain
/// the same bytes. Uses write-then-rename so concurrent readers see
/// either the old contents or the new contents — never a truncated /
/// partially-written file. Cheap-out via the byte-match check avoids
/// rewriting unchanged files across re-runs in a shared temp dir.
#[cfg(feature = "embed-model")]
fn write_if_changed(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() == data.len() as u64 {
            if let Ok(existing) = std::fs::read(path) {
                if existing == data {
                    return Ok(());
                }
            }
        }
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

#[cfg(all(test, feature = "embed-model"))]
mod tests {
    use super::write_if_changed;

    #[test]
    fn write_if_changed_rewrites_same_size_changed_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.bin");

        write_if_changed(&path, b"abc").expect("initial write");
        write_if_changed(&path, b"xyz").expect("rewrite same-size bytes");

        let bytes = std::fs::read(&path).expect("read rewritten bytes");
        assert_eq!(bytes, b"xyz");
    }
}
