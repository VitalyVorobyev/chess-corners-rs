# chess-corners-ml

ONNX-based ML refiner inference for the `chess-corners` pipeline.

This crate provides a small wrapper around `tract-onnx` that loads the refiner
model and runs batched inference on normalized intensity patches. It is a
low-level utility crate; most users should use the high-level API in the
`chess-corners` crate instead.

Features:

- `embed-model` *(default)* – embed the shipped ONNX model and metadata directly
  in the crate.

Basic usage:

```rust
use chess_corners_ml::{MlModel, ModelSource};

fn main() -> anyhow::Result<()> {
    let model = MlModel::load(ModelSource::EmbeddedDefault)?;
    let patch_size = model.patch_size();
    let patch_area = patch_size * patch_size;
    let patches = vec![0.0f32; patch_area]; // batch = 1
    let preds = model.infer_batch(&patches, 1)?;
    println!("pred = {:?}", preds[0]); // [dx, dy, conf_logit]
    Ok(())
}
```

Loading a custom model:

```rust
use chess_corners_ml::{MlModel, ModelSource};
use std::path::PathBuf;

let model = MlModel::load(ModelSource::Path(PathBuf::from("refiner.onnx")))?;
```

When loading from a path, the crate looks for a `fixtures/meta.json` file next
to the ONNX model to determine the patch size (falls back to 21 if missing).

For the full detector pipeline, ML patch extraction, and configuration, see the
`chess-corners` crate.
