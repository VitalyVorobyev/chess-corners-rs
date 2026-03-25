# ADR-005: Embedded ONNX Model with Lazy Extraction

## Status

Accepted (retrospective, introduced in v0.3.0)

## Context

The ML refiner requires an ONNX model file for inference. Distribution options considered:

1. **Separate file download** -- users must manage model path, complicates distribution
2. **Embedded in binary** -- single binary distribution, but increases binary size
3. **Network fetch** -- adds network dependency, fails offline

The primary use case is library consumers who want zero-configuration ML refinement.

## Decision

Embed the ONNX model bytes via `include_bytes!` (behind the `embed-model` feature, enabled by default in `chess-corners-ml`). On first use, extract the model to a temporary file and cache the loaded model in a `OnceLock` for the process lifetime.

An alternative `ModelSource::Path` allows advanced users to provide their own model file.

## Consequences

**Positive:**
- Single binary distribution with zero configuration required
- First-use extraction is a one-time cost, amortized across subsequent calls
- `OnceLock` ensures thread-safe lazy initialization

**Negative:**
- Binary size increases by the model size
- Model updates require recompilation (or using `ModelSource::Path`)
- Temp file cleanup is implicit (OS handles it)
- First invocation has extraction overhead
