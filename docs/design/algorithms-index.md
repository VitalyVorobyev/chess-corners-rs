# Atomic algorithms index

> **Purpose.** A single map of every distinct computational stage in the
> workspace, where it lives, and how the stages connect. This is a
> *pointer* document for navigating the code — it does **not** re-derive the
> math. The book (`book/src/`, Parts III–VIII) is the teaching reference;
> each row links to the relevant Part.

**How to read it.** Each algorithm has a stable ID (`R1`, `F2`, …) used in
the DAG and in cross-references. "In → Out" is the data contract; "Consumed
by" lists the downstream stages. Paths are relative to the repo root.

## Master table

| ID | Algorithm | Stage | Source | Book | Feature gate | In → Out | Consumed by |
|----|-----------|-------|--------|------|--------------|----------|-------------|
| **R1** | ChESS 16-sample ring response | Response | `crates/chess-corners-core/src/detect/chess/response.rs` | III | `simd`, `rayon` | `u8` image → `ResponseMap` (f32) | D1 |
| **R2** | Radon SAT ray response | Response | `crates/chess-corners-core/src/detect/radon/response.rs` | IV | `radon-sat-u32` | `u8` image → Radon response view | D2 |
| **H1** | Ring offsets / angles (radius 5, 10) | Response helper | `crates/chess-corners-core/src/detect/chess/ring.rs` | III | — | radius → coord table | R1, X1, O1 |
| **D1** | ChESS detection: threshold → NMS → cluster filter | Detection | `crates/chess-corners-core/src/detect/chess/detect.rs` | III | — | `ResponseMap` → `Vec<Corner>` | F1–F3, X1 |
| **D2** | Radon detection: threshold → NMS → cluster + 3-pt Gaussian fit | Detection | `crates/chess-corners-core/src/detect/radon/detect.rs` | IV | — | Radon view → `Vec<Corner>` | F4, X1 |
| **T1** | `DenseDetector` trait (drives both detectors) | Detection abstraction | `crates/chess-corners-core/src/detect/dense.rs` | III–IV | — | image → response + peaks | S3 |
| **F1** | CenterOfMass refiner | Refinement | `crates/chess-corners-core/src/refine/center_of_mass.rs` | V | — | response patch → subpixel `(x,y)` | X1 |
| **F2** | Förstner refiner (structure tensor) | Refinement | `crates/chess-corners-core/src/refine/forstner.rs` | V | — | image patch → subpixel `(x,y)` | X1 |
| **F3** | SaddlePoint refiner (quadratic Hessian) | Refinement | `crates/chess-corners-core/src/refine/saddle_point.rs` | V | — | response patch → subpixel `(x,y)` | X1 |
| **T2** | `CornerRefiner` trait + `Refiner` dispatch enum | Refinement abstraction | `crates/chess-corners-core/src/refine/mod.rs` | V | — | corner → refined corner | D1, D2 |
| **O1** | RingFit two-axis orientation (Gauss-Newton + σ-LUT) | Orientation | `crates/chess-corners-core/src/orientation/ring_fit/` | VI | `rayon` | ring samples → 2 axes + σ | X1 |
| **O2** | DiskFit two-axis orientation (full-disk, lazy RingFit fallback) | Orientation | `crates/chess-corners-core/src/orientation/disk_sector/` | VI | — | disk samples → 2 axes + σ | X1 |
| **U1** | Per-axis σ uncertainty calibration LUT | Orientation helper | `crates/chess-corners-core/src/orientation/ring_fit/uncertainty.rs` | VI | — | `(amp, rms)` → σ correction | O1, O2 |
| **X1** | Corner descriptors (axes, response) | Descriptors | `crates/chess-corners-core/src/orientation/descriptor.rs` | VI | `rayon` | corners + image → `CornerDescriptor[]` | output |
| **S1** | Box pyramid 2× downsample | Scale | `crates/box-image-pyramid/src/lib.rs` | VII | `par_pyramid` | `u8` image → pyramid levels | S3 |
| **S2** | Integer bilinear upscale (×2/3/4) | Scale | `crates/chess-corners/src/upscale.rs` | VII | — | low-res `u8` → upscaled `u8` | S1, R1/R2 |
| **S3** | Multiscale coarse-to-fine pipeline | Orchestration | `crates/chess-corners/src/multiscale.rs` | VII | — | image → `CornerDescriptor[]` | output |
| **M1** | ONNX ML refiner (optional) | Refinement (post) | `crates/chess-corners/src/ml_refiner.rs` | V | `ml-refiner` | corner + patch → refined corner | (post-F*) |

## Interconnection DAG

```
u8 image
  │
  ├─(opt S2) integer bilinear upscale ─┐
  ├─(opt S1) box pyramid downsample ───┤   orchestrated by S3 multiscale (coarse→fine)
  ▼                                     ▼
[R1] ChESS ring response   |   [R2] Radon SAT response          (via T1 DenseDetector)
        │                              │
        ▼                              ▼
[D1] detect: threshold→NMS→cluster   [D2] detect: …→Gaussian peak fit
        │                              │
        ▼                              ▼
[F1/F2/F3] subpixel refine            │            (via T2; [M1] optional post-step)
        └──────────────┬──────────────┘
                       ▼
        [O1] RingFit  /  [O2] DiskFit   (two-axis orientation; uses [U1] σ-LUT, [H1] ring tables)
                       │
                       ▼
        [X1] descriptors → CornerDescriptor[]   (S3 merges cross-level duplicates)
                       │
                       ▼
                    output
```

## Crate × stage map

| Crate | Stages it owns |
|-------|----------------|
| `box-image-pyramid` | S1 (fully standalone; zero chess coupling) |
| `chess-corners-core` | R1, R2, H1, D1, D2, T1, F1–F3, T2, O1, O2, U1, X1 |
| `chess-corners` (facade) | S2, S3 + the high-level `Detector`/config that selects R*/D*/F*/O* |
| `chess-corners-ml` | M1 (only with `ml-refiner`) |
| `chess-corners-py` / `-wasm` | bindings over the facade; own no algorithm |

## Feature-flag effects (perf-relevant)

| Flag | Crates | Effect on stages |
|------|--------|------------------|
| `rayon` | core, facade | row-parallel R1; parallel X1 descriptor build |
| `simd` | core, facade | portable-SIMD inner loop in R1 (nightly) |
| `par_pyramid` | facade, box-image-pyramid | SIMD/rayon in S1 downsample |
| `radon-sat-u32` | core | u32 SAT element in R2 (memory/lane trade-off) |
| `ml-refiner` | facade, py | enables M1 |

---

See also: [`architecture.md`](architecture.md) (crate layering & data flow),
[`../reference/detector-comparison.md`](../reference/detector-comparison.md)
(R1/D1 vs R2/D2), and
[`../reference/refiner-comparison.md`](../reference/refiner-comparison.md)
(F1–F3 accuracy/throughput, plus ML).
