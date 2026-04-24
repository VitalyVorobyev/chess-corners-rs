# ML Refiner Retraining Plan (v3)

`chess_refiner_v2.onnx` scores ~0.5 px mean error on
`crates/chess-corners/tests/refiner_benchmark.rs`, while CenterOfMass
hits 0.08 and RadonPeak 0.05. The ONNX/PyTorch parity test
(`crates/chess-corners-ml/tests/onnx_parity.rs`) is green, so the
failure is a train-test distribution gap, not a deployment bug. Files
reviewed: `tools/ml_refiner/{model,train,dataset,export_onnx,eval}.py`
and `tools/ml_refiner/synth/{render_corner,generate_dataset,augment,
homography,negatives}.py` plus `configs/{synth_v2,train_v2}.yaml`.

## 1. Diagnosis

- **D1 — Corner model is an infinite saddle, benchmark is hard cells.**
  `render_corner.render_ideal_corner_from_grid` (line 15) returns
  `0.5 + 0.5·tanh(x_r/(edge·s))·tanh(y_r/(edge·s))` — a smooth saddle
  filling the whole patch. The benchmark fixture
  (`refiner_benchmark.rs:34-65`) rasterises *hard* cells at 8× and
  adds a Gaussian blur; a 21×21 window around a corner at cell=5 shows
  up to 4 alternating cells with *neighbouring* corners the model has
  never seen. This is the dominant failure.
- **D2 — `scale: [0.7, 1.3]` does not control cell size.** It enters
  only through `edge_softness·scale` inside tanh, i.e. it sets the
  transition width (0.21–0.39 px). There is *no* parameter in the v2
  pipeline for pixel cell extent — the model never learns periodic
  structure.
- **D3 — Photometrics roughly cover the fixture but on the wrong
  shape.** `contrast [0.8, 1.2]`, `brightness [-10, 10]`,
  `gamma [0.8, 1.2]` → uint8 extrema ~30..225, close to the benchmark
  `dark=30, bright=230`. Range is fine; the step-vs-tanh intensity
  profile is not.
- **D4 — Homography warps the infinite saddle, adding no real
  structure.** `homography.sample_homography` (line 136) can produce
  strong perspective but the pre-warp image is still tanh-smooth, so
  F4 does not rescue D1.
- **D5 — Seed-offset coverage is adequate.** Benchmark seeds with
  `seed.round()` → sub-pixel truth in `[-0.5, 0.5)`; training
  `dx/dy_range: [-1.5, 1.5]` covers it but over-samples large
  offsets (mildly wasteful, not the root cause).
- **D6 — Normalisation and CoordConv match.** Rust
  `ml_refiner.rs::extract_patch_u8_to_f32` (line 466) does bilinear
  + `/255` with early OOB reject; Python `dataset._normalize_patches`
  (line 67) does the same. No mismatch.
- **D7 — Val split is self-same and cannot see the gap.**
  `dataset.ShardDataset` takes the last 10 % of shards (line 19),
  drawn from the same tanh generator, so val p95 is blind to the
  benchmark regime. `model_best.pt` was therefore selected on the
  wrong distribution.

## 2. Priority-ordered fixes

- **F1 (P0) — Replace tanh corners with AA-rasterised hard cells.**
  Mirror `synthetic_chessboard_aa` in `refiner_benchmark.rs` (8×
  supersample, cell-parity shading). Addresses D1, the single largest
  lever.
- **F2 (P0) — Parametrise `cell_size_px` in `[4.0, 12.0]` px.** The
  cell size must be an explicit random axis; v2's `scale` dial
  becomes redundant and is removed in hard-cells mode. Covers
  benchmark 5 and 8 with headroom.
- **F3 (P0) — Gaussian PSF after rasterisation, before photometric
  jitter.** `blur_sigma ∈ [0.3, 2.0]` matches the benchmark
  `gaussian_blur` (σ up to 1.5) plus real MTF floor.
- **F4 (P1) — Render a *tile grid* (not a single corner) so
  neighbouring saddles appear inside the 21×21 window.** Natural
  consequence of F1+F2.
- **F5 (P1) — Tighten seed offsets to `[-0.6, 0.6]`** (matches
  `seed.round()` plus a 0.1 px slack); add a secondary
  `[-0.15, 0.15]` mode for near-final convergence.
- **F6 (P2) — Relax homography to `scale_x/y [0.85, 1.15]`,
  `shear [-0.1, 0.1]`, `p [-0.0015, 0.0015]`.** Overlaps with real
  board perspectives; extreme v2 homographies are not representative.
- **F7 (P2) — Held-out val set rendered with the benchmark
  generator.** Used for `save_best`; fixes D7.
- **F8 (P3) — Revisit conf loss weighting.** `train.compute_loss`
  line 101 multiplies regression loss by `(0.2 + 0.8·conf)·is_pos`;
  with the wider blur range this starves gradients at high σ. Retune
  `conf_params.a` from 0.6 to 0.3 once F1-F5 land.

## 3. `configs/synth_v4.yaml` — key/value sketch

- `seed: 42`, `patch_size: 21`, `num_samples: 400_000`,
  `shard_size: 20_000`, `super_res: 8` (up from 4 to match benchmark).
- `render_mode: hard_cells` (new; `tanh` retained for ablation).
- `cell_size_px: [4.0, 12.0]` (F2).
- `rotation: [0.0, 6.2831853]`, `dx_range: [-0.6, 0.6]`,
  `dy_range: [-0.6, 0.6]` (F5).
- `blur_sigma: [0.3, 2.0]` (F3), `noise_sigma: [0.0, 10.0]` (unchanged).
- `contrast: [0.5, 1.5]`, `brightness: [-20, 20]`, `gamma: [0.7, 1.4]`
  (widen slightly for lighting while respecting fixture clipping).
- `edge_softness: 0.0` (unused in hard-cells mode).
- `homography`: `enabled: true`, `scale_x/y: [0.85, 1.15]`,
  `shear_range: [-0.1, 0.1]`, `p_range: [-0.0015, 0.0015]`,
  `tx/ty_range: [-0.6, 0.6]` (F6).
- `conf_params: {a: 0.3, b: 0.01}` (F8).
- `neg: {enabled: true, fraction: 0.2}`.

Companion `configs/val_hardcell_v4.yaml` with `cell_size_px ∈ {5, 8}`
only, negatives disabled, used by `eval.py` for model selection.

## 4. Evaluation protocol

**4a. Offline.** Held-out `data/val_hardcell_v4/` rendered with the
same AA-hard-cell pipeline as the benchmark. Sweep:
`cell ∈ {4,5,6,8,10,12}`, `blur_σ ∈ {0,0.5,1.0,1.5,2.0}`,
`noise_σ ∈ {0,5,10}`, photometric `contrast ∈ {0.6, 1.0}`, with a
7×7 sub-pixel offset grid. Report mean/p50/p90/p95/worst and
accept-rate (reject at > 1.0 px). Promotion gates:

- clean cell=8: mean < 0.10, p95 < 0.20 px;
- clean cell=5: mean < 0.15, p95 < 0.30 px;
- blur σ=1.5 cell=8: mean < 0.20 px;
- noise σ=10 cell=8: mean < 0.25 px.

**4b. Online.** `crates/chess-corners/tests/refiner_benchmark.rs`
becomes the integration gate. Add ML-column asserts:

- `sweep_clean`: ML mean < 0.15;
- `sweep_clean_small_cell`: ML mean < 0.20;
- `sweep_blur_1_5`: ML mean < 0.20;
- `sweep_noise_5` / `sweep_noise_10`: ML mean < 0.25.

## 5. Model architecture

`model.py` is a 5-layer conv stack (16→32→64 with two stride-2
downsamples), CoordConv, then `LazyLinear(64)→Linear(3)`. On 21×21
this gives a ~6×6×64 feature map. Capacity is adequate for sub-pixel
regression — a 0.5 px residual with a broken data distribution will
not shrink by scaling the net. **Verdict: failure is data, not
architecture.** Keep the topology; optional single tweak for v3: add
`GroupNorm(8, c)` after each conv (ONNX-opset-17 safe, deterministic
at inference) to stabilise training under wider contrast jitter. Adds
<1 % params.

## 6. Roll-out

- **M1 (1-2 days).** Implement F1 hard-cell generator (switch in
  `synth/render_corner.py`), regenerate `data/synth_v4` +
  `data/val_hardcell_v4`, extend `eval.py` to emit the §4a table.
  Visual smoke: `--preview 16` showing green cross on real saddle
  between cells.
- **M2 (2-3 days).** Train v3 with a fresh `configs/train_v3.yaml`,
  select best on `val_hardcell_v4` p95. Confirm §4a gates. Export
  ONNX via `export_onnx.py` (I/O contract unchanged). Run
  `cargo test -p chess-corners --test refiner_benchmark
  --all-features -- --nocapture` and confirm §4b gates.
- **M3 (0.5 day).** Replace
  `crates/chess-corners-ml/assets/ml/chess_refiner_v2.onnx` (5.8 KB
  graph + 713 KB weights = ~719 KB today) with `chess_refiner_v3.onnx`
  (budget ~730 KB with GroupNorm). Update
  `ModelSource::EmbeddedDefault` in
  `crates/chess-corners-ml/src/lib.rs`, regenerate fixtures under
  `assets/ml/fixtures/`, bump thresholds in `onnx_parity.rs` if
  needed. Keep v2 for one release for A/B.

## 7. Open questions

- Was v2 ever trained on real captures or OpenCV
  `findChessboardCornersSB` labels? The repo only shows an *eval*
  OpenCV comparison (`configs/compare_opencv_v1.yaml`,
  `eval/compare_opencv.py`), no training loader — if labels from real
  boards were used out-of-tree, we need them for v3 too.
- Where is `synth_v3.yaml`? `tools/ml_refiner/data/synth_v3` exists
  and `configs/train_v2.yaml` points to it, but no matching config is
  checked in. Config/data drift could mean v2 was trained against a
  different distribution than `synth_v2.yaml` describes.
- Is the 0.5 px benchmark error a *bias* (predictions collapse to 0)
  or *variance* (noisy around truth)? A Python eval on the benchmark
  fixture would disambiguate and confirm F1 is the biggest lever
  rather than F5.
- Any stash of real chessboard patches with high-resolution ground
  truth? 200-500 labelled real patches would validate sim-to-real
  transfer that synthetic data alone cannot.
- Did `tools/ml_refiner/data/synth_v4` (present on disk) influence v2
  training? If so the true training YAML differs from what's in
  `configs/`, changing the diagnosis weighting.
