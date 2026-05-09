# Two-Axis Orientation Fit — Benchmark Report

<!-- AUTO-GENERATED BELOW: do not edit by hand -->
Generated: <ts>  Commit: <git-rev>  Config: <name>

## Setup
This report is auto-rendered. Replace the placeholders by re-running
`python -m orientation_bench report --in <out_dir>`.

## Sweeps
| Sweep | Param | n/cell |
| --- | --- | --- |

## Baseline results (current `fit_two_axes`)
### Per-sweep summary
| Sweep | Param | Bias_a0 deg | RMSE_a0 deg | Bias_a1 deg | RMSE_a1 deg | z_std_a0 | z_std_a1 | failure% | detection% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Sigma calibration summary
- mean(z) across all cells: n/a
- std(z) across all cells: n/a
- |z|>2 fraction: n/a

### Plots
- CDF: plots/error_cdf_blur.png
- Error vs blur: plots/error_vs_blur.png

<!-- AUTO-GENERATED ABOVE -->

<!-- KEEP-MANUAL-START -->
## Hypothesis verdicts

Corrected baseline gate run: `out/orientation_bench_corrected_gate`
(`bench_default`, seed 1, n=300/cell patch, n=30/cell chess, first 8
cells for long sweeps).

Full-disk prototype gate run: `out/orientation_bench_disk_sector_gate6`
with status plots in `out/orientation_bench_disk_sector_status`.

Important correction: the first axis-skew report treated the patch
homography in the wrong direction. `H` is now consistently
source/patch → image for the GT Jacobian, and the renderer samples
through `H⁻¹`. The old 60°/120° “orthogonality catastrophe” was mostly
a benchmark artefact.

| # | Hypothesis | Verdict | Evidence |
|---|------------|---------|----------|
| H1 | β=4 fixed → blur-dependent variance | **CONFIRMED, but V1 failed** | Baseline blur RMSE remains U-shaped: 4.28°/4.34° at σ=0, 0.82°/0.83° near σ=1.5, 1.67°/1.73° at σ=4. AdaptiveBeta improves σ=0 but catastrophically regresses σ=4 to 17.3° and worsens noise/contrast. Do not ship V1. |
| H2 | 16 ring samples → underestimated σ | **CONFIRMED for precision, not accuracy** | Baseline noise sweep RMSE changes only 1.65°→2.18° from noise σ=0→20, but z_std sits around 1.19–1.30. The reported σ is systematically overconfident. |
| H3 | 2nd-harmonic seed biases to orthogonal axes | **PARTIAL after GT fix** | Corrected axis-skew baseline is acceptable for 60°–120° (≈1.75°–2.63° RMSE) but still fails at extreme 30°/150° (≈18°–21° RMSE). This is no longer the dominant normal-pose problem; it is an extreme-skew/outlier problem. |
| H4 | GN without LM diverges | **REFUTED for baseline** | Baseline failure stays ≤0.7% in corrected skew and 0% on blur/noise/contrast/chess_pose. AdaptiveBeta introduces failures up to 7%, so damping would be a V1 bandage, not a baseline fix. |
| H5 | CRLB σ is miscalibrated | **CONFIRMED and fixed by V6b** | Overall mean z_std drops from 1.44 (baseline) to 1.17 with `sigma_correction_lut`, while RMSE is bit-identical. Noise/contrast/chess_pose z_std land close to 1.0. Extreme 30°/150° skew remains overconfident because the angle itself is wrong. |
| H6 | Bilinear ring sampling aliases sharp edges | **WEAK** | Subpixel cells remain around 1.6°–1.8° RMSE. The signal is smaller than blur=0 variance and sigma calibration. |
| H7 | In-disk pixels unused | **CONFIRMED** | `disk_sector_py` uses disk pixels plus gradient-line evidence and reduces mean RMSE 2.68° → 1.38°. Extreme 30°/150° skew drops 19.1°/19.4° → 2.77°/3.03°, and blur=0 drops 4.31° → 1.48°. This is benchmark-only Python, not a public API yet. |
| H8 | Axis ordering/polarity instability | **BENCH CONVENTION, not a detector bug** | Axis matching must remain unordered/mod-π for metric aggregation. The previous 50% chess swap signal is expected from alternating checker parity and should not drive a detector change. |

## Recommended changes

Based on the corrected gate:

### Priority HIGH

- **Ship / keep V6b — `SigmaCorrectionLut`** for precision. It preserves all angles and fit residuals exactly, but calibrates σ much closer to empirical RMSE: overall mean z_std 1.44 → 1.17, noise/contrast/chess_pose mostly ≈0.95–1.22. This is the robust solution for downstream weighting without changing detector behavior.

### Priority MEDIUM

- **Port the successful full-disk prototype to Rust only after profiling the design.** `disk_sector_py` passes the corrected synthetic gate: no cell regresses by both >0.25° and >10%, failure stays at 0%, and mean z_std is 0.826. It is still too slow as Python and full-frame runs currently refine only top-response rows, so promote the concept, not the script.

### Priority LOW / drop

- **Drop V1 / AdaptiveBeta as currently implemented.** Corrected gate mean RMSE is 4.94° vs baseline 2.68°; it fails blur=4, noise≥5, low contrast, chess_pose, and raises failure rate.
- **Do not add LM damping to baseline.** Baseline convergence is not the measured issue.
- **Do not change axis-ordering/canonicalization based on swap_frac.** Metrics should pair axes unordered; checker parity makes fixed GT ordering misleading.

## Open questions

- Should `SigmaCorrectionLut` become the facade default after a larger run and real-image validation, or remain opt-in for one release?
- The full-disk prototype fixes the synthetic extreme-skew gate locally, but real-image validation is still missing.
- The blur=0 RMSE inflation is largely addressed by the disk/edge estimator in synthetic data; verify the same on real sharp board images before changing defaults.
<!-- KEEP-MANUAL-END -->
