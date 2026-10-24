# Asymmetry term: prototype and gates (UQ-26, UQ-27)

Gate for [design §4.2 / §8](../design/uncertainty-v2.md): can the
antisymmetric part of a SaddlePoint patch give a per-corner systematic term
(a), a bias correction (b), or is a multi-radius spread (c) enough?
Scripts: `tools/refiner_robustness/asymmetry/` (`core.py` numpy SaddlePoint
and asymmetry fit as linear pixel maps; `check.py`, `gain.py`,
`gate_synth.py`, `gate_real.py`). Raw output goes to
`tools/refiner_robustness/out/asym_*.json` (gitignored).

## Method

At the converged SaddlePoint estimate \(\hat c\), \(P(d) = I(\hat c + d)\)
(bilinear, integer \(d\)), \(S = \tfrac12[P(d) + P(-d)]\),
\(A = \tfrac12[P(d) - P(-d)]\) on one half-plane of the window.
Least squares \(A \approx b\cdot\nabla S + (g\cdot d)\,R(d)\) with
\(\nabla S\) by central differences. Variants of the ramp column \(R\):

- design text: \(R = S - m\) (\(m\) = window mean), 5×5;
- `+add`: \(R = S - m\) plus an additive ramp \(g'\cdot d\);
- `mul`: \(R = S\), a pure multiplicative illumination ramp, 5×5 / 7×7 / 9×9.

\(b\) moves with \(\hat c\) at gain \(G = \partial b/\partial \hat c\)
(≈ 1.13 I; central differences understate the bilinear slope), so the
estimate is the Newton step \(\beta = G^{-1}b\) toward \(b(c) = 0\).
\(G = X^+_{[b]}\,\partial A/\partial c\) analytically (bilinear derivative
taps); it matches finite differences to 0.005–0.04 (median), \(\beta\) to 0.002 px.
Noise maps (pixels → output, \(X\) held fixed): \(L_\beta = L_c + G^{-1}L_b\),
corrected position \(-G^{-1}L_b\), with \(L_c = -M^{-1}JRW\) the SaddlePoint
map and \(L_b = X^+_{[b]}QW\).

- (a) \(\Sigma = \Sigma_{1.5} + \big(\beta\beta^\top - \sigma^2 L_\beta L_\beta^\top\big)_+\)
  (positive part by eigenvalue clipping); the raw \(\beta\beta^\top\) is ~10% more
  conservative on symmetric conditions.
- (b) \(\hat c - \beta\); "gated" applies it only where the ramp is
  significant (\(g^\top\mathrm{Cov}(g)^{-1}g > 9.21\)).
- (c) SaddlePoint at radius 1, 2, 3; \(\Sigma = \Sigma_{1.5} + (D - E_n[D])_+\)
  with \(D\) the sample covariance of the three estimates.

Checks: the numpy SaddlePoint reproduces the bindings (positions 2e-4 px,
covariances 1e-4 relative median); Monte Carlo scatter / predicted variance
is 0.99–1.01 for SaddlePoint at radius 1/2/3, \(\beta\) and the corrected
position.

## Synthetic gate

Independent generator (`../run_synth.py` conditions), 3 boards × 12 draws,
noise σ supplied (realised), σ̂₀² = mean \(e^\top\Sigma^{-1}e\)/2. Cell 10,
noise 2. "+0.010 floor" is a control: 1.5.0 plus a constant 0.010 px floor.
RMS and bias in px; (b) = `mul` 9×9 ungated unless stated.

| condition | 1.5.0 | +0.010 floor | (a) design 5×5 | (a) mul 9×9 | (c) radii | RMS SP | RMS (b) | RMS (b) gated | bias SP | bias (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.96 | 0.67 | 0.83 | **0.77** | 0.78 | 0.021 | 0.023 | 0.022 | 0.008 | 0.013 |
| gauss 0.5 | 0.96 | 0.59 | 0.74 | **0.72** | 0.72 | 0.018 | 0.022 | 0.019 | 0.007 | 0.019 |
| gauss 2.0 | 0.94 | 0.83 | 0.82 | **0.74** | 0.74 | 0.039 | 0.063 | 0.041 | 0.007 | 0.012 |
| defocus disk | 0.94 | 0.61 | 0.78 | **0.74** | 0.74 | 0.019 | 0.020 | 0.019 | 0.007 | 0.014 |
| perspective+distortion | 0.99 | 0.71 | 0.85 | **0.76** | 0.80 | 0.023 | 0.029 | 0.029 | 0.009 | 0.022 |
| Poisson-Gaussian | 0.98 | 0.82 | 0.82 | **0.73** | 0.80 | 0.032 | 0.032 | 0.033 | 0.008 | 0.013 |
| gamma 1/2.2 | 0.80 | 0.60 | 0.68 | **0.63** | 0.68 | 0.022 | 0.025 | 0.023 | 0.008 | 0.013 |
| JPEG q90 / q75 | 0.93 / 1.23 | 0.70 / 0.99 | 0.76 / 0.96 | **0.72 / 0.88** | 0.72 / 0.87 | .024 / .032 | .025 / .034 | .024 / .034 | .014 / .025 | .017 / .028 |
| Bayer+demosaic | 2.36 | 1.42 | 1.82 | **1.64** | 1.69 | 0.027 | 0.029 | 0.029 | 0.009 | 0.012 |
| decentred defocus | 1.42 | 0.90 | 0.97 | **1.04** | 0.77 | 0.022 | 0.034 | 0.027 | 0.014 | 0.030 |
| vignetting 40% | 1.89 | 1.40 | 1.43 | **0.99** | 1.46 | 0.034 | 0.025 | 0.027 | 0.025 | 0.013 |
| ramp ±20% | 1.48 | 1.04 | 1.19 | **0.94** | 1.22 | 0.027 | 0.023 | 0.025 | 0.018 | 0.013 |
| shading ±5% / 5 cells | 3.26 | 2.28 | 2.06 | **1.11** | 2.00 | 0.040 | 0.024 | 0.025 | 0.034 | 0.014 |
| shading ±15% / 5 cells | 21.4 | 15.1 | 5.84 | **1.43** | 3.22 | 0.103 | 0.029 | 0.029 | 0.101 | 0.021 |
| coma mild / coma | 6.2 / 14.0 | 4.6 / 9.6 | 4.3 / 4.2 | **3.9 / 4.7** | 2.9 / 2.1 | .061 / .080 | .089 / .128 | .082 / .128 | .056 / .077 | .086 / .127 |
| combined A / B | 10.1 / 1.36 | 8.6 / 1.16 | 7.3 / 1.07 | **6.4 / 0.97** | 5.2 / 1.00 | .110 / .042 | .137 / .042 | .137 / .043 | .102 / .031 | .130 / .035 |

Grid (σ̂₀², 1.5.0 → (a) mul 9×9 → (c)), cells 8 / 12 at noise 2 and noise
1 / 8 at cell 10:

| condition | cell 8 | cell 12 | noise 1 | noise 8 |
|---|---|---|---|---|
| baseline | 0.96 → 0.76 → 0.78 | 0.97 → 0.78 → 0.79 | 0.90 → 0.75 → 0.68 | 1.01 → 0.78 → 0.84 |
| gauss 2.0 | 0.96 → 0.75 → 0.75 | 0.94 → 0.74 → 0.75 | 0.83 → **0.67** → 0.67 | 0.98 → 0.78 → 0.79 |
| decentred defocus | 1.44 → 1.05 → 0.77 | 1.44 → 1.05 → 0.77 | 1.92 → 1.06 → 0.60 | 1.04 → 0.78 → 0.84 |
| vignetting 40% | 1.90 → 0.98 → 1.46 | 1.93 → 1.00 → 1.49 | 3.38 → 1.15 → 2.07 | 1.08 → 0.80 → 0.90 |
| ramp ±20% | 1.47 → 0.93 → 1.21 | 1.47 → 0.93 → 1.21 | 2.24 → 1.17 → 1.69 | 1.03 → 0.78 → 0.86 |
| shading ±5% | 4.44 → 1.17 → 2.33 | 2.49 → 1.03 → 1.72 | 6.79 → 1.32 → 2.67 | 1.18 → 0.83 → 0.98 |
| shading ±15% | 31.8 → **1.54** → 3.43 | 15.1 → 1.36 → 3.13 | 53.6 → **1.99** → 3.57 | 2.53 → 1.02 → 1.75 |

Findings:

- The design's ramp column \((g\cdot d)(S - m)\) is wrong for a
  multiplicative gradient: it drops the \(m\,(g\cdot d)\) half, which then
  leaks into \(b\) (shading ±15%: 5.84). Adding a free additive ramp makes
  \(b\) nearly unidentifiable (7×7, blur 2: RMS 0.185 px, σ̂₀² 0.50). With
  \(R = S\) the fit removes 30–80% of the illumination bias (shading ±15%
  0.101 → 0.021 px, ramp 0.018 → 0.013).
- Window: 9×9 > 7×7 > 5×5 at equal model (shading ±15%: 1.43 / 2.07 / 3.98).
  The fit cannot separate a shift from a ramp by symmetry class (both are
  odd); it relies on the radial profile (step vs ramp), which a small
  window or strong blur hides.
- The per-corner term carries information a floor does not: a 0.010 px
  floor, more conservative on symmetric conditions (baseline 0.67), leaves
  shading ±15% at 15.
- The fit has its own model error (~0.013 px noise-free on symmetric
  corners, from bilinear resampling of pixel-integrated edges), so it pushes
  symmetric σ̂₀² to 0.72–0.78 (gamma 0.63, already 0.80 in 1.5.0). It does
  not describe PSF asymmetry: decentred defocus and coma are misattributed
  (correction makes them worse), though the term still covers decentred
  defocus (1.04) and cuts coma's σ̂₀² by 40–65% (most of coma's error is a
  per-image shift that a calibration absorbs).
- (c) cannot see shading: the three radii share most of the bias.

## Real data

`gate_real.py`: SaddlePoint matched to board-ID'd corners, noise σ from
`estimate_noise_sigma` (0.24 stereo → 1/12 gray-level² floor; 1.4 table).
Weighted bundle adjustment (whitened by \(\Sigma^{-1/2}\)), 2-fold CV by view
parity, held-out views' poses by uniform PnP. Held-out residual RMS per axis
(px) with the 95% view-bootstrap CI of the difference to uniform.
"+m" adds \(s^2 I\), \(s\) = per-axis RMS residual of a uniform calibration of
the training fold (the board/model error a consumer should add).

| case | stereo L (1529) | stereo R (1522) | table cam1 (6907) | table cam2 (6652) |
|---|---|---|---|---|
| uniform | 0.1164 | 0.1198 | 0.3131 | 0.2949 |
| w: 1.5.0 Σ | 0.1165 [−.0000, +.0001] | 0.1197 [−.0001, +.0000] | 0.3133 [−.0001, +.0005] | 0.2959 [+.0003, +.0017] |
| w: (a) mul 9×9 | 0.1154 [−.0039, +.0016] | 0.1195 [−.0022, +.0016] | 0.3139 [+.0002, +.0014] | 0.2963 [+.0006, +.0021] |
| w: (c) radii | 0.1168 [−.0016, +.0022] | 0.1189 [−.0024, +.0004] | 0.3140 [+.0002, +.0016] | 0.2958 [+.0004, +.0014] |
| w: 1.5.0 Σ +m | 0.1164 [−.0000, +.0000] | 0.1198 [−.0000, +.0000] | 0.3131 [−.0000, +.0000] | 0.2949 [−.0000, +.0000] |
| w: (a) +m | 0.1161 [−.0006, −.0000] | 0.1197 [−.0005, +.0003] | 0.3131 [−.0001, +.0001] | 0.2950 [−.0001, +.0002] |
| w: (c) +m | 0.1164 [−.0002, +.0000] | 0.1198 [−.0001, +.0001] | 0.3131 [−.0001, +.0000] | 0.2949 [−.0001, +.0000] |
| (b) corrected | 0.1152 [−.0029, +.0004] | 0.1228 [+.0008, +.0054] | 0.3158 [+.0012, +.0041] | 0.3022 [+.0043, +.0105] |
| (b) gated | 0.1152 [−.0029, +.0004] | 0.1228 [+.0007, +.0054] | 0.3166 [+.0022, +.0049] | 0.3022 [+.0044, +.0104] |

Spearman(predicted σ, held-out |residual|): 1.5.0 −0.00 / +0.11 / +0.13 /
+0.05; (a) +0.00 / +0.07 / +0.23 / +0.12; (c) −0.02 / +0.04 / +0.25 / +0.15.
Median predicted σ: 1.5.0 0.006–0.010 px, (a) 0.019–0.034 px.

- Board and model error (0.12–0.31 px per axis) dwarf any localisation
  term. Weighting by Σ alone is worse on the table sets (+0.3–0.5%,
  significant) — for the 1.5.0 Σ as well (cam2). With the model term added,
  every Σ is neutral to within 0.0003 px; (a) is marginally better on
  stereo L.
- The correction moves corners by 0.055 (stereo) and 0.11 px RMS (table)
  and worsens three of four sets: on real corners \(\beta\) is dominated
  by misattributed asymmetry (lens PSF, demosaicing), not by illumination.

## Verdict

| option | synthetic | real | gate |
|---|---|---|---|
| (a) design form (5×5, \(S - m\)) | shading ±5/±15%: 2.06 / 5.84 | — | **fail** |
| (a) `mul` 9×9 | asymmetric 0.94–1.43 at cell 10 / noise 2; symmetric 0.72–0.88 except gamma 0.63; misses at shading ±15% with cell 8 (1.54) or noise 1 (1.99), blur 2 at noise 1 (0.67) | neutral with +m; +0.3–0.5% alone on table (as 1.5.0) | **pass with exceptions** |
| (b) correction | removes shading bias; worse on blur 2, decentred, coma, perspective, baseline | worse on 3 of 4 sets | **fail** |
| (c) multi-radius | shading ±5/±15%: 2.00 / 3.22 | neutral with +m | **fail** |

The real-data clause ("not worse than uniform") fails for the 1.5.0 Σ too;
it should be read with the board/model term added, under which (a) passes.

## Design for UQ-27 (not shipped: real-data gate failed)

At the converged SaddlePoint estimate, per corner:

1. Bilinear samples \(P(d)\), \(d \in [-5, 5]^2\) (the blend of four
   integer-shifted 11×11 patches; 12×12 pixel reads).
2. On \(d \in [-4, 4]^2\), half plane (40 pairs): \(A\), \(S\),
   \(\nabla S\) by central differences; \(X = [\partial_x S, \partial_y S, d_x S, d_y S]\)
   (f64, or scale the ramp columns: cond ~2e3).
3. \(X^+ = (X^\top X)^{-1}X^\top\); \(b = X^+_{[b]}A\);
   \(G = X^+_{[b]}\,\partial A/\partial c\) from the bilinear derivative taps;
   \(\beta = G^{-1}b\).
4. \(L_\beta = L_c + G^{-1}X^+_{[b]}QW\) over the 12×12 block
   (\(L_c\) is the SaddlePoint map, needed explicitly);
   \(\Sigma = \Sigma_{1.5} + (\beta\beta^\top - \sigma_n^2 L_\beta L_\beta^\top)_+\).
5. No correction. Positions are unchanged.

Cost: ~4–5k multiply-adds per corner (sampling ~0.5k, LS ~1k, gain ~0.8k,
noise map and its covariance ~2k): about 3× the SaddlePoint refinement;
numpy timing puts one asymmetry fit plus gain at 0.9× a SaddlePoint
fixed-point pass. Needs a 9×9 window clear of neighbouring corners
(validated at cells ≥ 8 px).

Caveats: covers illumination gradients only; PSF asymmetry is covered loosely
(decentred defocus) or partly (coma); symmetric conditions become ~20% more
conservative; the absolute scale inherits the noise estimate (Bayer 1.64
with σ supplied). No measurable benefit on the real sets, where board and
model error dominate.

## Real-data gate (UQ-27)

Gate of [design §8](../design/uncertainty-v2.md): does the asymmetry term
track the per-corner remainder of the [`UQ-30`](real-data-validation.md)
decomposition (lens field, per-corner target term and, for the local
remainder, the per-view smooth pattern removed) better than the shipped
`Propagated` covariance ([uncertainty-model](uncertainty-model.md))?

### Pre-registered criteria

Fixed before any number below was computed. SaddlePoint (ChESS seeds) on the
stereo ChArUco set, both corner sets (**all**, and **chess** with the
blurred corners). \(\Sigma_P\): the bindings' `position_covariance` under
`Propagated`. Asymmetry term \(T = (\beta\beta^\top - \sigma_n^2 L_\beta L_\beta^\top)_+\)
(`mul` 9×9, at the same σₙ as \(\Sigma_P\)).

- **Form.** `Propagated`'s real-data floor (0.053 px) was fitted to absorb
  the real-image error beyond the synthetic model — the error \(T\) is meant
  to measure. Adding \(T\) on top counts it twice, so the gated form
  replaces the floor: \(\Sigma_A = \Sigma_P - f_P^2 I + T + f_A^2 I\), with
  \(f_A\) refitted by the shipped rule (the floor making the local-remainder
  σ̂₀² 1, max over the two sets). For a like-for-like comparison
  `Propagated` is also scored with its floor refitted by the same rule on
  the same positions. \(\Sigma_P + T\) is reported, not gated.
  Spearman of \(\sqrt{\operatorname{tr}\Sigma/2}\) does not depend on the
  floor.
- **Ranking (both required).** ΔSpearman = Spearman(\(\Sigma_A\)) −
  Spearman(\(\Sigma_P\)) between predicted σ and \(|e|\) is **≥ +0.10 on the
  local remainder** on both sets, and **> 0 on the cross-fold remainder** on
  both sets.
- **Weighting.** Σ-whitened rig calibration with \(\Sigma_A\) is not worse
  than with \(\Sigma_P\): held-out (2-fold CV) residual RMS and cross-fold
  remainder RMS each at most +0.5% over the \(\Sigma_P\)-weighted value, on
  both sets.
- **Verdict.** SHIP (opt-in tier, SaddlePoint only) iff both hold;
  otherwise NO-GO.

### Results

Scripts: `asymmetry/export_uq27.py` (term per observation; numpy
SaddlePoint matches the bindings to 6.5e-5 px median, 4e-4 px p99; 99.7–100%
valid fits), `asymmetry/gate_uq27.py` (decomposition, floors, weighted
calibrations, view-bootstrap CIs), `asymmetry/gate_synth_uq27.py`. Python
bindings at `UQ-24` (`Propagated` default). The uniform rig calibration
reproduces `real-data-validation.md` (.3536 / .3542 **all**, .4083 / .4088
**chess**).

Floors (local-remainder σ̂₀² = 1, **all** / **chess**, max): `Propagated`
refitted 0.038 / 0.0525 → **0.0525** (the shipped 0.053 reproduced);
\(\Sigma_A\) 0.034 / 0.044 → **0.044**. Median \(\sqrt{\operatorname{tr}T/2}\)
0.014 px (**all**), 0.018 px (**chess**); \(T\) is non-zero on 94% of the
corners.

| set | Σ | median σ px | σ̂₀² rem / x-fold / local | Spearman rem / x-fold / local |
|---|---|---|---|---|
| all | \(\Sigma_P\) (shipped) | .0537 | 1.63 / 1.87 / 0.53 | **.282 / .278 / .187** |
| all | `Propagated`, floor refitted | .0531 | 1.66 / 1.91 / 0.54 | .282 / .278 / .187 |
| all | \(\Sigma_A\) | .0470 | 2.05 / 2.36 / 0.66 | .118 / .125 / .127 |
| all | \(\Sigma_P + T\) | .0555 | 1.47 / 1.70 / 0.47 | .118 / .125 / .127 |
| chess | \(\Sigma_P\) (shipped) | .0539 | 2.17 / 2.52 / 0.98 | **.348 / .349 / .359** |
| chess | `Propagated`, floor refitted | .0534 | 2.21 / 2.56 / 1.00 | .348 / .349 / .359 |
| chess | \(\Sigma_A\) | .0485 | 2.30 / 2.64 / 1.00 | .259 / .271 / .307 |
| chess | \(\Sigma_P + T\) | .0568 | 1.71 / 1.97 / 0.75 | .259 / .271 / .307 |

Radon + SaddlePoint (**all**) gives the same numbers to ±0.005.
\(\sqrt{\operatorname{tr}T/2}\) alone ranks the local remainder at 0.085
(**all**) and 0.256 (**chess**), against 0.187 / 0.355 for \(\Sigma_P\); on
**all** it is uncorrelated with \(\Sigma_P\)'s σ (Spearman 0.000), on
**chess** it partly duplicates it (0.44, both grow with blur).

Σ-whitened rig calibration, RMS px; difference to \(\Sigma_P\)-weighted with
the 95% view-bootstrap CI:

| set | weighting | held-out CV | x-fold remainder | CV − P | x-fold − P |
|---|---|---|---|---|---|
| all | uniform | .3542 | .1061 | [−.0000, +.0000] | [−.0001, −.0000] |
| all | \(\Sigma_P\) | .3542 | .1062 | — | — |
| all | \(\Sigma_A\) | .3543 | .1069 (+0.7%) | [−.0000, +.0002] | [+.0002, +.0011] |
| all | \(\Sigma_P + T\) | .3543 | .1067 | [−.0000, +.0001] | [+.0001, +.0008] |
| chess | uniform | .4088 | .1348 | [−.0013, +.0000] | [−.0047, −.0006] |
| chess | \(\Sigma_P\) | .4093 | .1371 | — | — |
| chess | \(\Sigma_A\) | .4119 (+0.6%) | .1399 (+2.0%) | [+.0013, +.0040] | [+.0018, +.0038] |
| chess | \(\Sigma_P + T\) | .4114 | .1391 | [+.0010, +.0031] | [+.0012, +.0028] |

What \(\beta\) is on these corners (SaddlePoint; **all** / **chess**):

- It does not see the target error: \(\beta\) explains 0.2% / 0.1% of the
  board term's variance (and the board term 0.1% of \(\beta\)'s). Nor the
  smooth field (0.2%) or the per-view smooth pattern (0.0% / 1.7%).
- Its own structure: a per-view cubic takes 24% / 32% of its variance
  (6% / 4% for white noise), the per-id mean 13% / 31% (3% white). It has
  view- and id-level structure of its own that the decomposition's
  components do not share.
- It carries a little signed information about the remainder: slope of the
  remainder on \(\beta\) +0.34 / +0.12 (1 if \(\beta\) were the error),
  explaining 0.9–2.8% of its variance. Most of \(\beta\) is fit model error
  and misattributed asymmetry, consistent with the correction's failure
  above.

Synthetic, same baseline (σ supplied, shipped `Propagated` from the
bindings, \(\Sigma_A\) with the floors above; cell 10, noise 2 unless
stated; 3 boards × 8 draws):

| condition | RMS px/axis | \(\Sigma_P\) σ̂₀² | \(\Sigma_A\) σ̂₀² | Spearman P / A |
|---|---|---|---|---|
| baseline / gauss 2.0 / gamma / Bayer | .015 / .028 / .016 / .019 | 0.07 / 0.22 / 0.08 / 0.12 | 0.10 / 0.25 / 0.10 / 0.15 | ≤ .11 / ≤ .19 |
| decentred defocus | .016 | 0.09 | 0.11 | −.14 / −.09 |
| vignetting 40% / ramp ±20% | .024 / .019 | 0.18 / 0.12 | 0.19 / 0.14 | .28 / .49, .16 / .35 |
| shading ±5% | .028 | 0.26 | 0.23 | .12 / .65 |
| shading ±15% (cell 10 / cell 8 / noise 1) | .073 / .089 / .072 | 1.71 / 2.56 / 1.77 | 0.46 / 0.52 / 0.43 | .13 / .93 (all three ≥ .93) |
| coma / combined A / combined B | .056 / .078 / .029 | 1.05 / 1.74 / 0.25 | 0.92 / 1.78 / 0.27 | −.57 / −.43, .18 / −.04, .10 / .22 |

With the real-data floor `Propagated` already covers every synthetic
condition except strong shading (±15%: 1.7–2.6) and combined A (1.74, coma
plus demosaic); the term fixes shading only, where it ranks the error almost
perfectly (Spearman 0.93–0.96).

### Verdict: NO-GO

| criterion | all | chess | pass |
|---|---|---|---|
| ΔSpearman, local remainder ≥ +0.10 | −0.060 | −0.052 | **no** |
| ΔSpearman, x-fold remainder > 0 | −0.153 | −0.078 | **no** |
| weighting: CV ≤ +0.5% over \(\Sigma_P\) | +0.03% | +0.64% | **no** (chess) |
| weighting: x-fold remainder ≤ +0.5% | +0.66% | +2.0% | **no** |

On real corners the asymmetry term lowers the ranking of the remainder
instead of raising it, and weighting by it is significantly worse than by
`Propagated` (CIs exclude 0 on **chess**). The term is built to measure
illumination shading; on these corners its value is mostly the fit's own
model error and misattributed lens asymmetry (slope 0.12–0.34, ≤ 3% of the
remainder's variance), per-corner noise that dilutes `Propagated`'s blur
signal. The tier does not ship (`UQ-27` wontfix). The synthetic shading
gain is real; a tier for strongly shaded scenes would need real shaded data
to pass this gate. The design section above is kept as the record of what
would have shipped.
