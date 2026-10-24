# Uncertainty v2 — a position covariance for every localizer

Status: accepted · Workstream: UQ · Target: 1.6.0 (additive) · Evidence:
[`reference/refiner-robustness.md`](../reference/refiner-robustness.md),
[`reference/refiner-accuracy.md`](../reference/refiner-accuracy.md).

## 1. Problem

1.5.0 reports a position covariance only for ChESS + `SaddlePoint`: noise
propagated through the re-centred saddle fit plus a 0.006 px floor. The
robustness audit (`UQ-23`) shows:

- SaddlePoint is the most accurate localizer on an independent synthetic
  generator (18 of 19 conditions) and on four real calibration sets, but
  its "near-zero bias" holds only for point-symmetric corners.
  Illumination gradients and asymmetric PSFs bias it by 0.02–0.10 px, and
  its covariance does not cover that (σ̂₀² 1.5–22).
- On real, well-exposed calibration images the predicted σ sits at the
  floor and does not track the residual: the error there is systematic.
- The noise estimator misreads demosaiced images and declines on JPEG.
- Re-centring Förstner is not a fix: it stays dominated by SaddlePoint and
  is ill-conditioned under blur.

So a covariance that is useful on real images has to model the systematic
error per corner, for every localizer, and the noise estimate has to work on
the images people actually have.

## 2. Goals and non-goals

Goals:

1. A position covariance for every localizer: CenterOfMass, Förstner,
   SaddlePoint, the Radon peak fit, and SaddlePoint on Radon seeds.
2. Uncertainty is optional and tiered by cost.
3. The systematic part is modelled per corner from observables of the
   patch, calibrated on both synthetic generators, and validated on real
   data by whether weighting by it improves a held-out calibration.
4. A public per-image noise estimate (shipped in `UQ-18`) that works on
   demosaiced and JPEG images.
5. One derivation principle, finite-difference and Monte Carlo tests per
   method, σ̂₀² regression bands, perf budgets, book, bindings and demo
   updated together.

Non-goals: a covariance for the ML refiner; re-centring Förstner (`UQ-23`
outcome); correlated-noise propagation (handled by the noise estimate's
definition, §5); orientation σ (`UQ-06`); one refiner stage on
`DetectorConfig` (2.0, `UQ-19`).

## 3. Error model

\[ \hat x - x^* = J n + b, \qquad
   \Sigma = \max(\sigma_n^2, \tfrac1{12})\, J J^\top + \Sigma_{\text{sys}} . \]

- \(J\) is the exact Jacobian of the localizer with respect to the pixels of
  its support, including any resampling (bilinear re-centring, Radon's 2×
  upsample), at a fixed point via the implicit-function theorem. One
  principle for every localizer.
- \(\Sigma_{\text{sys}}\) models the systematic error \(b\) per corner (§4).
  No single image reveals a bias every localizer shares (a uniform coma
  shifts them all), so the downstream σ̂₀² stays the final check, and the
  docs say so.

## 4. Systematic term

Two sources, each with a per-corner observable:

1. **Window-offset bias of single-shot localizers** (CenterOfMass,
   Förstner, Radon peak fit). Bias is close to linear in the corner's
   offset from the window centre, \(b \approx (G - I)u\), with \(G\) set by
   the estimator and the edge profile (`refiner-accuracy.md` §Bias
   structure). Observables: \(u\) and an edge-sharpness measure of the patch.
   \(G(\text{sharpness})\) is calibrated per localizer on both generators.
   Where \(G\) is far from 0 the bias can be removed rather than reported;
   where it approaches 0 (Förstner at blur ≥ 1.5) the localizer carries no
   information about the offset and the term says so.
2. **Asymmetry of the corner** (illumination gradient, decentred PSF), at
   the converged SaddlePoint estimate. On a 9×9 window, the patch's
   antisymmetric part is fitted with a mislocation \(\delta\cdot\nabla S\)
   and a multiplicative illumination ramp \((g\cdot d)\,S\), where \(S\)
   is the symmetric part; \(\beta = G^{-1}\delta\) with the fit's gain
   \(G\) is a per-corner bias estimate with a known noise map, and
   \(\Sigma \mathrel{+}= (\beta\beta^\top - \sigma_n^2 L_\beta L_\beta^\top)_+\).
   Prototype and evidence: [`reference/asymmetry-term.md`](../reference/asymmetry-term.md)
   (`UQ-26`). It calibrates shading, ramps and vignetting on synthetic
   boards (σ̂₀² 0.94–1.43 from 1.48–21), leaves symmetric conditions about
   20% conservative, covers asymmetric PSFs only partly, and costs about
   3× the SaddlePoint refinement. Using β as a correction was rejected: it
   raises error on blur, coma and perspective, and on three of four real
   sets. A multi-radius spread was rejected: the radii share the bias.

Which of these ship as correction versus as covariance, and in which tier,
is decided by the `UQ-26` gate (§8).

## 5. Noise estimate

`estimate_noise_sigma_u8` (`UQ-18`) returns the σ that makes the white-noise
propagation correct: on correlated noise (demosaicing, JPEG) that is an
effective σ, not the per-pixel σ. `UQ-25` makes it read correctly on Bayer-
demosaiced images and stop declining on JPEG, validated against σ̂₀² under
those formations, not against the per-pixel σ.

## 6. Tiers

A detector-independent stage on `DetectorConfig`, like `orientation_method`:

```rust
#[non_exhaustive]
pub enum Uncertainty {
    /// No covariance; no noise estimate.
    Off,
    /// Noise propagation plus the per-localizer systematic model (§4.1).
    Propagated,
    /// Propagated, plus the per-corner asymmetry term (§4.2).
    // Variant added by UQ-27 if the UQ-26 gate passes; named for what it
    // measures.
}
```

- `Off` — `position_covariance` is `None`; no Jacobians, no noise estimate.
  Must stay within 2% p95 of 1.5.0 frame time.
- `Propagated` — the default. Every localizer reports a covariance.
  Its cost per localizer and per frame is measured and published.
- The asymmetry tier costs one extra pass over each patch plus a small
  least-squares solve.

`DetectorConfig::noise_sigma` keeps its meaning and is ignored under `Off`.
The per-corner output stays one total `position_covariance`.

## 7. Validation

1. Both synthetic generators (`optical_chessboard`, and the audit's
   supersampled generator with PSF, illumination and sensor effects).
2. Real data: OpenCV calibration on the real sets, 2-fold cross-validated.
   The usefulness metric is the **held-out residual and parameter stability
   when the calibration weights corners by Σ versus uniformly**, plus the
   rank correlation between predicted σ and per-corner held-out residual.
   σ̂₀² on real data includes board and model error and is reported, not
   gated.
3. Gates per localizer and tier: Jacobian vs central finite differences;
   Monte Carlo scatter vs \(JJ^\top\) (z-score std 0.9–1.1); σ̂₀² bands on a
   fixed grid of both generators (fast subset in `cargo test`, full grid
   `--ignored`); `Off` and `Propagated` do not move positions (snapshots).
4. Bindings parity for covariance arrays and the noise estimate.
5. Criterion: ns/corner per localizer and tier; frame time per tier.

## 8. Decision gates

- **UQ-26 (asymmetry):** synthetic part passed (§4.2); correction and
  multi-radius spread rejected. On the real sets the board and model error
  (0.12–0.31 px per axis) hides any localization term: weighting by Σ with
  that error added as \(s^2 I\) is neutral within 0.0003 px for every Σ.
  The tier ships (`UQ-27`) only if the residual decomposition of `UQ-30`,
  which removes the field and board components, shows the asymmetry term
  tracking the per-corner remainder better than `Propagated`.
- **Radon default localizer (`UQ-22`):** decided on the real sets with
  Radon + SaddlePoint included.

## 9. Surface and semver

Additive in 1.6.0: `Uncertainty`, `DetectorConfig::uncertainty`
(`DetectorConfig` is `#[non_exhaustive]`). Behaviour change in the
CHANGELOG: every localizer reports a covariance by default; SaddlePoint's
covariance gains the systematic model. C ABI: new config field →
`cc_abi_version` bump, header regenerated.

## 10. Docs and demo

Book Part VIII rewritten around §3–§7 with the real-data evidence, and
the overstated 1.5.0 statements listed in `refiner-robustness.md`
corrected; Part V and Part IX gain the real-data comparison; Part IV
§4.5 states Radon's shading robustness. The demo shows every localizer's
covariance, the tier, and the noise estimate, on lab-ui (`EllipseSet` item
colours, non-pickable `PointSet`).

## 11. Phases

| ID | Phase | Deps |
|---|---|---|
| UQ-23 | Robustness audit: independent generator + real data | done |
| UQ-24 | `Uncertainty` stage; Jacobian noise term for CenterOfMass, Förstner, Radon peak fit; systematic model §4.1 | UQ-18 |
| UQ-25 | Noise estimate on demosaiced and JPEG images | UQ-18 |
| UQ-26 | Asymmetry term §4.2: prototype and gate | UQ-23 |
| UQ-27 | Ship the asymmetry tier (and correction if gated in) | UQ-24, UQ-26 |
| UQ-28 | Book, README, CHANGELOG, reference docs | UQ-24–27 |
| UQ-29 | Demo: per-localizer covariance, tiers, noise readout | UQ-24 |
| UQ-30 | Absolute validation data with detector-separable truth (absorbs UQ-08) | — |
