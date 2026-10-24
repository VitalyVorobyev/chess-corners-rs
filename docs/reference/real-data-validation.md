# Real-data validation of corner localization and covariance

Internal evidence for `UQ-30`. On real calibration images there is no
ground truth, and the calibration residual mixes the localizer's error with
target error and lens-model error. This study removes the components that are
not the detector's, using the structure of one dataset, and measures what is
left per localizer.

Scripts: `tools/refiner_robustness/real_validation/` (`detect.py`,
`calibrate.py`, `analyze.py`; the dataset path is an argument, outputs go to
the gitignored `out/`). Aggregate numbers only below.

## Data

A private stereo ChArUco dataset on an accurate 22×22 target (1.35 mm
squares, ~90 px per square), two 2048×1536 monochrome cameras on a fixed rig
(~24° convergence, focal length ~21 500 px, narrow field), white squares
saturated. 27 board poses per camera from a two-axis rotation stage (each
axis at 0, ±9°, ±14°) plus two ±5 mm Z steps; the stage pose is known per
view. Corner identity from OpenCV's ChArUco detector; a localizer's detection
is assigned to the corner within 2 px.

Localizers: ChESS + CenterOfMass / Förstner / SaddlePoint, the Radon peak
fit, Radon + SaddlePoint (`RadonRefiner::SaddlePoint`), `cv2.cornerSubPix`
11×11 (cSP-11) seeded from the rounded reference corners. Python bindings
1.5.0 (`feat/uncertainty-v2`), default configs, orientation off.

Two corner sets: **all** — corners found by every localizer (9 478
observations; Radon finds 50–100% of the corners per view, ~70% overall);
**chess** — corners found by the four non-Radon localizers (13 433). Tables
are on **all** unless stated.

## 1. Pose models

| model | pose parameters | what is shared |
|---|---|---|
| free | 6 per view and camera | nothing (OpenCV `calibrateCamera` per camera, as in `refiner-robustness.md`) |
| rig | 6 per view | one cam1→cam2 transform |
| stage | 17 in total | rig, plus the board pose from the stage reading: a fixed base pose, two rotation axes (unknown direction and position) and the Z direction |

Camera model: OpenCV k1 k2 p1 p2 k3 per camera. 2-fold cross-validation
splits the views as a checkerboard on the (Rx, Ry) grid.

### The stage-constrained model does not fit

The stage poses are not reproducible at the detector's scale. Fitting stage
models to the rig poses (pixel error of 25 board points through cam1):

| stage model | pose parameters | RMS px |
|---|---|---|
| rigid two-axis | 17 | 5.04 |
| + angle scale per axis | 19 | 5.03 |
| + free 6-dof offset per Rx level | 41 | 2.68 |
| + free 6-dof offset per Ry level | 41 | 4.94 |

The rigid model's pose error clusters by Rx level: offsets of 5–130 µm and
0.3–1.8 mrad between levels, 7–10 µm scatter within a level (the views that
share an Rx setting), and the zero pose and Z-step views disagree with the
other zero-Rx views by ~100 µm. On this target 1 px ≈ 15 µm, so the stage
repeats to ~0.1 mm (~7 px), twenty times the calibration residual. Joint
pixel-space calibration with the rigid stage gives 4.51 px fitted and 5.00 px
predicted on held-out views for every localizer. A stage would need sub-µm
repeatability to constrain poses at the level that separates localizers
(0.005–0.03 px ≈ 0.1–0.5 µm).

The rig constraint does hold: one shared extrinsic costs 0.002 px over free
per-camera poses, and its held-out residual (one pose per held-out view for
both cameras) is within 0.001 px of the fitted one. The decomposition below
uses the rig model.

## 2. Calibration residual

RMS px per point, fitted / 2-fold cross-validated.

| localizer | free | rig | stage |
|---|---|---|---|
| CenterOfMass | .3756 / .3763 | .3780 / .3788 | 4.514 / 5.001 |
| Förstner | .4215 / .4222 | .4242 / .4252 | 4.519 / 5.004 |
| SaddlePoint | .3513 / .3519 | .3536 / .3542 | 4.513 / 4.999 |
| Radon | .3564 / .3569 | .3589 / .3595 | 4.515 / 5.001 |
| Radon + SaddlePoint | **.3508 / .3513** | **.3531 / .3537** | 4.513 / 4.999 |
| cSP-11 | .3525 / .3528 | .3547 / .3553 | 4.514 / 5.000 |

The audit's numbers (`refiner-robustness.md`, free model) are reproduced
within 0.004 px. On the **chess** set (adds the ~30% of corners Radon
misses):

| localizer | free | rig |
|---|---|---|
| CenterOfMass | .4205 / .4209 | .4207 / .4213 |
| Förstner | .5741 / .5748 | .5768 / .5778 |
| SaddlePoint | **.4081 / .4085** | **.4083 / .4088** |
| cSP-11 | .4678 / .4680 | .4703 / .4709 |

The corners Radon misses are the blurred ones: they sit farther from the
image centre (median 674 vs 557 px), the miss is mostly per corner id across
views (the far side of the tilted board), and SaddlePoint's predicted σ there
is twice as large (median 0.0164 vs 0.0087 px). There cSP-11 and Förstner
degrade far more than SaddlePoint and CenterOfMass (rig residual on missed /
found corners: cSP-11 .64 / .38, Förstner .82 / .44, SaddlePoint .49 / .37,
CoM .48 / .39).

## 3. Residual decomposition

\[ r = f_{\text{cam}}(u) + J_b\,\beta_{\text{id}} + e \]

- **smooth** \(f_{\text{cam}}\): per camera, a cubic tensor B-spline over the
  image (5×4 basis functions per component, chosen by the lowest cross-fold
  remainder among 5×4 … 16×12). Lens-model error plus any smooth target
  deformation: a corner id falls within ~60 px of the same image position in
  every view, so the two are not separable here.
- **board** \(\beta_{\text{id}}\): per ChArUco corner id, a 3-D displacement of
  the corner on the target (in-plane and out-of-plane, mm), pooled over views
  and cameras through the per-observation Jacobian \(J_b\). Out-of-plane
  displacement is observable through the ±14° tilts and the 24° rig
  convergence.
- **remainder** \(e\): what is left.

Fitted alternately, smooth first. *x-fold*: smooth and board terms estimated
on one fold of views and removed from the other.

| localizer | total | smooth | board | remainder | remainder x-fold |
|---|---|---|---|---|---|
| CenterOfMass | .3780 | .190 | .265 | .1680 | .1787 |
| Förstner | .4242 | .188 | .270 | .2522 | .2701 |
| SaddlePoint | .3536 | .190 | .265 | .0992 | .1061 |
| Radon | .3589 | .191 | .265 | .1156 | .1239 |
| Radon + SaddlePoint | .3531 | .190 | .265 | **.0974** | **.1043** |
| cSP-11 | .3547 | .190 | .265 | .1036 | .1110 |

- The smooth and board terms are the same for every localizer (to 0.005 px):
  they are properties of the lens and the target. Together they are ~90%
  of the residual variance of the better localizers (80% for CoM).
- Board term: 4.2 µm RMS in-plane and 2.0 µm out-of-plane per corner. It
  repeats across folds (variance explained 0.98 in-plane, r = 0.93
  out-of-plane, SaddlePoint) and across cameras (0.89), which see each corner
  at image positions ~100 px apart, so it is on the target, not in the lens.
  Out-of-plane terms lower the remainder by 0.007 px (in-plane only: .1063 /
  .1123 for SaddlePoint).
- The split between smooth and board depends on the basis (8×6: .207 / .247;
  board first: .340 / .044), the remainder does not (SaddlePoint .0990–.0992
  in-sample).
- On the **chess** set (remainder / x-fold: SaddlePoint .124 / .135, CoM
  .166 / .178, cSP-11 .232 / .251, Förstner .418 / .446) the board terms of
  cSP-11 and Förstner stop matching SaddlePoint's (variance explained 0.47 and
  0.56; 4.4–4.7 µm RMS difference). Blur is repeatable per corner (the same
  region is defocused in views of similar tilt), so a localizer's blur bias
  lands in its board term, and the remainder then understates that
  localizer's error. The raw-residual differences are the safer comparison
  there.

### What the remainder contains

| localizer | remainder | per-view cubic part (white-noise level) | local remainder | Z-step views | tilt 0 | tilt 9° | tilt 14° |
|---|---|---|---|---|---|---|---|
| CenterOfMass | .1680 | .085 (.040) | .1451 | .226 | .158 | .159 | .168 |
| Förstner | .2522 | .099 (.060) | .2321 | .305 | .243 | .248 | .251 |
| SaddlePoint | .0992 | .082 (.024) | .0567 | .190 | .081 | .084 | .098 |
| Radon | .1156 | .084 (.028) | .0798 | .202 | .097 | .104 | .114 |
| Radon + SaddlePoint | .0974 | .081 (.023) | .0535 | .190 | .073 | .084 | .095 |
| cSP-11 | .1036 | .082 (.025) | .0631 | .195 | .081 | .091 | .101 |

- A smooth pattern *per view* (a cubic in image coordinates fitted to each
  view and camera) takes 0.08 px RMS for every localizer, 3.5× what such a
  fit takes from white noise. The Z-step views (±5 mm, defocused) carry
  0.19 px for every localizer, and the error rises with tilt. This part is
  common to all localizers and changes from view to view: defocus- and
  depth-dependent imaging that a fixed camera model does not describe.
- *Local remainder* (remainder minus the per-view cubic): what is left at
  the corner scale.

### Common and localizer-specific error

Pairwise RMS difference of the remainders (px per point). Common error
cancels in the difference, so each entry bounds the specific error of both
localizers from above unless they are anti-correlated:

| | CoM | Förstner | SaddlePoint | Radon | cSP-11 |
|---|---|---|---|---|---|
| CenterOfMass | — | .216 | .142 | .154 | .150 |
| Förstner | .216 | — | .237 | .244 | .246 |
| SaddlePoint | .142 | .237 | — | .069 | **.042** |
| Radon | .154 | .244 | .069 | — | .069 |
| cSP-11 | .150 | .246 | **.042** | .069 | — |

Radon + SaddlePoint differs from ChESS + SaddlePoint by 0.018 px (same
refiner, different seeds).

- SaddlePoint's localizer-specific error is at most 0.042 px per point
  (0.030 px per axis); cSP-11's too. An N-cornered hat over the five
  independent localizers puts it at 0.05 px but is not consistent with the
  pairwise bound (correlated errors between the gradient-based estimators), so
  only the bound is used. On the **chess** set the closest pair is
  SaddlePoint–CoM (.130), a loose bound; the hat gives 0.07 px there.
- Extra error over SaddlePoint, \(\sqrt{\text{rms}^2 - \text{rms}_{SP}^2}\),
  is the same on the raw residual and on the remainder (CoM .134 / .136,
  Förstner .234 / .232, Radon .061 / .059, cSP-11 .028 / .030): the removed
  components are common. What changes is the ratio: remainder 1.0 : 1.04 :
  1.17 : 1.69 : 2.54 (Radon+SP or SP : cSP-11 : Radon : CoM : Förstner)
  versus 1.0 : 1.003 : 1.015 : 1.07 : 1.20 on the raw residual. The ranking
  is the audit's, now resolved in size, not only in sign.

## 4. ChArUco neighbourhood

A ChArUco corner touches two marker squares; marker bits break the point
symmetry of the pattern from ~0.25 square (22 px) outwards. The antisymmetric
first moment of the ideal pattern around each corner (radius 0.5, 0.75, 1
square) explains ≤ 1.1% of the board term's variance for every localizer,
and ≤ 3.5% of the difference between a localizer's board term and
SaddlePoint's. The board terms of all localizers agree (variance of each
explained by SaddlePoint's: 0.91 Förstner, 0.98 CoM, ≥ 0.99 the others;
RMS difference 0.3–0.7 µm, Förstner 1.3 µm). On the **chess** set, where the
board terms of cSP-11 and Förstner diverge from SaddlePoint's (blur, §3), the
neighbourhood moment still explains ≤ 1.7% of either the board term or the
divergence. No ChArUco-specific detector
bias is visible at this scale (~90 px squares; every localizer's support is
inside the 11 px white margin around the corner); the board term is target
error.

## 5. Stereo consistency

cam2←cam1 from independent PnP on each camera per view (intrinsics from the
rig fit), scatter over the 27 views:

| localizer | rotation mrad | translation µm (x / y / z) | after removing smooth + board |
|---|---|---|---|
| CenterOfMass | 0.461 | 128 (70 / 102 / 34) | 0.311 mrad / 84 µm |
| Förstner | 0.484 | 135 (74 / 105 / 40) | 0.310 / 86 |
| SaddlePoint | 0.467 | 129 (64 / 108 / 29) | 0.316 / 86 |
| Radon | 0.472 | 131 (67 / 107 / 35) | 0.307 / 83 |
| Radon + SaddlePoint | 0.467 | 129 (64 / 109 / 29) | 0.315 / 86 |
| cSP-11 | 0.469 | 130 (65 / 108 / 32) | 0.320 / 87 |

With a ~5° field of view each PnP's tilt and position are weakly determined,
and on the **all** set the scatter is set by the per-view common error, not by
the localizer (Förstner's 2.5× larger remainder moves it by 4%).

On the **chess** set (more corners per view, including the blurred ones) it
does separate them: rotation / translation scatter SaddlePoint 0.263 mrad /
75 µm, CoM 0.252 / 71, cSP-11 0.309 / 87, Förstner 0.410 / 112; after
removing smooth + board 0.179 / 54, 0.148 / 45, 0.260 / 76, 0.296 / 84. With
27 views the scatter's own sampling error is ~14%, so SaddlePoint and CoM tie;
cSP-11 and Förstner are worse, in line with their blur sensitivity.

## 6. Covariance validation (SaddlePoint, 1.5.0 model)

Predicted σ (median 0.0087 px per axis) against the error at each stage of
the decomposition:

| error used | σ̂₀² all / chess | Spearman(predicted σ, \|e\|) all / chess |
|---|---|---|
| raw rig residual | 910 / 843 | −0.09 / 0.12 |
| remainder | 55 / 44 | 0.24 / 0.34 |
| remainder, x-fold | 64 / 50 | 0.22 / 0.35 |
| local remainder (dof-corrected) | 20 / 19 | 0.13 / 0.36 |
| specific error, aggregate | ≤ 10 (pairwise bound) / ~10 (hat), ≤ 37 (bound) | — |

Median predicted σ 0.0087 px (all), 0.0100 (chess). Radon + SaddlePoint
gives the same numbers as ChESS + SaddlePoint.

- The 1.5.0 covariance is overconfident by about 10× in variance (~3× in σ)
  against the localizer's own error, not by the ~900× the raw residual
  suggests. The rest is target and imaging error that no per-corner
  localizer covariance should carry.
- Predicted σ ranks the error once the common terms are removed, and more so
  when blurred corners are present (0.35 on **chess**): its blur and contrast
  dependence is real signal. Part of the 0.24 on **all** is view-level
  imaging (blurrier views have larger σ and larger common error); at the
  corner scale it is 0.13 there.
- Weighting the rig calibration by Σ does not help, and hurts when σ varies:
  held-out residual .3542 uniform vs .3544 weighted (**all**), .4088 vs .4186
  (**chess**); x-fold remainder .1061 vs .1076 and .1348 vs .1485. The
  weights follow σ over a 2× range while the error each corner carries is
  mostly the common terms, so down-weighting the blurred corners throws away
  geometry without removing error.

## 7. Consequences

1. On sharp, well-exposed images of an accurate target the calibration
   residual is ~90% target and lens error; the localizer is visible only
   after removing them. A per-corner board term (3-D) and a smooth field term
   are both needed and both repeat across folds and cameras.
2. Real-data localizer ranking, extra error over SaddlePoint on the corners
   every localizer finds: Radon + SaddlePoint ≈ SaddlePoint < cSP-11
   (+0.03 px) < Radon peak fit (+0.06) < CenterOfMass (+0.14) < Förstner
   (+0.23). With blurred corners included: SaddlePoint < CoM (+0.10) <
   cSP-11 (+0.23) < Förstner (+0.41). This supports SaddlePoint as Radon's
   refiner (`UQ-22`); Radon itself misses the blurred ~30% of the corners
   here.
3. For `UQ-24`–`UQ-27`, the real-data target for a localizer covariance is
   the specific error: ≤ 0.03 px per axis for SaddlePoint on sharp corners
   (σ̂₀² ≈ 10 with 1.5.0). A per-view common error of ~0.06 px per axis
   (defocus, depth) remains that no per-corner term on a single image can
   see; it belongs to the downstream model. Covariance weighting should be
   judged against the remainder, not the raw residual, and on sets with a
   real range of blur.
4. A stage-constrained pose model needs a stage with sub-µm repeatability;
   this one repeats to ~0.1 mm.

## Rerunning with new covariances

`detect.py` stores every localizer's `position_covariance` from the
installed bindings (NaN rows become "no covariance"), so after a binding that
reports covariances for more localizers is installed in the venv, rerun the
three scripts and the covariance table covers each localizer that has one.
A covariance computed elsewhere goes in with
`calibrate.py --cov LOC=file.npz` (`cam` 'cam1'/'cam2', `token` the
view token, `id` the ChArUco corner id, `cov` (N, 2, 2) px²), which replaces
that localizer's covariance before the Σ-weighted fit and the analysis.

## Limits

One dataset, one camera type, one target. The stage could not supply poses.
The smooth/board split is basis-dependent (the remainder is not). The
pairwise bound assumes the specific errors of SaddlePoint and cSP-11 are not
anti-correlated. Radon recall limits the **all** set to ~70% of the corners.
