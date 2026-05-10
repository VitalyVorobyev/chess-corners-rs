# Full-Disk Sector Orientation

The old orientation fit uses 16 samples on a ring. That is fast, but it is a weak signal for projective chess corners: two visibly different crossings can produce similar ring samples, and the fit has a strong tendency to drift toward orthogonal axes. `AdaptiveBeta` tried to fix one part of this by fitting edge sharpness, but it added an unstable degree of freedom without adding more image evidence.

The full-disk sector estimator uses the same detected corner center, then recomputes the two local line directions from all pixels in a disk around that center. It started as a benchmark-only Python method and now also exists as the opt-in Rust method `OrientationMethod::FullDiskSector`. The default detector behavior is still unchanged.

## Core Idea

A chess corner is modeled as two crossing transition lines through the detected center. For a pixel `p`, the signed distances to those lines are `d0(p)` and `d1(p)`. The local intensity model is:

```text
I(p) = mu + A * tanh(d0(p) / w) * tanh(d1(p) / w)
```

`mu` is the local mean level, `A` is the sector contrast, and `w` is a discrete edge width. The estimator does not fit a continuous blur or beta parameter. It tries a small fixed set of widths: `0.35`, `0.70`, `1.40`, and `2.80` pixels.

This matters because the hard part is not estimating edge sharpness. The hard part is choosing the two correct non-orthogonal lines from image evidence. The full disk gives the estimator many more pixels than the ring and makes acute or obtuse projective crossings observable.

## Candidate Search

The method uses three sources of candidate line directions:

- peaks from a Sobel edge-direction histogram inside the support disk;
- the baseline axes with small angular offsets;
- a sparse global angular grid as a deterministic safety net.

Candidate pairs are allowed when their line separation is between `12°` and `168°` in directed-axis terms, equivalently at least `12°` away from degenerate parallel lines in the undirected line representation.

Each pair is scored over the full disk. The score combines normalized model residual with a gradient-support term, so a pair must both explain the intensities and align with visible edge evidence. The best few pairs are refined by deterministic local angular search down to `0.25°`.

The measured implementation uses an `8 px` local support cap. This was important in the Rust port: blindly expanding the support to `16 px` for radius-10 descriptors pulled in too much surrounding board structure and was slower on full-frame candidate sets. Full-frame Rust detection also runs the disk model only for the strongest 80 response candidates; the rest keep the sigma-LUT fallback.

## Acceptance And Fallback

The disk estimate is only published when it passes conservative gates:

- enough valid support pixels in the disk;
- sufficient recovered contrast;
- sufficient model correlation;
- finite residual and valid line separation;
- either a better normalized residual than baseline, strong edge-supported disagreement with baseline, or a clearly non-orthogonal crossing.

If these gates fail, the detector keeps the sigma-LUT baseline orientation. When the disk model disagrees strongly but is not confident enough, the benchmark keeps the baseline angles and inflates angular sigma instead of returning an overconfident wrong direction.

## Current Synthetic Status

On the current synthetic benchmark, the Rust opt-in method removes the previous extreme-skew failure:

```text
baseline worst RMSE:       20.93 deg
sigma-LUT worst RMSE:      20.93 deg
disk-sector worst RMSE:     4.11 deg
disk-sector mean RMSE:      1.15 deg
disk-sector mean z std:     0.805
disk-sector failure frac:   0.0
```

The largest remaining synthetic errors are not catastrophic failures. They are practical uncertainty cases around sharp or strongly skewed corners where small center errors and finite support still move the recovered axes by a few degrees.

## Practical Interpretation

The benchmark result says the task is solvable locally. The useful signal is in the disk, not only on the ring. The current method should still be treated as opt-in until it is measured on real images, but it is already a practical orientation estimator for synthetic projective corners.
