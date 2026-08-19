# Method specification

## Interpretation

The method estimates a cross-sectional **lesion-associated deformation proxy**.
It is not a measurement of pressure, not the causal displacement caused by the
original stroke, and not a premorbid reconstruction. Chronic tissue collapse,
ventricular expansion, remodeling, natural asymmetry, and residual registration
error can contribute.

## Deformation construction

Let `F(x)` be SVReg's atlas-to-subject inverse coordinate map. A robust affine
map `F_A(x) = Mx + b` is fitted using valid voxels in the hemisphere
contralateral to the lesion, with iterative median-absolute-deviation outlier
rejection. The nonlinear residual is returned to atlas axes:

```text
u(x) = M^-1 [F(x) - F_A(x)].
```

For the current one-millimeter orthogonal atlas, voxel displacement and
atlas-axis millimeters coincide. The implementation nevertheless applies atlas
voxel spacing explicitly and validates geometry.

If `R` reflects positions across the atlas midline and flips a vector's
left-right component, the lesional asymmetry field is:

```text
u_L(x) = u(x) - R u(Rx).
```

Only the lesional hemisphere is retained. The contralesional half of every
stored effect map is exactly zero. Both members of a mirrored pair must be valid
and outside the lesion-inpainting target.

Magnitude is `||u_L||`. Radial displacement is the dot product between `u_L`
and the outward normal from the lesion distance transform. Local volume change
uses `J = det(I + grad(u))`; log-Jacobian asymmetry is the ipsilesional log-J
minus its reflected contralesional homolog. Nonpositive Jacobians are invalid.

Summaries are calculated within 3-5, 5-10, 10-20, and 20-40 mm shells. The
predictive model uses eight prespecified features in the combined 3-20 mm shell:
median and 95th-percentile magnitude, median and mean-absolute radial
displacement, outward and inward radial integrals, and positive and negative
log-Jacobian integrals.

## Quality control

A case passes if:

- lesion laterality index is at least 0.80;
- nonpositive-Jacobian fraction is at most 0.05;
- at least 1,000 valid voxels remain 3-20 mm from the lesion.

Direct/non-inpainted and inpainted SVReg fields are normalized identically. The
magnitude of their difference is registration sensitivity and is never treated
as biological deformation.

## Prediction

The primary reference model includes age at stroke, log-transformed days from
stroke to WAB, lesion volume, left and right language-network lesion burden, and
lesion laterality. The incremental model adds the eight deformation summaries.

Missing predictors are median-imputed, standardized, and fit with ridge
regression inside each training fold. A 4-fold inner loop selects the ridge
penalty by MAE. A 5-fold outer loop is repeated 20 times, with identical outer
splits for every model. Participants never cross folds because exactly one scan
per participant is retained.

The primary effect is the participant-level mean absolute-error advantage:
reference-model error minus augmented-model error, averaged over repeats.
Positive values favor the augmented model. Its uncertainty is a 5,000-sample
participant bootstrap 95% interval. A paired Wilcoxon signed-rank test is also
reported and Holm-adjusted, but it tests a rank/distributional contrast rather
than the mean advantage. A superiority statement requires the mean-advantage
interval to exclude zero.
