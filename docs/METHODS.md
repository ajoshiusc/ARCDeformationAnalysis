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
predictive model uses eight fixed features in the combined 3-20 mm shell:
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

## Stationary log-velocity embedding

The lesional-only mirrored-difference field is not assumed to be the original
SVReg diffeomorphism. For exploratory log-domain analysis it is embedded using
one cohort-wide numerical protocol:

- sample the vector field and valid mask on a 4 mm grid;
- taper the field from zero to full weight over 4 voxels inward from the valid
  boundary using a raised cosine;
- apply Gaussian smoothing with sigma 2.5 voxels (10 mm);
- zero-pad by 6 voxels (24 mm) on each side;
- require `det(I + grad(u)) > 0` everywhere;
- approximate `v = log(Id + u)` with six scaling-and-squaring steps and at most
  six symmetric residual corrections;
- require a positive-Jacobian `exp(v)` reconstruction and relative
  reconstruction RMSE no greater than 0.02.

The resulting `v` is a stationary registration parameter. It is neither the
time-dependent velocity used by an LDDMM optimizer nor a measured physical
tissue velocity.

On the padded periodic grid, Fourier coefficients are projected parallel and
perpendicular to each nonzero wave vector to obtain curl-free and
divergence-free components. The zero-frequency coefficient is harmonic. The
model uses total log-velocity RMS and the curl-free and divergence-free energy
fractions. Tapering, padding, and the periodic boundary convention are part of
the estimand because Hodge components are boundary-condition dependent.

No output is pressure. Pressure recovery requires a constitutive mechanical
model, tissue parameters, loads, and boundary conditions absent from ARC.

### Numerical sensitivity

Eight fixed one-factor variants test Gaussian smoothing at 8 and 12 mm,
raised-cosine taper widths of 12 and 20 mm, padding of 16 and 32 mm, and 3- and
5-mm grids. The grid variants preserve the primary 10-mm smoothing, 16-mm taper,
and approximately 24-mm padding in physical units. Every variant uses the same
positive-Jacobian and relative exponentiation-RMSE threshold as the primary
analysis. Expected numerical failures are retained as QC failures rather than
causing the cohort run to abort. Aggregate outputs report QC counts, rank
stability against the primary descriptors, and unadjusted AQ-association
directions; case-level variant manifests remain private.

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

An intercept-only benchmark, participant-bootstrap intervals for each model's
absolute MAE, a left-dominant-lesion-only rerun, and 12 Holm-controlled
descriptive Spearman correlations are also reported. Hodge models test the
three log-domain descriptors after conventional lesion features and again after
the displacement summaries.

Six exploratory partial Spearman correlations residualize the ranks of each
deformation/Hodge exposure and AQ on ranked conventional clinical and lesion
features. Their 95% intervals use 5,000 participant bootstraps. Two-sided
residual-permutation p-values use 5,000 permutations and are Holm-adjusted over
the six-test family. WAB aphasia subtype is not modeled: it is derived from WAB,
is missing in 29 participants, and includes classes with only four or five
cases. The joined analysis table contains no independent language-task outcome.
