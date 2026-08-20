# Frozen result provenance

## Input identity

The aggregate results in `results/reference/` were regenerated on 2026-08-19.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `lesion_mass_effect_v2/mass_effect_manifest.csv` | 1,385,289 | `fc812c6d668e85ebdb999f1f6c4d4418bce745d94e2090eb2898503afe905098` |
| `aq_mass_effect_inputs_v2/case_metrics.csv` | 60,336 | `521f37aab0a6da1b4a284a77abb720a0eda4e40ecb1deaff23ebcbeb236c2ca5` |
| `lesion_uncertainty/uncertainty_manifest.csv` | 596,670 | `937e735b1e9da51e3b39b89946e028420f95d17e312c3713856cf30694916aba` |
| generated private `hodge_manifest.csv` | 92,572 | `660f308c412785b800672f08e76989c5464528c6e1f39e2719e903673a01488a` |

Reference aggregate hashes:

- `analysis_config.json`: `cb1938ccd17e87dbf5961e242f25067b0d424ee5c4275ab7a28554979e7c0167`;
- `hodge_summary.json`: `b920630fab7857b8c04ca0cef09876e4ac5a66b09b431587466c3e821de01acd`;
- `hodge_config.json`: `a5fb64aaba64fdd83ffc95e1a96ff968b2033df17a93bd469e9898a803166d85`;
- `adjusted_deformation_associations.csv`: `91ecd750a403e47c19c0c4f08aff7a70642b34d17312e48475a6cf4efeb8d69a`;
- `hodge_parameter_sensitivity.csv`: `04346ffdf849a9c039fcd051bd3d4bb3a0cba30644c8a061508c05ce45ceeca8`;
- `hodge_parameter_sensitivity.json`: `b99c7d27ae1cd43313b5b062ec89d4b2244344c7617398e8528aae1d3d726d48`;
- `cohort_summary.json`: `df5a21bc1afb1bda8474f8797aac578beadbd8826861711243eddb2a87ca32cc`;
- `reproduction_check.json`: `70f0e3f8f85976406c8c45716f03fb279708c46127e1ab39becf2e8b7f223f0f`.

## Cohort flow

- 214 deformation cases and 214 unique participants were present.
- 211 cases matched the one-scan-per-participant clinical table.
- One matched case failed laterality QC (laterality index 0.090).
- 210 independent participants entered every model: 209 left-dominant and 1
  right-dominant lesion.
- Lesion-uncertainty and QC-passing Hodge features had 100% coverage in the
  analysis cohort.
- All 214 regularized log-velocity embeddings passed the positive-Jacobian and
  exponentiation criteria.

## Frozen numerical configuration

Deformation method: `contralateral_control_lesional_only_v2`.

Log-domain/Hodge method: `log_svf_tapered_periodic_fft_hhd_v1`:

- 4 mm analysis grid;
- 16 mm raised-cosine boundary taper;
- 10 mm Gaussian smoothing;
- 24 mm zero padding;
- 6 scaling-and-squaring steps and at most 6 residual updates;
- relative exponentiation RMSE threshold 0.02;
- positive input and reconstructed Jacobians required;
- periodic Fourier Hodge projection on the padded grid.

Prediction:

- five shuffled outer folds repeated 20 times;
- four inner folds;
- ridge alpha grid of 17 log-spaced values from `1e-4` to `1e4`;
- 5,000 participant bootstrap samples;
- seed 2026;
- identical outer splits for all feature families.

Adjusted associations residualized ranked exposures and ranked AQ on the six
conventional clinical/lesion variables. They used 5,000 participant bootstraps,
5,000 two-sided residual permutations, and Holm correction across six tests.

The Hodge sensitivity analysis changed one numerical factor at a time: 8/12-mm
smoothing, 12/20-mm taper, 16/32-mm padding, and 3/5-mm grids. Grid variants
preserved the primary physical regularization scales. All used the primary
positive-Jacobian and 0.02 exponentiation-error criteria.

## Results

The conventional lesion model had MAE 16.42 AQ points. Adding displacement
features yielded MAE 16.04 and a mean participant-level advantage of 0.38
points (95% bootstrap CI −0.50 to 1.17). Because the interval includes zero,
incremental prediction was not established.

Adding Hodge features to the lesion model yielded MAE 16.48 and an advantage of
−0.06 points (95% CI −0.42 to 0.29). Adding Hodge features after displacement
yielded MAE 16.12 and a direct advantage of −0.08 points (95% CI −0.35 to
0.21). Neither interval excluded zero.

The curl-free energy fraction was descriptively associated with AQ (Spearman
rho −0.240, bootstrap CI −0.365 to −0.104, Holm-adjusted p = 0.002), but Hodge
features did not add held-out predictive value. This is not a pressure result.

After adjustment for conventional clinical and lesion variables, the curl-free
result was partial Spearman rho −0.159 (bootstrap CI −0.290 to −0.028; Holm
residual-permutation p = 0.116), and the divergence-free result was rho 0.138
(0.011 to 0.261; Holm p = 0.186). None of the six adjusted associations survived
multiplicity correction.

Across eight numerical variants, the minimum rank correlation with the primary
descriptor was 0.991. Curl-free correlations with AQ ranged from −0.263 to
−0.220, and divergence-free correlations ranged from 0.149 to 0.206. The 8-mm
smoothing variant passed log-domain QC in 207/214 cases, the 12-mm taper in
212/214, and the 5-mm grid in 213/214; every other variant passed 214/214.

## Independent checks

Relative to the initial frozen analysis, every numeric value in the six legacy
model summaries and 120 legacy repeat-level metric rows is exact (maximum delta
0.0). Legacy paired-comparison point estimates, bootstrap intervals, Wilcoxon
statistics, and raw p-values are also exact. Holm-adjusted p-values were
recomputed because the comparison family expanded. The subsequent adjusted-
association and Hodge-parameter robustness extension left all prior aggregate
model, comparison, unadjusted-association, coefficient, left-dominant
sensitivity files byte-identical. Previously reported cohort-summary values were
unchanged; the file was extended only with aggregate WAB-type availability and
class counts used to document why subtype prediction was not attempted.

The read-only spatial audit loaded 1,070 NIfTI maps (five per case), found zero
nonzero or nonfinite contralesional values, and found zero v1-to-v2 change in
the ten predictive deformation/QC features. All 214 completion markers were
present and no processing-error record existed.

## ANTs/MNI152 pipeline-sensitivity rerun

The independent ANTs/MNI152 analysis was regenerated on 2026-08-20 for the
same explicit 214-case selection. The inpainting manifest contained additional
acquisitions; the primary SVReg manifest was used as a required one-to-one
selection contract so that those acquisitions could not enter registration or
modeling. All 214 registrations passed registration QC, 213 passed the combined
registration/deformation criteria, and 210 clinically matched participants
entered the models.

The TemplateFlow reference identities were:

- T1w SHA-256 `e0bdd27231960b3e930e86cf72b0d6bcf4a7d9e5195fb97e4c8f826f8d59c6e7`;
- brain-mask SHA-256 `6bae185e10e6bcd871e0caedfaa88a362eaefaccab2e81b96f7fa8f36b7ad6f0`.

Median brain-mask Dice was 0.9715 (minimum 0.8668), median point round-trip
RMSE was 0.0749 mm (maximum 0.0939), and every raw SyN warp had zero observed
nonpositive-Jacobian fraction. All 214 ANTs-derived log-domain embeddings
passed QC; median exponentiation relative RMSE was 0.0102. A complete repeat of
one case with the same seed and thread count reproduced every audited scalar,
nine output images, and both warp fields exactly.

The ANTs deformation model had MAE 15.61 versus 16.42 for the conventional
lesion model. Its participant-level mean advantage was 0.81 AQ points (95%
bootstrap CI 0.004 to 1.608; Holm-adjusted signed-rank p = 0.038). Hodge
features alone had advantage -0.12 (-0.53 to 0.30). On identical participants
and outer splits, ANTs-minus-SVReg MAE for the deformation model was -0.44
(-1.20 to 0.27), so the direct pipeline contrast was inconclusive.

Cross-pipeline rank agreement varied substantially: rho was 0.178 for median
near-lesion magnitude, 0.130 for log-velocity RMS, 0.257 for curl-free energy
fraction, and 0.259 for divergence-free energy fraction. Agreement for the
eight displacement descriptors ranged from 0.178 to 0.842. These results are
interpreted as whole-pipeline sensitivity because registration software,
reference atlas, and grid resolution changed together.

Selected aggregate file identities:

- `analysis_provenance.json`: `0fd80567577429669e5b5aa01cfe10499efd3f5a0952410c8a967b4f0276b8a4`;
- `model_summary.csv`: `2c177b6ac0e18e2083cf51deb4a10295f364adfd6b1327fb6227759cca886893`;
- `paired_comparisons.csv`: `e00dc14ead1a29c7bb5e8bc58902039aed0af1226caa66c0ee4b1e603e3cbcd7`;
- `descriptor_agreement.csv`: `52e00828c0906955fee4575aff1229031e986b9dd5407e743bbf3ca39aee7377`;
- `predictive_method_comparison.csv`: `0d89bd65a50cc418f51dba49ec92a3db072f275e1d4be16d93ede446f520ec21`.

The public freeze strips machine-local paths and retains only basenames,
checksums, aggregate rows, software versions, and numerical settings. It does
not retain case identifiers, participant rows, images, transforms, joined
design matrices, or out-of-fold predictions.
