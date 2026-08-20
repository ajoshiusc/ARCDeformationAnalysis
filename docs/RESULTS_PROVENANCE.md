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

- `analysis_config.json`: `a1c75e7ed59cea627d0daaba562c17ce922c0840a36b114cd690c445ceceb76b`;
- `hodge_summary.json`: `b920630fab7857b8c04ca0cef09876e4ac5a66b09b431587466c3e821de01acd`;
- `hodge_config.json`: `14d7eebce814d6cd820278a375feb41df0ee9df1585a506dcb525a92cabb3ce2`.

## Cohort flow

- 214 deformation cases and 214 unique participants were present.
- 211 cases matched the one-scan-per-participant clinical table.
- One matched case failed laterality QC (`sub-M2150`, laterality index 0.090).
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

## Independent checks

Relative to the initial frozen analysis, every numeric value in the six legacy
model summaries and 120 legacy repeat-level metric rows is exact (maximum delta
0.0). Legacy paired-comparison point estimates, bootstrap intervals, Wilcoxon
statistics, and raw p-values are also exact. Holm-adjusted p-values were
recomputed because the comparison family expanded.

The read-only spatial audit loaded 1,070 NIfTI maps (five per case), found zero
nonzero or nonfinite contralesional values, and found zero v1-to-v2 change in
the ten predictive deformation/QC features. All 214 completion markers were
present and no processing-error record existed.
