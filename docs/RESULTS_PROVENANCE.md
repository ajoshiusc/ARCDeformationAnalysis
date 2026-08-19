# Frozen result provenance

## Input identity

The aggregate results in `results/reference/` were generated from the ARC files
below. SHA-256 values were calculated on 2026-08-19.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `lesion_mass_effect_v2/mass_effect_manifest.csv` | 1,385,289 | `fc812c6d668e85ebdb999f1f6c4d4418bce745d94e2090eb2898503afe905098` |
| `aq_mass_effect_inputs_v2/case_metrics.csv` | 60,336 | `521f37aab0a6da1b4a284a77abb720a0eda4e40ecb1deaff23ebcbeb236c2ca5` |
| `lesion_uncertainty/uncertainty_manifest.csv` | 596,670 | `937e735b1e9da51e3b39b89946e028420f95d17e312c3713856cf30694916aba` |
| `aq_mass_effect_comparison_v2/analysis_config.json` | 5,342 | `9b1e09136c3c6f3e3ef43724b21dfa678647a397dab813c9b2138119bb507ab4` |

## Cohort flow

- 214 deformation cases and 214 unique participants were present.
- 211 matched the one-scan-per-participant clinical table.
- One matched case failed laterality QC (`sub-M2150`, laterality index 0.090).
- 210 independent participants entered every model.
- The analysis cohort contained 209 left-dominant and 1 right-dominant lesion.
- Optional uncertainty features had 100% coverage.

## Frozen model configuration

- deformation method: `contralateral_control_lesional_only_v2`;
- outer CV: 5 folds, shuffled, 20 repeats;
- inner CV: 4 folds;
- ridge alpha grid: 17 log-spaced values from `1e-4` to `1e4`;
- bootstrap: 5,000 participant resamples;
- random seed: 2026.

## Primary result

Conventional lesion model MAE was 16.42 AQ points. Adding deformation produced
MAE 16.04 and a participant-level mean MAE advantage of 0.38 points (95%
bootstrap CI -0.50 to 1.17). Because the interval includes zero, this is a
negative incremental-prediction result. The Holm-adjusted signed-rank p-value
was 0.023, a different rank-based estimand; it is retained for transparency but
does not override the prespecified mean-effect interval.

Adding deformation after lesion uncertainty changed MAE by -0.06 points (95%
CI -0.77 to 0.61; signed-rank p = 0.723), providing no evidence of additive
value.

## Independent reproduction checks

The refactored package reran the complete 20-repeat nested-CV analysis on
2026-08-19. All 18 model-summary columns, all 7 repeat-level metric columns,
and all 10 original paired-comparison columns were exactly equal to the frozen
ARC outputs (maximum numerical delta 0.0). The new table adds one explicit
Boolean field recording whether the primary mean-advantage interval excludes
zero.

The read-only spatial audit loaded 1,070 NIfTI maps (five maps for each of 214
cases), found zero nonzero contralesional values, zero nonfinite contralesional
values, and zero change from v1 to v2 in the ten predictive deformation/QC
features. All 214 completion markers were present and no processing-error
record existed.
