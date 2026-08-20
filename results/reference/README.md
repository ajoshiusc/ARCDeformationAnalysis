# Aggregate reference results

These are frozen, non-participant-level outputs from the verified ARC run. They
allow every manuscript value and statistical figure to be regenerated without
redistributing participant clinical data or images.

- `model_summary.csv`: performance aggregated over 20 repeated outer-CV splits;
- `model_mae_inference.csv`: participant-bootstrap MAE intervals;
- `paired_comparisons.csv`: aggregate paired-error contrasts and intervals;
- `metrics_by_repeat.csv`: aggregate performance for each repeat and model;
- `deformation_associations.csv`: 12 descriptive correlations with bootstrap
  intervals and Holm-adjusted p-values;
- `adjusted_deformation_associations.csv`: six partial rank correlations with
  participant-bootstrap intervals and Holm-adjusted residual-permutation
  p-values;
- `hodge_parameter_sensitivity.csv` and `.json`: aggregate QC, rank stability,
  AQ-association direction, settings, and hashes for eight one-factor numerical
  variants;
- `left_only_model_summary.csv` and `left_only_paired_comparisons.csv`:
  left-dominant-lesion sensitivity results;
- `coefficient_summary.csv`: aggregate standardized ridge-coefficient stability;
- `cohort_summary.json`: aggregate cohort, displacement, and Hodge summaries;
- `hodge_summary.json`: 214-case numerical decomposition audit;
- `hodge_config.json`: fixed log-domain and Hodge settings plus input hash;
- `cohort_audit.json`: full 1,070-map spatial-support audit;
- `legacy_feature_comparison.csv`: v1/v2 predictive-feature identity check;
- `analysis_config.json`: model configuration and input provenance;
- `reproduction_check.json`: exact legacy-output and new-analysis checks;
- `figures/`: technical paper figures.

The primary and variant case-level Hodge manifests, joined design, fold
coefficients, and out-of-fold predictions are generated under `results/runs/`
and intentionally ignored.
