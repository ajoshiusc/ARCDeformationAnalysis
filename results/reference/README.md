# Aggregate reference results

These files are frozen, non-participant-level outputs from the verified ARC run.
They allow the manuscript to compile and its reported values to be checked
without distributing subject-level clinical data or images.

- `model_summary.csv`: performance aggregated over 20 repeated outer-CV splits;
- `paired_comparisons.csv`: participant-level paired error contrasts;
- `metrics_by_repeat.csv`: aggregate performance for each repeat and model;
- `coefficient_summary.csv`: aggregate standardized ridge-coefficient stability;
- `cohort_summary.json`: aggregate cohort/deformation descriptive statistics;
- `cohort_audit.json`: full 1,070-map spatial-support audit;
- `legacy_feature_comparison.csv`: v1/v2 predictive-feature identity check;
- `analysis_config.json`: exact model configuration and input provenance;
- `figures/`: paper figures generated from the frozen analysis.

Regenerate a complete private run under `results/runs/`; that directory is
ignored by git because it contains case identifiers and predictions.
