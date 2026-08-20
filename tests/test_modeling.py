from __future__ import annotations

import numpy as np
import pandas as pd

from arc_deformation.constants import (
    CLINICAL_FEATURES,
    DEFORMATION_FEATURES,
    HODGE_FEATURES,
    LESION_FEATURES,
)
from arc_deformation.modeling import (
    ModelConfig,
    _partial_rank_correlation,
    adjusted_deformation_associations,
    holm_adjust,
    model_feature_sets,
    paired_comparisons,
    repeated_nested_cv,
)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def _synthetic_design(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(n)],
            "subject": [f"sub-{index:03d}" for index in range(n)],
            "session": ["ses-1"] * n,
            "wab_aq": rng.normal(60, 15, n),
        }
    )
    for feature in CLINICAL_FEATURES + LESION_FEATURES + DEFORMATION_FEATURES:
        frame[feature] = rng.normal(size=n)
    frame["wab_aq"] += 8 * frame["lesion_volume_ml"]
    return frame


def test_repeated_nested_cv_is_complete_and_deterministic() -> None:
    design = _synthetic_design()
    standard = CLINICAL_FEATURES + LESION_FEATURES
    sets = {
        "lesion_standard": standard,
        "lesion_plus_mass_effect": standard + DEFORMATION_FEATURES,
    }
    config = ModelConfig(
        outer_folds=3,
        inner_folds=2,
        repeats=2,
        bootstrap_samples=100,
        seed=11,
        alpha_grid=(0.1, 1.0),
    )
    first, metrics, _ = repeated_nested_cv(design, sets, config)
    second, _, _ = repeated_nested_cv(design, sets, config)
    assert len(first) == 30 * 2 * 2
    assert len(metrics) == 4
    assert first["predicted_aq"].notna().all()
    np.testing.assert_allclose(first["predicted_aq"], second["predicted_aq"])


def test_intercept_only_benchmark_uses_training_fold_mean() -> None:
    design = _synthetic_design()
    config = ModelConfig(
        outer_folds=3,
        inner_folds=2,
        repeats=1,
        bootstrap_samples=100,
        seed=19,
        alpha_grid=(1.0,),
    )
    predictions, metrics, coefficients = repeated_nested_cv(
        design, {"intercept_only": ()}, config
    )
    observed = design.set_index("subject")["wab_aq"]
    for fold, group in predictions.groupby("outer_fold"):
        held_out = set(group["subject"])
        expected = observed.loc[~observed.index.isin(held_out)].mean()
        np.testing.assert_allclose(group["predicted_aq"], expected)
        assert fold in {1, 2, 3}
    assert len(metrics) == 1
    assert coefficients.empty


def test_hodge_feature_sets_are_incremental() -> None:
    design = _synthetic_design()
    for feature in HODGE_FEATURES:
        design[feature] = np.linspace(0.1, 0.9, len(design))
    sets = model_feature_sets(design, uncertainty_used=False, hodge_used=True)
    assert sets["lesion_plus_hodge"] == (CLINICAL_FEATURES + LESION_FEATURES + HODGE_FEATURES)
    assert sets["lesion_plus_mass_effect_plus_hodge"] == (
        CLINICAL_FEATURES + LESION_FEATURES + DEFORMATION_FEATURES + HODGE_FEATURES
    )


def test_paired_comparison_reports_mean_interval_estimand() -> None:
    rows = []
    for subject in range(25):
        for repeat in (1, 2):
            rows.extend(
                [
                    {
                        "subject": f"sub-{subject}",
                        "model": "reference",
                        "absolute_error": 4.0 + subject / 100,
                        "repeat": repeat,
                    },
                    {
                        "subject": f"sub-{subject}",
                        "model": "better",
                        "absolute_error": 3.0 + subject / 100,
                        "repeat": repeat,
                    },
                ]
            )
    result = paired_comparisons(pd.DataFrame(rows), "reference", 200, 3).iloc[0]
    assert result["mean_mae_advantage_points"] == 1.0
    assert bool(result["mean_advantage_ci_excludes_zero"])


def test_partial_rank_correlation_removes_ranked_covariate_effect() -> None:
    rng = np.random.default_rng(31)
    covariate = np.linspace(-2, 2, 80)
    frame = pd.DataFrame(
        {
            "exposure": covariate + rng.normal(0, 0.1, len(covariate)),
            "outcome": covariate + rng.normal(0, 0.1, len(covariate)),
            "covariate": covariate,
        }
    )
    crude = frame[["exposure", "outcome"]].corr(method="spearman").iloc[0, 1]
    adjusted, _, _ = _partial_rank_correlation(frame, "exposure", "outcome", ("covariate",))
    assert crude > 0.9
    assert abs(adjusted) < 0.25


def test_adjusted_associations_are_complete_and_deterministic() -> None:
    design = _synthetic_design(40)
    first = adjusted_deformation_associations(design, "wab_aq", 100, 41)
    second = adjusted_deformation_associations(design, "wab_aq", 100, 41)
    assert len(first) == 3
    assert first["n_subjects"].eq(40).all()
    assert first["permutation_p_value_holm"].between(0, 1).all()
    pd.testing.assert_frame_equal(first, second)
