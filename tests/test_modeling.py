from __future__ import annotations

import numpy as np
import pandas as pd

from arc_deformation.constants import CLINICAL_FEATURES, DEFORMATION_FEATURES, LESION_FEATURES
from arc_deformation.modeling import (
    ModelConfig,
    holm_adjust,
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
