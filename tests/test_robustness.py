from __future__ import annotations

import numpy as np
import pandas as pd

from arc_deformation.robustness import (
    SENSITIVITY_FEATURES,
    compare_hodge_variant,
    default_sensitivity_variants,
)


def test_default_sensitivity_variants_are_unique_one_factor_checks() -> None:
    variants = default_sensitivity_variants()
    names = [variant.name for variant in variants]
    assert len(variants) == 8
    assert len(set(names)) == len(names)
    assert {"grid_3mm", "grid_5mm", "padding_16mm", "padding_32mm"}.issubset(names)


def test_identical_hodge_variant_has_unit_rank_stability() -> None:
    size = 24
    primary = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(size)],
            "wab_aq": np.linspace(20, 90, size),
        }
    )
    variant = pd.DataFrame(
        {
            "case_id": primary["case_id"],
            "velocity_qc_pass": True,
        }
    )
    for offset, feature in enumerate(SENSITIVITY_FEATURES):
        values = np.linspace(0.1 + offset, 1.1 + offset, size)
        primary[f"hhd_{feature}"] = values
        variant[feature] = values
    comparison = compare_hodge_variant(primary, variant, "identical")
    assert comparison["common_analysis_cases"].eq(size).all()
    np.testing.assert_allclose(comparison["spearman_vs_primary"], 1.0)
    np.testing.assert_allclose(comparison["median_absolute_difference"], 0.0)

    variant.loc[0, "velocity_qc_pass"] = False
    filtered = compare_hodge_variant(primary, variant, "one-failure")
    assert filtered["common_analysis_cases"].eq(size - 1).all()
