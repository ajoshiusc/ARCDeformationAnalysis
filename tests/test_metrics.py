from __future__ import annotations

import numpy as np

from arc_deformation.field import NormalizedField
from arc_deformation.metrics import CaseMetadata, calculate_case_metrics, safe_summary


def test_safe_summary_ignores_nonfinite_values() -> None:
    summary = safe_summary(np.array([1.0, 2.0, np.nan, np.inf, 3.0]), "x")
    assert summary["x_n_voxels"] == 3
    assert summary["x_median"] == 2.0


def test_case_metrics_apply_qc_thresholds() -> None:
    shape = (21, 21, 21)
    lesion = np.zeros(shape, dtype=bool)
    lesion[4:7, 8:13, 8:13] = True
    target = lesion.copy()
    valid = np.zeros(shape, dtype=bool)
    valid[:10] = True
    distance = np.full(shape, 5.0, dtype=np.float32)
    magnitude = np.where(valid, 2.0, 0.0).astype(np.float32)
    radial = np.where(valid, -1.0, 0.0).astype(np.float32)
    logj = np.where(valid, 0.1, 0.0).astype(np.float32)
    effect = np.zeros((*shape, 3), dtype=np.float32)
    effect[..., 0] = magnitude
    normalized = NormalizedField(
        displacement_mm=effect,
        valid=valid,
        affine_coefficients=np.vstack([np.eye(3), np.zeros(3)]),
        affine_fit_rmse_subject_voxels=1.0,
        affine_fit_points=10_000,
        affine_fit_inlier_fraction=0.99,
        jacobian=np.ones(shape, dtype=np.float32),
        folding_fraction=0.01,
    )
    metadata = CaseMetadata("case", "sub-1", "ses-1", "left", 1.0, 75, 0)
    metrics = calculate_case_metrics(
        metadata,
        lesion,
        target,
        normalized,
        effect,
        logj,
        valid,
        distance,
        radial,
        magnitude,
        voxel_volume_mm3=1.0,
    )
    assert metrics["deformation_qc_pass"] is True
    assert metrics["mass_effect_3_20mm_magnitude_mm_median"] == 2.0
    assert metrics["mass_effect_3_20mm_radial_mm_median"] == -1.0
    assert metrics["effect_field_support"] == "lesional_hemisphere_only"
