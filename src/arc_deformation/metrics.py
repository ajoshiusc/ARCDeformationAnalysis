"""Voxel and case-level summaries of the lesion-associated deformation proxy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from arc_deformation.constants import METHOD_VERSION, SHELLS_MM
from arc_deformation.field import NormalizedField, hemisphere_mask


@dataclass(frozen=True)
class CaseMetadata:
    case_id: str
    subject: str
    session: str
    lesion_side: str
    lesion_laterality_index: float
    lesion_left_voxels: int
    lesion_right_voxels: int


def safe_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_n_voxels": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_p95": math.nan,
        }
    p25, median, p75, p95 = np.percentile(finite, [25, 50, 75, 95])
    return {
        f"{prefix}_n_voxels": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(median),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p95": float(p95),
    }


def shell_key(lower: float, upper: float) -> str:
    return f"shell_{lower:g}_{upper:g}mm".replace(".", "p")


def calculate_case_metrics(
    metadata: CaseMetadata,
    lesion: np.ndarray,
    target: np.ndarray,
    normalized: NormalizedField,
    effect: np.ndarray,
    log_jacobian_asymmetry: np.ndarray,
    valid: np.ndarray,
    distance_mm: np.ndarray,
    radial_mm: np.ndarray,
    magnitude_mm: np.ndarray,
    voxel_volume_mm3: float,
    registration_sensitivity_mm: np.ndarray | None = None,
    minimum_laterality: float = 0.80,
    smoothing_mm: float = 2.0,
) -> dict[str, object]:
    """Calculate all registered case outputs from one common valid-support mask."""
    expected_shape = lesion.shape
    arrays = (target, valid, distance_mm, radial_mm, magnitude_mm, log_jacobian_asymmetry)
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("Scalar metric arrays do not share lesion geometry")
    if effect.shape != (*expected_shape, 3):
        raise ValueError("Effect field does not share lesion geometry")

    analysis = valid & hemisphere_mask(expected_shape, metadata.lesion_side)
    metrics: dict[str, object] = {
        "case_id": metadata.case_id,
        "subject": metadata.subject,
        "session": metadata.session,
        "method_version": METHOD_VERSION,
        "interpretation": (
            "lesional-only deformation relative to the mirrored contralesional "
            "within-participant control"
        ),
        "not_physical_ground_truth": True,
        "effect_field_support": "lesional_hemisphere_only",
        "contralesional_effect_value": 0.0,
        "lesion_side": metadata.lesion_side,
        "lesion_laterality_index": metadata.lesion_laterality_index,
        "minimum_supported_laterality": minimum_laterality,
        "laterality_supported": metadata.lesion_laterality_index >= minimum_laterality,
        "lesion_left_voxels_atlas": metadata.lesion_left_voxels,
        "lesion_right_voxels_atlas": metadata.lesion_right_voxels,
        "lesion_volume_atlas_ml": float(lesion.sum() * voxel_volume_mm3 / 1000.0),
        "inpainting_target_volume_atlas_ml": float(target.sum() * voxel_volume_mm3 / 1000.0),
        "valid_analysis_volume_ml": float(analysis.sum() * voxel_volume_mm3 / 1000.0),
        "smoothing_mm": smoothing_mm,
        "contralateral_affine_fit_rmse_subject_voxels": (
            normalized.affine_fit_rmse_subject_voxels
        ),
        # ARC's processed/inpainted source grid is 1 mm isotropic, so this is
        # numerically identical to the explicitly named voxel-coordinate RMSE.
        "contralateral_affine_fit_rmse_mm": normalized.affine_fit_rmse_subject_voxels,
        "contralateral_affine_fit_points": normalized.affine_fit_points,
        "contralateral_affine_fit_inlier_fraction": normalized.affine_fit_inlier_fraction,
        "normalized_field_folding_fraction": normalized.folding_fraction,
        "affine_coefficients_atlas_to_subject_voxels": normalized.affine_coefficients.tolist(),
    }
    metrics.update(safe_summary(magnitude_mm[analysis], "ipsilateral_magnitude_mm"))
    metrics.update(safe_summary(radial_mm[analysis], "ipsilateral_radial_mm"))
    metrics.update(
        safe_summary(log_jacobian_asymmetry[analysis], "ipsilateral_log_jacobian_asymmetry")
    )

    for lower, upper in SHELLS_MM:
        region = analysis & (distance_mm >= lower) & (distance_mm < upper)
        key = shell_key(lower, upper)
        metrics.update(safe_summary(magnitude_mm[region], f"{key}_magnitude_mm"))
        metrics.update(safe_summary(radial_mm[region], f"{key}_radial_mm"))
        metrics.update(
            safe_summary(log_jacobian_asymmetry[region], f"{key}_log_jacobian_asymmetry")
        )
        count = int(region.sum())
        metrics[f"{key}_outward_fraction"] = (
            float(np.mean(radial_mm[region] > 0)) if count else math.nan
        )
        metrics[f"{key}_inward_fraction"] = (
            float(np.mean(radial_mm[region] < 0)) if count else math.nan
        )

    near = analysis & (distance_mm >= 3.0) & (distance_mm < 20.0)
    near_radial = radial_mm[near]
    near_magnitude = magnitude_mm[near]
    near_logj = log_jacobian_asymmetry[near]
    metrics.update(safe_summary(near_magnitude, "mass_effect_3_20mm_magnitude_mm"))
    metrics.update(safe_summary(near_radial, "mass_effect_3_20mm_radial_mm"))
    metrics.update(safe_summary(near_logj, "mass_effect_3_20mm_log_jacobian_asymmetry"))
    metrics["mass_effect_3_20mm_mean_absolute_radial_mm"] = (
        float(np.mean(np.abs(near_radial))) if near_radial.size else math.nan
    )
    scale = voxel_volume_mm3 / 1000.0
    metrics["mass_effect_3_20mm_outward_integral_ml_mm"] = float(
        np.sum(np.clip(near_radial, 0, None), dtype=np.float64) * scale
    )
    metrics["mass_effect_3_20mm_inward_integral_ml_mm"] = float(
        np.sum(np.clip(-near_radial, 0, None), dtype=np.float64) * scale
    )
    metrics["mass_effect_3_20mm_magnitude_integral_ml_mm"] = float(
        np.sum(near_magnitude, dtype=np.float64) * scale
    )
    metrics["mass_effect_3_20mm_logjac_expansion_integral_ml"] = float(
        np.sum(np.clip(near_logj, 0, None), dtype=np.float64) * scale
    )
    metrics["mass_effect_3_20mm_logjac_compression_integral_ml"] = float(
        np.sum(np.clip(-near_logj, 0, None), dtype=np.float64) * scale
    )

    if registration_sensitivity_mm is not None:
        if registration_sensitivity_mm.shape != expected_shape:
            raise ValueError("Registration sensitivity geometry differs")
        metrics.update(
            safe_summary(
                registration_sensitivity_mm[near], "registration_sensitivity_3_20mm_mm"
            )
        )
    else:
        metrics["registration_sensitivity_3_20mm_mm_n_voxels"] = 0

    signal = float(metrics.get("mass_effect_3_20mm_magnitude_mm_median", math.nan))
    sensitivity = float(metrics.get("registration_sensitivity_3_20mm_mm_median", math.nan))
    metrics["mass_effect_to_registration_sensitivity_ratio"] = (
        signal / sensitivity
        if np.isfinite(signal) and np.isfinite(sensitivity) and sensitivity > 0
        else math.nan
    )
    metrics["deformation_qc_pass"] = bool(
        metrics["laterality_supported"]
        and normalized.folding_fraction <= 0.05
        and near_magnitude.size >= 1000
    )
    metrics["deformation_qc_criteria"] = {
        "laterality_fraction_minimum": minimum_laterality,
        "normalized_field_folding_fraction_maximum": 0.05,
        "mass_effect_3_20mm_valid_voxels_minimum": 1000,
    }
    return metrics
