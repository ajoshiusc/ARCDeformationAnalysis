"""Prespecified method identifiers and model features."""

from __future__ import annotations

METHOD_VERSION = "contralateral_control_lesional_only_v2"
SHELLS_MM = ((3.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0))

CLINICAL_FEATURES = ("age_at_stroke", "log1p_wab_days")
LESION_FEATURES = (
    "lesion_volume_ml",
    "left_language_lesion_ml",
    "right_language_lesion_ml",
    "lesion_laterality_index",
)
DEFORMATION_FEATURES = (
    "me_mass_effect_3_20mm_magnitude_mm_median",
    "me_mass_effect_3_20mm_magnitude_mm_p95",
    "me_mass_effect_3_20mm_radial_mm_median",
    "me_mass_effect_3_20mm_mean_absolute_radial_mm",
    "me_mass_effect_3_20mm_outward_integral_ml_mm",
    "me_mass_effect_3_20mm_inward_integral_ml_mm",
    "me_mass_effect_3_20mm_logjac_expansion_integral_ml",
    "me_mass_effect_3_20mm_logjac_compression_integral_ml",
)
REGISTRATION_QC_FEATURES = (
    "me_registration_sensitivity_3_20mm_mm_median",
    "me_contralateral_affine_fit_rmse_mm",
)
UNCERTAINTY_FEATURES = (
    "unc_expected_lesion_volume_ml",
    "unc_expected_left_lesion_volume_ml",
    "unc_expected_right_lesion_volume_ml",
    "unc_maximum_lesion_probability",
    "unc_entropy_mass_ml",
    "unc_left_entropy_mass_ml",
    "unc_right_entropy_mass_ml",
    "unc_candidate_mean_entropy",
    "unc_boundary_mean_entropy",
    "unc_high_uncertainty_volume_ml",
    "unc_volume_p25_ml",
    "unc_volume_p75_ml",
)
