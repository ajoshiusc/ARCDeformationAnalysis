"""Read-only cohort and spatial-support audit for deformation derivatives."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from arc_deformation.constants import DEFORMATION_FEATURES, METHOD_VERSION
from arc_deformation.field import contralateral_mask
from arc_deformation.io import (
    atomic_csv,
    atomic_json,
    localize_arc_path,
    read_table,
    require_unique,
    truthy,
)

MAP_COLUMNS = (
    "mass_effect_vector_path",
    "mass_effect_magnitude_path",
    "mass_effect_radial_path",
    "log_jacobian_asymmetry_path",
    "valid_mask_path",
)


def validate_manifest(
    frame: pd.DataFrame,
    expected_cases: int | None = None,
    expected_method: str = METHOD_VERSION,
) -> dict[str, object]:
    require_unique(frame, "case_id", "mass-effect manifest")
    require_unique(frame, "subject", "mass-effect manifest")
    required = {
        "session",
        "method_version",
        "effect_field_support",
        "contralesional_effect_value",
        "lesion_side",
        "lesion_laterality_index",
        "laterality_supported",
        "normalized_field_folding_fraction",
        "mass_effect_3_20mm_magnitude_mm_n_voxels",
        *MAP_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Mass-effect manifest is incomplete: {missing}")
    if expected_cases is not None and len(frame) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, found {len(frame)}")
    versions = sorted(frame["method_version"].dropna().astype(str).unique())
    if versions != [expected_method]:
        raise ValueError(f"Expected only method {expected_method!r}, found {versions!r}")
    if not frame["effect_field_support"].eq("lesional_hemisphere_only").all():
        raise ValueError("At least one field is not marked lesional-only")
    values = pd.to_numeric(frame["contralesional_effect_value"], errors="coerce")
    if not values.eq(0).all():
        raise ValueError("At least one record does not declare zero contralesional effect")
    qc = (
        truthy(frame["laterality_supported"])
        & pd.to_numeric(frame["normalized_field_folding_fraction"], errors="coerce").le(0.05)
        & pd.to_numeric(frame["mass_effect_3_20mm_magnitude_mm_n_voxels"], errors="coerce").ge(
            1000
        )
    )
    return {
        "cases": len(frame),
        "subjects": int(frame["subject"].nunique()),
        "method_version": expected_method,
        "qc_pass": int(qc.sum()),
        "qc_fail": int((~qc).sum()),
        "lesion_side_counts": {
            str(key): int(value) for key, value in frame["lesion_side"].value_counts().items()
        },
    }


def audit_zero_contralesional_support(frame: pd.DataFrame, arc_root: Path) -> dict[str, object]:
    """Read every stored map and prove that reference-side values are exactly zero."""
    checked = 0
    nonzero = 0
    maximum = 0.0
    for record in frame.to_dict("records"):
        side = str(record["lesion_side"])
        reference_shape: tuple[int, int, int] | None = None
        reference_mask: np.ndarray | None = None
        for column in MAP_COLUMNS:
            path = localize_arc_path(record[column], arc_root)
            data = np.asarray(nib.load(path).dataobj)
            shape = tuple(int(value) for value in data.shape[:3])
            if reference_shape is None:
                reference_shape = shape
                reference_mask = contralateral_mask(shape, side, midline_buffer_voxels=0)
            elif shape != reference_shape:
                raise ValueError(f"Map geometry mismatch for {record['case_id']}: {path}")
            assert reference_mask is not None
            values = np.asarray(data[reference_mask], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite contralesional values in {path}")
            nonzero += int(np.count_nonzero(values))
            if values.size:
                maximum = max(maximum, float(np.max(np.abs(values))))
            checked += 1
    if nonzero:
        raise ValueError(
            f"Found {nonzero} nonzero contralesional values; maximum absolute value {maximum}"
        )
    return {
        "stored_maps_checked": checked,
        "contralesional_nonzero_values": nonzero,
        "contralesional_maximum_absolute_value": maximum,
    }


def compare_legacy_features(
    current: pd.DataFrame,
    legacy: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    require_unique(legacy, "case_id", "legacy mass-effect manifest")
    if set(current["case_id"]) != set(legacy["case_id"]):
        raise ValueError("Current and legacy manifests contain different case sets")
    unprefixed = tuple(feature.removeprefix("me_") for feature in DEFORMATION_FEATURES)
    extra = ("registration_sensitivity_3_20mm_mm_median", "contralateral_affine_fit_rmse_mm")
    features = unprefixed + extra
    joined = legacy[["case_id", *features]].merge(
        current[["case_id", *features]],
        on="case_id",
        suffixes=("_legacy", "_current"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for feature in features:
        old = pd.to_numeric(joined[f"{feature}_legacy"], errors="coerce").to_numpy(float)
        new = pd.to_numeric(joined[f"{feature}_current"], errors="coerce").to_numpy(float)
        both_missing = np.isnan(old) & np.isnan(new)
        comparable = np.isfinite(old) & np.isfinite(new)
        incompatible = ~(both_missing | comparable)
        delta = np.abs(old[comparable] - new[comparable])
        rows.append(
            {
                "feature": feature,
                "n_cases": len(joined),
                "n_comparable": int(comparable.sum()),
                "n_incompatible_missingness": int(incompatible.sum()),
                "n_above_tolerance": int(np.count_nonzero(delta > tolerance)),
                "maximum_absolute_delta": float(delta.max()) if delta.size else 0.0,
                "absolute_tolerance": tolerance,
            }
        )
    result = pd.DataFrame(rows)
    if result["n_incompatible_missingness"].sum() or result["n_above_tolerance"].sum():
        raise ValueError("Current and legacy predictive deformation features differ")
    return result


def run_audit(
    manifest_path: Path,
    output_dir: Path,
    arc_root: Path,
    expected_cases: int | None = None,
    check_maps: bool = False,
    legacy_manifest: Path | None = None,
) -> dict[str, object]:
    frame = read_table(manifest_path)
    report = validate_manifest(frame, expected_cases)
    manifest_root = Path(manifest_path).parent
    failures = list(manifest_root.rglob("processing_error.json"))
    markers = list(manifest_root.glob("sub-*/mass_effect_complete"))
    report.update(
        {
            "status": "pass",
            "failure_records": len(failures),
            "completion_markers": len(markers),
        }
    )
    if failures:
        raise ValueError(f"Found {len(failures)} processing_error.json records")
    if expected_cases is not None and markers and len(markers) != expected_cases:
        raise ValueError(f"Expected {expected_cases} completion markers, found {len(markers)}")
    if check_maps:
        report.update(audit_zero_contralesional_support(frame, arc_root))
    if legacy_manifest:
        comparison = compare_legacy_features(frame, read_table(legacy_manifest))
        atomic_csv(Path(output_dir) / "legacy_feature_comparison.csv", comparison)
        report["legacy_maximum_absolute_feature_delta"] = float(
            comparison["maximum_absolute_delta"].max()
        )
    atomic_json(Path(output_dir) / "cohort_audit.json", report)
    return report
