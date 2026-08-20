from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arc_deformation.audit import validate_manifest
from arc_deformation.constants import METHOD_VERSION
from arc_deformation.extract import collect_metrics
from arc_deformation.io import atomic_json, ensure_output_outside_data, localize_arc_path


def test_output_inside_data_root_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match="Refusing"):
        ensure_output_outside_data(data_root / "derivatives" / "new", data_root)


def test_manifest_validation_counts_qc() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["case-1"],
            "subject": ["sub-1"],
            "session": ["ses-1"],
            "method_version": [METHOD_VERSION],
            "effect_field_support": ["lesional_hemisphere_only"],
            "contralesional_effect_value": [0.0],
            "lesion_side": ["left"],
            "lesion_laterality_index": [1.0],
            "laterality_supported": [True],
            "normalized_field_folding_fraction": [0.01],
            "mass_effect_3_20mm_magnitude_mm_n_voxels": [2000],
            "mass_effect_vector_path": ["a"],
            "mass_effect_magnitude_path": ["b"],
            "mass_effect_radial_path": ["c"],
            "log_jacobian_asymmetry_path": ["d"],
            "valid_mask_path": ["e"],
        }
    )
    result = validate_manifest(frame, expected_cases=1)
    assert result["qc_pass"] == 1
    assert result["qc_fail"] == 0


def test_atomic_json_writes_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    atomic_json(output, {"finite": np.float32(1.5), "missing": float("nan")})
    assert "NaN" not in output.read_text(encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": 1.5,
        "missing": None,
    }


def test_arc_path_localization_is_machine_independent(tmp_path: Path) -> None:
    arc_root = tmp_path / "ARC"
    local_file = arc_root / "derivatives" / "example.nii.gz"
    local_file.parent.mkdir(parents=True)
    local_file.touch()
    relocated = "/different/system/data/ARC/derivatives/example.nii.gz"
    assert localize_arc_path(relocated, arc_root) == local_file.resolve()


def test_collect_metrics_is_sorted_and_rejects_duplicates(tmp_path: Path) -> None:
    for subject, case_id in (("sub-2", "case-2"), ("sub-1", "case-1")):
        path = tmp_path / subject / "mass_effect_metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"subject": subject, "case_id": case_id}), encoding="utf-8")
    manifest = collect_metrics(tmp_path)
    assert pd.read_csv(manifest)["case_id"].tolist() == ["case-1", "case-2"]

    duplicate = tmp_path / "duplicate" / "mass_effect_metrics.json"
    duplicate.parent.mkdir()
    duplicate.write_text(
        json.dumps({"subject": "sub-3", "case_id": "case-1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate case IDs"):
        collect_metrics(tmp_path)
