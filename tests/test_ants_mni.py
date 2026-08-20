from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from ants_mni152.freeze_results import _public_provenance
from arc_deformation.ants_mni import (
    _indices_to_lps,
    _lps_to_indices,
    select_case_frame,
    validate_symmetric_mni_geometry,
)


@dataclass
class _ImageGeometry:
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: np.ndarray


def test_ants_physical_index_conversion_roundtrip() -> None:
    geometry = _ImageGeometry(
        origin=(98.0, 134.0, -72.0),
        spacing=(2.0, 2.0, 2.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    indices = np.array([[0, 0, 0], [49, 67, 36], [98, 116, 94]], dtype=float)
    np.testing.assert_allclose(
        _lps_to_indices(geometry, _indices_to_lps(geometry, indices)), indices
    )


def test_symmetric_mni_geometry_validation(tmp_path) -> None:
    shape = (9, 7, 5)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (-8.0, -6.0, -4.0)
    template = tmp_path / "template.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), template)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask)
    result = validate_symmetric_mni_geometry(template, mask)
    assert result["orientation"] == "RAS"
    assert result["midline_world_x_mm"] == 0.0
    assert result["brain_mask_reflection_dice"] == 1.0


def test_case_selection_is_exact_and_rejects_missing_ids() -> None:
    inpainting = pd.DataFrame({"case_id": ["case-1", "case-2", "case-3"]})
    selected = select_case_frame(inpainting, pd.DataFrame({"case_id": ["case-3", "case-1"]}))
    assert selected["case_id"].tolist() == ["case-1", "case-3"]
    with pytest.raises(ValueError, match="absent"):
        select_case_frame(inpainting, pd.DataFrame({"case_id": ["case-4"]}))


def test_public_provenance_removes_local_paths_but_retains_hashes() -> None:
    private = {
        "template_path": "/tmp/template.nii.gz",
        "input_manifest": "/protected/mass_effect_manifest.csv",
        "input_sha256": "abc123",
        "inputs": {
            "clinical_table": "/protected/clinical.csv",
            "clinical": {"path": "/protected/clinical.csv", "sha256": "def456"},
        },
    }
    public = _public_provenance(private)
    assert public["template_path_basename"] == "template.nii.gz"
    assert public["input_manifest_basename"] == "mass_effect_manifest.csv"
    assert public["input_sha256"] == "abc123"
    assert public["inputs"]["clinical_table_basename"] == "clinical.csv"
    assert public["inputs"]["clinical"] == {
        "path_basename": "clinical.csv",
        "sha256": "def456",
    }
    assert "/protected" not in str(public)
