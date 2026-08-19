from __future__ import annotations

import numpy as np
import pytest

from arc_deformation.field import (
    NormalizedField,
    fit_robust_affine,
    hemisphere_mask,
    jacobian_determinant,
    lesion_effect_field,
    lesion_laterality,
    mirrored_vector,
)


def test_lesion_laterality_and_hemisphere_masks() -> None:
    lesion = np.zeros((9, 5, 5), dtype=bool)
    lesion[1:3, 2, 2] = True
    side, index, left, right = lesion_laterality(lesion)
    assert (side, index, left, right) == ("left", 1.0, 2, 0)
    assert hemisphere_mask(lesion.shape, "left")[1, 2, 2]
    assert not hemisphere_mask(lesion.shape, "left")[7, 2, 2]


def test_mirrored_vector_flips_left_right_component() -> None:
    field = np.zeros((5, 2, 2, 3), dtype=np.float32)
    field[0, ..., :] = (1, 2, 3)
    mirrored = mirrored_vector(field)
    np.testing.assert_array_equal(mirrored[4, 0, 0], (-1, 2, 3))


def test_lesion_effect_is_exactly_zero_on_control_side() -> None:
    shape = (9, 5, 5)
    displacement = np.zeros((*shape, 3), dtype=np.float32)
    displacement[:4, ..., 1] = 2.0
    normalized = NormalizedField(
        displacement_mm=displacement,
        valid=np.ones(shape, dtype=bool),
        affine_coefficients=np.vstack([np.eye(3), np.zeros(3)]),
        affine_fit_rmse_subject_voxels=0.0,
        affine_fit_points=100,
        affine_fit_inlier_fraction=1.0,
        jacobian=np.ones(shape, dtype=np.float32),
        folding_fraction=0.0,
    )
    effect, logj, valid = lesion_effect_field(normalized, np.zeros(shape, dtype=bool), "left")
    control = hemisphere_mask(shape, "right")
    assert np.count_nonzero(effect[control]) == 0
    assert np.count_nonzero(logj[control]) == 0
    assert not valid[control].any()
    assert np.count_nonzero(effect[hemisphere_mask(shape, "left")]) > 0


def test_jacobian_for_uniform_x_expansion() -> None:
    shape = (12, 8, 6)
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    displacement = np.zeros((*shape, 3), dtype=np.float32)
    displacement[..., 0] = 0.1 * x
    jacobian = jacobian_determinant(displacement, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(jacobian, 1.1, atol=1e-6)


def test_robust_affine_recovers_known_map_with_outliers() -> None:
    shape = (16, 14, 12)
    coordinates = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)
    linear = np.array([[1.1, 0.02, 0.0], [0.0, 0.9, 0.03], [0.0, 0.0, 1.05]])
    offset = np.array([4.0, -2.0, 1.0])
    coordinate_map = coordinates @ linear.T + offset
    coordinate_map[0, :3, :3] += 50
    coefficients, _, _, fraction = fit_robust_affine(
        coordinate_map, np.ones(shape, dtype=bool), subsample=1, minimum_points=100
    )
    expected = np.vstack([linear.T, offset])
    np.testing.assert_allclose(coefficients, expected, atol=1e-5)
    assert fraction < 1.0


def test_empty_lesion_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        lesion_laterality(np.zeros((5, 5, 5), dtype=bool))
