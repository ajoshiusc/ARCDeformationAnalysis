"""Pure numerical operations for contralateral-normalized deformation fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


@dataclass(frozen=True)
class NormalizedField:
    """Affine-removed atlas-aligned displacement and its quality diagnostics."""

    displacement_mm: np.ndarray
    valid: np.ndarray
    affine_coefficients: np.ndarray
    affine_fit_rmse_subject_voxels: float
    affine_fit_points: int
    affine_fit_inlier_fraction: float
    jacobian: np.ndarray
    folding_fraction: float
    affine_fit_rmse_mm: float | None = None


def sample_image(data: np.ndarray, coordinate_map: np.ndarray, order: int) -> np.ndarray:
    """Sample a scalar image at the voxel coordinates stored in a coordinate map."""
    if coordinate_map.ndim != 4 or coordinate_map.shape[-1] != 3:
        raise ValueError(f"Expected an XxYxZx3 coordinate map, got {coordinate_map.shape}")
    return map_coordinates(
        data,
        [coordinate_map[..., axis] for axis in range(3)],
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def coordinate_map_validity(
    coordinate_map: np.ndarray,
    target_shape: tuple[int, int, int],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Return map samples that are finite, in bounds, and inside both brains."""
    if coordinate_map.shape[:3] != target_mask.shape:
        raise ValueError("Coordinate map and target mask geometry differ")
    if source_mask.shape != target_shape:
        raise ValueError("Source mask and source image geometry differ")
    valid = target_mask.astype(bool) & np.isfinite(coordinate_map).all(axis=-1)
    for axis, size in enumerate(target_shape):
        valid &= coordinate_map[..., axis] >= 0
        valid &= coordinate_map[..., axis] <= size - 1
    valid &= sample_image(source_mask.astype(np.float32), coordinate_map, order=0) > 0.5
    return valid


def lesion_laterality(lesion: np.ndarray) -> tuple[str, float, int, int]:
    """Determine dominant side and absolute left-right lesion imbalance."""
    if lesion.ndim != 3:
        raise ValueError("Lesion mask must be three-dimensional")
    midpoint = (lesion.shape[0] - 1) / 2.0
    x = np.arange(lesion.shape[0], dtype=float)[:, None, None]
    left = int(np.count_nonzero(lesion & (x < midpoint)))
    right = int(np.count_nonzero(lesion & (x > midpoint)))
    total = left + right
    if total == 0:
        raise ValueError("Lesion is empty after atlas mapping")
    side = "left" if left > right else "right" if right > left else "bilateral"
    return side, float(abs(left - right) / total), left, right


def hemisphere_mask(
    shape: tuple[int, int, int], side: str, midline_buffer_voxels: float = 0.0
) -> np.ndarray:
    """Construct a left or right mask along the symmetric atlas first axis."""
    midpoint = (shape[0] - 1) / 2.0
    x = np.arange(shape[0], dtype=float)[:, None, None]
    if side == "left":
        base = x < midpoint - midline_buffer_voxels
    elif side == "right":
        base = x > midpoint + midline_buffer_voxels
    else:
        raise ValueError(f"Hemisphere must be 'left' or 'right', got {side!r}")
    return np.broadcast_to(base, shape)


def contralateral_mask(
    shape: tuple[int, int, int], lesion_side: str, midline_buffer_voxels: float = 2.0
) -> np.ndarray:
    opposite = "right" if lesion_side == "left" else "left" if lesion_side == "right" else ""
    if not opposite:
        raise ValueError("A dominant lesion side is required")
    return hemisphere_mask(shape, opposite, midline_buffer_voxels)


def fit_robust_affine(
    coordinate_map: np.ndarray,
    fit_mask: np.ndarray,
    subsample: int = 8,
    minimum_points: int = 10_000,
) -> tuple[np.ndarray, float, int, float]:
    """Fit atlas-voxel to subject-voxel affine with iterative MAD rejection."""
    if subsample < 1:
        raise ValueError("subsample must be positive")
    coordinates = np.argwhere(fit_mask)
    if coordinates.shape[0] < minimum_points:
        raise ValueError(
            f"Too few affine-fit voxels: {coordinates.shape[0]} < {minimum_points}"
        )
    coordinates = coordinates[::subsample]
    target = coordinate_map[tuple(coordinates.T)].astype(np.float64)
    design = np.column_stack([coordinates.astype(np.float64), np.ones(len(coordinates))])
    keep = np.ones(len(coordinates), dtype=bool)
    initial_count = len(coordinates)

    for _ in range(5):
        coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        residual = np.linalg.norm(target - design @ coefficients, axis=1)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        scale = max(1e-6, 1.4826 * mad)
        new_keep = residual <= center + 4.5 * scale
        if np.array_equal(new_keep, keep):
            break
        if int(new_keep.sum()) < max(4, minimum_points // subsample):
            break
        keep = new_keep

    coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    residual = np.linalg.norm(target[keep] - design[keep] @ coefficients, axis=1)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return coefficients, rmse, int(keep.sum()), float(keep.sum() / initial_count)


def affine_prediction(shape: tuple[int, int, int], coefficients: np.ndarray) -> np.ndarray:
    if coefficients.shape != (4, 3):
        raise ValueError(f"Expected 4x3 affine coefficients, got {coefficients.shape}")
    x, y, z = np.indices(shape, dtype=np.float32, sparse=True)
    prediction = np.empty((*shape, 3), dtype=np.float32)
    for component in range(3):
        prediction[..., component] = (
            coefficients[0, component] * x
            + coefficients[1, component] * y
            + coefficients[2, component] * z
            + coefficients[3, component]
        )
    return prediction


def smooth_vector_masked(
    field: np.ndarray,
    valid: np.ndarray,
    sigma_voxels: float | tuple[float, float, float],
) -> np.ndarray:
    """Normalized Gaussian smoothing that does not bleed zeros across invalid regions."""
    sigma = np.asarray(sigma_voxels, dtype=float)
    if np.any(sigma < 0):
        raise ValueError("Gaussian sigma must be nonnegative")
    if np.all(sigma == 0):
        result = field.astype(np.float32, copy=True)
        result[~valid] = 0
        return result
    weight = gaussian_filter(valid.astype(np.float32), sigma=sigma_voxels, mode="constant")
    stable = valid & (weight > 0.25)
    result = np.zeros_like(field, dtype=np.float32)
    for component in range(3):
        numerator = gaussian_filter(
            np.where(valid, field[..., component], 0.0),
            sigma=sigma_voxels,
            mode="constant",
        )
        result[..., component][stable] = numerator[stable] / weight[stable]
    return result


def jacobian_determinant(
    displacement_mm: np.ndarray, spacing_mm: tuple[float, float, float]
) -> np.ndarray:
    """Calculate det(I + grad(u)) for atlas-axis displacement in millimeters."""
    if displacement_mm.ndim != 4 or displacement_mm.shape[-1] != 3:
        raise ValueError("Displacement must have shape XxYxZx3")
    gradients = [
        [
            np.asarray(value, dtype=np.float32)
            for value in np.gradient(displacement_mm[..., component], *spacing_mm, edge_order=1)
        ]
        for component in range(3)
    ]
    f00, f01, f02 = 1.0 + gradients[0][0], gradients[0][1], gradients[0][2]
    f10, f11, f12 = gradients[1][0], 1.0 + gradients[1][1], gradients[1][2]
    f20, f21, f22 = gradients[2][0], gradients[2][1], 1.0 + gradients[2][2]
    determinant = (
        f00 * (f11 * f22 - f12 * f21)
        - f01 * (f10 * f22 - f12 * f20)
        + f02 * (f10 * f21 - f11 * f20)
    )
    return np.asarray(determinant, dtype=np.float32)


def normalize_coordinate_map(
    coordinate_map: np.ndarray,
    valid: np.ndarray,
    lesion_side: str,
    atlas_spacing_mm: tuple[float, float, float],
    source_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    smoothing_mm: float = 2.0,
    fit_subsample: int = 8,
    minimum_fit_points: int = 10_000,
) -> NormalizedField:
    """Remove a contralaterally fit global affine from an inverse coordinate map."""
    if not np.allclose(source_spacing_mm, source_spacing_mm[0], atol=1e-4, rtol=1e-4):
        raise ValueError(
            "Physical affine-fit RMSE currently requires an isotropic source grid; "
            f"found {source_spacing_mm}"
        )
    if any(not np.isfinite(value) or value <= 0 for value in source_spacing_mm):
        raise ValueError(f"Invalid source spacing: {source_spacing_mm}")
    fit_mask = valid & contralateral_mask(valid.shape, lesion_side)
    coefficients, rmse, fit_points, inlier_fraction = fit_robust_affine(
        coordinate_map, fit_mask, fit_subsample, minimum_fit_points
    )
    prediction = affine_prediction(valid.shape, coefficients)
    residual_subject_voxels = coordinate_map - prediction
    linear = coefficients[:3, :].T
    condition = float(np.linalg.cond(linear))
    if not np.isfinite(condition) or condition > 100:
        raise ValueError(f"Ill-conditioned fitted affine: condition={condition:.2f}")
    residual_atlas_voxels = residual_subject_voxels @ np.linalg.inv(linear).T
    displacement_mm = residual_atlas_voxels * np.asarray(atlas_spacing_mm, dtype=np.float32)
    displacement_mm = np.asarray(displacement_mm, dtype=np.float32)
    displacement_mm[~valid] = 0
    sigma = tuple(smoothing_mm / spacing for spacing in atlas_spacing_mm)
    displacement_mm = smooth_vector_masked(displacement_mm, valid, sigma)
    jacobian = jacobian_determinant(displacement_mm, atlas_spacing_mm)
    folding_fraction = float(np.mean(jacobian[valid] <= 0))
    return NormalizedField(
        displacement_mm=displacement_mm,
        valid=valid,
        affine_coefficients=coefficients,
        affine_fit_rmse_subject_voxels=rmse,
        affine_fit_points=fit_points,
        affine_fit_inlier_fraction=inlier_fraction,
        jacobian=jacobian,
        folding_fraction=folding_fraction,
        affine_fit_rmse_mm=rmse * float(source_spacing_mm[0]),
    )


def mirrored_vector(field: np.ndarray) -> np.ndarray:
    """Reflect a vector field across the first-axis atlas midline."""
    mirrored = np.flip(field, axis=0).copy()
    mirrored[..., 0] *= -1
    return mirrored


def lesion_effect_field(
    field: NormalizedField,
    target: np.ndarray,
    lesion_side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract mirrored control geometry and retain only lesional support."""
    paired_valid = field.valid & np.flip(field.valid, axis=0)
    paired_valid &= ~target & ~np.flip(target, axis=0)
    asymmetry = field.displacement_mm - mirrored_vector(field.displacement_mm)

    positive_jacobian = (field.jacobian > 0) & field.valid
    log_jacobian = np.zeros(field.valid.shape, dtype=np.float32)
    log_jacobian[positive_jacobian] = np.log(field.jacobian[positive_jacobian])
    paired_jacobian = positive_jacobian & np.flip(positive_jacobian, axis=0)
    paired_jacobian &= ~target & ~np.flip(target, axis=0)
    log_jacobian_asymmetry = log_jacobian - np.flip(log_jacobian, axis=0)

    valid = paired_valid & paired_jacobian & hemisphere_mask(target.shape, lesion_side)
    asymmetry[~valid] = 0
    log_jacobian_asymmetry[~valid] = 0
    return asymmetry.astype(np.float32), log_jacobian_asymmetry, valid


def lesion_distance_and_normals(
    lesion: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    distance = distance_transform_edt(~lesion, sampling=spacing_mm).astype(np.float32)
    gradients = np.gradient(distance, *spacing_mm, edge_order=1)
    normal = np.stack(gradients, axis=-1).astype(np.float32)
    norm = np.linalg.norm(normal, axis=-1)
    stable = norm > 1e-6
    normal[stable] /= norm[stable, None]
    normal[~stable] = 0
    return distance, normal
