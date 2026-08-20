"""Log-domain stationary-velocity approximations for regularized deformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from arc_deformation.field import jacobian_determinant


@dataclass(frozen=True)
class VelocityLog:
    """A validated stationary-velocity approximation and reconstruction audit."""

    velocity_mm: np.ndarray
    reconstructed_displacement_mm: np.ndarray
    iterations: int
    relative_reconstruction_rmse: float
    displacement_minimum_jacobian: float
    reconstruction_minimum_jacobian: float


def _validate_vector(field: np.ndarray, spacing_mm: tuple[float, float, float]) -> None:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected XxYxZx3 vector field, got {field.shape}")
    if not np.isfinite(field).all():
        raise ValueError("Vector field contains nonfinite values")
    if any(not np.isfinite(value) or value <= 0 for value in spacing_mm):
        raise ValueError(f"Invalid vector-field spacing: {spacing_mm}")


def compose_displacements(
    outer_mm: np.ndarray,
    inner_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Return displacement of ``(Id + outer) o (Id + inner)``.

    Fields use physical axis-aligned millimeters on the same regular grid.
    Identity continuation is used beyond the finite computational domain.
    """
    outer_mm = np.asarray(outer_mm, dtype=np.float32)
    inner_mm = np.asarray(inner_mm, dtype=np.float32)
    _validate_vector(outer_mm, spacing_mm)
    if inner_mm.shape != outer_mm.shape:
        raise ValueError("Composed displacement fields must share a grid")
    _validate_vector(inner_mm, spacing_mm)
    coordinates = np.indices(outer_mm.shape[:3], dtype=np.float32)
    for axis, spacing in enumerate(spacing_mm):
        coordinates[axis] += inner_mm[..., axis] / spacing
    sampled_outer = np.empty_like(outer_mm)
    for component in range(3):
        sampled_outer[..., component] = map_coordinates(
            outer_mm[..., component],
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return inner_mm + sampled_outer


def exponentiate_stationary_velocity(
    velocity_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
    squaring_steps: int = 6,
) -> np.ndarray:
    """Compute ``exp(v) - Id`` by scaling and squaring."""
    velocity_mm = np.asarray(velocity_mm, dtype=np.float32)
    _validate_vector(velocity_mm, spacing_mm)
    if squaring_steps < 0:
        raise ValueError("squaring_steps must be nonnegative")
    displacement = velocity_mm / float(2**squaring_steps)
    for _ in range(squaring_steps):
        displacement = compose_displacements(displacement, displacement, spacing_mm)
    return np.asarray(displacement, dtype=np.float32)


def _relative_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean((reference - estimate) ** 2, dtype=np.float64)))
    denominator = float(np.sqrt(np.mean(reference**2, dtype=np.float64)))
    return numerator / max(denominator, 1e-12)


def logarithm_stationary_velocity(
    displacement_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
    squaring_steps: int = 6,
    maximum_iterations: int = 8,
    tolerance: float = 0.02,
    relaxation: float = 0.75,
) -> VelocityLog:
    """Approximate ``log(Id + u)`` with symmetric residual corrections.

    The operation is accepted only for a positive-Jacobian input embedding. The
    returned reconstruction error makes the stationary-flow approximation
    falsifiable on every case; it is not assumed to be the velocity used by the
    registration optimizer or a physical time derivative.
    """
    displacement_mm = np.asarray(displacement_mm, dtype=np.float32)
    _validate_vector(displacement_mm, spacing_mm)
    if maximum_iterations < 1 or not 0 < tolerance < 1 or not 0 < relaxation <= 1:
        raise ValueError("Invalid stationary-velocity logarithm settings")
    displacement_jacobian = jacobian_determinant(displacement_mm, spacing_mm)
    minimum_jacobian = float(np.min(displacement_jacobian))
    if minimum_jacobian <= 0:
        raise ValueError(
            "A stationary-velocity logarithm requires a positive-Jacobian "
            f"embedding; minimum Jacobian is {minimum_jacobian:.6g}"
        )

    velocity = displacement_mm.copy()
    reconstruction = exponentiate_stationary_velocity(velocity, spacing_mm, squaring_steps)
    error = _relative_rmse(displacement_mm, reconstruction)
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        if error <= tolerance:
            break
        inverse = exponentiate_stationary_velocity(-velocity, spacing_mm, squaring_steps)
        right_residual = compose_displacements(displacement_mm, inverse, spacing_mm)
        left_residual = compose_displacements(inverse, displacement_mm, spacing_mm)
        velocity += relaxation * 0.5 * (right_residual + left_residual)
        reconstruction = exponentiate_stationary_velocity(velocity, spacing_mm, squaring_steps)
        error = _relative_rmse(displacement_mm, reconstruction)
    reconstruction_jacobian = jacobian_determinant(reconstruction, spacing_mm)
    return VelocityLog(
        velocity_mm=np.asarray(velocity, dtype=np.float32),
        reconstructed_displacement_mm=reconstruction,
        iterations=iterations,
        relative_reconstruction_rmse=error,
        displacement_minimum_jacobian=minimum_jacobian,
        reconstruction_minimum_jacobian=float(np.min(reconstruction_jacobian)),
    )
