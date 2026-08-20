from __future__ import annotations

import numpy as np

from arc_deformation.hodge import fourier_hodge_components
from arc_deformation.velocity import (
    exponentiate_stationary_velocity,
    logarithm_stationary_velocity,
)


def test_stationary_velocity_log_reconstructs_smooth_diffeomorphism() -> None:
    shape = (17, 15, 13)
    x, y, z = np.indices(shape, dtype=np.float32)
    velocity = np.zeros((*shape, 3), dtype=np.float32)
    velocity[..., 0] = 0.18 * np.sin(np.pi * x / (shape[0] - 1))
    velocity[..., 1] = 0.12 * np.sin(np.pi * y / (shape[1] - 1))
    velocity[..., 2] = 0.08 * np.sin(np.pi * z / (shape[2] - 1))
    displacement = exponentiate_stationary_velocity(velocity, (1.0, 1.0, 1.0), 5)
    recovered = logarithm_stationary_velocity(
        displacement,
        (1.0, 1.0, 1.0),
        squaring_steps=5,
        maximum_iterations=6,
        tolerance=0.005,
    )
    assert recovered.relative_reconstruction_rmse < 0.005
    assert recovered.displacement_minimum_jacobian > 0
    assert recovered.reconstruction_minimum_jacobian > 0


def test_periodic_hodge_separates_parallel_and_perpendicular_modes() -> None:
    shape = (24, 20, 16)
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    parallel = np.zeros((*shape, 3), dtype=np.float32)
    parallel[..., 0] = np.sin(2 * np.pi * x / shape[0])
    curl_free, divergence_free, harmonic = fourier_hodge_components(parallel, (1.0, 1.0, 1.0))
    assert np.sum(curl_free**2) / np.sum(parallel**2) > 0.999
    assert np.sum(divergence_free**2) / np.sum(parallel**2) < 1e-6
    assert np.sum(harmonic**2) / np.sum(parallel**2) < 1e-6

    perpendicular = np.zeros((*shape, 3), dtype=np.float32)
    perpendicular[..., 1] = np.sin(2 * np.pi * x / shape[0])
    curl_free, divergence_free, harmonic = fourier_hodge_components(
        perpendicular, (1.0, 1.0, 1.0)
    )
    assert np.sum(divergence_free**2) / np.sum(perpendicular**2) > 0.999
    assert np.sum(curl_free**2) / np.sum(perpendicular**2) < 1e-6
    assert np.sum(harmonic**2) / np.sum(perpendicular**2) < 1e-6
