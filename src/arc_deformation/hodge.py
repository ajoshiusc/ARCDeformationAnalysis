"""Exploratory Helmholtz--Hodge descriptors of lesion-associated displacement."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import fft
from scipy.ndimage import distance_transform_edt, gaussian_filter

from arc_deformation.constants import HODGE_METHOD_VERSION, METHOD_VERSION
from arc_deformation.io import (
    atomic_csv,
    atomic_json,
    localize_arc_path,
    read_table,
    require_unique,
    sha256_file,
)
from arc_deformation.velocity import logarithm_stationary_velocity


@dataclass(frozen=True)
class HodgeConfig:
    """Numerical choices defining the reproducible periodic-grid projection."""

    stride: int = 4
    padding: int = 6
    boundary_taper_width_voxels: float = 4.0
    displacement_smoothing_sigma_voxels: float = 2.5
    velocity_squaring_steps: int = 6
    velocity_maximum_iterations: int = 6
    velocity_reconstruction_tolerance: float = 0.02


@dataclass(frozen=True)
class HodgeDecomposition:
    """Cropped component fields plus whole-padded-grid energy diagnostics."""

    tapered: np.ndarray
    stationary_velocity: np.ndarray
    curl_free: np.ndarray
    divergence_free: np.ndarray
    harmonic: np.ndarray
    valid: np.ndarray
    features: dict[str, float | int | str]


def _validate_inputs(
    field: np.ndarray,
    valid: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: HodgeConfig,
) -> None:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected XxYxZx3 field, got {field.shape}")
    if valid.shape != field.shape[:3]:
        raise ValueError("Hodge field and valid mask geometry differ")
    if (
        config.stride < 1
        or config.padding < 0
        or config.boundary_taper_width_voxels < 0
        or config.displacement_smoothing_sigma_voxels < 0
    ):
        raise ValueError("stride must be positive and regularization settings nonnegative")
    if config.velocity_squaring_steps < 0 or config.velocity_maximum_iterations < 1:
        raise ValueError("Invalid stationary-velocity scaling or iteration count")
    if any(not np.isfinite(value) or value <= 0 for value in spacing_mm):
        raise ValueError(f"Invalid field spacing: {spacing_mm}")
    if not np.isfinite(field[valid]).all():
        raise ValueError("Hodge field has nonfinite values on valid support")
    if not valid.any():
        raise ValueError("Hodge valid mask is empty")


def _energy(field: np.ndarray) -> float:
    return float(np.sum(np.asarray(field, dtype=np.float64) ** 2, dtype=np.float64))


def _regularized_embedding(
    field: np.ndarray,
    valid: np.ndarray,
    config: HodgeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample, taper at the valid-domain boundary, smooth, and pad."""
    sampled = field[:: config.stride, :: config.stride, :: config.stride].copy()
    mask = valid[:: config.stride, :: config.stride, :: config.stride]
    sampled[~mask] = 0
    if config.boundary_taper_width_voxels > 0:
        distance = distance_transform_edt(mask)
        normalized_distance = np.clip(distance / config.boundary_taper_width_voxels, 0, 1)
        taper = 0.5 - 0.5 * np.cos(np.pi * normalized_distance)
    else:
        taper = mask.astype(np.float32)
    tapered = sampled * taper[..., None]
    if config.displacement_smoothing_sigma_voxels > 0:
        tapered = np.stack(
            [
                gaussian_filter(
                    tapered[..., component],
                    sigma=config.displacement_smoothing_sigma_voxels,
                    mode="constant",
                )
                for component in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
    pad_width = ((config.padding, config.padding),) * 3 + ((0, 0),)
    return np.pad(tapered, pad_width, mode="constant"), mask, taper


def fourier_hodge_components(
    vector: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return curl-free, divergence-free, and harmonic periodic components."""
    vector = np.asarray(vector, dtype=np.float32)
    _validate_inputs(vector, np.ones(vector.shape[:3], dtype=bool), spacing_mm, HodgeConfig())
    transformed = fft.fftn(vector, axes=(0, 1, 2), workers=1)
    frequencies = [
        2 * np.pi * fft.fftfreq(size, d=spacing)
        for size, spacing in zip(vector.shape[:3], spacing_mm, strict=True)
    ]
    kx, ky, kz = np.meshgrid(*frequencies, indexing="ij", sparse=True)
    wave_vectors = (kx, ky, kz)
    k_squared = kx**2 + ky**2 + kz**2
    safe_k_squared = np.where(k_squared > 0, k_squared, 1.0)
    dot = sum(wave_vectors[axis] * transformed[..., axis] for axis in range(3))
    parallel_hat = np.empty_like(transformed)
    for axis in range(3):
        parallel_hat[..., axis] = np.where(
            k_squared > 0,
            wave_vectors[axis] * dot / safe_k_squared,
            0,
        )
    harmonic_hat = np.zeros_like(transformed)
    harmonic_hat[0, 0, 0, :] = transformed[0, 0, 0, :]
    perpendicular_hat = transformed - parallel_hat - harmonic_hat
    parallel = fft.ifftn(parallel_hat, axes=(0, 1, 2), workers=1).real
    perpendicular = fft.ifftn(perpendicular_hat, axes=(0, 1, 2), workers=1).real
    harmonic = fft.ifftn(harmonic_hat, axes=(0, 1, 2), workers=1).real
    return (
        parallel.astype(np.float32),
        perpendicular.astype(np.float32),
        harmonic.astype(np.float32),
    )


def periodic_hodge_decomposition(
    field: np.ndarray,
    valid: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: HodgeConfig | None = None,
) -> HodgeDecomposition:
    """Project a tapered field on a padded periodic grid.

    The log is a stationary registration parameter, not physical tissue velocity.
    Its boundary convention is explicit because Hodge components are not unique
    on a bounded anatomical domain without boundary conditions. The regularized
    embedding must have a positive Jacobian before its stationary log is accepted.
    """
    config = config or HodgeConfig()
    field = np.asarray(field, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    _validate_inputs(field, valid, spacing_mm, config)

    padded, mask, taper = _regularized_embedding(field, valid, config)
    if config.padding:
        embedding_crop = tuple(slice(config.padding, -config.padding) for _ in range(3)) + (
            slice(None),
        )
        tapered = padded[embedding_crop]
    else:
        tapered = padded

    physical_spacing = tuple(value * config.stride for value in spacing_mm)
    velocity_log = logarithm_stationary_velocity(
        padded,
        physical_spacing,
        squaring_steps=config.velocity_squaring_steps,
        maximum_iterations=config.velocity_maximum_iterations,
        tolerance=config.velocity_reconstruction_tolerance,
    )
    velocity = velocity_log.velocity_mm
    parallel, perpendicular, harmonic = fourier_hodge_components(velocity, physical_spacing)
    reconstruction = parallel + perpendicular + harmonic
    residual = velocity - reconstruction
    if config.padding:
        crop = tuple(slice(config.padding, -config.padding) for _ in range(3)) + (slice(None),)
    else:
        crop = (slice(None),) * 4
    cropped_velocity = velocity[crop]
    cropped_parallel = parallel[crop]
    cropped_perpendicular = perpendicular[crop]
    cropped_harmonic = harmonic[crop]

    total_energy = _energy(velocity)
    if total_energy <= np.finfo(float).eps:
        raise ValueError("Hodge field has negligible tapered energy")
    parallel_energy = _energy(parallel)
    perpendicular_energy = _energy(perpendicular)
    harmonic_energy = _energy(harmonic)
    valid_count = int(mask.sum())
    total_rms = float(
        np.sqrt(np.mean(np.sum(cropped_velocity[mask] ** 2, axis=-1), dtype=np.float64))
    )
    residual_rmse = float(np.sqrt(_energy(residual) / padded[..., 0].size))
    features: dict[str, float | int | str] = {
        "hodge_method_version": HODGE_METHOD_VERSION,
        "stride": config.stride,
        "padding": config.padding,
        "boundary_taper_width_voxels": config.boundary_taper_width_voxels,
        "displacement_smoothing_sigma_voxels": (config.displacement_smoothing_sigma_voxels),
        "velocity_squaring_steps": config.velocity_squaring_steps,
        "velocity_maximum_iterations": config.velocity_maximum_iterations,
        "velocity_reconstruction_tolerance": config.velocity_reconstruction_tolerance,
        "valid_downsampled_voxels": valid_count,
        "representation": "stationary_velocity_log_of_tapered_displacement",
        "total_rms_mm": total_rms,
        "curl_free_energy_fraction": parallel_energy / total_energy,
        "divergence_free_energy_fraction": perpendicular_energy / total_energy,
        "harmonic_energy_fraction": harmonic_energy / total_energy,
        "curl_free_rms_mm": float(
            np.sqrt(np.mean(np.sum(cropped_parallel[mask] ** 2, axis=-1), dtype=np.float64))
        ),
        "divergence_free_rms_mm": float(
            np.sqrt(
                np.mean(
                    np.sum(cropped_perpendicular[mask] ** 2, axis=-1),
                    dtype=np.float64,
                )
            )
        ),
        "harmonic_rms_mm": float(
            np.sqrt(np.mean(np.sum(cropped_harmonic[mask] ** 2, axis=-1), dtype=np.float64))
        ),
        "hodge_reconstruction_relative_rmse": residual_rmse / max(total_rms, 1e-12),
        "velocity_log_iterations": velocity_log.iterations,
        "velocity_reconstruction_relative_rmse": (velocity_log.relative_reconstruction_rmse),
        "displacement_minimum_jacobian": (velocity_log.displacement_minimum_jacobian),
        "velocity_reconstruction_minimum_jacobian": (
            velocity_log.reconstruction_minimum_jacobian
        ),
        "velocity_qc_pass": bool(
            velocity_log.relative_reconstruction_rmse
            <= config.velocity_reconstruction_tolerance
            and velocity_log.reconstruction_minimum_jacobian > 0
        ),
        "energy_fraction_sum": (parallel_energy + perpendicular_energy + harmonic_energy)
        / total_energy,
        "taper_mean_on_valid_domain": float(np.mean(taper[mask])),
    }
    return HodgeDecomposition(
        tapered=tapered,
        stationary_velocity=cropped_velocity,
        curl_free=cropped_parallel,
        divergence_free=cropped_perpendicular,
        harmonic=cropped_harmonic,
        valid=mask,
        features=features,
    )


def _one_case(
    record: dict[str, object], arc_root: Path, config: HodgeConfig
) -> dict[str, object]:
    vector_path = localize_arc_path(str(record["mass_effect_vector_path"]), arc_root)
    valid_path = localize_arc_path(str(record["valid_mask_path"]), arc_root)
    vector_image = nib.load(vector_path)
    valid_image = nib.load(valid_path)
    if vector_image.shape[:3] != valid_image.shape[:3] or not np.allclose(
        vector_image.affine, valid_image.affine, atol=1e-4, rtol=1e-5
    ):
        raise ValueError(f"Hodge input geometry differs for {record['case_id']}")
    field = np.asarray(vector_image.dataobj, dtype=np.float32)
    valid = np.asarray(valid_image.dataobj) > 0
    spacing = tuple(float(value) for value in vector_image.header.get_zooms()[:3])
    decomposition = periodic_hodge_decomposition(field, valid, spacing, config)
    return {
        "case_id": str(record["case_id"]),
        "subject": str(record["subject"]),
        "session": str(record["session"]),
        **decomposition.features,
    }


def run_hodge_extraction(
    mass_effect_manifest: Path,
    arc_root: Path,
    output_dir: Path,
    config: HodgeConfig | None = None,
    n_jobs: int = 1,
) -> Path:
    """Create a deterministic case-level Hodge descriptor manifest."""
    config = config or HodgeConfig()
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")
    frame = read_table(mass_effect_manifest)
    require_unique(frame, "case_id", "mass-effect manifest")
    required = {
        "case_id",
        "subject",
        "session",
        "method_version",
        "mass_effect_vector_path",
        "valid_mask_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Mass-effect manifest lacks Hodge inputs: {missing}")
    versions = sorted(frame["method_version"].dropna().astype(str).unique())
    if versions != [METHOD_VERSION]:
        raise ValueError(f"Hodge extraction requires {METHOD_VERSION!r}, found {versions}")
    records = frame.sort_values(["subject", "case_id"]).to_dict("records")
    if n_jobs == 1:
        results = [_one_case(record, Path(arc_root), config) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(
                executor.map(
                    _one_case,
                    records,
                    [Path(arc_root)] * len(records),
                    [config] * len(records),
                )
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "hodge_manifest.csv"
    result_frame = pd.DataFrame(results)
    atomic_csv(manifest, result_frame)
    summary_columns = (
        "total_rms_mm",
        "curl_free_energy_fraction",
        "divergence_free_energy_fraction",
        "harmonic_energy_fraction",
        "velocity_reconstruction_relative_rmse",
        "displacement_minimum_jacobian",
        "velocity_reconstruction_minimum_jacobian",
        "hodge_reconstruction_relative_rmse",
        "energy_fraction_sum",
    )
    summaries: dict[str, object] = {}
    for column in summary_columns:
        values = result_frame[column].to_numpy(float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries[column] = {
            "minimum": float(np.min(values)),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "maximum": float(np.max(values)),
        }
    atomic_json(
        output_dir / "hodge_summary.json",
        {
            "method_version": HODGE_METHOD_VERSION,
            "cases": len(result_frame),
            "velocity_qc_pass_cases": int(result_frame["velocity_qc_pass"].sum()),
            "summary": summaries,
        },
    )
    atomic_json(
        output_dir / "hodge_config.json",
        {
            "method_version": HODGE_METHOD_VERSION,
            "interpretation": (
                "Stationary log-velocity representation of a tapered displacement "
                "embedding followed by periodic-grid decomposition; not physical "
                "velocity, pressure, force, or tissue mechanics"
            ),
            "config": asdict(config),
            "cases": len(results),
            "input_manifest": str(Path(mass_effect_manifest).resolve()),
            "input_manifest_sha256": sha256_file(mass_effect_manifest),
        },
    )
    return manifest
