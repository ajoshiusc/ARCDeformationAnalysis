"""Explicit, portable extraction of one lesion-associated deformation case."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from arc_deformation.field import (
    NormalizedField,
    coordinate_map_validity,
    lesion_distance_and_normals,
    lesion_effect_field,
    lesion_laterality,
    normalize_coordinate_map,
    sample_image,
)
from arc_deformation.io import atomic_csv, atomic_json
from arc_deformation.metrics import CaseMetadata, calculate_case_metrics


@dataclass(frozen=True)
class ExtractionInputs:
    case_id: str
    subject: str
    session: str
    inverse_map: Path
    subject_t1: Path
    subject_mask: Path
    atlas_t1: Path
    atlas_mask: Path
    lesion_mask: Path
    inpainting_target: Path
    output_dir: Path
    raw_inverse_map: Path | None = None
    raw_subject_t1: Path | None = None
    raw_subject_mask: Path | None = None


def same_geometry(
    first: nib.spatialimages.SpatialImage, second: nib.spatialimages.SpatialImage
) -> bool:
    return first.shape[:3] == second.shape[:3] and np.allclose(
        first.affine, second.affine, atol=1e-4, rtol=1e-5
    )


def _load_required(paths: list[Path | None]) -> None:
    missing = [str(path) for path in paths if path is None or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing extraction inputs: {missing}")


def _validate_atlas(image: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid atlas spacing: {spacing}")
    linear = image.affine[:3, :3]
    gram = linear.T @ linear
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-5):
        raise ValueError("Atlas axes must be orthogonal for midline reflection")
    return spacing


def _normalized_from_files(
    inverse_path: Path,
    subject_t1_path: Path,
    subject_mask_path: Path,
    atlas_mask: np.ndarray,
    lesion_side: str,
    spacing: tuple[float, float, float],
    smoothing_mm: float,
    fit_subsample: int,
) -> NormalizedField:
    inverse = np.asarray(nib.load(inverse_path).dataobj, dtype=np.float32)
    subject_image = nib.load(subject_t1_path)
    subject_mask_image = nib.load(subject_mask_path)
    if not same_geometry(subject_image, subject_mask_image):
        raise ValueError("Subject T1 and mask geometry differ")
    source_spacing = tuple(float(value) for value in subject_image.header.get_zooms()[:3])
    if not np.allclose(source_spacing, (1.0, 1.0, 1.0), atol=1e-3):
        raise ValueError(
            "The ARC v2 method requires the processed source coordinate grid to be 1 mm "
            f"isotropic; found {source_spacing}"
        )
    subject_mask = np.asarray(subject_mask_image.dataobj) > 0
    valid = coordinate_map_validity(inverse, subject_image.shape, subject_mask, atlas_mask)
    return normalize_coordinate_map(
        inverse,
        valid,
        lesion_side,
        spacing,
        smoothing_mm=smoothing_mm,
        fit_subsample=fit_subsample,
    )


def _save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header["descrip"] = description[:79]
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(image, temporary)
    os.replace(temporary, path)


def _make_qc(
    atlas_t1: np.ndarray,
    lesion: np.ndarray,
    target: np.ndarray,
    magnitude: np.ndarray,
    radial: np.ndarray,
    log_jacobian: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = int(np.argmax(np.sum(lesion, axis=(0, 1))))
    anatomy = np.rot90(atlas_t1[:, :, index])
    lesion_slice = np.rot90(lesion[:, :, index])
    target_slice = np.rot90(target[:, :, index])
    valid_slice = np.rot90(valid[:, :, index])
    foreground = atlas_t1[np.isfinite(atlas_t1) & (atlas_t1 > 0)]
    lower, upper = np.percentile(foreground, [1, 99]) if foreground.size else (0, 1)

    def robust_limit(volume: np.ndarray, floor: float) -> float:
        values = np.abs(volume[valid])
        values = values[np.isfinite(values)]
        return max(floor, float(np.percentile(values, 99))) if values.size else floor

    radial_limit = robust_limit(radial, 1.0)
    jacobian_limit = robust_limit(log_jacobian, 0.1)
    overlays = (
        (magnitude, "magma", 0.0, robust_limit(magnitude, 1.0)),
        (radial, "coolwarm", -radial_limit, radial_limit),
        (log_jacobian, "PuOr_r", -jacobian_limit, jacobian_limit),
    )
    titles = ("Atlas anatomy", "Magnitude (mm)", "Radial (mm)", "Log-J asymmetry")
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, title in zip(axes, titles, strict=True):
        axis.imshow(anatomy, cmap="gray", vmin=lower, vmax=upper)
        axis.contour(lesion_slice, levels=[0.5], colors="#00ffff", linewidths=1.2)
        axis.contour(
            target_slice,
            levels=[0.5],
            colors="#ffd92f",
            linewidths=0.8,
            linestyles="--",
        )
        axis.set_title(title)
        axis.axis("off")
    for axis, (volume, color_map, vmin, vmax) in zip(axes[1:], overlays, strict=True):
        values = np.rot90(volume[:, :, index])
        artist = axis.imshow(
            np.ma.masked_where(~valid_slice, values),
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
        )
        figure.colorbar(artist, ax=axis, fraction=0.046)
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def extract_case(
    inputs: ExtractionInputs,
    smoothing_mm: float = 2.0,
    fit_subsample: int = 8,
    minimum_laterality: float = 0.80,
    make_qc: bool = True,
) -> dict[str, object]:
    """Extract and write one case; all writes are confined to `output_dir`."""
    _load_required(
        [
            inputs.inverse_map,
            inputs.subject_t1,
            inputs.subject_mask,
            inputs.atlas_t1,
            inputs.atlas_mask,
            inputs.lesion_mask,
            inputs.inpainting_target,
        ]
    )
    atlas_image = nib.load(inputs.atlas_t1)
    atlas_mask_image = nib.load(inputs.atlas_mask)
    if not same_geometry(atlas_image, atlas_mask_image):
        raise ValueError("Atlas T1 and mask geometry differ")
    spacing = _validate_atlas(atlas_image)
    atlas_mask = np.asarray(atlas_mask_image.dataobj) > 0

    inverse = np.asarray(nib.load(inputs.inverse_map).dataobj, dtype=np.float32)
    subject_image = nib.load(inputs.subject_t1)
    lesion_image = nib.load(inputs.lesion_mask)
    target_image = nib.load(inputs.inpainting_target)
    if not same_geometry(subject_image, lesion_image) or not same_geometry(
        subject_image, target_image
    ):
        raise ValueError("Subject T1, lesion, and inpainting target geometry differ")
    lesion = (
        sample_image(
            (np.asarray(lesion_image.dataobj) > 0).astype(np.float32), inverse, order=0
        )
        > 0.5
    )
    target = (
        sample_image(
            (np.asarray(target_image.dataobj) > 0).astype(np.float32), inverse, order=0
        )
        > 0.5
    )
    target |= lesion
    side, laterality, left_voxels, right_voxels = lesion_laterality(lesion)

    normalized = _normalized_from_files(
        inputs.inverse_map,
        inputs.subject_t1,
        inputs.subject_mask,
        atlas_mask,
        side,
        spacing,
        smoothing_mm,
        fit_subsample,
    )
    effect, log_jacobian, valid = lesion_effect_field(normalized, target, side)
    distance, normals = lesion_distance_and_normals(lesion, spacing)
    radial = np.sum(effect * normals, axis=-1, dtype=np.float32)
    magnitude = np.linalg.norm(effect, axis=-1).astype(np.float32)
    radial[~valid] = 0
    magnitude[~valid] = 0

    sensitivity = None
    raw_paths = (inputs.raw_inverse_map, inputs.raw_subject_t1, inputs.raw_subject_mask)
    if any(path is not None for path in raw_paths):
        _load_required(list(raw_paths))
        raw = _normalized_from_files(
            inputs.raw_inverse_map,  # type: ignore[arg-type]
            inputs.raw_subject_t1,  # type: ignore[arg-type]
            inputs.raw_subject_mask,  # type: ignore[arg-type]
            atlas_mask,
            side,
            spacing,
            smoothing_mm,
            fit_subsample,
        )
        raw_effect, _, raw_valid = lesion_effect_field(raw, target, side)
        shared = valid & raw_valid
        sensitivity = np.linalg.norm(effect - raw_effect, axis=-1).astype(np.float32)
        sensitivity[~shared] = 0

    metadata = CaseMetadata(
        case_id=inputs.case_id,
        subject=inputs.subject,
        session=inputs.session,
        lesion_side=side,
        lesion_laterality_index=laterality,
        lesion_left_voxels=left_voxels,
        lesion_right_voxels=right_voxels,
    )
    voxel_volume = float(abs(np.linalg.det(atlas_image.affine[:3, :3])))
    metrics = calculate_case_metrics(
        metadata,
        lesion,
        target,
        normalized,
        effect,
        log_jacobian,
        valid,
        distance,
        radial,
        magnitude,
        voxel_volume,
        sensitivity,
        minimum_laterality,
        smoothing_mm,
    )
    case_dir = inputs.output_dir / inputs.subject / inputs.session / inputs.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    prefix = case_dir / inputs.case_id
    paths = {
        "mass_effect_vector_path": Path(f"{prefix}_deformation_atlas_mm.nii.gz"),
        "mass_effect_magnitude_path": Path(f"{prefix}_magnitude_atlas_mm.nii.gz"),
        "mass_effect_radial_path": Path(f"{prefix}_radial_atlas_mm.nii.gz"),
        "log_jacobian_asymmetry_path": Path(f"{prefix}_logjac_asymmetry.nii.gz"),
        "valid_mask_path": Path(f"{prefix}_valid_mask.nii.gz"),
        "lesion_atlas_path": Path(f"{prefix}_lesion_atlas.nii.gz"),
        "target_atlas_path": Path(f"{prefix}_excluded_target_atlas.nii.gz"),
    }
    _save_nifti(
        effect,
        atlas_image,
        paths["mass_effect_vector_path"],
        np.float32,
        "Lesional-only deformation proxy, atlas-axis mm",
    )
    _save_nifti(
        magnitude,
        atlas_image,
        paths["mass_effect_magnitude_path"],
        np.float32,
        "Lesional-only deformation magnitude, mm",
    )
    _save_nifti(
        radial,
        atlas_image,
        paths["mass_effect_radial_path"],
        np.float32,
        "Radial deformation: outward positive, mm",
    )
    _save_nifti(
        log_jacobian,
        atlas_image,
        paths["log_jacobian_asymmetry_path"],
        np.float32,
        "Lesional minus mirrored contralesional log-J",
    )
    _save_nifti(
        valid,
        atlas_image,
        paths["valid_mask_path"],
        np.uint8,
        "Valid lesional deformation proxy support",
    )
    _save_nifti(
        lesion,
        atlas_image,
        paths["lesion_atlas_path"],
        np.uint8,
        "Lesion mapped to symmetric atlas",
    )
    _save_nifti(
        target,
        atlas_image,
        paths["target_atlas_path"],
        np.uint8,
        "Excluded inpainting target mapped to atlas",
    )
    if sensitivity is not None:
        sensitivity_path = Path(f"{prefix}_registration_sensitivity_mm.nii.gz")
        _save_nifti(
            sensitivity,
            atlas_image,
            sensitivity_path,
            np.float32,
            "Inpainted minus direct registration sensitivity, mm",
        )
        metrics["registration_sensitivity_path"] = str(sensitivity_path)
    metrics.update({key: str(path) for key, path in paths.items()})
    metrics.update(
        {
            "source_inverse_map": str(inputs.inverse_map),
            "source_lesion_mask": str(inputs.lesion_mask),
            "source_inpainting_target": str(inputs.inpainting_target),
        }
    )
    if make_qc:
        qc_path = Path(f"{prefix}_qc.png")
        _make_qc(
            np.asarray(atlas_image.dataobj, dtype=np.float32),
            lesion,
            target,
            magnitude,
            radial,
            log_jacobian,
            valid,
            qc_path,
        )
        metrics["subject_qc_png"] = str(qc_path)
    atomic_json(case_dir / "mass_effect_metrics.json", metrics)
    return metrics


def collect_metrics(output_dir: Path) -> Path:
    """Rebuild a deterministic cohort manifest from completed case JSON files."""
    records: list[dict[str, object]] = []
    for path in sorted(Path(output_dir).rglob("mass_effect_metrics.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    if not records:
        raise ValueError(f"No mass_effect_metrics.json files found under {output_dir}")
    frame = pd.DataFrame(records).sort_values(["subject", "case_id"]).reset_index(drop=True)
    if frame["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs found while collecting metrics")
    manifest = Path(output_dir) / "mass_effect_manifest.csv"
    atomic_csv(manifest, frame)
    return manifest
