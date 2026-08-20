"""Independent ANTs registration sensitivity analysis in symmetric MNI152 space."""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from arc_deformation.constants import ANTS_METHOD_VERSION
from arc_deformation.field import (
    coordinate_map_validity,
    lesion_distance_and_normals,
    lesion_effect_field,
    lesion_laterality,
    normalize_coordinate_map,
)
from arc_deformation.io import (
    atomic_csv,
    atomic_json,
    ensure_output_outside_data,
    localize_arc_path,
    read_table,
    require_unique,
    sha256_file,
)
from arc_deformation.metrics import CaseMetadata, calculate_case_metrics


@dataclass(frozen=True)
class AntsMNIConfig:
    """Frozen registration, extraction, and registration-QC settings."""

    registration_transform: str = "antsRegistrationSyNQuickRepro[s]"
    registration_spacing_mm: float = 2.0
    random_seed: int = 2026
    single_precision: bool = True
    mask_all_stages: bool = True
    smoothing_mm: float = 2.0
    affine_fit_subsample: int = 2
    minimum_laterality: float = 0.80
    minimum_near_lesion_volume_mm3: float = 1000.0
    minimum_brain_mask_dice: float = 0.70
    maximum_cycle_rmse_mm: float = 0.50
    maximum_raw_warp_folding_fraction: float = 0.001
    cycle_sample_points: int = 2000


def _ants_module() -> Any:
    try:
        import ants
    except ImportError as error:
        raise RuntimeError(
            "ANTs analysis requires the optional dependencies: pip install -e '.[ants]'"
        ) from error
    return ants


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def validate_symmetric_mni_geometry(template_path: Path, mask_path: Path) -> dict[str, object]:
    """Reject templates for which an axis-0 array flip is not x=0 reflection."""
    template = nib.load(template_path)
    mask_image = nib.load(mask_path)
    if template.shape[:3] != mask_image.shape[:3] or not np.allclose(
        template.affine, mask_image.affine, atol=1e-5, rtol=1e-6
    ):
        raise ValueError("MNI152 template and brain-mask geometry differ")
    orientation = nib.aff2axcodes(template.affine)
    if orientation != ("R", "A", "S"):
        raise ValueError(f"Expected canonical RAS MNI152 geometry, found {orientation}")
    if template.shape[0] % 2 != 1:
        raise ValueError("Symmetric MNI152 first dimension must be odd")
    linear = template.affine[:3, :3]
    gram = linear.T @ linear
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-5):
        raise ValueError("MNI152 axes must be orthogonal")
    midpoint = (template.shape[0] - 1) // 2
    center = np.array([midpoint, (template.shape[1] - 1) / 2, (template.shape[2] - 1) / 2])
    center_world = nib.affines.apply_affine(template.affine, center)
    if not np.isclose(center_world[0], 0.0, atol=1e-4):
        raise ValueError(f"Array midpoint is not x=0: x={center_world[0]:.6g} mm")
    endpoints = nib.affines.apply_affine(
        template.affine,
        np.array([[0, center[1], center[2]], [template.shape[0] - 1, center[1], center[2]]]),
    )
    if not np.allclose(endpoints[0, 0], -endpoints[1, 0], atol=1e-4):
        raise ValueError("Axis-0 array flip is not reflection through x=0")
    mask = np.asarray(mask_image.dataobj) > 0
    denominator = int(mask.sum() + np.flip(mask, axis=0).sum())
    symmetry_dice = float(2 * np.count_nonzero(mask & np.flip(mask, axis=0)) / denominator)
    if symmetry_dice < 0.999:
        raise ValueError(f"MNI152 brain mask is not symmetric: Dice={symmetry_dice:.6f}")
    return {
        "shape": list(template.shape[:3]),
        "spacing_mm": [float(value) for value in template.header.get_zooms()[:3]],
        "orientation": "".join(orientation),
        "midline_world_x_mm": float(center_world[0]),
        "brain_mask_reflection_dice": symmetry_dice,
    }


def _nib_affine_from_ants(image: Any) -> np.ndarray:
    direction = np.asarray(image.direction, dtype=float)
    spacing = np.asarray(image.spacing, dtype=float)
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = np.asarray(image.origin, dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def _indices_to_lps(image: Any, indices: np.ndarray) -> np.ndarray:
    direction = np.asarray(image.direction, dtype=float)
    scaled = indices.astype(float) * np.asarray(image.spacing, dtype=float)
    return np.asarray(image.origin, dtype=float) + scaled @ direction.T


def _lps_to_indices(image: Any, points: np.ndarray) -> np.ndarray:
    direction = np.asarray(image.direction, dtype=float)
    relative = points.astype(float) - np.asarray(image.origin, dtype=float)
    return (relative @ direction) / np.asarray(image.spacing, dtype=float)


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine, header)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(image, temporary)
    os.replace(temporary, path)


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first) > 0
    second = np.asarray(second) > 0
    denominator = int(first.sum() + second.sum())
    return float(2 * np.count_nonzero(first & second) / denominator) if denominator else 1.0


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=float)[mask]
    second_values = np.asarray(second, dtype=float)[mask]
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    if finite.sum() < 2:
        return float("nan")
    return float(np.corrcoef(first_values[finite], second_values[finite])[0, 1])


def _coordinate_map_and_cycle(
    fixed: Any,
    moving: Any,
    fixed_mask: np.ndarray,
    forward_transforms: list[str],
    inverse_transforms: list[str],
    sample_points: int,
) -> tuple[np.ndarray, float, float, int]:
    ants = _ants_module()
    indices = np.argwhere(fixed_mask)
    fixed_lps = _indices_to_lps(fixed, indices)
    points = pd.DataFrame(fixed_lps, columns=["x", "y", "z"])
    mapped = ants.apply_transforms_to_points(3, points, forward_transforms)
    moving_lps = mapped[["x", "y", "z"]].to_numpy(float)
    coordinate_map = np.full((*fixed.shape, 3), np.nan, dtype=np.float32)
    coordinate_map[tuple(indices.T)] = _lps_to_indices(moving, moving_lps).astype(np.float32)

    step = max(1, len(points) // sample_points)
    sampled_moving = mapped.iloc[::step].reset_index(drop=True)
    roundtrip = ants.apply_transforms_to_points(3, sampled_moving, inverse_transforms)
    fixed_sample = fixed_lps[::step]
    difference = roundtrip[["x", "y", "z"]].to_numpy(float) - fixed_sample
    error = np.linalg.norm(difference, axis=1)
    return (
        coordinate_map,
        float(np.sqrt(np.mean(error**2))),
        float(np.max(error)),
        len(error),
    )


def _case_paths(output_dir: Path, record: dict[str, object]) -> tuple[Path, Path]:
    case_dir = (
        output_dir / str(record["subject"]) / str(record["session"]) / str(record["case_id"])
    )
    return case_dir, case_dir / "ants_metrics.json"


def select_case_frame(inpainting: pd.DataFrame, selection: pd.DataFrame | None) -> pd.DataFrame:
    """Apply an exact, unique case-ID selection to a larger acquisition manifest."""
    require_unique(inpainting, "case_id", "inpainting manifest")
    if selection is None:
        return inpainting.copy()
    require_unique(selection, "case_id", "case-selection manifest")
    selected_ids = set(selection["case_id"].astype(str))
    available_ids = set(inpainting["case_id"].astype(str))
    missing_selected = sorted(selected_ids - available_ids)
    if missing_selected:
        raise ValueError(
            "Case-selection manifest contains IDs absent from the inpainting "
            f"manifest, including {missing_selected[:5]}"
        )
    result = inpainting.loc[inpainting["case_id"].astype(str).isin(selected_ids)].copy()
    if len(result) != len(selection):
        raise RuntimeError("Case selection did not produce an exact one-to-one cohort")
    return result


def extract_ants_mni_case(
    record: dict[str, object],
    arc_root: Path,
    template_path: Path,
    template_mask_path: Path,
    output_dir: Path,
    config: AntsMNIConfig,
) -> dict[str, object]:
    """Register one lesion-filled ARC image bidirectionally and extract descriptors."""
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ["ANTS_RANDOM_SEED"] = str(config.random_seed)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-ants-matplotlib")
    ants = _ants_module()

    case_dir, metrics_path = _case_paths(output_dir, record)
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("method_version") == ANTS_METHOD_VERSION:
            return existing
    case_dir.mkdir(parents=True, exist_ok=True)

    moving_path = localize_arc_path(str(record["brain_inpainted_t1_mni"]), arc_root)
    moving_mask_path = localize_arc_path(str(record["skullstrip_mask_mni"]), arc_root)
    lesion_path = localize_arc_path(str(record["lesion_mask_mni"]), arc_root)
    target_path = localize_arc_path(str(record["dilated_lesion_mask_mni"]), arc_root)

    fixed_native = ants.image_read(str(template_path))
    fixed_mask_native = ants.image_read(str(template_mask_path))
    moving_native = ants.image_read(str(moving_path))
    moving_mask_native = ants.image_read(str(moving_mask_path))
    lesion_native = ants.image_read(str(lesion_path))
    target_native = ants.image_read(str(target_path))
    spacing = (config.registration_spacing_mm,) * 3
    fixed = ants.resample_image(fixed_native, spacing, use_voxels=False, interp_type=0)
    fixed_mask_image = ants.resample_image(
        fixed_mask_native, spacing, use_voxels=False, interp_type=1
    )
    moving = ants.resample_image(moving_native, spacing, use_voxels=False, interp_type=0)
    moving_mask_image = ants.resample_image(
        moving_mask_native, spacing, use_voxels=False, interp_type=1
    )
    fixed_mask = fixed_mask_image.numpy() > 0
    moving_mask = moving_mask_image.numpy() > 0
    fixed = fixed * (fixed_mask_image > 0)
    moving = moving * (moving_mask_image > 0)

    started = time.monotonic()
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=config.registration_transform,
        mask=fixed_mask_image,
        moving_mask=moving_mask_image,
        mask_all_stages=config.mask_all_stages,
        random_seed=config.random_seed,
        singleprecision=config.single_precision,
        outprefix=str(case_dir / "ants_"),
        verbose=False,
    )
    elapsed = time.monotonic() - started
    forward = [str(path) for path in registration["fwdtransforms"]]
    inverse = [str(path) for path in registration["invtransforms"]]

    warped_mask_image = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask_native,
        transformlist=forward,
        interpolator="genericLabel",
        singleprecision=config.single_precision,
    )
    lesion_image = ants.apply_transforms(
        fixed=fixed,
        moving=lesion_native,
        transformlist=forward,
        interpolator="genericLabel",
        singleprecision=config.single_precision,
    )
    target_image = ants.apply_transforms(
        fixed=fixed,
        moving=target_native,
        transformlist=forward,
        interpolator="genericLabel",
        singleprecision=config.single_precision,
    )
    lesion = lesion_image.numpy() > 0
    target = (target_image.numpy() > 0) | lesion
    side, laterality, left_voxels, right_voxels = lesion_laterality(lesion)

    coordinate_map, cycle_rmse, cycle_maximum, cycle_points = _coordinate_map_and_cycle(
        fixed,
        moving,
        fixed_mask,
        forward,
        inverse,
        config.cycle_sample_points,
    )
    valid = coordinate_map_validity(coordinate_map, moving.shape, moving_mask, fixed_mask)
    normalized = normalize_coordinate_map(
        coordinate_map,
        valid,
        side,
        tuple(float(value) for value in fixed.spacing),
        source_spacing_mm=tuple(float(value) for value in moving.spacing),
        smoothing_mm=config.smoothing_mm,
        fit_subsample=config.affine_fit_subsample,
    )
    effect, log_jacobian, effect_valid = lesion_effect_field(normalized, target, side)
    distance, normals = lesion_distance_and_normals(
        lesion, tuple(float(value) for value in fixed.spacing)
    )
    radial = np.sum(effect * normals, axis=-1, dtype=np.float32)
    magnitude = np.linalg.norm(effect, axis=-1).astype(np.float32)
    radial[~effect_valid] = 0
    magnitude[~effect_valid] = 0

    warped_mask = warped_mask_image.numpy() > 0
    overlap = _dice(fixed_mask, warped_mask)
    correlation_mask = fixed_mask & warped_mask
    intensity_correlation = _correlation(
        fixed.numpy(), registration["warpedmovout"].numpy(), correlation_mask
    )
    raw_jacobian = ants.create_jacobian_determinant_image(
        fixed, forward[0], do_log=False, geom=False
    ).numpy()
    raw_folding_fraction = float(np.mean(raw_jacobian[fixed_mask] <= 0))
    raw_minimum_jacobian = float(np.min(raw_jacobian[fixed_mask]))
    registration_qc = bool(
        overlap >= config.minimum_brain_mask_dice
        and cycle_rmse <= config.maximum_cycle_rmse_mm
        and raw_folding_fraction <= config.maximum_raw_warp_folding_fraction
    )

    metadata = CaseMetadata(
        case_id=str(record["case_id"]),
        subject=str(record["subject"]),
        session=str(record["session"]),
        lesion_side=side,
        lesion_laterality_index=laterality,
        lesion_left_voxels=left_voxels,
        lesion_right_voxels=right_voxels,
    )
    voxel_volume = float(np.prod(fixed.spacing))
    metrics = calculate_case_metrics(
        metadata,
        lesion,
        target,
        normalized,
        effect,
        log_jacobian,
        effect_valid,
        distance,
        radial,
        magnitude,
        voxel_volume,
        minimum_laterality=config.minimum_laterality,
        smoothing_mm=config.smoothing_mm,
        method_version=ANTS_METHOD_VERSION,
        minimum_near_lesion_volume_mm3=config.minimum_near_lesion_volume_mm3,
    )
    metrics["registration_qc_pass"] = registration_qc
    metrics["deformation_qc_pass"] = bool(metrics["deformation_qc_pass"] and registration_qc)
    metrics.update(
        {
            "registration_backend": "ANTsPy/ANTs",
            "registration_transform": config.registration_transform,
            "registration_spacing_mm": config.registration_spacing_mm,
            "registration_elapsed_seconds": elapsed,
            "registration_random_seed": config.random_seed,
            "registration_single_threaded": True,
            "registration_single_precision": config.single_precision,
            "registration_mask_all_stages": config.mask_all_stages,
            "registration_brain_mask_dice": overlap,
            "registration_intensity_pearson_r": intensity_correlation,
            "registration_cycle_rmse_mm": cycle_rmse,
            "registration_cycle_maximum_mm": cycle_maximum,
            "registration_cycle_sample_points": cycle_points,
            "registration_raw_warp_minimum_jacobian": raw_minimum_jacobian,
            "registration_raw_warp_folding_fraction": raw_folding_fraction,
            "registration_qc_criteria": {
                "brain_mask_dice_minimum": config.minimum_brain_mask_dice,
                "cycle_rmse_mm_maximum": config.maximum_cycle_rmse_mm,
                "raw_warp_folding_fraction_maximum": (config.maximum_raw_warp_folding_fraction),
            },
            "ants_point_transform_semantics": (
                "forward transforms map fixed MNI points to moving subject points; "
                "inverse transforms map moving points to fixed points"
            ),
            "source_inpainted_t1": str(moving_path),
            "source_brain_mask": str(moving_mask_path),
            "source_lesion_mask": str(lesion_path),
            "source_inpainting_target": str(target_path),
        }
    )

    affine = _nib_affine_from_ants(fixed)
    prefix = case_dir / str(record["case_id"])
    output_paths = {
        "mass_effect_vector_path": Path(f"{prefix}_ants_deformation_mni152_mm.nii.gz"),
        "mass_effect_magnitude_path": Path(f"{prefix}_ants_magnitude_mni152_mm.nii.gz"),
        "mass_effect_radial_path": Path(f"{prefix}_ants_radial_mni152_mm.nii.gz"),
        "log_jacobian_asymmetry_path": Path(f"{prefix}_ants_logjac_asymmetry.nii.gz"),
        "valid_mask_path": Path(f"{prefix}_ants_valid_mask.nii.gz"),
        "lesion_atlas_path": Path(f"{prefix}_ants_lesion_mni152.nii.gz"),
        "target_atlas_path": Path(f"{prefix}_ants_excluded_target_mni152.nii.gz"),
        "subject_to_mni152_path": Path(f"{prefix}_subject_to_mni152.nii.gz"),
        "mni152_to_subject_path": Path(f"{prefix}_mni152_to_subject.nii.gz"),
    }
    _save_nifti(effect, affine, output_paths["mass_effect_vector_path"], np.float32)
    _save_nifti(magnitude, affine, output_paths["mass_effect_magnitude_path"], np.float32)
    _save_nifti(radial, affine, output_paths["mass_effect_radial_path"], np.float32)
    _save_nifti(log_jacobian, affine, output_paths["log_jacobian_asymmetry_path"], np.float32)
    _save_nifti(effect_valid, affine, output_paths["valid_mask_path"], np.uint8)
    _save_nifti(lesion, affine, output_paths["lesion_atlas_path"], np.uint8)
    _save_nifti(target, affine, output_paths["target_atlas_path"], np.uint8)
    ants.image_write(registration["warpedmovout"], str(output_paths["subject_to_mni152_path"]))
    ants.image_write(registration["warpedfixout"], str(output_paths["mni152_to_subject_path"]))
    metrics.update({key: str(path) for key, path in output_paths.items()})
    metrics["ants_forward_transforms"] = forward
    metrics["ants_inverse_transforms"] = inverse
    atomic_json(metrics_path, metrics)
    return metrics


def run_ants_mni_cohort(
    inpainting_manifest: Path,
    arc_root: Path,
    template_path: Path,
    template_mask_path: Path,
    output_dir: Path,
    config: AntsMNIConfig | None = None,
    n_jobs: int = 1,
    selection_manifest: Path | None = None,
) -> Path:
    """Run or resume the independent ANTs analysis and write an aggregate manifest."""
    config = config or AntsMNIConfig()
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")
    output_dir = ensure_output_outside_data(output_dir, arc_root)
    geometry = validate_symmetric_mni_geometry(template_path, template_mask_path)
    request: dict[str, object] = {
        "method_version": ANTS_METHOD_VERSION,
        "config": asdict(config),
        "template_geometry": geometry,
        "template_path": str(Path(template_path).resolve()),
        "template_sha256": sha256_file(template_path),
        "template_mask_path": str(Path(template_mask_path).resolve()),
        "template_mask_sha256": sha256_file(template_mask_path),
        "inpainting_manifest": str(Path(inpainting_manifest).resolve()),
        "inpainting_manifest_sha256": sha256_file(inpainting_manifest),
        "selection_manifest": (
            str(Path(selection_manifest).resolve()) if selection_manifest else None
        ),
        "selection_manifest_sha256": (
            sha256_file(selection_manifest) if selection_manifest else None
        ),
        "software": {
            "python": platform.python_version(),
            "antspyx": _package_version("antspyx"),
            "templateflow": _package_version("templateflow"),
            "nibabel": _package_version("nibabel"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
        },
    }
    request_path = output_dir / "ants_mni_request.json"
    if request_path.is_file():
        with request_path.open(encoding="utf-8") as handle:
            previous_request = json.load(handle)
        if previous_request != request:
            raise ValueError(
                "Refusing to mix an existing ANTs output directory with a different "
                "template, input manifest, method version, or numerical configuration"
            )
    else:
        atomic_json(request_path, request)
    frame = select_case_frame(
        read_table(inpainting_manifest),
        read_table(selection_manifest) if selection_manifest is not None else None,
    )
    required = {
        "case_id",
        "subject",
        "session",
        "brain_inpainted_t1_mni",
        "skullstrip_mask_mni",
        "lesion_mask_mni",
        "dilated_lesion_mask_mni",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Inpainting manifest lacks ANTs inputs: {missing}")
    records = frame.sort_values(["subject", "case_id"]).to_dict("records")
    arguments = (
        records,
        [Path(arc_root)] * len(records),
        [Path(template_path)] * len(records),
        [Path(template_mask_path)] * len(records),
        [Path(output_dir)] * len(records),
        [config] * len(records),
    )
    if n_jobs == 1:
        results = [
            extract_ants_mni_case(record, *values)
            for record, *values in zip(*arguments, strict=True)
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(extract_ants_mni_case, *arguments))
    result_frame = (
        pd.DataFrame(results).sort_values(["subject", "case_id"]).reset_index(drop=True)
    )
    manifest = output_dir / "mass_effect_manifest.csv"
    atomic_csv(manifest, result_frame)
    atomic_json(
        output_dir / "ants_mni_config.json",
        {
            **request,
            "cases": len(result_frame),
            "registration_qc_pass_cases": int(result_frame["registration_qc_pass"].sum()),
            "deformation_qc_pass_cases": int(result_frame["deformation_qc_pass"].sum()),
            "reproducibility": (
                "fixed random seed; one ITK thread per registration; deterministic "
                "case ordering; process-level parallelism only"
            ),
        },
    )
    return manifest


def copy_templateflow_reference(output_dir: Path) -> tuple[Path, Path]:
    """Resolve and copy the exact TemplateFlow MNI152NLin2009aSym references."""
    try:
        from templateflow.api import get
    except ImportError as error:
        raise RuntimeError("TemplateFlow is required: pip install -e '.[ants]'") from error
    template = Path(str(get("MNI152NLin2009aSym", resolution=1, suffix="T1w")))
    mask = Path(str(get("MNI152NLin2009aSym", resolution=1, desc="brain", suffix="mask")))
    reference_dir = Path(output_dir) / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    copied_template = reference_dir / template.name
    copied_mask = reference_dir / mask.name
    shutil.copy2(template, copied_template)
    shutil.copy2(mask, copied_mask)
    return copied_template, copied_mask
