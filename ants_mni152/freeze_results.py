"""Freeze only de-identified aggregate ANTs outputs for the public repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arc_deformation.io import atomic_csv, atomic_json, read_table, sha256_file

QC_METRICS = {
    "Brain-mask Dice": "registration_brain_mask_dice",
    "Cycle RMSE (mm)": "registration_cycle_rmse_mm",
    "Cycle maximum (mm)": "registration_cycle_maximum_mm",
    "Intensity Pearson r": "registration_intensity_pearson_r",
    "Raw minimum Jacobian": "registration_raw_warp_minimum_jacobian",
    "Raw folding fraction": "registration_raw_warp_folding_fraction",
    "Normalized-field folding fraction": "normalized_field_folding_fraction",
    "Affine-fit RMSE (mm)": "contralateral_affine_fit_rmse_mm",
}

MODEL_FILES = (
    "model_summary.csv",
    "paired_comparisons.csv",
    "model_mae_inference.csv",
    "deformation_associations.csv",
    "adjusted_deformation_associations.csv",
    "left_only_model_summary.csv",
    "left_only_paired_comparisons.csv",
    "coefficient_summary.csv",
    "cohort_summary.json",
    "metrics_by_repeat.csv",
)


def _copy_files(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, destination / name)


def _public_provenance(value: Any) -> Any:
    """Remove machine-local paths while retaining file identities and settings."""
    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, item in value.items():
            local_path = isinstance(item, str) and Path(item).is_absolute()
            if (
                key == "path"
                or key.endswith("_path")
                or key.endswith("_manifest")
                or local_path
            ):
                if item is not None:
                    public[f"{key}_basename"] = Path(str(item)).name
                continue
            public[key] = _public_provenance(item)
        return public
    if isinstance(value, list):
        return [_public_provenance(item) for item in value]
    return value


def _copy_public_json(source: Path, destination: Path) -> None:
    """Copy JSON after stripping nonportable local filesystem locations."""
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    atomic_json(destination, _public_provenance(payload))


def freeze(private_root: Path, reference_dir: Path) -> None:
    """Create a reviewable public snapshot without case identifiers or paths."""
    registration_dir = private_root / "registration"
    manifest = read_table(registration_dir / "mass_effect_manifest.csv")
    reference_dir.mkdir(parents=True, exist_ok=True)
    registration_reference = reference_dir / "registration"
    registration_reference.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    histograms: dict[str, object] = {}
    for label, column in QC_METRICS.items():
        values = pd.to_numeric(manifest[column], errors="coerce").dropna().to_numpy(float)
        minimum, q1, median, q3, maximum = np.percentile(values, [0, 25, 50, 75, 100])
        summary_rows.append(
            {
                "metric": label,
                "column": column,
                "n": len(values),
                "minimum": float(minimum),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "maximum": float(maximum),
            }
        )
        counts, edges = np.histogram(values, bins=20)
        histograms[column] = {
            "counts": counts.astype(int).tolist(),
            "edges": edges.astype(float).tolist(),
        }
    atomic_csv(
        registration_reference / "registration_qc_summary.csv", pd.DataFrame(summary_rows)
    )
    atomic_json(registration_reference / "registration_qc_histograms.json", histograms)
    atomic_json(
        registration_reference / "registration_flow.json",
        {
            "manifest_cases": len(manifest),
            "registration_qc_pass_cases": int(manifest["registration_qc_pass"].sum()),
            "deformation_qc_pass_cases": int(manifest["deformation_qc_pass"].sum()),
            "left_dominant_cases": int(manifest["lesion_side"].eq("left").sum()),
            "right_dominant_cases": int(manifest["lesion_side"].eq("right").sum()),
            "participant_level_rows_retained": False,
        },
    )
    _copy_public_json(
        registration_dir / "ants_mni_config.json",
        registration_reference / "ants_mni_config.json",
    )

    hodge_reference = reference_dir / "hodge"
    hodge_reference.mkdir(parents=True, exist_ok=True)
    shutil.copy2(private_root / "hodge" / "hodge_summary.json", hodge_reference)
    _copy_public_json(
        private_root / "hodge" / "hodge_config.json",
        hodge_reference / "hodge_config.json",
    )
    _copy_files(private_root / "model", reference_dir / "model", MODEL_FILES)
    _copy_public_json(
        private_root / "model" / "analysis_config.json",
        reference_dir / "model" / "analysis_config.json",
    )
    _copy_files(
        private_root / "comparison",
        reference_dir / "comparison",
        (
            "descriptor_agreement.csv",
            "predictive_method_comparison.csv",
        ),
    )
    _copy_public_json(
        private_root / "comparison" / "comparison_provenance.json",
        reference_dir / "comparison" / "comparison_provenance.json",
    )
    reproduction_check = private_root / "reproduction_check.json"
    if reproduction_check.is_file():
        shutil.copy2(reproduction_check, reference_dir / reproduction_check.name)

    public_files = sorted(
        path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.name != "analysis_provenance.json"
    )
    atomic_json(
        reference_dir / "analysis_provenance.json",
        {
            "freeze_policy": (
                "No case identifiers, participant rows, image paths, transforms, joined "
                "design data, fold coefficients, or out-of-fold predictions are retained"
            ),
            "private_input_hashes": {
                "registration_manifest_sha256": sha256_file(
                    registration_dir / "mass_effect_manifest.csv"
                ),
                "hodge_manifest_sha256": sha256_file(
                    private_root / "hodge" / "hodge_manifest.csv"
                ),
                "predictions_sha256": sha256_file(
                    private_root / "model" / "aq_mass_effect_predictions_long.csv"
                ),
            },
            "public_output_hashes": {
                str(path.relative_to(reference_dir)): sha256_file(path) for path in public_files
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.private_root, args.reference_dir)


if __name__ == "__main__":
    main()
