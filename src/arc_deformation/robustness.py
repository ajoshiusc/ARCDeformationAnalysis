"""Parameter-sensitivity analysis for stationary log-velocity Hodge descriptors."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from arc_deformation.constants import HODGE_METHOD_VERSION
from arc_deformation.hodge import HodgeConfig, extract_hodge_case
from arc_deformation.io import (
    atomic_csv,
    atomic_json,
    read_table,
    require_unique,
    sha256_file,
    truthy,
)
from arc_deformation.modeling import ModelConfig, build_design

SENSITIVITY_METHOD_VERSION = "hodge_parameter_sensitivity_v1"
SENSITIVITY_FEATURES = (
    "total_rms_mm",
    "curl_free_energy_fraction",
    "divergence_free_energy_fraction",
)


@dataclass(frozen=True)
class HodgeSensitivityVariant:
    """A one-factor perturbation around the primary Hodge configuration."""

    name: str
    rationale: str
    config: HodgeConfig


def default_sensitivity_variants() -> tuple[HodgeSensitivityVariant, ...]:
    """Return fixed, physically interpretable one-factor sensitivity settings."""
    primary = HodgeConfig()
    return (
        HodgeSensitivityVariant(
            "smoothing_8mm",
            "Gaussian smoothing sigma reduced from 10 to 8 mm",
            replace(primary, displacement_smoothing_sigma_voxels=2.0),
        ),
        HodgeSensitivityVariant(
            "smoothing_12mm",
            "Gaussian smoothing sigma increased from 10 to 12 mm",
            replace(primary, displacement_smoothing_sigma_voxels=3.0),
        ),
        HodgeSensitivityVariant(
            "taper_12mm",
            "Raised-cosine boundary taper reduced from 16 to 12 mm",
            replace(primary, boundary_taper_width_voxels=3.0),
        ),
        HodgeSensitivityVariant(
            "taper_20mm",
            "Raised-cosine boundary taper increased from 16 to 20 mm",
            replace(primary, boundary_taper_width_voxels=5.0),
        ),
        HodgeSensitivityVariant(
            "padding_16mm",
            "Periodic-grid zero padding reduced from 24 to 16 mm",
            replace(primary, padding=4),
        ),
        HodgeSensitivityVariant(
            "padding_32mm",
            "Periodic-grid zero padding increased from 24 to 32 mm",
            replace(primary, padding=8),
        ),
        HodgeSensitivityVariant(
            "grid_3mm",
            "Analysis grid reduced from 4 to 3 mm while preserving physical regularization",
            HodgeConfig(
                stride=3,
                padding=8,
                boundary_taper_width_voxels=16 / 3,
                displacement_smoothing_sigma_voxels=10 / 3,
            ),
        ),
        HodgeSensitivityVariant(
            "grid_5mm",
            "Analysis grid increased from 4 to 5 mm while preserving physical regularization",
            HodgeConfig(
                stride=5,
                padding=5,
                boundary_taper_width_voxels=16 / 5,
                displacement_smoothing_sigma_voxels=2.0,
            ),
        ),
    )


def compare_hodge_variant(
    primary_design: pd.DataFrame,
    variant: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    """Compare one case-level variant with primary descriptors on shared QC cases."""
    primary_columns = [
        "case_id",
        "wab_aq",
        *(f"hhd_{feature}" for feature in SENSITIVITY_FEATURES),
    ]
    required_primary = set(primary_columns)
    required_variant = {"case_id", "velocity_qc_pass"} | set(SENSITIVITY_FEATURES)
    if missing := sorted(required_primary - set(primary_design.columns)):
        raise ValueError(f"Primary design lacks sensitivity fields: {missing}")
    if missing := sorted(required_variant - set(variant.columns)):
        raise ValueError(f"Variant manifest lacks sensitivity fields: {missing}")
    eligible = variant.loc[
        truthy(variant["velocity_qc_pass"]), ["case_id", *SENSITIVITY_FEATURES]
    ]
    joined = primary_design[primary_columns].merge(
        eligible,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for feature in SENSITIVITY_FEATURES:
        primary_values = pd.to_numeric(joined[f"hhd_{feature}"], errors="coerce")
        variant_values = pd.to_numeric(joined[feature], errors="coerce")
        outcome = pd.to_numeric(joined["wab_aq"], errors="coerce")
        finite = primary_values.notna() & variant_values.notna() & outcome.notna()
        primary_array = primary_values.loc[finite].to_numpy(float)
        variant_array = variant_values.loc[finite].to_numpy(float)
        outcome_array = outcome.loc[finite].to_numpy(float)
        rows.append(
            {
                "variant": variant_name,
                "feature": feature,
                "common_analysis_cases": int(finite.sum()),
                "spearman_vs_primary": float(spearmanr(primary_array, variant_array).statistic),
                "median_absolute_difference": float(
                    np.median(np.abs(variant_array - primary_array))
                ),
                "variant_spearman_vs_aq": float(
                    spearmanr(variant_array, outcome_array).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_sensitivity_case(
    record: dict[str, object], arc_root: Path, config: HodgeConfig
) -> dict[str, object]:
    """Extract one variant while retaining explicit numerical QC failures."""
    try:
        result = extract_hodge_case(record, arc_root, config)
    except ValueError as error:
        if "positive-Jacobian embedding" not in str(error):
            raise
        return {
            "case_id": str(record["case_id"]),
            "subject": str(record["subject"]),
            "session": str(record["session"]),
            "hodge_method_version": HODGE_METHOD_VERSION,
            "velocity_qc_pass": False,
            "sensitivity_failure_reason": "nonpositive_input_jacobian",
            **{feature: np.nan for feature in SENSITIVITY_FEATURES},
        }
    result["sensitivity_failure_reason"] = (
        "" if bool(result["velocity_qc_pass"]) else "reconstruction_qc_failed"
    )
    return result


def _run_sensitivity_variant(
    records: list[dict[str, object]],
    arc_root: Path,
    variant: HodgeSensitivityVariant,
    output_dir: Path,
    n_jobs: int,
) -> Path:
    """Run one variant without aborting the cohort at an expected QC failure."""
    if n_jobs == 1:
        results = [
            _safe_sensitivity_case(record, arc_root, variant.config) for record in records
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(
                executor.map(
                    _safe_sensitivity_case,
                    records,
                    [arc_root] * len(records),
                    [variant.config] * len(records),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    manifest = output_dir / "hodge_manifest.csv"
    atomic_csv(manifest, frame)
    failures = (
        frame["sensitivity_failure_reason"]
        .replace("", "none")
        .value_counts(dropna=False)
        .to_dict()
    )
    atomic_json(
        output_dir / "variant_summary.json",
        {
            "method_version": SENSITIVITY_METHOD_VERSION,
            "variant": variant.name,
            "rationale": variant.rationale,
            "config": asdict(variant.config),
            "cases": len(frame),
            "velocity_qc_pass_cases": int(truthy(frame["velocity_qc_pass"]).sum()),
            "failure_counts": {str(key): int(value) for key, value in failures.items()},
        },
    )
    return manifest


def run_hodge_sensitivity(
    mass_effect_manifest: Path,
    clinical_table: Path,
    primary_hodge_manifest: Path,
    arc_root: Path,
    output_dir: Path,
    n_jobs: int = 1,
    variants: tuple[HodgeSensitivityVariant, ...] | None = None,
) -> Path:
    """Run fixed Hodge perturbations and write aggregate robustness summaries."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")
    variants = variants or default_sensitivity_variants()
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("Hodge sensitivity variant names must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mass = read_table(mass_effect_manifest)
    require_unique(mass, "case_id", "mass-effect manifest")
    clinical = read_table(clinical_table)
    primary_hodge = read_table(primary_hodge_manifest)
    primary_design, audit = build_design(
        mass,
        clinical,
        uncertainty=None,
        config=ModelConfig(),
        hodge=primary_hodge,
    )

    comparison_frames: list[pd.DataFrame] = []
    manifest_hashes: dict[str, str] = {}
    qc_counts: dict[str, int] = {}
    records = mass.sort_values(["subject", "case_id"]).to_dict("records")
    for variant in variants:
        variant_dir = output_dir / variant.name
        manifest = _run_sensitivity_variant(
            records,
            Path(arc_root),
            variant,
            variant_dir,
            n_jobs,
        )
        frame = read_table(manifest)
        qc_counts[variant.name] = int(truthy(frame["velocity_qc_pass"]).sum())
        manifest_hashes[variant.name] = sha256_file(manifest)
        comparison_frames.append(compare_hodge_variant(primary_design, frame, variant.name))

    comparisons = pd.concat(comparison_frames, ignore_index=True)
    comparison_path = output_dir / "hodge_parameter_sensitivity.csv"
    atomic_csv(comparison_path, comparisons)
    feature_summary: dict[str, object] = {}
    for feature, group in comparisons.groupby("feature", sort=False):
        feature_summary[str(feature)] = {
            "minimum_spearman_vs_primary": float(group["spearman_vs_primary"].min()),
            "maximum_median_absolute_difference": float(
                group["median_absolute_difference"].max()
            ),
            "minimum_variant_spearman_vs_aq": float(group["variant_spearman_vs_aq"].min()),
            "maximum_variant_spearman_vs_aq": float(group["variant_spearman_vs_aq"].max()),
        }
    atomic_json(
        output_dir / "hodge_parameter_sensitivity.json",
        {
            "method_version": SENSITIVITY_METHOD_VERSION,
            "interpretation": (
                "One-factor numerical sensitivity analysis; Hodge descriptors are "
                "registration geometry, not physical velocity or pressure"
            ),
            "primary_analysis_cases": len(primary_design),
            "primary_hodge_manifest": str(Path(primary_hodge_manifest).resolve()),
            "primary_hodge_manifest_sha256": sha256_file(primary_hodge_manifest),
            "mass_effect_manifest_sha256": sha256_file(mass_effect_manifest),
            "clinical_table_sha256": sha256_file(clinical_table),
            "primary_design_audit": audit,
            "variants": [
                {
                    "name": variant.name,
                    "rationale": variant.rationale,
                    "config": asdict(variant.config),
                    "qc_pass_cases": qc_counts[variant.name],
                    "manifest_sha256": manifest_hashes[variant.name],
                }
                for variant in variants
            ],
            "feature_summary": feature_summary,
        },
    )
    return comparison_path
