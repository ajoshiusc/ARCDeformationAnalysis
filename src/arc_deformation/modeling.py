"""Leakage-resistant repeated nested cross-validation for WAB-AQ."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arc_deformation.constants import (
    CLINICAL_FEATURES,
    DEFORMATION_FEATURES,
    LESION_FEATURES,
    METHOD_VERSION,
    REGISTRATION_QC_FEATURES,
    UNCERTAINTY_FEATURES,
)
from arc_deformation.io import (
    atomic_csv,
    atomic_json,
    prefix_nonkeys,
    read_table,
    require_unique,
    sha256_file,
    truthy,
)


@dataclass(frozen=True)
class ModelConfig:
    outcome: str = "wab_aq"
    expected_method_version: str = METHOD_VERSION
    outer_folds: int = 5
    inner_folds: int = 4
    repeats: int = 20
    bootstrap_samples: int = 5000
    seed: int = 2026
    n_jobs: int = 1
    require_deformation_qc: bool = True
    maximum_folding_fraction: float = 0.05
    minimum_near_lesion_voxels: int = 1000
    minimum_uncertainty_coverage: float = 0.90
    alpha_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(-4, 4, 17).tolist())
    )


def build_design(
    mass: pd.DataFrame,
    clinical: pd.DataFrame,
    uncertainty: pd.DataFrame | None,
    config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join inputs, enforce one case per participant, and apply prespecified QC."""
    if "participant_id" in clinical and "subject" not in clinical:
        clinical = clinical.rename(columns={"participant_id": "subject"})
    require_unique(mass, "case_id", "mass-effect manifest")
    require_unique(clinical, "case_id", "clinical table")
    if config.outcome not in clinical:
        raise ValueError(f"Clinical table has no outcome column {config.outcome!r}")
    if "method_version" not in mass:
        raise ValueError("Mass-effect manifest has no method_version provenance")
    versions = sorted(mass["method_version"].dropna().astype(str).str.strip().unique())
    if versions != [config.expected_method_version]:
        raise ValueError(
            "Mass-effect manifest is stale or mixes methods: expected only "
            f"{config.expected_method_version!r}, found {versions!r}"
        )

    mass_rows = len(mass)
    mass_prefixed = prefix_nonkeys(mass, "me_")
    clinical_payload = clinical.drop(columns=["subject", "session"], errors="ignore")
    design = mass_prefixed.merge(
        clinical_payload, on="case_id", how="inner", validate="one_to_one"
    )
    matched_rows = len(design)

    qc_columns = (
        "me_laterality_supported",
        "me_normalized_field_folding_fraction",
        "me_mass_effect_3_20mm_magnitude_mm_n_voxels",
    )
    missing = [column for column in qc_columns if column not in design]
    if missing:
        raise ValueError(f"Mass-effect QC fields are missing: {missing}")
    if config.require_deformation_qc:
        supported = truthy(design["me_laterality_supported"])
        folding = pd.to_numeric(design["me_normalized_field_folding_fraction"], errors="coerce")
        valid_voxels = pd.to_numeric(
            design["me_mass_effect_3_20mm_magnitude_mm_n_voxels"], errors="coerce"
        )
        design = design.loc[
            supported
            & folding.le(config.maximum_folding_fraction)
            & valid_voxels.ge(config.minimum_near_lesion_voxels)
        ].copy()
    qc_rows = len(design)

    uncertainty_used = False
    uncertainty_coverage = 0.0
    if uncertainty is not None:
        require_unique(uncertainty, "case_id", "uncertainty manifest")
        uncertainty_coverage = float(design["case_id"].isin(uncertainty["case_id"]).mean())
        if uncertainty_coverage >= config.minimum_uncertainty_coverage:
            payload = prefix_nonkeys(uncertainty, "unc_").drop(
                columns=["subject", "session"], errors="ignore"
            )
            design = design.merge(payload, on="case_id", how="inner", validate="one_to_one")
            uncertainty_used = True

    design[config.outcome] = pd.to_numeric(design[config.outcome], errors="coerce")
    design["age_at_stroke"] = pd.to_numeric(design["age_at_stroke"], errors="coerce")
    wab_days = pd.to_numeric(design["wab_days"], errors="coerce")
    design["log1p_wab_days"] = np.log1p(wab_days)
    numeric = design.select_dtypes(include=[np.number]).columns
    design.loc[:, numeric] = design.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    design = design.dropna(subset=[config.outcome])
    if design["subject"].duplicated().any():
        examples = design.loc[design["subject"].duplicated(False), "subject"].head().tolist()
        raise ValueError(f"Multiple cases remain for participants including {examples}")
    design = design.sort_values(["subject", "case_id"]).reset_index(drop=True)
    audit: dict[str, object] = {
        "mass_effect_rows": mass_rows,
        "mass_effect_method_versions": versions,
        "mass_effect_clinical_matches": matched_rows,
        "rows_after_deformation_qc": qc_rows,
        "uncertainty_coverage_before_matching": uncertainty_coverage,
        "uncertainty_models_used": uncertainty_used,
        "analysis_subjects": len(design),
    }
    return design, audit


def model_feature_sets(
    design: pd.DataFrame, uncertainty_used: bool
) -> dict[str, tuple[str, ...]]:
    standard = CLINICAL_FEATURES + LESION_FEATURES
    sets: dict[str, tuple[str, ...]] = {
        "clinical_only": CLINICAL_FEATURES,
        "lesion_standard": standard,
        "lesion_plus_mass_effect": standard + DEFORMATION_FEATURES,
    }
    if all(
        column in design and design[column].notna().any() for column in REGISTRATION_QC_FEATURES
    ):
        sets["lesion_plus_mass_effect_and_registration_qc"] = (
            standard + DEFORMATION_FEATURES + REGISTRATION_QC_FEATURES
        )
    if uncertainty_used:
        uncertainty_set = standard + UNCERTAINTY_FEATURES
        sets["lesion_plus_uncertainty"] = uncertainty_set
        sets["lesion_uncertainty_plus_mass_effect"] = uncertainty_set + DEFORMATION_FEATURES
    for model, features in sets.items():
        missing = [feature for feature in features if feature not in design]
        empty = [
            feature
            for feature in features
            if feature in design and not design[feature].notna().any()
        ]
        if missing or empty:
            raise ValueError(f"Invalid feature set {model}: missing={missing}, empty={empty}")
    return sets


def _validate_config(config: ModelConfig, n: int) -> None:
    if n < max(20, config.outer_folds):
        raise ValueError(f"Only {n} participants remain; at least 20 are required")
    if not 2 <= config.outer_folds <= n:
        raise ValueError("outer_folds must be between 2 and cohort size")
    if config.inner_folds < 2 or config.repeats < 1 or config.bootstrap_samples < 100:
        raise ValueError("Need inner_folds >= 2, repeats >= 1, bootstrap_samples >= 100")
    if not 0 <= config.maximum_folding_fraction <= 1:
        raise ValueError("maximum_folding_fraction must be in [0, 1]")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in config.alpha_grid):
        raise ValueError("Every ridge alpha must be positive and finite")


def pearson_r(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.isclose(np.std(observed), 0) or np.isclose(np.std(predicted), 0):
        return math.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def repeated_nested_cv(
    design: pd.DataFrame,
    feature_sets: dict[str, tuple[str, ...]],
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run shared-split repeated outer CV with all preprocessing nested inside."""
    n = len(design)
    _validate_config(config, n)
    y = design[config.outcome].to_numpy(float)
    identifiers = design[["case_id", "subject", "session"]].to_dict("records")
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for repeat in range(1, config.repeats + 1):
        repeat_seed = config.seed + 1009 * (repeat - 1)
        outer = list(
            KFold(config.outer_folds, shuffle=True, random_state=repeat_seed).split(
                np.arange(n)
            )
        )
        for model, features in feature_sets.items():
            x = design[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            predicted = np.full(n, np.nan)
            folds = np.full(n, -1, dtype=int)
            chosen_alpha = np.full(n, np.nan)
            for fold_number, (train, test) in enumerate(outer, start=1):
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge()),
                    ]
                )
                search = GridSearchCV(
                    pipeline,
                    {"ridge__alpha": config.alpha_grid},
                    scoring="neg_mean_absolute_error",
                    cv=KFold(
                        min(config.inner_folds, len(train)),
                        shuffle=True,
                        random_state=repeat_seed + fold_number,
                    ),
                    n_jobs=config.n_jobs,
                    refit=True,
                )
                search.fit(x[train], y[train])
                predicted[test] = search.predict(x[test])
                alpha = float(search.best_params_["ridge__alpha"])
                chosen_alpha[test] = alpha
                folds[test] = fold_number
                coefficients = search.best_estimator_.named_steps["ridge"].coef_
                coefficient_rows.extend(
                    {
                        "repeat": repeat,
                        "outer_fold": fold_number,
                        "model": model,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                        "alpha": alpha,
                    }
                    for feature, coefficient in zip(features, coefficients, strict=True)
                )
            if not np.isfinite(predicted).all() or np.any(folds < 1):
                raise RuntimeError(
                    f"Incomplete out-of-fold predictions for {model}, repeat {repeat}"
                )
            prediction_rows.extend(
                {
                    **identifiers[index],
                    "repeat": repeat,
                    "outer_fold": int(folds[index]),
                    "model": model,
                    "observed_aq": float(y[index]),
                    "predicted_aq": float(predicted[index]),
                    "absolute_error": float(abs(y[index] - predicted[index])),
                    "alpha": float(chosen_alpha[index]),
                }
                for index in range(n)
            )
            metric_rows.append(
                {
                    "repeat": repeat,
                    "model": model,
                    "n": n,
                    "mae": float(mean_absolute_error(y, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
                    "r2": float(r2_score(y, predicted)),
                    "pearson_r": pearson_r(y, predicted),
                }
            )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "n": int(group["n"].iloc[0])}
        for metric in ("mae", "rmse", "r2", "pearson_r"):
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd_across_repeats"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_p025_across_repeats"] = float(np.percentile(values, 2.5))
            row[f"{metric}_p975_across_repeats"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Step-down Holm family-wise error adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparisons(
    predictions: pd.DataFrame,
    reference: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compare participant errors averaged across repeated CV splits."""
    errors = (
        predictions.groupby(["subject", "model"], as_index=False)["absolute_error"]
        .mean()
        .pivot(index="subject", columns="model", values="absolute_error")
    )
    if reference not in errors:
        raise ValueError(f"Reference model {reference!r} is unavailable")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model in errors.columns:
        if model == reference:
            continue
        pair = errors[[reference, model]].dropna()
        advantage = pair[reference].to_numpy(float) - pair[model].to_numpy(float)
        indices = rng.integers(0, len(advantage), size=(bootstrap_samples, len(advantage)))
        bootstrap = advantage[indices].mean(axis=1)
        if np.allclose(advantage, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(advantage, zero_method="pratt", alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(
            {
                "reference_model": reference,
                "comparison_model": model,
                "n_subjects": len(advantage),
                "mean_mae_advantage_points": float(np.mean(advantage)),
                "bootstrap_ci025": float(lower),
                "bootstrap_ci975": float(upper),
                "mean_advantage_ci_excludes_zero": bool(lower > 0 or upper < 0),
                "subjects_improved_fraction": float(np.mean(advantage > 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = holm_adjust(result["p_value"].to_numpy(float))
    return result


def aggregate_cohort_summary(
    design: pd.DataFrame, audit: dict[str, object]
) -> dict[str, object]:
    """Create deidentified descriptive statistics used by the manuscript."""

    def quantiles(column: str, digits: int) -> dict[str, float | int]:
        values = pd.to_numeric(design[column], errors="coerce").dropna().to_numpy(float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return {
            "n": len(values),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "digits": digits,
        }

    side = design["me_lesion_side"].value_counts()
    folding = quantiles("me_normalized_field_folding_fraction", 2)
    for key in ("median", "q1", "q3"):
        folding[key] = 100 * float(folding[key])
    return {
        "manifest_cases": int(audit["mass_effect_rows"]),
        "clinical_matches": int(audit["mass_effect_clinical_matches"]),
        "analysis_subjects": len(design),
        "left_dominant": int(side.get("left", 0)),
        "right_dominant": int(side.get("right", 0)),
        "age_at_stroke": quantiles("age_at_stroke", 1),
        "wab_days": quantiles("wab_days", 0),
        "wab_aq": quantiles("wab_aq", 1),
        "lesion_volume_ml": quantiles("lesion_volume_ml", 1),
        "magnitude_median_mm": quantiles("me_mass_effect_3_20mm_magnitude_mm_median", 2),
        "radial_median_mm": quantiles("me_mass_effect_3_20mm_radial_mm_median", 2),
        "absolute_radial_mean_mm": quantiles(
            "me_mass_effect_3_20mm_mean_absolute_radial_mm", 2
        ),
        "folding_percent": folding,
        "registration_sensitivity_median_mm": quantiles(
            "me_registration_sensitivity_3_20mm_mm_median", 2
        ),
        "signal_sensitivity_ratio": quantiles(
            "me_mass_effect_to_registration_sensitivity_ratio", 2
        ),
    }


def run_modeling(
    mass_effect_manifest: Path,
    clinical_table: Path,
    output_dir: Path,
    uncertainty_manifest: Path | None = None,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    """Run the complete model comparison and write machine-readable outputs."""
    config = config or ModelConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mass = read_table(mass_effect_manifest)
    clinical = read_table(clinical_table)
    uncertainty = read_table(uncertainty_manifest) if uncertainty_manifest else None
    design, audit = build_design(mass, clinical, uncertainty, config)
    features = model_feature_sets(design, bool(audit["uncertainty_models_used"]))
    predictions, metrics, coefficients = repeated_nested_cv(design, features, config)
    summary = summarize_metrics(metrics)
    comparisons = paired_comparisons(
        predictions,
        "lesion_standard",
        config.bootstrap_samples,
        config.seed + 99991,
    )
    if {
        "lesion_plus_uncertainty",
        "lesion_uncertainty_plus_mass_effect",
    }.issubset(features):
        direct = paired_comparisons(
            predictions,
            "lesion_plus_uncertainty",
            config.bootstrap_samples,
            config.seed + 199982,
        )
        direct = direct.loc[
            direct["comparison_model"].eq("lesion_uncertainty_plus_mass_effect")
        ].copy()
        direct["p_value_holm"] = direct["p_value"]
        comparisons = pd.concat([comparisons, direct], ignore_index=True)

    coefficient_summary = (
        coefficients.groupby(["model", "feature"])["standardized_coefficient"]
        .agg(mean="mean", std="std", median="median", count="count")
        .reset_index()
    )
    atomic_csv(output_dir / "aq_mass_effect_design.csv", design)
    atomic_csv(output_dir / "aq_mass_effect_predictions_long.csv", predictions)
    atomic_csv(output_dir / "metrics_by_repeat.csv", metrics)
    atomic_csv(output_dir / "model_summary.csv", summary)
    atomic_csv(output_dir / "paired_comparisons.csv", comparisons)
    atomic_csv(output_dir / "coefficients.csv", coefficients)
    atomic_csv(output_dir / "coefficient_summary.csv", coefficient_summary)
    atomic_json(output_dir / "cohort_summary.json", aggregate_cohort_summary(design, audit))
    provenance: dict[str, object] = {
        **audit,
        "model_config": asdict(config),
        "feature_sets": {name: list(values) for name, values in features.items()},
        "inputs": {
            "mass_effect_manifest": str(Path(mass_effect_manifest).resolve()),
            "mass_effect_manifest_sha256": sha256_file(mass_effect_manifest),
            "clinical_table": str(Path(clinical_table).resolve()),
            "clinical_table_sha256": sha256_file(clinical_table),
            "uncertainty_manifest": str(Path(uncertainty_manifest).resolve())
            if uncertainty_manifest
            else None,
            "uncertainty_manifest_sha256": sha256_file(uncertainty_manifest)
            if uncertainty_manifest
            else None,
        },
        "primary_estimand": (
            "participant mean absolute-error advantage: lesion_standard minus "
            "lesion_plus_mass_effect, averaged across repeats"
        ),
        "superiority_rule": "primary participant-bootstrap CI must exclude zero",
        "interpretation_warning": (
            "Cross-sectional lesion-associated deformation proxy; not physical ground-truth "
            "mass effect or pressure."
        ),
    }
    atomic_json(output_dir / "analysis_config.json", provenance)
    return provenance
