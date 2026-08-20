"""Generate publication figures, tables, and LaTeX macros from aggregate results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arc_deformation.io import atomic_text, read_table

MODEL_NAMES = {
    "intercept_only": "Intercept only",
    "clinical_only": "Clinical",
    "lesion_standard": "Conventional lesion",
    "lesion_plus_mass_effect": "Lesion + deformation",
    "lesion_plus_mass_effect_and_registration_qc": "Deformation + registration QC",
    "lesion_plus_uncertainty": "Lesion + uncertainty",
    "lesion_uncertainty_plus_mass_effect": "Uncertainty + deformation",
    "lesion_plus_hodge": "Lesion + log-velocity Hodge",
    "lesion_plus_mass_effect_plus_hodge": "Deformation + log-velocity Hodge",
}


def _comparison(comparisons: pd.DataFrame, reference: str, comparison: str) -> pd.Series:
    selected = comparisons.loc[
        comparisons["reference_model"].eq(reference)
        & comparisons["comparison_model"].eq(comparison)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one {reference} vs {comparison} row, found {len(selected)}")
    return selected.iloc[0]


def _latex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _signed(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _descriptive_macros(lines: list[str], prefix: str, values: dict[str, object]) -> None:
    digits = int(values.get("digits", 1))
    for suffix, key in (("Median", "median"), ("QOne", "q1"), ("QThree", "q3")):
        lines.append(_latex_command(prefix + suffix, _signed(float(values[key]), digits)))


def write_result_macros(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    mae_inference: pd.DataFrame,
    associations: pd.DataFrame,
    adjusted_associations: pd.DataFrame,
    left_summary: pd.DataFrame,
    left_comparisons: pd.DataFrame,
    cohort: dict[str, object],
    hodge_sensitivity: dict[str, object],
    path: Path,
) -> None:
    """Write manuscript values from machine-readable aggregate outputs."""
    models = summary.set_index("model")
    required = {"clinical_only", "lesion_standard", "lesion_plus_mass_effect"}
    if not required.issubset(models.index):
        raise ValueError(f"Model summary is missing {sorted(required - set(models.index))}")
    inference = mae_inference.set_index("model")
    stems = {
        "intercept_only": "Intercept",
        "clinical_only": "Clinical",
        "lesion_standard": "Standard",
        "lesion_plus_mass_effect": "Deformation",
        "lesion_plus_mass_effect_and_registration_qc": "DeformationQC",
        "lesion_plus_uncertainty": "Uncertainty",
        "lesion_uncertainty_plus_mass_effect": "Combined",
        "lesion_plus_hodge": "Hodge",
        "lesion_plus_mass_effect_plus_hodge": "DeformationHodge",
    }
    reported_sex = cohort["reported_sex"]
    wab_type = cohort["wab_type"]
    wab_type_counts = wab_type["counts"]
    lines = [
        "% Generated from frozen aggregate results; do not edit by hand.",
        _latex_command("AnalysisN", str(int(models.loc["lesion_standard", "n"]))),
        _latex_command("ManifestN", str(int(cohort["manifest_cases"]))),
        _latex_command("ClinicalMatchN", str(int(cohort["clinical_matches"]))),
        _latex_command("LeftLesionN", str(int(cohort["left_dominant"]))),
        _latex_command("RightLesionN", str(int(cohort["right_dominant"]))),
        _latex_command("FemaleN", str(int(reported_sex["female"]))),
        _latex_command("MaleN", str(int(reported_sex["male"]))),
        _latex_command("WABTypeMissingN", str(int(wab_type["missing"]))),
        _latex_command("WABTypeBrocaN", str(int(wab_type_counts.get("Broca", 0)))),
        _latex_command("WABTypeAnomicN", str(int(wab_type_counts.get("Anomic", 0)))),
        _latex_command("WABTypeConductionN", str(int(wab_type_counts.get("Conduction", 0)))),
        _latex_command("WABTypeGlobalN", str(int(wab_type_counts.get("Global", 0)))),
        _latex_command("WABTypeWernickeN", str(int(wab_type_counts.get("Wernicke", 0)))),
        _latex_command(
            "WABTypeTranscorticalMotorN",
            str(int(wab_type_counts.get("TranscorticalMotor", 0))),
        ),
    ]
    descriptive = {
        "Age": "age_at_stroke",
        "WABDays": "wab_days",
        "AQ": "wab_aq",
        "LesionVolume": "lesion_volume_ml",
        "Magnitude": "magnitude_median_mm",
        "Radial": "radial_median_mm",
        "AbsRadial": "absolute_radial_mean_mm",
        "FoldingPct": "folding_percent",
        "Sensitivity": "registration_sensitivity_median_mm",
        "SignalSensitivityRatio": "signal_sensitivity_ratio",
    }
    for macro, key in descriptive.items():
        _descriptive_macros(lines, macro, cohort[key])

    for model, stem in stems.items():
        if model not in models.index:
            continue
        row = models.loc[model]
        lines.extend(
            [
                _latex_command(f"{stem}MAE", f"{float(row['mae_mean']):.2f}"),
                _latex_command(f"{stem}RMSE", f"{float(row['rmse_mean']):.2f}"),
                _latex_command(f"{stem}RSquared", _signed(float(row["r2_mean"]), 3)),
                _latex_command(f"{stem}Correlation", _signed(float(row["pearson_r_mean"]), 3)),
            ]
        )
        if model in inference.index:
            infer = inference.loc[model]
            lines.extend(
                [
                    _latex_command(f"{stem}MAECILow", f"{float(infer['bootstrap_ci025']):.2f}"),
                    _latex_command(
                        f"{stem}MAECIHigh", f"{float(infer['bootstrap_ci975']):.2f}"
                    ),
                ]
            )

    primary = _comparison(comparisons, "lesion_standard", "lesion_plus_mass_effect")
    lines.extend(
        [
            _latex_command(
                "DeformationAdvantage", _signed(float(primary["mean_mae_advantage_points"]), 2)
            ),
            _latex_command("DeformationCILow", _signed(float(primary["bootstrap_ci025"]), 2)),
            _latex_command("DeformationCIHigh", _signed(float(primary["bootstrap_ci975"]), 2)),
            _latex_command("DeformationP", _p_value(float(primary["p_value_holm"]))),
            _latex_command(
                "DeformationImprovedFraction",
                f"{100 * float(primary['subjects_improved_fraction']):.1f}\\%",
            ),
        ]
    )
    if "lesion_plus_mass_effect_and_registration_qc" in models.index:
        row = _comparison(
            comparisons,
            "lesion_standard",
            "lesion_plus_mass_effect_and_registration_qc",
        )
        lines.extend(
            [
                _latex_command(
                    "DeformationQCAdvantage",
                    _signed(float(row["mean_mae_advantage_points"]), 2),
                ),
                _latex_command("DeformationQCCILow", _signed(float(row["bootstrap_ci025"]), 2)),
                _latex_command(
                    "DeformationQCCIHigh", _signed(float(row["bootstrap_ci975"]), 2)
                ),
            ]
        )
    if {
        "lesion_plus_uncertainty",
        "lesion_uncertainty_plus_mass_effect",
    }.issubset(models.index):
        row = _comparison(
            comparisons,
            "lesion_plus_uncertainty",
            "lesion_uncertainty_plus_mass_effect",
        )
        lines.extend(
            [
                _latex_command(
                    "AfterUncertaintyAdvantage",
                    _signed(float(row["mean_mae_advantage_points"]), 2),
                ),
                _latex_command(
                    "AfterUncertaintyCILow", _signed(float(row["bootstrap_ci025"]), 2)
                ),
                _latex_command(
                    "AfterUncertaintyCIHigh", _signed(float(row["bootstrap_ci975"]), 2)
                ),
                _latex_command("AfterUncertaintyP", _p_value(float(row["p_value_holm"]))),
            ]
        )
    if "lesion_plus_hodge" in models.index:
        hodge = _comparison(comparisons, "lesion_standard", "lesion_plus_hodge")
        after_deformation = _comparison(
            comparisons,
            "lesion_plus_mass_effect",
            "lesion_plus_mass_effect_plus_hodge",
        )
        lines.extend(
            [
                _latex_command(
                    "HodgeAdvantage", _signed(float(hodge["mean_mae_advantage_points"]), 2)
                ),
                _latex_command("HodgeCILow", _signed(float(hodge["bootstrap_ci025"]), 2)),
                _latex_command("HodgeCIHigh", _signed(float(hodge["bootstrap_ci975"]), 2)),
                _latex_command("HodgeP", _p_value(float(hodge["p_value_holm"]))),
                _latex_command(
                    "HodgeAfterDeformationAdvantage",
                    _signed(float(after_deformation["mean_mae_advantage_points"]), 2),
                ),
                _latex_command(
                    "HodgeAfterDeformationCILow",
                    _signed(float(after_deformation["bootstrap_ci025"]), 2),
                ),
                _latex_command(
                    "HodgeAfterDeformationCIHigh",
                    _signed(float(after_deformation["bootstrap_ci975"]), 2),
                ),
                _latex_command(
                    "HodgeAfterDeformationP",
                    _p_value(float(after_deformation["p_value_holm"])),
                ),
            ]
        )

    hodge_summary = cohort.get("log_velocity_hodge")
    if isinstance(hodge_summary, dict):
        hodge_descriptives = {
            "HodgeTotalRMS": "total_rms_mm",
            "HodgeCurlFraction": "curl_free_energy_fraction",
            "HodgeDivergenceFraction": "divergence_free_energy_fraction",
            "HodgeHarmonicFraction": "harmonic_energy_fraction",
            "VelocityReconstructionError": "velocity_reconstruction_relative_rmse",
            "VelocityMinimumJacobian": "minimum_input_jacobian",
        }
        for macro, key in hodge_descriptives.items():
            _descriptive_macros(lines, macro, hodge_summary[key])
        lines.extend(
            [
                _latex_command("HodgeExtractionN", str(hodge_summary["extraction_cases"])),
                _latex_command("HodgeQCPassN", str(hodge_summary["extraction_qc_pass_cases"])),
                _latex_command("HodgeAnalysisN", str(hodge_summary["analysis_qc_pass_n"])),
            ]
        )

    association_stems = {
        "lesion_volume_vs_aq": "LesionVolumeAQ",
        "magnitude_vs_aq": "MagnitudeAQ",
        "absolute_radial_vs_aq": "AbsoluteRadialAQ",
        "radial_direction_vs_aq": "RadialDirectionAQ",
        "hodge_total_rms_vs_aq": "HodgeTotalAQ",
        "hodge_curl_free_fraction_vs_aq": "HodgeCurlAQ",
        "hodge_divergence_free_fraction_vs_aq": "HodgeDivergenceAQ",
    }
    association_index = associations.set_index("association")
    for association, stem in association_stems.items():
        if association not in association_index.index:
            continue
        row = association_index.loc[association]
        lines.extend(
            [
                _latex_command(stem + "Rho", _signed(float(row["spearman_rho"]), 3)),
                _latex_command(stem + "CILow", _signed(float(row["bootstrap_ci025"]), 3)),
                _latex_command(stem + "CIHigh", _signed(float(row["bootstrap_ci975"]), 3)),
                _latex_command(stem + "P", _p_value(float(row["p_value_holm"]))),
            ]
        )

    adjusted_stems = {
        "adjusted_magnitude_vs_aq": "AdjustedMagnitudeAQ",
        "adjusted_absolute_radial_vs_aq": "AdjustedAbsoluteRadialAQ",
        "adjusted_radial_direction_vs_aq": "AdjustedRadialDirectionAQ",
        "adjusted_hodge_total_rms_vs_aq": "AdjustedHodgeTotalAQ",
        "adjusted_hodge_curl_free_fraction_vs_aq": "AdjustedHodgeCurlAQ",
        "adjusted_hodge_divergence_free_fraction_vs_aq": "AdjustedHodgeDivergenceAQ",
    }
    adjusted_index = adjusted_associations.set_index("association")
    for association, stem in adjusted_stems.items():
        if association not in adjusted_index.index:
            continue
        row = adjusted_index.loc[association]
        lines.extend(
            [
                _latex_command(stem + "Rho", _signed(float(row["partial_spearman_rho"]), 3)),
                _latex_command(stem + "CILow", _signed(float(row["bootstrap_ci025"]), 3)),
                _latex_command(stem + "CIHigh", _signed(float(row["bootstrap_ci975"]), 3)),
                _latex_command(stem + "P", _p_value(float(row["permutation_p_value_holm"]))),
            ]
        )

    sensitivity_summary = hodge_sensitivity["feature_summary"]
    sensitivity_macros = {
        "SensitivityTotalStabilityMinimum": (
            sensitivity_summary["total_rms_mm"]["minimum_spearman_vs_primary"]
        ),
        "SensitivityCurlStabilityMinimum": (
            sensitivity_summary["curl_free_energy_fraction"]["minimum_spearman_vs_primary"]
        ),
        "SensitivityDivergenceStabilityMinimum": (
            sensitivity_summary["divergence_free_energy_fraction"][
                "minimum_spearman_vs_primary"
            ]
        ),
        "SensitivityCurlAQMinimum": (
            sensitivity_summary["curl_free_energy_fraction"]["minimum_variant_spearman_vs_aq"]
        ),
        "SensitivityCurlAQMaximum": (
            sensitivity_summary["curl_free_energy_fraction"]["maximum_variant_spearman_vs_aq"]
        ),
        "SensitivityDivergenceAQMinimum": (
            sensitivity_summary["divergence_free_energy_fraction"][
                "minimum_variant_spearman_vs_aq"
            ]
        ),
        "SensitivityDivergenceAQMaximum": (
            sensitivity_summary["divergence_free_energy_fraction"][
                "maximum_variant_spearman_vs_aq"
            ]
        ),
    }
    lines.extend(
        _latex_command(name, _signed(float(value), 3))
        for name, value in sensitivity_macros.items()
    )
    sensitivity_variants = {
        str(record["name"]): record for record in hodge_sensitivity["variants"]
    }
    lines.extend(
        [
            _latex_command("SensitivityVariantN", str(len(sensitivity_variants))),
            _latex_command(
                "SensitivitySmoothingEightPassN",
                str(sensitivity_variants["smoothing_8mm"]["qc_pass_cases"]),
            ),
            _latex_command(
                "SensitivitySmoothingEightFailureN",
                str(
                    int(hodge_sensitivity["primary_design_audit"]["hodge_manifest_rows"])
                    - int(sensitivity_variants["smoothing_8mm"]["qc_pass_cases"])
                ),
            ),
            _latex_command(
                "SensitivityTaperTwelvePassN",
                str(sensitivity_variants["taper_12mm"]["qc_pass_cases"]),
            ),
            _latex_command(
                "SensitivityGridFivePassN",
                str(sensitivity_variants["grid_5mm"]["qc_pass_cases"]),
            ),
        ]
    )

    left_models = left_summary.set_index("model")
    if "lesion_plus_mass_effect" in left_models.index:
        left_primary = _comparison(
            left_comparisons, "lesion_standard", "lesion_plus_mass_effect"
        )
        lines.extend(
            [
                _latex_command("LeftOnlyN", str(int(left_models.iloc[0]["n"]))),
                _latex_command(
                    "LeftOnlyDeformationAdvantage",
                    _signed(float(left_primary["mean_mae_advantage_points"]), 2),
                ),
                _latex_command(
                    "LeftOnlyDeformationCILow",
                    _signed(float(left_primary["bootstrap_ci025"]), 2),
                ),
                _latex_command(
                    "LeftOnlyDeformationCIHigh",
                    _signed(float(left_primary["bootstrap_ci975"]), 2),
                ),
            ]
        )
    atomic_text(path, "\n".join(lines) + "\n")


def write_model_table(
    summary: pd.DataFrame,
    mae_inference: pd.DataFrame,
    path: Path,
    selected: tuple[str, ...] | None = None,
) -> None:
    intervals = mae_inference.set_index("model")
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & MAE (95\\% CI) & RMSE & $R^2$ & Pearson $r$ \\\\",
        "\\midrule",
    ]
    for record in summary.to_dict("records"):
        model = str(record["model"])
        if selected is not None and model not in selected:
            continue
        interval = intervals.loc[model]
        name = MODEL_NAMES.get(model, model.replace("_", " "))
        rows.append(
            f"{name} & {float(record['mae_mean']):.2f} "
            f"({float(interval['bootstrap_ci025']):.2f}, "
            f"{float(interval['bootstrap_ci975']):.2f}) & "
            f"{float(record['rmse_mean']):.2f} & {float(record['r2_mean']):.3f} "
            f"& {float(record['pearson_r_mean']):.3f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(rows) + "\n")


def write_association_table(associations: pd.DataFrame, path: Path) -> None:
    labels = {
        "lesion_volume_vs_aq": "Lesion volume vs AQ",
        "magnitude_vs_lesion_volume": "Deformation magnitude vs lesion volume",
        "absolute_radial_vs_lesion_volume": "Absolute radial deformation vs lesion volume",
        "magnitude_vs_registration_sensitivity": "Magnitude vs registration sensitivity",
        "absolute_radial_vs_registration_sensitivity": "Absolute radial vs registration sensitivity",
        "magnitude_vs_aq": "Deformation magnitude vs AQ",
        "absolute_radial_vs_aq": "Absolute radial deformation vs AQ",
        "radial_direction_vs_aq": "Signed radial deformation vs AQ",
        "hodge_total_rms_vs_lesion_volume": "Log-velocity RMS vs lesion volume",
        "hodge_total_rms_vs_aq": "Log-velocity RMS vs AQ",
        "hodge_curl_free_fraction_vs_aq": "Curl-free energy fraction vs AQ",
        "hodge_divergence_free_fraction_vs_aq": "Divergence-free energy fraction vs AQ",
    }
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Association & $n$ & Spearman $\\rho$ & 95\\% CI & Holm $p$ \\\\",
        "\\midrule",
    ]
    for record in associations.to_dict("records"):
        label = labels.get(record["association"], record["association"].replace("_", " "))
        rows.append(
            f"{label} & {int(record['n_subjects'])} & {float(record['spearman_rho']):.3f} "
            f"& ({float(record['bootstrap_ci025']):.3f}, "
            f"{float(record['bootstrap_ci975']):.3f}) & "
            f"\\({_p_value(float(record['p_value_holm']))}\\) \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(rows) + "\n")


def write_adjusted_association_table(associations: pd.DataFrame, path: Path) -> None:
    """Write the conventional-feature-adjusted exploratory associations."""
    labels = {
        "adjusted_magnitude_vs_aq": "Deformation magnitude",
        "adjusted_absolute_radial_vs_aq": "Absolute radial deformation",
        "adjusted_radial_direction_vs_aq": "Signed radial deformation",
        "adjusted_hodge_total_rms_vs_aq": "Log-velocity RMS",
        "adjusted_hodge_curl_free_fraction_vs_aq": "Curl-free energy fraction",
        "adjusted_hodge_divergence_free_fraction_vs_aq": "Divergence-free energy fraction",
    }
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Exposure & $n$ & Partial $\\rho$ & 95\\% CI & Holm permutation $p$ \\\\",
        "\\midrule",
    ]
    for record in associations.to_dict("records"):
        label = labels.get(record["association"], str(record["association"]))
        rows.append(
            f"{label} & {int(record['n_subjects'])} & "
            f"{float(record['partial_spearman_rho']):.3f} & "
            f"({float(record['bootstrap_ci025']):.3f}, "
            f"{float(record['bootstrap_ci975']):.3f}) & "
            f"\\({_p_value(float(record['permutation_p_value_holm']))}\\) \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(rows) + "\n")


def write_hodge_sensitivity_table(
    comparisons: pd.DataFrame,
    summary: dict[str, object],
    path: Path,
) -> None:
    """Write per-variant numerical stability and outcome-association summaries."""
    labels = {
        "smoothing_8mm": "Smoothing $\\sigma=8$ mm",
        "smoothing_12mm": "Smoothing $\\sigma=12$ mm",
        "taper_12mm": "Taper 12 mm",
        "taper_20mm": "Taper 20 mm",
        "padding_16mm": "Padding 16 mm",
        "padding_32mm": "Padding 32 mm",
        "grid_3mm": "Grid 3 mm",
        "grid_5mm": "Grid 5 mm",
    }
    qc = {str(record["name"]): int(record["qc_pass_cases"]) for record in summary["variants"]}
    indexed = comparisons.set_index(["variant", "feature"])
    rows = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Variant & QC & $n$ & RMS stability & Curl stability & Curl--AQ & Div--AQ \\\\",
        "\\midrule",
    ]
    for variant in labels:
        total = indexed.loc[(variant, "total_rms_mm")]
        curl = indexed.loc[(variant, "curl_free_energy_fraction")]
        divergence = indexed.loc[(variant, "divergence_free_energy_fraction")]
        rows.append(
            f"{labels[variant]} & {qc[variant]}/214 & "
            f"{int(total['common_analysis_cases'])} & "
            f"{float(total['spearman_vs_primary']):.3f} & "
            f"{float(curl['spearman_vs_primary']):.3f} & "
            f"{float(curl['variant_spearman_vs_aq']):.3f} & "
            f"{float(divergence['variant_spearman_vs_aq']):.3f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(rows) + "\n")


def write_paired_comparison_table(comparisons: pd.DataFrame, path: Path) -> None:
    """Write every prespecified paired MAE comparison for the supplement."""
    rows = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Reference & Comparison model & $n$ & Mean advantage (95\\% CI) & Improved & Holm $p$ \\\\",
        "\\midrule",
    ]
    for record in comparisons.to_dict("records"):
        reference = MODEL_NAMES.get(
            str(record["reference_model"]), str(record["reference_model"]).replace("_", " ")
        )
        comparison = MODEL_NAMES.get(
            str(record["comparison_model"]),
            str(record["comparison_model"]).replace("_", " "),
        )
        rows.append(
            f"{reference} & {comparison} & {int(record['n_subjects'])} & "
            f"{float(record['mean_mae_advantage_points']):.2f} "
            f"({float(record['bootstrap_ci025']):.2f}, "
            f"{float(record['bootstrap_ci975']):.2f}) & "
            f"{100 * float(record['subjects_improved_fraction']):.1f}\\% & "
            f"\\({_p_value(float(record['p_value_holm']))}\\) \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(rows) + "\n")


def _configure_matplotlib() -> None:
    """Apply a consistent, journal-compatible style to generated figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_model_figure(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_stem: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    preferred = [
        "intercept_only",
        "clinical_only",
        "lesion_standard",
        "lesion_plus_mass_effect",
        "lesion_plus_hodge",
        "lesion_plus_mass_effect_plus_hodge",
    ]
    available = set(metrics["model"])
    order = [model for model in preferred if model in available]
    colors = dict(zip(order, plt.cm.viridis(np.linspace(0.10, 0.90, len(order))), strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    values = [metrics.loc[metrics["model"].eq(model), "mae"].to_numpy(float) for model in order]
    boxes = axes[0].boxplot(
        values,
        tick_labels=[MODEL_NAMES.get(model, model) for model in order],
        patch_artist=True,
        showfliers=False,
    )
    for patch, model in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.8)
    axes[0].tick_params(axis="x", rotation=28, labelsize=8)
    axes[0].set_ylabel("Outer-CV MAE (AQ points)")
    axes[0].set_title("A. Performance across 20 repeated splits")
    axes[0].grid(axis="y", alpha=0.2)

    pairs = [
        ("lesion_standard", "lesion_plus_mass_effect"),
        ("lesion_standard", "lesion_plus_hodge"),
        ("lesion_plus_mass_effect", "lesion_plus_mass_effect_plus_hodge"),
    ]
    shown = pd.DataFrame([_comparison(comparisons, *pair) for pair in pairs])
    shown["label"] = [
        "Deformation after lesion",
        "Log-velocity Hodge after lesion",
        "Log-velocity Hodge after deformation",
    ]
    y = np.arange(len(shown))
    mean = shown["mean_mae_advantage_points"].to_numpy(float)
    lower = shown["bootstrap_ci025"].to_numpy(float)
    upper = shown["bootstrap_ci975"].to_numpy(float)
    axes[1].errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lower, upper - mean]),
        fmt="o",
        color="#2166ac",
        ecolor="#6baed6",
        capsize=4,
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, shown["label"], fontsize=8)
    axes[1].set_xlabel("Mean MAE advantage of the added feature family")
    axes[1].set_title("B. Participant-bootstrap 95% intervals")
    axes[1].grid(axis="x", alpha=0.2)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def make_association_figure(associations: pd.DataFrame, output_stem: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    order = [
        "lesion_volume_vs_aq",
        "magnitude_vs_aq",
        "absolute_radial_vs_aq",
        "radial_direction_vs_aq",
        "hodge_total_rms_vs_aq",
        "hodge_curl_free_fraction_vs_aq",
        "hodge_divergence_free_fraction_vs_aq",
    ]
    labels = {
        "lesion_volume_vs_aq": "Lesion volume",
        "magnitude_vs_aq": "Deformation magnitude",
        "absolute_radial_vs_aq": "Absolute radial deformation",
        "radial_direction_vs_aq": "Signed radial deformation",
        "hodge_total_rms_vs_aq": "Log-velocity RMS",
        "hodge_curl_free_fraction_vs_aq": "Curl-free energy fraction",
        "hodge_divergence_free_fraction_vs_aq": "Divergence-free energy fraction",
    }
    indexed = associations.set_index("association")
    selected = indexed.loc[[name for name in order if name in indexed.index]].copy()
    y = np.arange(len(selected))[::-1]
    mean = selected["spearman_rho"].to_numpy(float)
    lower = selected["bootstrap_ci025"].to_numpy(float)
    upper = selected["bootstrap_ci975"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lower, upper - mean]),
        fmt="o",
        color="#7f2704",
        ecolor="#d94801",
        capsize=4,
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, [labels[name] for name in selected.index])
    axis.set_xlabel("Spearman association with WAB-AQ (95% bootstrap CI)")
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def make_adjustment_figure(
    associations: pd.DataFrame,
    adjusted_associations: pd.DataFrame,
    output_stem: Path,
) -> None:
    """Compare unadjusted and conventional-feature-adjusted AQ associations."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    pairs = [
        ("magnitude_vs_aq", "adjusted_magnitude_vs_aq", "Deformation magnitude"),
        (
            "absolute_radial_vs_aq",
            "adjusted_absolute_radial_vs_aq",
            "Absolute radial deformation",
        ),
        (
            "radial_direction_vs_aq",
            "adjusted_radial_direction_vs_aq",
            "Signed radial deformation",
        ),
        ("hodge_total_rms_vs_aq", "adjusted_hodge_total_rms_vs_aq", "Log-velocity RMS"),
        (
            "hodge_curl_free_fraction_vs_aq",
            "adjusted_hodge_curl_free_fraction_vs_aq",
            "Curl-free energy fraction",
        ),
        (
            "hodge_divergence_free_fraction_vs_aq",
            "adjusted_hodge_divergence_free_fraction_vs_aq",
            "Divergence-free energy fraction",
        ),
    ]
    crude = associations.set_index("association")
    adjusted = adjusted_associations.set_index("association")
    y = np.arange(len(pairs))[::-1]
    figure, axis = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    for offset, source, value_column, label, color, marker in (
        (0.12, crude, "spearman_rho", "Unadjusted", "#b35806", "o"),
        (-0.12, adjusted, "partial_spearman_rho", "Adjusted", "#2166ac", "s"),
    ):
        names = [pair[0] if label == "Unadjusted" else pair[1] for pair in pairs]
        selected = source.loc[names]
        mean = selected[value_column].to_numpy(float)
        lower = selected["bootstrap_ci025"].to_numpy(float)
        upper = selected["bootstrap_ci975"].to_numpy(float)
        axis.errorbar(
            mean,
            y + offset,
            xerr=np.vstack([mean - lower, upper - mean]),
            fmt=marker,
            color=color,
            ecolor=color,
            alpha=0.9,
            capsize=3,
            markersize=5,
            label=label,
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, [pair[2] for pair in pairs])
    axis.set_xlabel("Association with WAB-AQ (95% participant-bootstrap interval)")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def make_hodge_sensitivity_figure(
    sensitivity: pd.DataFrame,
    associations: pd.DataFrame,
    output_stem: Path,
) -> None:
    """Visualize Hodge descriptor stability and association-direction sensitivity."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    variants = [
        ("primary", "Primary"),
        ("smoothing_8mm", "Smoothing 8 mm"),
        ("smoothing_12mm", "Smoothing 12 mm"),
        ("taper_12mm", "Taper 12 mm"),
        ("taper_20mm", "Taper 20 mm"),
        ("padding_16mm", "Padding 16 mm"),
        ("padding_32mm", "Padding 32 mm"),
        ("grid_3mm", "Grid 3 mm"),
        ("grid_5mm", "Grid 5 mm"),
    ]
    feature_style = {
        "total_rms_mm": ("Total RMS", "#4d4d4d", "o"),
        "curl_free_energy_fraction": ("Curl-free fraction", "#b35806", "s"),
        "divergence_free_energy_fraction": ("Divergence-free fraction", "#2166ac", "^"),
    }
    indexed = sensitivity.set_index(["variant", "feature"])
    association_index = associations.set_index("association")
    primary_aq = {
        "curl_free_energy_fraction": float(
            association_index.loc["hodge_curl_free_fraction_vs_aq", "spearman_rho"]
        ),
        "divergence_free_energy_fraction": float(
            association_index.loc["hodge_divergence_free_fraction_vs_aq", "spearman_rho"]
        ),
    }
    y = np.arange(len(variants))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), constrained_layout=True)
    for feature, (label, color, marker) in feature_style.items():
        stability = [
            1.0
            if variant == "primary"
            else float(indexed.loc[(variant, feature), "spearman_vs_primary"])
            for variant, _ in variants
        ]
        axes[0].plot(stability, y, marker=marker, color=color, linewidth=1, label=label)
    axes[0].set_yticks(y, [label for _, label in variants])
    axes[0].set_xlim(0.985, 1.001)
    axes[0].set_xlabel("Spearman rank correlation with primary setting")
    axes[0].set_title("A. Descriptor stability")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False, loc="lower left")

    for feature in ("curl_free_energy_fraction", "divergence_free_energy_fraction"):
        label, color, marker = feature_style[feature]
        values = [
            primary_aq[feature]
            if variant == "primary"
            else float(indexed.loc[(variant, feature), "variant_spearman_vs_aq"])
            for variant, _ in variants
        ]
        axes[1].plot(values, y, marker=marker, color=color, linewidth=1, label=label)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Unadjusted Spearman association with WAB-AQ")
    axes[1].set_title("B. Association-direction sensitivity")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(frameon=False, loc="lower right")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def generate_report(results_dir: Path, output_dir: Path) -> None:
    """Generate all aggregate paper assets from one completed model run."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_table(results_dir / "model_summary.csv")
    comparisons = read_table(results_dir / "paired_comparisons.csv")
    metrics = read_table(results_dir / "metrics_by_repeat.csv")
    mae_inference = read_table(results_dir / "model_mae_inference.csv")
    associations = read_table(results_dir / "deformation_associations.csv")
    adjusted_associations = read_table(results_dir / "adjusted_deformation_associations.csv")
    sensitivity_comparisons = read_table(results_dir / "hodge_parameter_sensitivity.csv")
    left_summary = read_table(results_dir / "left_only_model_summary.csv")
    left_comparisons = read_table(results_dir / "left_only_paired_comparisons.csv")
    with (results_dir / "cohort_summary.json").open(encoding="utf-8") as handle:
        cohort = json.load(handle)
    with (results_dir / "hodge_parameter_sensitivity.json").open(encoding="utf-8") as handle:
        hodge_sensitivity = json.load(handle)
    write_result_macros(
        summary,
        comparisons,
        mae_inference,
        associations,
        adjusted_associations,
        left_summary,
        left_comparisons,
        cohort,
        hodge_sensitivity,
        output_dir / "results.tex",
    )
    selected = (
        "intercept_only",
        "clinical_only",
        "lesion_standard",
        "lesion_plus_mass_effect",
        "lesion_plus_hodge",
        "lesion_plus_mass_effect_plus_hodge",
    )
    write_model_table(summary, mae_inference, output_dir / "model_table.tex", selected)
    write_model_table(summary, mae_inference, output_dir / "model_table_all.tex")
    write_association_table(associations, output_dir / "association_table.tex")
    write_adjusted_association_table(
        adjusted_associations, output_dir / "adjusted_association_table.tex"
    )
    write_hodge_sensitivity_table(
        sensitivity_comparisons,
        hodge_sensitivity,
        output_dir / "hodge_sensitivity_table.tex",
    )
    write_paired_comparison_table(comparisons, output_dir / "paired_comparison_table.tex")
    write_paired_comparison_table(
        left_comparisons, output_dir / "left_only_comparison_table.tex"
    )
    make_model_figure(metrics, comparisons, output_dir / "model_comparison")
    make_association_figure(associations, output_dir / "association_forest")
    make_adjustment_figure(
        associations, adjusted_associations, output_dir / "association_adjustment"
    )
    make_hodge_sensitivity_figure(
        sensitivity_comparisons, associations, output_dir / "hodge_sensitivity"
    )
