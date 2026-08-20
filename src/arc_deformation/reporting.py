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
    left_summary: pd.DataFrame,
    left_comparisons: pd.DataFrame,
    cohort: dict[str, object],
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
    lines = [
        "% Generated from frozen aggregate results; do not edit by hand.",
        _latex_command("AnalysisN", str(int(models.loc["lesion_standard", "n"]))),
        _latex_command("ManifestN", str(int(cohort["manifest_cases"]))),
        _latex_command("ClinicalMatchN", str(int(cohort["clinical_matches"]))),
        _latex_command("LeftLesionN", str(int(cohort["left_dominant"]))),
        _latex_command("RightLesionN", str(int(cohort["right_dominant"]))),
        _latex_command("FemaleN", str(int(reported_sex["female"]))),
        _latex_command("MaleN", str(int(reported_sex["male"]))),
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


def make_model_figure(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_stem: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-deformation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    left_summary = read_table(results_dir / "left_only_model_summary.csv")
    left_comparisons = read_table(results_dir / "left_only_paired_comparisons.csv")
    with (results_dir / "cohort_summary.json").open(encoding="utf-8") as handle:
        cohort = json.load(handle)
    write_result_macros(
        summary,
        comparisons,
        mae_inference,
        associations,
        left_summary,
        left_comparisons,
        cohort,
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
    make_model_figure(metrics, comparisons, output_dir / "model_comparison")
    make_association_figure(associations, output_dir / "association_forest")
