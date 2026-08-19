"""Generate publication figures, tables, and LaTeX macros from aggregate results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arc_deformation.io import atomic_text, read_table

MODEL_NAMES = {
    "clinical_only": "Clinical",
    "lesion_standard": "Conventional lesion",
    "lesion_plus_mass_effect": "Lesion + deformation",
    "lesion_plus_mass_effect_and_registration_qc": "Deformation + registration QC",
    "lesion_plus_uncertainty": "Lesion + uncertainty",
    "lesion_uncertainty_plus_mass_effect": "Uncertainty + deformation",
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
    rendered = f"{value:.{digits}f}"
    return f"${rendered}$" if value < 0 else rendered


def write_result_macros(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    cohort: dict[str, object],
    path: Path,
) -> None:
    models = summary.set_index("model")
    required = {"clinical_only", "lesion_standard", "lesion_plus_mass_effect"}
    if not required.issubset(models.index):
        raise ValueError(f"Model summary is missing {sorted(required - set(models.index))}")
    stems = {
        "clinical_only": "Clinical",
        "lesion_standard": "Standard",
        "lesion_plus_mass_effect": "Deformation",
        "lesion_plus_mass_effect_and_registration_qc": "DeformationQC",
        "lesion_plus_uncertainty": "Uncertainty",
        "lesion_uncertainty_plus_mass_effect": "Combined",
    }
    lines = [
        "% Generated from frozen aggregate results; do not edit by hand.",
        _latex_command("AnalysisN", str(int(models.loc["lesion_standard", "n"]))),
        _latex_command("ManifestN", str(int(cohort["manifest_cases"]))),
        _latex_command("ClinicalMatchN", str(int(cohort["clinical_matches"]))),
        _latex_command("LeftLesionN", str(int(cohort["left_dominant"]))),
        _latex_command("RightLesionN", str(int(cohort["right_dominant"]))),
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
        values = cohort[key]
        assert isinstance(values, dict)
        digits = int(values.get("digits", 1))
        lines.extend(
            [
                _latex_command(f"{macro}Median", _signed(float(values["median"]), digits)),
                _latex_command(f"{macro}QOne", _signed(float(values["q1"]), digits)),
                _latex_command(f"{macro}QThree", _signed(float(values["q3"]), digits)),
            ]
        )
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
    primary = _comparison(comparisons, "lesion_standard", "lesion_plus_mass_effect")
    lines.extend(
        [
            _latex_command(
                "DeformationAdvantage", _signed(float(primary["mean_mae_advantage_points"]), 2)
            ),
            _latex_command("DeformationCILow", _signed(float(primary["bootstrap_ci025"]), 2)),
            _latex_command("DeformationCIHigh", _signed(float(primary["bootstrap_ci975"]), 2)),
            _latex_command("DeformationP", f"{float(primary['p_value_holm']):.3f}"),
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
        direct = _comparison(
            comparisons,
            "lesion_plus_uncertainty",
            "lesion_uncertainty_plus_mass_effect",
        )
        lines.extend(
            [
                _latex_command(
                    "AfterUncertaintyAdvantage",
                    _signed(float(direct["mean_mae_advantage_points"]), 2),
                ),
                _latex_command(
                    "AfterUncertaintyCILow", _signed(float(direct["bootstrap_ci025"]), 2)
                ),
                _latex_command(
                    "AfterUncertaintyCIHigh", _signed(float(direct["bootstrap_ci975"]), 2)
                ),
                _latex_command("AfterUncertaintyP", f"{float(direct['p_value_holm']):.3f}"),
            ]
        )
    atomic_text(path, "\n".join(lines) + "\n")


def write_model_table(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & MAE & RMSE & $R^2$ & Pearson $r$ \\\\",
        "\\midrule",
    ]
    for record in summary.to_dict("records"):
        name = MODEL_NAMES.get(str(record["model"]), str(record["model"]).replace("_", " "))
        rows.append(
            f"{name} & {float(record['mae_mean']):.2f} & {float(record['rmse_mean']):.2f} "
            f"& {float(record['r2_mean']):.3f} & {float(record['pearson_r_mean']):.3f} \\\\"
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

    order = metrics["model"].drop_duplicates().tolist()
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

    shown = comparisons.loc[comparisons["reference_model"].eq("lesion_standard")].copy()
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
    axes[1].set_yticks(
        y,
        [MODEL_NAMES.get(model, model) for model in shown["comparison_model"]],
        fontsize=8,
    )
    axes[1].set_xlabel("Mean MAE advantage vs conventional lesion model")
    axes[1].set_title("B. Participant-bootstrap 95% intervals")
    axes[1].grid(axis="x", alpha=0.2)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def generate_report(results_dir: Path, output_dir: Path) -> None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_table(results_dir / "model_summary.csv")
    comparisons = read_table(results_dir / "paired_comparisons.csv")
    metrics = read_table(results_dir / "metrics_by_repeat.csv")
    with (results_dir / "cohort_summary.json").open(encoding="utf-8") as handle:
        cohort = json.load(handle)
    write_result_macros(summary, comparisons, cohort, output_dir / "results.tex")
    write_model_table(summary, output_dir / "model_table.tex")
    make_model_figure(metrics, comparisons, output_dir / "model_comparison")
