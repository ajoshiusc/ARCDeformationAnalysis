"""Generate the standalone ANTs supplemental material from aggregate outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arc_deformation.io import atomic_text, read_table
from arc_deformation.reporting import MODEL_NAMES

ASSOCIATION_NAMES = {
    "lesion_volume_vs_aq": "Lesion volume vs AQ",
    "magnitude_vs_lesion_volume": "Magnitude vs lesion volume",
    "magnitude_vs_aq": "Magnitude vs AQ",
    "absolute_radial_vs_lesion_volume": "Absolute radial vs lesion volume",
    "absolute_radial_vs_aq": "Absolute radial vs AQ",
    "radial_direction_vs_aq": "Signed radial vs AQ",
    "magnitude_vs_registration_sensitivity": "Magnitude vs registration sensitivity",
    "absolute_radial_vs_registration_sensitivity": "Absolute radial vs registration sensitivity",
    "hodge_total_rms_vs_lesion_volume": "Log-velocity RMS vs lesion volume",
    "hodge_total_rms_vs_aq": "Log-velocity RMS vs AQ",
    "hodge_curl_free_fraction_vs_aq": "Curl-free fraction vs AQ",
    "hodge_divergence_free_fraction_vs_aq": "Divergence-free fraction vs AQ",
}

ADJUSTED_ASSOCIATION_NAMES = {
    "adjusted_magnitude_vs_aq": "Magnitude vs AQ",
    "adjusted_absolute_radial_vs_aq": "Absolute radial vs AQ",
    "adjusted_radial_direction_vs_aq": "Signed radial vs AQ",
    "adjusted_hodge_total_rms_vs_aq": "Log-velocity RMS vs AQ",
    "adjusted_hodge_curl_free_fraction_vs_aq": "Curl-free fraction vs AQ",
    "adjusted_hodge_divergence_free_fraction_vs_aq": "Divergence-free fraction vs AQ",
}

DESCRIPTOR_NAMES = {
    "mass_effect_3_20mm_magnitude_mm_median": "Median magnitude",
    "mass_effect_3_20mm_magnitude_mm_p95": "95th-percentile magnitude",
    "mass_effect_3_20mm_radial_mm_median": "Median signed radial",
    "mass_effect_3_20mm_mean_absolute_radial_mm": "Mean absolute radial",
    "mass_effect_3_20mm_outward_integral_ml_mm": "Outward radial integral",
    "mass_effect_3_20mm_inward_integral_ml_mm": "Inward radial integral",
    "mass_effect_3_20mm_logjac_expansion_integral_ml": "Log-Jacobian expansion integral",
    "mass_effect_3_20mm_logjac_compression_integral_ml": "Log-Jacobian compression integral",
    "total_rms_mm": "Log-velocity RMS",
    "curl_free_energy_fraction": "Curl-free energy fraction",
    "divergence_free_energy_fraction": "Divergence-free energy fraction",
}


def _command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("<", "$<$")
        .replace(">", "$>$")
    )


def _write_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    alignment = "l" + "r" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(_escape(value) for value in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    atomic_text(path, "\n".join(lines) + "\n")


def _select_comparison(frame: pd.DataFrame, reference: str, comparison: str) -> pd.Series:
    rows = frame.loc[
        frame["reference_model"].eq(reference) & frame["comparison_model"].eq(comparison)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one comparison {reference} vs {comparison}, got {len(rows)}"
        )
    return rows.iloc[0]


def generate(reference_dir: Path, generated_dir: Path) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    registration_dir = reference_dir / "registration"
    registration_summary = read_table(registration_dir / "registration_qc_summary.csv")
    registration_index = registration_summary.set_index("column")
    with (registration_dir / "registration_flow.json").open() as handle:
        registration_flow = json.load(handle)
    with (registration_dir / "registration_qc_histograms.json").open() as handle:
        registration_histograms = json.load(handle)
    with (reference_dir / "hodge" / "hodge_summary.json").open() as handle:
        hodge_summary = json.load(handle)
    model_dir = reference_dir / "model"
    model_summary = read_table(model_dir / "model_summary.csv")
    comparisons = read_table(model_dir / "paired_comparisons.csv")
    associations = read_table(model_dir / "deformation_associations.csv")
    adjusted = read_table(model_dir / "adjusted_deformation_associations.csv")
    agreement = read_table(reference_dir / "comparison" / "descriptor_agreement.csv")
    predictive = read_table(reference_dir / "comparison" / "predictive_method_comparison.csv")
    with (registration_dir / "ants_mni_config.json").open() as handle:
        registration_config = json.load(handle)

    primary = _select_comparison(comparisons, "lesion_standard", "lesion_plus_mass_effect")
    hodge_comparison = _select_comparison(comparisons, "lesion_standard", "lesion_plus_hodge")
    adjusted_index = adjusted.set_index("association")
    agreement_index = agreement.set_index("descriptor")
    prediction_index = predictive.set_index("model")
    macros = [
        "% Generated from aggregate ANTs outputs; do not edit by hand.",
        _command("AntsManifestN", str(registration_flow["manifest_cases"])),
        _command("AntsRegistrationPassN", str(registration_flow["registration_qc_pass_cases"])),
        _command("AntsDeformationPassN", str(registration_flow["deformation_qc_pass_cases"])),
        _command("AntsAnalysisN", str(int(model_summary["n"].iloc[0]))),
        _command("AntsHodgePassN", str(hodge_summary["velocity_qc_pass_cases"])),
        _command(
            "AntsBrainDiceMedian",
            f"{registration_index.loc['registration_brain_mask_dice', 'median']:.3f}",
        ),
        _command(
            "AntsCycleMedian",
            f"{registration_index.loc['registration_cycle_rmse_mm', 'median']:.3f}",
        ),
        _command(
            "AntsCycleMaximum",
            f"{registration_index.loc['registration_cycle_rmse_mm', 'maximum']:.3f}",
        ),
        _command(
            "AntsRawFoldingMaximum",
            f"{100 * registration_index.loc['registration_raw_warp_folding_fraction', 'maximum']:.3f}\\%",
        ),
        _command("AntsDeformationAdvantage", f"{primary['mean_mae_advantage_points']:.2f}"),
        _command("AntsDeformationCILow", f"{primary['bootstrap_ci025']:.3f}"),
        _command("AntsDeformationCIHigh", f"{primary['bootstrap_ci975']:.3f}"),
        _command("AntsDeformationP", _p(float(primary["p_value_holm"]))),
        _command("AntsHodgeAdvantage", f"{hodge_comparison['mean_mae_advantage_points']:.2f}"),
        _command("AntsHodgeCILow", f"{hodge_comparison['bootstrap_ci025']:.2f}"),
        _command("AntsHodgeCIHigh", f"{hodge_comparison['bootstrap_ci975']:.2f}"),
        _command("AntsHodgeP", _p(float(hodge_comparison["p_value_holm"]))),
    ]
    macro_associations = {
        "adjusted_magnitude_vs_aq": "AntsAdjustedMagnitude",
        "adjusted_hodge_curl_free_fraction_vs_aq": "AntsAdjustedCurlFree",
        "adjusted_hodge_divergence_free_fraction_vs_aq": "AntsAdjustedDivergenceFree",
    }
    for association, stem in macro_associations.items():
        row = adjusted_index.loc[association]
        macros.extend(
            [
                _command(stem + "Rho", f"{row['partial_spearman_rho']:.3f}"),
                _command(stem + "P", _p(float(row["permutation_p_value_holm"]))),
            ]
        )
    macro_agreements = {
        "mass_effect_3_20mm_magnitude_mm_median": "AntsMagnitudeAgreement",
        "total_rms_mm": "AntsTotalRMSAgreement",
        "curl_free_energy_fraction": "AntsCurlFreeAgreement",
        "divergence_free_energy_fraction": "AntsDivergenceFreeAgreement",
    }
    for descriptor, macro in macro_agreements.items():
        macros.append(_command(macro, f"{agreement_index.loc[descriptor, 'spearman_rho']:.3f}"))
    for model, stem in {
        "lesion_plus_mass_effect": "AntsVsSvregDeformation",
        "lesion_plus_hodge": "AntsVsSvregHodge",
        "lesion_plus_mass_effect_plus_hodge": "AntsVsSvregCombined",
    }.items():
        row = prediction_index.loc[model]
        macros.extend(
            [
                _command(stem + "Difference", f"{row['ants_minus_svreg_mean_mae']:.2f}"),
                _command(stem + "CILow", f"{row['bootstrap_ci_low']:.2f}"),
                _command(stem + "CIHigh", f"{row['bootstrap_ci_high']:.2f}"),
            ]
        )
    atomic_text(generated_dir / "results.tex", "\n".join(macros) + "\n")

    qc_rows = [
        [
            row.metric,
            *(
                f"{value:.4f}"
                for value in (row.minimum, row.q1, row.median, row.q3, row.maximum)
            ),
        ]
        for row in registration_summary.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "registration_qc_table.tex",
        ["Metric", "Min", "Q1", "Median", "Q3", "Max"],
        qc_rows,
    )

    agreement_rows = [
        [
            str(row.family).capitalize(),
            DESCRIPTOR_NAMES.get(row.descriptor, row.descriptor),
            int(row.n_common_cases),
            f"{row.spearman_rho:.3f}",
            f"{row.bootstrap_ci_low:.3f}",
            f"{row.bootstrap_ci_high:.3f}",
        ]
        for row in agreement.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "descriptor_agreement_table.tex",
        ["Family", "Descriptor", "n", "$\\rho$", "CI low", "CI high"],
        agreement_rows,
    )

    association_rows = [
        [
            ASSOCIATION_NAMES.get(row.association, row.association),
            int(row.n_subjects),
            f"{row.spearman_rho:.3f}",
            f"{row.bootstrap_ci025:.3f}",
            f"{row.bootstrap_ci975:.3f}",
            _p(float(row.p_value_holm)),
        ]
        for row in associations.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "association_table.tex",
        ["Association", "n", "$\\rho$", "CI low", "CI high", "Holm p"],
        association_rows,
    )

    adjusted_rows = [
        [
            ADJUSTED_ASSOCIATION_NAMES.get(row.association, row.association),
            int(row.n_subjects),
            f"{row.partial_spearman_rho:.3f}",
            f"{row.bootstrap_ci025:.3f}",
            f"{row.bootstrap_ci975:.3f}",
            _p(float(row.permutation_p_value_holm)),
        ]
        for row in adjusted.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "adjusted_association_table.tex",
        ["Association", "n", "Partial $\\rho$", "CI low", "CI high", "Holm p"],
        adjusted_rows,
    )

    model_rows = [
        [
            MODEL_NAMES.get(row.model, row.model),
            int(row.n),
            f"{row.mae_mean:.2f}",
            f"{row.rmse_mean:.2f}",
            f"{row.r2_mean:.3f}",
            f"{row.pearson_r_mean:.3f}",
        ]
        for row in model_summary.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "model_table.tex",
        ["Model", "n", "MAE", "RMSE", "$R^2$", "$r$"],
        model_rows,
    )

    comparison_rows = [
        [
            MODEL_NAMES.get(row.reference_model, row.reference_model),
            MODEL_NAMES.get(row.comparison_model, row.comparison_model),
            f"{row.mean_mae_advantage_points:.2f}",
            (
                f"{row.bootstrap_ci025:.3f}"
                if 0 < abs(row.bootstrap_ci025) < 0.01
                else f"{row.bootstrap_ci025:.2f}"
            ),
            f"{row.bootstrap_ci975:.2f}",
            _p(float(row.p_value_holm)),
        ]
        for row in comparisons.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "paired_comparison_table.tex",
        ["Reference", "Comparison", "Advantage", "CI low", "CI high", "Holm p"],
        comparison_rows,
    )

    predictive_rows = [
        [
            MODEL_NAMES.get(row.model, row.model),
            f"{row.svreg_mean_mae:.2f}",
            f"{row.ants_mean_mae:.2f}",
            f"{row.ants_minus_svreg_mean_mae:.2f}",
            f"{row.bootstrap_ci_low:.2f}",
            f"{row.bootstrap_ci_high:.2f}",
            _p(float(row.wilcoxon_p_value)),
        ]
        for row in predictive.itertuples(index=False)
    ]
    _write_table(
        generated_dir / "predictive_method_table.tex",
        ["Model", "SVReg MAE", "ANTs MAE", "$\\Delta$MAE", "CI low", "CI high", "p"],
        predictive_rows,
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc-ants-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def histogram(axis: object, column: str, color: str) -> None:
        payload = registration_histograms[column]
        edges = np.asarray(payload["edges"], dtype=float)
        counts = np.asarray(payload["counts"], dtype=float)
        axis.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    histogram(axes[0], "registration_brain_mask_dice", "#4477AA")
    axes[0].axvline(
        registration_config["config"]["minimum_brain_mask_dice"],
        color="black",
        linestyle="--",
    )
    axes[0].set(xlabel="Brain-mask Dice", ylabel="Cases")
    histogram(axes[1], "registration_cycle_rmse_mm", "#66CCEE")
    axes[1].axvline(
        registration_config["config"]["maximum_cycle_rmse_mm"],
        color="black",
        linestyle="--",
    )
    axes[1].set(xlabel="Round-trip RMSE (mm)")
    histogram(axes[2], "registration_raw_warp_minimum_jacobian", "#228833")
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(xlabel="Raw-warp minimum Jacobian")
    figure.savefig(generated_dir / "registration_qc.pdf", bbox_inches="tight")
    figure.savefig(generated_dir / "registration_qc.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    order = np.arange(len(agreement))[::-1]
    estimates = agreement["spearman_rho"].to_numpy(float)
    low = agreement["bootstrap_ci_low"].to_numpy(float)
    high = agreement["bootstrap_ci_high"].to_numpy(float)
    axis.errorbar(
        estimates,
        order,
        xerr=np.vstack([estimates - low, high - estimates]),
        fmt="o",
        color="#AA3377",
        ecolor="#777777",
        capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(
        order,
        [DESCRIPTOR_NAMES.get(value, value) for value in agreement["descriptor"]],
    )
    axis.set(xlabel="ANTs--SVReg Spearman rank agreement", xlim=(-1, 1))
    figure.savefig(generated_dir / "descriptor_agreement.pdf", bbox_inches="tight")
    figure.savefig(generated_dir / "descriptor_agreement.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.reference_dir, args.generated_dir)


if __name__ == "__main__":
    main()
