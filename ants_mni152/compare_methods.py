"""Aggregate-only agreement and predictive comparisons for ANTs versus SVReg."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from arc_deformation.constants import DEFORMATION_FEATURES, HODGE_FEATURES
from arc_deformation.io import atomic_csv, atomic_json, read_table, sha256_file


def _bootstrap_spearman(
    first: np.ndarray, second: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    estimate = float(spearmanr(first, second).statistic)
    indices = rng.integers(0, len(first), size=(samples, len(first)))
    values = np.empty(samples, dtype=float)
    for index, sample in enumerate(indices):
        values[index] = spearmanr(first[sample], second[sample]).statistic
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return estimate, float(low), float(high)


def descriptor_agreement(
    svreg_mass: pd.DataFrame,
    ants_mass: pd.DataFrame,
    svreg_hodge: pd.DataFrame,
    ants_hodge: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute participant-bootstrap rank agreement on common QC-passing cases."""
    svreg_mass = svreg_mass.loc[
        svreg_mass["deformation_qc_pass"].astype(str).str.lower().eq("true")
    ]
    ants_mass = ants_mass.loc[
        ants_mass["deformation_qc_pass"].astype(str).str.lower().eq("true")
    ]
    svreg_hodge = svreg_hodge.loc[
        svreg_hodge["velocity_qc_pass"].astype(str).str.lower().eq("true")
    ]
    ants_hodge = ants_hodge.loc[
        ants_hodge["velocity_qc_pass"].astype(str).str.lower().eq("true")
    ]
    mass = svreg_mass.merge(ants_mass, on="case_id", suffixes=("_svreg", "_ants"))
    hodge = svreg_hodge.merge(ants_hodge, on="case_id", suffixes=("_svreg", "_ants"))
    specifications = [
        (
            "displacement",
            feature.removeprefix("me_"),
            feature.removeprefix("me_"),
            mass,
        )
        for feature in DEFORMATION_FEATURES
    ] + [
        ("hodge", feature.removeprefix("hhd_"), feature.removeprefix("hhd_"), hodge)
        for feature in HODGE_FEATURES
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for family, label, column, frame in specifications:
        first = pd.to_numeric(frame[f"{column}_svreg"], errors="coerce").to_numpy(float)
        second = pd.to_numeric(frame[f"{column}_ants"], errors="coerce").to_numpy(float)
        finite = np.isfinite(first) & np.isfinite(second)
        estimate, low, high = _bootstrap_spearman(
            first[finite], second[finite], bootstrap_samples, rng
        )
        rows.append(
            {
                "family": family,
                "descriptor": label,
                "n_common_cases": int(finite.sum()),
                "spearman_rho": estimate,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def predictive_method_comparison(
    svreg_predictions: pd.DataFrame,
    ants_predictions: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compare repeat-averaged errors when cases and shared splits are identical."""
    keys = ["case_id", "subject", "repeat", "outer_fold", "model"]
    common_models = sorted(
        set(svreg_predictions["model"]).intersection(ants_predictions["model"])
    )
    svreg_predictions = svreg_predictions.loc[
        svreg_predictions["model"].isin(common_models)
    ].copy()
    ants_predictions = ants_predictions.loc[
        ants_predictions["model"].isin(common_models)
    ].copy()
    merged = svreg_predictions.merge(
        ants_predictions,
        on=keys,
        suffixes=("_svreg", "_ants"),
        validate="one_to_one",
    )
    if len(merged) != len(svreg_predictions) or len(merged) != len(ants_predictions):
        raise ValueError("ANTs and SVReg predictions do not use the same cases and splits")
    baseline = merged.loc[merged["model"].eq("lesion_standard")]
    if not np.allclose(
        baseline["predicted_aq_svreg"], baseline["predicted_aq_ants"], atol=1e-12
    ):
        raise ValueError("Shared lesion-standard predictions differ; splits are not comparable")
    participant = (
        merged.groupby(["subject", "model"], as_index=False)[
            ["absolute_error_svreg", "absolute_error_ants"]
        ]
        .mean()
        .assign(
            ants_minus_svreg_mae=lambda frame: (
                frame["absolute_error_ants"] - frame["absolute_error_svreg"]
            )
        )
    )
    rng = np.random.default_rng(seed)
    rows = []
    for model, group in participant.groupby("model", sort=False):
        difference = group["ants_minus_svreg_mae"].to_numpy(float)
        samples = rng.choice(
            difference, size=(bootstrap_samples, len(difference)), replace=True
        )
        low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
        statistic = (
            None
            if np.allclose(difference, 0)
            else wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        )
        rows.append(
            {
                "model": model,
                "n_subjects": len(difference),
                "svreg_mean_mae": float(group["absolute_error_svreg"].mean()),
                "ants_mean_mae": float(group["absolute_error_ants"].mean()),
                "ants_minus_svreg_mean_mae": float(difference.mean()),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "wilcoxon_p_value": float(statistic.pvalue) if statistic else 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_comparison(
    svreg_mass_path: Path,
    ants_mass_path: Path,
    svreg_hodge_path: Path,
    ants_hodge_path: Path,
    svreg_predictions_path: Path,
    ants_predictions_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 5000,
    seed: int = 2026,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svreg_mass": svreg_mass_path,
        "ants_mass": ants_mass_path,
        "svreg_hodge": svreg_hodge_path,
        "ants_hodge": ants_hodge_path,
        "svreg_predictions": svreg_predictions_path,
        "ants_predictions": ants_predictions_path,
    }
    agreement = descriptor_agreement(
        read_table(svreg_mass_path),
        read_table(ants_mass_path),
        read_table(svreg_hodge_path),
        read_table(ants_hodge_path),
        bootstrap_samples,
        seed,
    )
    prediction = predictive_method_comparison(
        read_table(svreg_predictions_path),
        read_table(ants_predictions_path),
        bootstrap_samples,
        seed + 1,
    )
    atomic_csv(output_dir / "descriptor_agreement.csv", agreement)
    atomic_csv(output_dir / "predictive_method_comparison.csv", prediction)
    atomic_json(
        output_dir / "comparison_provenance.json",
        {
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "inputs": {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in paths.items()
            },
            "interpretation": (
                "Atlas-registration-pipeline sensitivity comparison on common participants; "
                "registration software, reference atlas, and grid differ together; rank "
                "agreement does not establish physical validity or equivalence"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svreg-mass", type=Path, required=True)
    parser.add_argument("--ants-mass", type=Path, required=True)
    parser.add_argument("--svreg-hodge", type=Path, required=True)
    parser.add_argument("--ants-hodge", type=Path, required=True)
    parser.add_argument("--svreg-predictions", type=Path, required=True)
    parser.add_argument("--ants-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_comparison(
        args.svreg_mass,
        args.ants_mass,
        args.svreg_hodge,
        args.ants_hodge,
        args.svreg_predictions,
        args.ants_predictions,
        args.output_dir,
        args.bootstrap_samples,
        args.seed,
    )


if __name__ == "__main__":
    main()
