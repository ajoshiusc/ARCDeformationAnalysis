"""Run the complete ANTs/MNI152 registration-pipeline sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_methods import run_comparison
from freeze_results import freeze
from generate_supplement import generate

from arc_deformation.ants_mni import (
    AntsMNIConfig,
    copy_templateflow_reference,
    run_ants_mni_cohort,
)
from arc_deformation.constants import ANTS_METHOD_VERSION
from arc_deformation.hodge import HodgeConfig, run_hodge_extraction
from arc_deformation.modeling import ModelConfig, run_modeling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-root", type=Path, required=True)
    parser.add_argument("--inpainting-manifest", type=Path, required=True)
    parser.add_argument("--clinical-table", type=Path, required=True)
    parser.add_argument("--uncertainty-manifest", type=Path, required=True)
    parser.add_argument("--svreg-manifest", type=Path, required=True)
    parser.add_argument("--svreg-hodge-manifest", type=Path, required=True)
    parser.add_argument("--svreg-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--template-t1", type=Path)
    parser.add_argument("--template-mask", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--registration-jobs", type=int, default=8)
    parser.add_argument("--hodge-jobs", type=int, default=8)
    parser.add_argument("--model-jobs", type=int, default=4)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent.parent
    if args.output_root.resolve().is_relative_to(repository_root):
        raise ValueError(
            "--output-root contains participant-level images and transforms and must be "
            "outside the repository"
        )

    if (args.reference_dir is None) != (args.generated_dir is None):
        raise ValueError("Provide both --reference-dir and --generated-dir or neither")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.template_t1 is None) != (args.template_mask is None):
        raise ValueError("Provide both --template-t1 and --template-mask or neither")
    if args.template_t1 is None:
        template_t1, template_mask = copy_templateflow_reference(args.output_root)
    else:
        template_t1, template_mask = args.template_t1, args.template_mask

    registration_root = args.output_root / "registration"
    mass_manifest = run_ants_mni_cohort(
        args.inpainting_manifest,
        args.arc_root,
        template_t1,
        template_mask,
        registration_root,
        AntsMNIConfig(),
        args.registration_jobs,
        selection_manifest=args.svreg_manifest,
    )
    hodge_root = args.output_root / "hodge"
    hodge_manifest = run_hodge_extraction(
        mass_manifest,
        args.arc_root,
        hodge_root,
        HodgeConfig(stride=2),
        args.hodge_jobs,
        expected_method_version=ANTS_METHOD_VERSION,
    )
    model_root = args.output_root / "model"
    run_modeling(
        mass_manifest,
        args.clinical_table,
        model_root,
        uncertainty_manifest=args.uncertainty_manifest,
        config=ModelConfig(
            expected_method_version=ANTS_METHOD_VERSION,
            minimum_near_lesion_voxels=125,
            n_jobs=args.model_jobs,
        ),
        hodge_manifest=hodge_manifest,
    )
    run_comparison(
        args.svreg_manifest,
        mass_manifest,
        args.svreg_hodge_manifest,
        hodge_manifest,
        args.svreg_predictions,
        model_root / "aq_mass_effect_predictions_long.csv",
        args.output_root / "comparison",
    )
    if args.reference_dir is not None:
        freeze(args.output_root, args.reference_dir)
        generate(args.reference_dir, args.generated_dir)


if __name__ == "__main__":
    main()
