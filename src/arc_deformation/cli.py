"""Command-line interface for extraction, audit, modeling, and paper assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arc_deformation.audit import run_audit
from arc_deformation.config import config_value, load_config
from arc_deformation.extract import ExtractionInputs, collect_metrics, extract_case
from arc_deformation.hodge import HodgeConfig, run_hodge_extraction
from arc_deformation.io import ensure_output_outside_data
from arc_deformation.modeling import ModelConfig, run_modeling
from arc_deformation.reporting import generate_report


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _resolve(value: Any, config: dict[str, Any], section: str, key: str) -> Any:
    return value if value is not None else config_value(config, section, key)


def _load_optional(path: Path | None) -> dict[str, Any]:
    return load_config(path) if path else {}


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mass-effect-manifest", type=Path)
    parser.add_argument("--clinical-table", type=Path)
    parser.add_argument("--uncertainty-manifest", type=Path)
    parser.add_argument("--hodge-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--outcome")
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-jobs", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arc-deformation", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit a completed derivative read-only")
    audit.add_argument("--config", type=Path)
    audit.add_argument("--arc-root", type=Path)
    audit.add_argument("--manifest", type=Path)
    audit.add_argument("--output-dir", type=Path)
    audit.add_argument("--expected-cases", type=int)
    audit.add_argument("--check-maps", action=argparse.BooleanOptionalAction, default=None)
    audit.add_argument("--legacy-manifest", type=Path)

    model = subparsers.add_parser("model", help="Run repeated nested-CV model comparison")
    _add_model_arguments(model)

    hodge = subparsers.add_parser(
        "hodge", help="Compute stationary log-velocity and Hodge descriptors"
    )
    hodge.add_argument("--config", type=Path)
    hodge.add_argument("--arc-root", type=Path)
    hodge.add_argument("--mass-effect-manifest", type=Path)
    hodge.add_argument("--output-dir", type=Path)
    hodge.add_argument("--stride", type=int)
    hodge.add_argument("--padding", type=int)
    hodge.add_argument("--boundary-taper-width-voxels", type=float)
    hodge.add_argument("--displacement-smoothing-sigma-voxels", type=float)
    hodge.add_argument("--velocity-squaring-steps", type=int)
    hodge.add_argument("--velocity-maximum-iterations", type=int)
    hodge.add_argument("--velocity-reconstruction-tolerance", type=float)
    hodge.add_argument("--n-jobs", type=int)

    report = subparsers.add_parser("report", help="Generate aggregate manuscript assets")
    report.add_argument("--results-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)

    collect = subparsers.add_parser(
        "collect", help="Collect completed case metrics into a cohort manifest"
    )
    collect.add_argument("--output-dir", type=Path, required=True)

    reproduce = subparsers.add_parser("reproduce", help="Audit, model, and generate a report")
    reproduce.add_argument("--config", type=Path, required=True)

    extract = subparsers.add_parser(
        "extract-case", help="Extract one explicit deformation case"
    )
    extract.add_argument("--case-id", required=True)
    extract.add_argument("--subject", required=True)
    extract.add_argument("--session", required=True)
    extract.add_argument("--inverse-map", type=Path, required=True)
    extract.add_argument("--subject-t1", type=Path, required=True)
    extract.add_argument("--subject-mask", type=Path, required=True)
    extract.add_argument("--atlas-t1", type=Path, required=True)
    extract.add_argument("--atlas-mask", type=Path, required=True)
    extract.add_argument("--lesion-mask", type=Path, required=True)
    extract.add_argument("--inpainting-target", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--arc-root", type=Path)
    extract.add_argument("--raw-inverse-map", type=Path)
    extract.add_argument("--raw-subject-t1", type=Path)
    extract.add_argument("--raw-subject-mask", type=Path)
    extract.add_argument("--smoothing-mm", type=float, default=2.0)
    extract.add_argument("--fit-subsample", type=int, default=8)
    extract.add_argument("--minimum-laterality", type=float, default=0.80)
    extract.add_argument("--make-qc", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _required_path(value: Any, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required either on the command line or in config")
    return Path(value).expanduser()


def _model_from_args(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    arc_root_value = config_value(config, "paths", "arc_root")
    arc_root = _path(arc_root_value)
    mass = _required_path(
        _resolve(args.mass_effect_manifest, config, "paths", "mass_effect_manifest"),
        "mass_effect_manifest",
    )
    clinical = _required_path(
        _resolve(args.clinical_table, config, "paths", "clinical_table"),
        "clinical_table",
    )
    uncertainty_value = _resolve(
        args.uncertainty_manifest, config, "paths", "uncertainty_manifest"
    )
    uncertainty = _path(str(uncertainty_value)) if uncertainty_value else None
    hodge_value = _resolve(args.hodge_manifest, config, "paths", "hodge_manifest")
    hodge = _path(str(hodge_value)) if hodge_value else None
    output_value = _resolve(args.output_dir, config, "paths", "output_root")
    output = _required_path(output_value, "output_dir")
    if args.output_dir is None and config_value(config, "paths", "output_root"):
        output = output / "aq_comparison"
    output = ensure_output_outside_data(output, arc_root)
    defaults = ModelConfig()
    model_config = ModelConfig(
        outcome=str(_resolve(args.outcome, config, "model", "outcome") or defaults.outcome),
        outer_folds=int(
            _resolve(args.outer_folds, config, "model", "outer_folds") or defaults.outer_folds
        ),
        inner_folds=int(
            _resolve(args.inner_folds, config, "model", "inner_folds") or defaults.inner_folds
        ),
        repeats=int(_resolve(args.repeats, config, "model", "repeats") or defaults.repeats),
        bootstrap_samples=int(
            _resolve(args.bootstrap_samples, config, "model", "bootstrap_samples")
            or defaults.bootstrap_samples
        ),
        seed=int(_resolve(args.seed, config, "model", "seed") or defaults.seed),
        n_jobs=int(_resolve(args.n_jobs, config, "model", "n_jobs") or defaults.n_jobs),
        require_deformation_qc=bool(
            config_value(config, "model", "require_deformation_qc", True)
        ),
        maximum_folding_fraction=float(
            config_value(config, "model", "maximum_folding_fraction", 0.05)
        ),
        minimum_near_lesion_voxels=int(
            config_value(config, "model", "minimum_near_lesion_voxels", 1000)
        ),
        minimum_uncertainty_coverage=float(
            config_value(config, "model", "minimum_uncertainty_coverage", 0.90)
        ),
        minimum_hodge_coverage=float(
            config_value(config, "model", "minimum_hodge_coverage", 0.90)
        ),
    )
    run_modeling(
        mass,
        clinical,
        output,
        uncertainty_manifest=uncertainty,
        config=model_config,
        hodge_manifest=hodge,
    )
    return output


def _hodge_from_args(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    defaults = HodgeConfig()
    arc_root = _required_path(_resolve(args.arc_root, config, "paths", "arc_root"), "arc_root")
    mass = _required_path(
        _resolve(args.mass_effect_manifest, config, "paths", "mass_effect_manifest"),
        "mass_effect_manifest",
    )
    output_value = _resolve(args.output_dir, config, "paths", "output_root")
    output = _required_path(output_value, "output_dir")
    if args.output_dir is None and config_value(config, "paths", "output_root"):
        output = output / "hodge"
    output = ensure_output_outside_data(output, arc_root)

    def setting(argument: Any, key: str, default: Any) -> Any:
        return argument if argument is not None else config_value(config, "hodge", key, default)

    hodge_config = HodgeConfig(
        stride=int(setting(args.stride, "stride", defaults.stride)),
        padding=int(setting(args.padding, "padding", defaults.padding)),
        boundary_taper_width_voxels=float(
            setting(
                args.boundary_taper_width_voxels,
                "boundary_taper_width_voxels",
                defaults.boundary_taper_width_voxels,
            )
        ),
        displacement_smoothing_sigma_voxels=float(
            setting(
                args.displacement_smoothing_sigma_voxels,
                "displacement_smoothing_sigma_voxels",
                defaults.displacement_smoothing_sigma_voxels,
            )
        ),
        velocity_squaring_steps=int(
            setting(
                args.velocity_squaring_steps,
                "velocity_squaring_steps",
                defaults.velocity_squaring_steps,
            )
        ),
        velocity_maximum_iterations=int(
            setting(
                args.velocity_maximum_iterations,
                "velocity_maximum_iterations",
                defaults.velocity_maximum_iterations,
            )
        ),
        velocity_reconstruction_tolerance=float(
            setting(
                args.velocity_reconstruction_tolerance,
                "velocity_reconstruction_tolerance",
                defaults.velocity_reconstruction_tolerance,
            )
        ),
    )
    n_jobs = int(setting(args.n_jobs, "n_jobs", 1))
    return run_hodge_extraction(mass, arc_root, output, hodge_config, n_jobs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        generate_report(args.results_dir, args.output_dir)
        print(args.output_dir)
        return 0
    if args.command == "collect":
        print(collect_metrics(args.output_dir))
        return 0
    if args.command == "audit":
        config = _load_optional(args.config)
        arc_root = _required_path(
            _resolve(args.arc_root, config, "paths", "arc_root"), "arc_root"
        )
        manifest = _required_path(
            _resolve(args.manifest, config, "paths", "mass_effect_manifest"), "manifest"
        )
        output_value = _resolve(args.output_dir, config, "paths", "output_root")
        output = _required_path(output_value, "output_dir")
        if args.output_dir is None and config_value(config, "paths", "output_root"):
            output = output / "audit"
        output = ensure_output_outside_data(output, arc_root)
        expected = _resolve(args.expected_cases, config, "audit", "expected_cases")
        check_maps = _resolve(args.check_maps, config, "audit", "check_maps")
        report = run_audit(
            manifest,
            output,
            arc_root,
            int(expected) if expected is not None else None,
            bool(check_maps),
            args.legacy_manifest,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "model":
        output = _model_from_args(args, _load_optional(args.config))
        print(output)
        return 0
    if args.command == "hodge":
        manifest = _hodge_from_args(args, _load_optional(args.config))
        print(manifest)
        return 0
    if args.command == "reproduce":
        config = load_config(args.config)
        arc_root = _required_path(config_value(config, "paths", "arc_root"), "arc_root")
        output_root = ensure_output_outside_data(
            _required_path(config_value(config, "paths", "output_root"), "output_root"),
            arc_root,
        )
        run_audit(
            _required_path(
                config_value(config, "paths", "mass_effect_manifest"), "mass_effect_manifest"
            ),
            output_root / "audit",
            arc_root,
            int(config_value(config, "audit", "expected_cases", 214)),
            bool(config_value(config, "audit", "check_maps", False)),
        )
        hodge_namespace = argparse.Namespace(
            arc_root=None,
            mass_effect_manifest=None,
            output_dir=None,
            stride=None,
            padding=None,
            boundary_taper_width_voxels=None,
            displacement_smoothing_sigma_voxels=None,
            velocity_squaring_steps=None,
            velocity_maximum_iterations=None,
            velocity_reconstruction_tolerance=None,
            n_jobs=None,
        )
        hodge_manifest = _hodge_from_args(hodge_namespace, config)
        model_namespace = argparse.Namespace(
            mass_effect_manifest=None,
            clinical_table=None,
            uncertainty_manifest=None,
            hodge_manifest=hodge_manifest,
            output_dir=None,
            outcome=None,
            outer_folds=None,
            inner_folds=None,
            repeats=None,
            bootstrap_samples=None,
            seed=None,
            n_jobs=None,
        )
        model_output = _model_from_args(model_namespace, config)
        generate_report(model_output, output_root / "paper_assets")
        print(output_root)
        return 0
    if args.command == "extract-case":
        output = ensure_output_outside_data(args.output_dir, args.arc_root)
        inputs = ExtractionInputs(
            case_id=args.case_id,
            subject=args.subject,
            session=args.session,
            inverse_map=args.inverse_map,
            subject_t1=args.subject_t1,
            subject_mask=args.subject_mask,
            atlas_t1=args.atlas_t1,
            atlas_mask=args.atlas_mask,
            lesion_mask=args.lesion_mask,
            inpainting_target=args.inpainting_target,
            output_dir=output,
            raw_inverse_map=args.raw_inverse_map,
            raw_subject_t1=args.raw_subject_t1,
            raw_subject_mask=args.raw_subject_mask,
        )
        metrics = extract_case(
            inputs,
            args.smoothing_mm,
            args.fit_subsample,
            args.minimum_laterality,
            args.make_qc,
        )
        print(json.dumps({"case_id": metrics["case_id"], "output_dir": str(output)}, indent=2))
        return 0
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
