# ARC Deformation Analysis

Production-oriented code and manuscript sources for contralateral-normalized
deformation analysis in the [Aphasia Recovery Cohort](https://doi.org/10.18112/openneuro.ds004884.v1.0.2).
The package constructs a lesional-only displacement proxy, validates a
stationary log-velocity embedding, performs a periodic Helmholtz--Hodge
decomposition, and tests incremental prediction of Western Aphasia Battery
Aphasia Quotient (WAB-AQ). An independent ANTs/SyN subanalysis repeats the
complete workflow in symmetric MNI152 space and quantifies
registration-pipeline sensitivity.

The result is conservative. In 210 QC-passing participants, adding displacement
features to a conventional lesion model changed held-out MAE by 0.38 AQ points
(95% participant-bootstrap CI −0.50 to 1.17). Log-velocity Hodge features did
not improve prediction after lesion or displacement features. Hodge descriptor
ranks were stable across eight smoothing, taper, padding, and grid perturbations,
but adjusted component--AQ associations did not survive multiplicity correction.
The ANTs/MNI152 deformation model showed a small 0.81-point advantage (95% CI
0.004 to 1.608), unlike the inconclusive SVReg result; weak agreement for some
cross-pipeline descriptors makes this a pipeline-sensitivity finding rather
than a registration-invariant biomarker.
The analysis does not estimate pressure, physical tissue velocity, force, or
causal mass effect.

## Repository scope

- `src/arc_deformation/`: installable extraction, audit, log-domain, Hodge,
  modeling, and reporting package;
- `tests/`: deterministic geometry, decomposition, QC, and modeling tests;
- `config/`: portable full-run configuration;
- `results/reference/`: aggregate, non-participant-level frozen results;
- `paper/`: Imaging Neuroscience manuscript, supplement, cover letter, and
  verified bibliography;
- `ants_mni152/`: ANTs forward/inverse mappings, aggregate pipeline comparison,
  and standalone Supplementary Material 2;
- `docs/`: input contracts, method specification, lesion-delineation boundary,
  and provenance.

The repository is self-contained from the documented derivative-input boundary.
It does not redistribute raw ARC images, trained nnU-Net weights, BrainSuite, or
participant-level outputs. Lesion delineation code is public in the pinned
[TR2StrokeSeg source](https://github.com/ajoshiusc/TR2StrokeSeg/tree/286cd508f1bbbcbc3c5182db32db41a5b4797eb7);
see `docs/LESION_DELINEATION.md` for the exact handoff.

## Install and test

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check src tests ants_mni152
```

Install the pinned ANTs and TemplateFlow dependencies only when running the
independent registration analysis:

```bash
.venv/bin/pip install -e '.[ants,dev]'
```

## Reproduce the analysis

Copy `config/arc.example.toml` to `config/local.toml`, change only local paths,
and run:

```bash
.venv/bin/python -m arc_deformation.cli reproduce --config config/local.toml
```

The full command performs a read-only derivative audit, extracts log-velocity
Hodge descriptors, runs repeated nested cross-validation and sensitivity
analyses, and creates aggregate paper assets. It refuses to place outputs below
the ARC data root.

Individual stages are also available:

```bash
arc-deformation audit --config config/local.toml
arc-deformation hodge --config config/local.toml
arc-deformation hodge-sensitivity --config config/local.toml \
  --primary-hodge-manifest results/runs/hodge/hodge_manifest.csv
arc-deformation model --config config/local.toml \
  --hodge-manifest results/runs/hodge/hodge_manifest.csv
arc-deformation report \
  --results-dir results/runs/aq_comparison \
  --output-dir paper/generated
```

`arc-deformation extract-case --help` documents the explicit inputs for one
deformation case. `arc-deformation collect` rebuilds a deterministically sorted
cohort manifest.

The complete ANTs analysis is isolated in `ants_mni152/`. Its driver downloads
the identified TemplateFlow reference when explicit template paths are not
given, writes every participant transform outside the ARC root and repository,
then freezes only de-identified aggregate results:

```bash
.venv/bin/python ants_mni152/run_analysis.py --help
```

See `ants_mni152/README.md` for the exact path contract, transform-direction
convention, QC thresholds, resume behavior, and full command.

## Build the submission

```bash
make submission
```

This regenerates every table, macro, and statistical figure from aggregate CSV
and JSON outputs, then builds:

- `paper/main.pdf`;
- `paper/supplement.pdf`;
- `ants_mni152/paper/supplement_ants.pdf`;
- `paper/submission.pdf` (combined manuscript and both supplements for first
  submission);
- `paper/cover_letter.pdf`.

The final author list, funding, and competing-interest declaration are factual
metadata that must be confirmed by the author before upload; they are never
inferred by the code.

## Interpretation boundary

The stored displacement is an affine-removed, mirrored-difference,
lesional-only registration proxy. Its regularized stationary logarithm is a
group parameter with displacement units over an arbitrary unit interval—not a
measured velocity in mm/s. Hodge components depend on the stated taper,
smoothing, padding, and periodic boundary convention. Pressure reconstruction
would require a constitutive biomechanical model, material parameters, loads,
and boundary conditions that ARC does not provide.

## Reproducibility and privacy

The frozen analysis used 20 repetitions of five-fold outer CV, four-fold inner
CV, ridge penalties selected entirely within training folds, 5,000 participant
bootstrap resamples, 5,000 residual permutations for adjusted associations,
eight one-factor Hodge perturbations, and seed 2026. Input SHA-256 values and
result provenance are in `docs/RESULTS_PROVENANCE.md`.

Raw images, NIfTI derivatives, joined clinical/design tables, per-participant
Hodge descriptors, coefficients by fold, and out-of-fold predictions are
ignored. Only aggregate metrics, aggregate coefficient summaries, audit hashes,
and manuscript figures are committed.

## Citation

See `CITATION.cff` and `paper/references.bib`. The public repository is
<https://github.com/ajoshiusc/ARCDeformationAnalysis>.
