# ARC Deformation Analysis

Reproducible analysis of contralateral-normalized, lesion-associated deformation
in the Aphasia Recovery Cohort (ARC). The repository extracts a cross-sectional
registration-derived deformation proxy, audits its spatial support, and tests its
incremental value for predicting Western Aphasia Battery Aphasia Quotient (WAB-AQ).

The central result is deliberately conservative: in 210 QC-passing ARC
participants, adding deformation to a conventional lesion model changed held-out
mean absolute error by **0.38 AQ points** (95% participant-bootstrap CI
**-0.50 to 1.17**). The interval crosses zero, so the analysis does not establish
incremental predictive benefit.

## What this repository contains

- `src/arc_deformation/`: installable analysis package;
- `tests/`: unit and smoke tests for geometry, QC, and modeling;
- `config/`: portable ARC configuration example;
- `results/reference/`: aggregate, non-participant-level results from the frozen run;
- `paper/`: standalone LaTeX manuscript and verified bibliography;
- `docs/`: method, data-governance, and provenance details.

Raw ARC images, subject-level tables, NIfTI derivatives, and out-of-fold
participant predictions are intentionally not committed.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Audit the frozen ARC deformation derivative without writing into ARC:

```bash
arc-deformation audit \
  --arc-root /home/ajoshi/project2_ajoshi_1183/data/ARC \
  --manifest /home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/lesion_mass_effect_v2/mass_effect_manifest.csv \
  --output-dir results/runs/audit
```

Reproduce the predictive comparison:

```bash
arc-deformation model \
  --mass-effect-manifest /home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/lesion_mass_effect_v2/mass_effect_manifest.csv \
  --clinical-table /home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/aq_mass_effect_inputs_v2/case_metrics.csv \
  --uncertainty-manifest /home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/lesion_uncertainty/uncertainty_manifest.csv \
  --output-dir results/runs/aq_comparison \
  --n-jobs 4
```

Case extraction is deliberately explicit and scheduler-friendly: run
`arc-deformation extract-case --help` for one acquisition, then rebuild a
deterministically sorted cohort manifest with
`arc-deformation collect --output-dir results/runs/deformation`.

Build the checked-in manuscript from the frozen aggregate results:

```bash
make paper
```

For a full local run, copy `config/arc.example.toml` to `config/local.toml`,
update paths, and run `make reproduce CONFIG=config/local.toml`.

## Interpretation boundary

The output is a **cross-sectional lesion-associated deformation proxy**, not
physical ground-truth mass effect and not pressure. A chronic scan cannot
separate premorbid asymmetry, tissue collapse, ventricular expansion, remodeling,
and residual registration error. Synthetic inpainting-target voxels and their
mirrors are excluded from every biological summary.

## Reproducibility

The reference run used one scan per participant, 5-fold outer CV repeated 20
times, 4-fold inner CV for ridge-penalty selection, seed 2026, and identical
outer splits for all models. Input SHA-256 values and exact result provenance are
listed in `docs/RESULTS_PROVENANCE.md`.

## Citation

See `CITATION.cff`. Cite the ARC data descriptor and BrainSuite/SVReg methods as
listed in `paper/references.bib`.
