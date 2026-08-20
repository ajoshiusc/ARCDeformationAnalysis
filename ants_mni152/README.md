# Independent ANTs registration analysis in MNI152 space

This subanalysis repeats the deformation, stationary log-velocity,
Helmholtz--Hodge, association, and nested-cross-validation analyses with an
independent atlas-registration pipeline. It is a pipeline sensitivity
analysis, not a second biological cohort.

The inpainting derivative may contain multiple acquisitions per participant.
The primary SVReg manifest is therefore also the explicit case-selection
manifest: all and only its 214 one-scan-per-participant case IDs are registered.
The driver rejects missing or duplicate selections.

The fixed reference is TemplateFlow's symmetric
`MNI152NLin2009aSym` 1-mm T1w template and brain mask. Lesion-filled ARC T1w
images receive one ANTs rigid, affine, and SyN fit that supplies forward and
inverse mappings.
Registration and field extraction use a 2-mm grid; the Hodge analysis uses a
2-voxel stride, preserving the primary analysis's 4-mm physical analysis grid.
Every registration uses one ITK thread, random seed 2026, brain masks at all
stages, and ANTsPy's `antsRegistrationSyNQuickRepro[s]` preset.

## Why this formulation

- Holding lesion-filled inputs, masks, and downstream statistics constant while
  changing software, atlas, and grid quantifies whole-pipeline sensitivity; it
  does not isolate a single ANTs-versus-SVReg software effect.
- A dense fixed-to-moving point map is constructed explicitly. ANTs point
  transforms and image-resampling transforms have counterintuitive direction
  conventions, so both directions and round-trip error are recorded.
- The code rejects a template unless an axis-0 array flip is exactly a
  reflection through world x=0. This prevents biased contralateral comparisons
  on templates whose midpoint lies between voxels.
- Physical QC thresholds are resolution invariant: the near-lesion support
  threshold is 1,000 mm³ (125 voxels at 2-mm isotropic resolution), and
  affine-fit RMSE is converted from source voxels to millimetres.
- Registration QC requires brain-mask Dice at least 0.70, fixed-to-moving-to-fixed
  cycle RMSE no greater than 0.50 mm, and raw SyN nonpositive-Jacobian fraction
  no greater than 0.001. The original deformation laterality and normalized-field
  folding criteria still apply.

The quick reproducible preset was selected after an end-to-end one-case
engineering benchmark. Relative to the full reproducible preset, it reduced
elapsed time from 348.3 to 87.8 seconds while brain-mask Dice changed from
0.9012 to 0.9008 and intensity correlation from 0.5643 to 0.5493. These
measurements guided computational design; they are not cohort evidence.

## Installation and execution

```bash
python -m venv .venv
.venv/bin/pip install -e '.[ants,dev]'
cp ants_mni152/config.example.toml ants_mni152/config.local.toml
```

The complete driver uses explicit paths so protected participant data are never
copied into Git:

```bash
.venv/bin/python ants_mni152/run_analysis.py \
  --arc-root /path/to/ARC \
  --inpainting-manifest /path/to/stroke_inpainting/manifest.csv \
  --clinical-table /path/to/case_metrics.csv \
  --uncertainty-manifest /path/to/uncertainty_manifest.csv \
  --svreg-manifest /path/to/svreg/mass_effect_manifest.csv \
  --svreg-hodge-manifest /path/to/svreg/hodge_manifest.csv \
  --svreg-predictions /path/to/svreg/aq_mass_effect_predictions_long.csv \
  --output-root /path/outside/ARC/ants_mni152 \
  --reference-dir ants_mni152/results/reference \
  --generated-dir ants_mni152/paper/generated
```

The driver is resumable at the participant level. It writes participant images
and transforms only under `--output-root`, then produces aggregate Hodge,
modeling, agreement, and predictive-comparison tables. When the two final
arguments are supplied, a separate freeze step writes only de-identified
aggregate reference results, histogram counts, and provenance to the repository
and regenerates the supplemental assets from that public snapshot.

## Direction convention

For image resampling, the ANTs forward transform list brings the moving subject
image into fixed MNI152 space. For point data, the same list maps fixed MNI152
physical points to moving subject physical points. The latter is the inverse
coordinate map required by this analysis. The inverse list maps moving points
back to fixed points and is used for cycle QC.

## Interpretation

ANTs/SyN estimates a diffeomorphic registration parameter. Subsequent
stationary-log and Hodge quantities remain geometric descriptors of a
regularized, lesional-only asymmetry field. They are not measured tissue
velocity, pressure, force, or material properties.
