# Data contract and governance

## Source dataset

The analysis reads the Aphasia Recovery Cohort (ARC) BIDS dataset. The mounted
copy used for the frozen run was:

```text
/home/ajoshi/project2_ajoshi_1183/data/ARC
```

The dataset identifies itself as OpenNeuro `ds004884`, version 1.0.2, DOI
`10.18112/openneuro.ds004884.v1.0.2`, with a CC0 license. Its
`dataset_description.json` reports University of South Carolina IRB approval
for the source studies and exemption of the anonymized released dataset.

The repository treats ARC and every ARC derivative as read-only. Commands
require a separate output directory and refuse to place analysis products under
the ARC root.

## Required cohort inputs

| Input | Required keys / fields | Role |
|---|---|---|
| `mass_effect_manifest.csv` | unique `case_id`, `subject`, `session`, method/QC columns, deformation summaries | Primary imaging predictors |
| `case_metrics.csv` | unique `case_id`, `wab_aq`, age, assessment delay, lesion burden/location | Outcome and reference predictors |
| `uncertainty_manifest.csv` | unique `case_id`, soft lesion and entropy summaries | Optional sensitivity model |

The model rejects duplicate cases, mixed deformation-method versions, multiple
cases per participant after matching, and missing outcome values. All models in
a comparison use the same QC-passing participants and the same outer folds.

## Case extraction inputs

`arc-deformation extract-case` accepts explicit paths rather than assuming a
site-specific directory layout:

- BrainSuite atlas-to-subject inverse coordinate map;
- processed subject T1 and brain mask on the inverse-map target grid;
- symmetric atlas T1 and brain mask;
- lesion mask and dilated inpainting target on the processed subject grid;
- optional direct/non-inpainted BrainSuite maps for registration sensitivity.

Lesion and target masks are mapped into atlas space. The target is always
expanded to include the lesion. A target voxel, its reflected homolog, invalid
map samples, nonpositive-Jacobian pairs, and the reference hemisphere are all
excluded from the stored lesion-effect field.

## Files that must not be committed

- raw BIDS images;
- subject-level NIfTI derivatives;
- joined clinical/design tables;
- long-form out-of-fold predictions;
- any local credentials or cluster paths not already public dataset metadata.

The `.gitignore` enforces these exclusions. Only aggregate model metrics,
aggregate coefficient summaries, provenance hashes, and manuscript figures are
checked in.
