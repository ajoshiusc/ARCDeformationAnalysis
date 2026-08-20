# Data contract and governance

## Source dataset

The analysis reads a local, read-only copy of the Aphasia Recovery Cohort (ARC)
BIDS dataset. Local mount paths are deliberately excluded from the public
configuration; input identities are recorded by release and SHA-256 instead.

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
| `hodge_manifest.csv` | unique `case_id`, method/QC fields, aggregate log-velocity/Hodge descriptors | Generated secondary predictors |
| variant `hodge_manifest.csv` files | unique `case_id`, variant setting, QC, and descriptors | Private numerical-sensitivity intermediates |
| `stroke_inpainting/manifest.csv` | unique `case_id`; lesion-filled T1w, brain mask, lesion mask, dilated target paths | Independent ANTs registration inputs |
| TemplateFlow `MNI152NLin2009aSym` | resolution-1 T1w image and brain mask | Fixed ANTs reference and reflection geometry |

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
- participant-level ANTs affine matrices, forward/inverse warps, warped images,
  and QC mosaics;
- joined clinical/design tables;
- primary and numerical-variant case-level Hodge manifests;
- long-form out-of-fold predictions;
- any local credentials or cluster paths not already public dataset metadata.

The `.gitignore` enforces these exclusions. Only aggregate model metrics,
aggregate coefficient summaries, provenance hashes, and manuscript figures are
checked in.

## Reproducibility boundary

This repository contains all code from the deformation-derivative inputs to the
paper outputs. It does not duplicate external applications, source imaging, or
trained weights. In particular:

- lesion training, inference, probability export, and validation code is pinned
  in `docs/LESION_DELINEATION.md`;
- nnU-Net and trained checkpoints are external software/data dependencies;
- BrainSuite/SVReg is an external application;
- ANTsPy 0.6.1 and TemplateFlow 25.1.2 are pinned optional dependencies; the
  retrieved MNI152 reference files are identified by SHA-256 in the ANTs
  provenance record;
- the ARC images and participant-level derivatives remain under the ARC data
  root and are read-only.

This boundary is intentional: checking participant images or multi-gigabyte
model checkpoints into the public analysis repository would be neither portable
nor appropriate. Portable basenames, versions, hashes, and commands needed at
the boundary are recorded instead; site-local paths stay in ignored
configuration files.
