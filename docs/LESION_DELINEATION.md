# Lesion delineation provenance

The deformation paper repository consumes completed lesion and uncertainty
derivatives; it does not silently imply that lesions were hand drawn or that the
raw-to-lesion stage is implemented here.

## Public source

The nnU-Net v2 training and ARC inference source is available at:

```text
https://github.com/ajoshiusc/TR2StrokeSeg
commit 286cd508f1bbbcbc3c5182db32db41a5b4797eb7
```

Relevant entry points at that immutable revision are:

- `src/data_preparation/prepare_atlas2.py`: converts ATLAS v2 T1/mask pairs to
  nnU-Net format;
- `src/training/train_nnunet.py`: plans, preprocesses, and trains the
  full-resolution nnU-Net;
- `stroke_brainsuite_analysis/run_arc_lesion_uncertainty.py`: runs ARC inference,
  exports lesion probability, binary entropy, the 0.5 hard mask, scalar
  uncertainty features, QC images, and a locked case manifest;
- `stroke_brainsuite_analysis/analyze_arc_lesion_dice.py`: registers available
  same-session expert T2 lesion masks and calculates overlap/surface metrics with
  registration QC.

The source is GPL-2.0, matching this repository. nnU-Net itself is an external
dependency and should be installed at the version recorded by the upstream
environment. Trained checkpoints are not source code and are not redistributed
here.

## Frozen ARC handoff

The paper analysis reads:

```text
ARC/derivatives/lesion_uncertainty/uncertainty_manifest.csv
ARC/derivatives/aq_mass_effect_inputs_v2/case_metrics.csv
ARC/derivatives/lesion_mass_effect_v2/mass_effect_manifest.csv
```

The first manifest records probability/mask outputs and uncertainty summaries.
The case table records the lesion volumes and language-network lesion burdens
used by the conventional model. The mass-effect manifest records the lesion and
exclusion masks used during deformation extraction. All joins require a unique
`case_id`; the analysis fails on duplicates or stale method versions.

## What “self-contained” means here

From these documented derivative inputs, this repository is self-contained:
audit, deformation case extraction, stationary logarithm, Hodge decomposition,
statistics, figures, and LaTeX generation are all included and tested. Starting
from raw ARC T1-weighted images additionally requires the pinned lesion code,
trained nnU-Net weights, and BrainSuite/SVReg. The paper and README state this
boundary explicitly.
