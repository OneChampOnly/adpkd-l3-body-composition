# Data

## Not released

The CT images, the manual segmentation masks and the patient-level clinical table are
**not** distributed. Their use and disclosure are governed by institutional review board
approval and applicable privacy regulations. De-identified data may be made available from
the corresponding author on reasonable request and with the permission of the
data-providing institution.

This covers:

- source and preprocessed CT volumes, and the nnU-Net raw/preprocessed trees
- manual L3 vertebral masks and manual body composition masks
- the clinical table (age, sex, height, weight, BMI, TKV, htTKV)
- case manifests, `splits_final.json`, and every per-case result CSV — all keyed by
  hospital record number

`.gitignore` blocks these paths, but check `git status` before your first push.

## No identifiers in this repository

Case identifiers are hospital record numbers. They were removed from the code and are read
from the environment instead:

| Variable | Needed by |
|---|---|
| `ADPKD_CLINICAL_CSV` | `analysis/sex_stratified/`, `analysis/external_reference/` |
| `ADPKD_REPLACEMENT_IDS`, `ADPKD_RETIRED_IDS` | `analysis/public_model/scripts/` — the two cases replaced at re-annotation, comma-separated |
| `--cases NAME:FOLD:CASE_ID` | `figures/segmentation_map/` — which cases to render |

Before publishing, confirm the repository is clean:

```bash
grep -rnE '\b[0-9]{7,8}\b' --include='*.py' --include='*.sh' .
grep -rn '/home/\|/data/' --include='*.py' --include='*.sh' .
```

Both should return nothing.

## Cohort, for context

300 abdominal CT scans from patients with ADPKD at one institution; 159 women, 141 men.
Contrast phase: 250 precontrast, 1 arterial, 1 delayed, 48 undetermined. Matrix 512 × 512.
z-axis spacing ranged 1.6–6.0 mm, median 5 mm (5 mm in 203 scans, 3 mm in 74, and 23 scans
spread across 1.6, 2.0, 2.4, 2.5, 3.75 and 6.0 mm).

Ground truth: SMA, VFAT, SFAT and IMAT annotated in 3D Slicer at both the L3 inferior
endplate and the L3 middle section, reviewed by two board-certified physicians.

## Bringing your own data

The pipeline expects, per case, a CT volume in NIfTI with original spacing and Hounsfield
units preserved. No global HU normalisation, no isotropic resampling. `code/01`–`code/11`
build the dataset, the nnU-Net raw trees and the manifests; `config.json` names the source
folders. With the released weights you can skip straight to `code/16_run_staged_pipeline.py`.
