# Reproducing the analysis

Three entry points, depending on what you have.

| You have | Start at |
|---|---|
| Released weights + your own CT | [Inference only](#inference-only) |
| The original data | [Full reproduction](#full-reproduction) |
| Per-case result CSVs | [Tables and figures](#tables-and-figures) |

## Setup

```bash
export ADPKD_ROOT="$PWD"
cp config/config.example.json config.json        # edit the paths inside

# inference with the released weights
export nnUNet_results="$PWD/weights/nnUNet_results"
export nnUNet_raw="$PWD/weights/nnUNet_results"
export nnUNet_preprocessed="$PWD/weights/nnUNet_results"
export NNUNETV2_PREDICT_BIN="$(command -v nnUNetv2_predict)"

# only for the container launchers in scripts/
export IMAGE=/path/to/nnunet-container.sqsh
export YOLO_ENV=/path/to/conda/envs/adpkd-analysis
export ENROOT_RESTRICT_DEV=yes                   # do not omit
```

## Inference only

With `weights/` populated ([docs/WEIGHTS.md](WEIGHTS.md)) and `config/model_registry.json`
written, and `NNUNET_RESULTS` exported to `weights/nnUNet_results` (see the README quick
start — without it the runners look in the in-tree `work/nnUNet_results` instead):

```bash
python code/16_run_staged_pipeline.py --folds 0 1 2 3 4 --branches inferior middle
```

Stage by stage this is: L3 segmentation over the fold, then one detector load per branch,
then one body-composition call per branch. `code/13_run_end_to_end_5fold.py` does the same
work case by case, which is slower but easier to follow.

Outputs land under `results/`: per-case L3 masks, the selected section as NIfTI, the
four-class body composition mask, and the CSAs.

**Fold matching is mandatory.** A case must be processed by the models of the fold in which
it was a validation case, or the result is contaminated by training exposure. The registry
enforces this; do not hand-edit it.

## Full reproduction

### 1. Data preparation

```bash
python code/01_build_dataset.py            # clean per-case dataset
python code/02_prepare_l3_nnunet.py        # nnU-Net raw, L3 vertebra
python code/03_refresh_label_stats.py
python code/04_build_yolo_datasets.py      # per-fold YOLO sets from the MANUAL L3 masks
python code/05_render_bbox_qc.py           # QC, optional
```

The re-annotation round (`06`, `07`, `09`) applies only to the cases whose reference
section a reviewer moved. It needs `PRIOR_NNUNET_RESULTS`.

```bash
python code/10_prepare_branch_nnunet.py    # nnU-Net raw, one per level
python code/11_build_manifests.py
```

### 2. Training

```bash
bash scripts/run_l3_5fold_enroot.sh          # 3d_fullres, 500 epochs
bash scripts/run_yolo_5fold_enroot.sh        # YOLOv8n, both branches
bash scripts/run_bodycomp_5fold_enroot.sh    # 2d, 500 epochs, both branches
```

Then register the checkpoints:

```bash
python code/12_prepare_model_registry.py
```

This writes `checkpoint_best.pth` into the registry. That is deliberate — see
[docs/WEIGHTS.md](WEIGHTS.md#which-checkpoint-and-why-it-matters).

### 3. End-to-end inference and evaluation

```bash
python code/16_run_staged_pipeline.py --folds 0 1 2 3 4 --branches inferior middle
python code/14_evaluate_end_to_end.py        # end-to-end scores
python code/15_evaluate_detector_only.py     # detectors alone
python code/17_evaluate_detector_on_gt_mask.py   # detectors on the manual L3 mask
```

### 4. Supplementary analyses

```bash
bash analysis/offset/run_offset_direct.sh         # ±1, ±2 section offsets (Supp. Table 6)
bash analysis/z5mm/run_z5mm_direct.sh             # uniform 5 mm grid   (condition 2)
bash analysis/z3mm/run_z3mm_direct.sh             # uniform 3 mm grid   (condition 3)
bash analysis/public_model/run_public_direct.sh   # needs PUBLIC_MODEL_REPO
python analysis/external_reference/scripts/01_summarize_clinical_csv.py
python analysis/sex_stratified/compute_sex_stratified.py
```

The two grid runs are the comparison conditions of Table 2. Each resamples every volume
along z, then re-runs L3 segmentation and detection with the same fold-matched weights;
they are reported for comparison only, because the detectors were trained at the original
spacing and a uniform grid adds quantization error of up to half the target spacing.

The offset run at **offset 0** is the segmentation-only condition: the manual reference
section as input, the fold-matched `checkpoint_best`, one dedicated inference run. Table 3
and the Table 4 segmentation-only row are built from it, so that every published number
comes from the same weights and the same inference route.

## Tables and figures

```bash
python results/generate_final_results.py     # every published table from the per-case CSVs

python figures/bland_altman/make_bland_altman_inferior_end2end.py     # Figure 2
python figures/segmentation_map/make_inferior_segmentation_maps.py \
       --cases figure3:FOLD:CASE_ID figure4:FOLD:CASE_ID              # Figures 3, 4
python figures/figure_candidates/make_figure_candidates.py            # candidate cases

python figures/supplementary/make_supp_fig1.py \
       --cases CASE_ID:FOLD:CORONAL_OFFSET CASE_ID:FOLD:CORONAL_OFFSET   # Supp. Figure 1
python figures/supplementary/make_supp_fig2.py --case CASE_ID --fold N   # Supp. Figure 2
python figures/supplementary/sweep_coronal.py CASE_ID FOLD [OFFSETS...]  # pick the plane
```

Supplementary Figure 1 shows two scans whose reference section was placed on an adjacent
vertebra. `sweep_coronal.py` renders a contact sheet of candidate coronal planes so the
locator panel can be cut where the vertebrae are actually visible; the offsets used for the
published figure were +20 and +25 voxels from the L3 centroid.

Supplementary Figure 2 runs the same body composition model twice on one scan — once on the
manual reference section, once on the section the pipeline chose — so the cost of a stage-1
failure is visible with the segmentation model held fixed.

Figures write to `$ADPKD_FIGDIR`, or to `figures_out/` if that is unset.

Figure 2 is the **end-to-end** analysis at the L3 inferior level, read from
`results/evaluation/per_case_end_to_end.csv` with `branch == "inferior"`.

Figures 3 and 4 need explicit case identifiers, which are hospital record numbers and are
not distributed. `make_figure_candidates.py` ranks candidates by total absolute CSA error
among exactly localised cases, which is how the published cases were chosen.

## Verification

```bash
python analysis/verification/recompute_best_weights.py   # Tables 2 and 3 from checkpoint_best
python analysis/verification/recompute_downstream.py     # Supp. Tables 3, 5, 6, 7
python analysis/verification/verify_sex_and_l3.py        # sex-stratified, exact P values
bash  analysis/verification/run_ckpt_test.sh             # best vs final checkpoint

python analysis/verification/evaluate_in_original_mm.py  # localization error in original mm
python analysis/verification/level_misid.py              # 20 mm rule -> Table 2, Supp. Table 2
python analysis/verification/compare_grids.py            # native vs 5 mm vs 3 mm
python analysis/verification/l3_body_height.py           # L3 height, the basis of the 20 mm rule
```

Run these four in that order: `level_misid.py` consumes the per-case table that
`evaluate_in_original_mm.py` writes beside it. Together they produce Table 2, Supplementary
Table 2 and Supplementary Table 3, and the threshold sensitivity check behind the 20 mm
level-misidentification rule. All four write their CSVs next to themselves, and those CSVs
are git-ignored because some carry case identifiers.

### A note on table numbers in the verification scripts

`recompute_best_weights.py` and `recompute_downstream.py` predate the final table
numbering, and their docstrings and printed headings still use the draft numbers. The
mapping to the published article is:

| Printed heading | Published as |
|---|---|
| Table 2 (segmentation-only) | **Table 3** |
| Supplementary Table 3 (external reference) | **Supplementary Table 5** |
| Supplementary Tables 5 and 6 (public model) | **Supplementary Tables 7 and 8** |
| Supplementary Table 7 (sex-stratified) | **Supplementary Table 9** |

The numbers they compute are unaffected; only the labels moved.

`verify_sex_and_l3.py` also checks that the 159:141 weighted averages reproduce the pooled
values, which catches a mis-joined sex column.

## Runtime

On one H100, per fold: L3 training several hours, each body-composition model several
hours, detectors under an hour. End-to-end inference over 60 validation cases is minutes.
Two folds run concurrently, one per GPU — provided `ENROOT_RESTRICT_DEV=yes` is set.
