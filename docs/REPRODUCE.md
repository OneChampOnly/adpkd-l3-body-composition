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
export IMAGE=/path/to/nnunet-container.sqsh      # launcher scripts
export YOLO_ENV=/path/to/conda/envs/ultralytics  # launcher scripts
export ENROOT_RESTRICT_DEV=yes                   # do not omit

cp config/config.example.json config.json        # edit the paths inside
```

## Inference only

With `weights/` populated ([docs/WEIGHTS.md](WEIGHTS.md)) and `config/model_registry.json`
written:

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
bash analysis/offset/run_offset_direct.sh         # ±1, ±2 section offsets (Supp. Table 4)
bash analysis/z5mm/run_z5mm_direct.sh             # uniform 5 mm grid (Supp. Results)
bash analysis/public_model/run_public_direct.sh   # needs PUBLIC_MODEL_REPO
python analysis/external_reference/scripts/01_summarize_clinical_csv.py
python analysis/sex_stratified/compute_sex_stratified.py
```

The offset run at **offset 0** is the segmentation-only condition: the manual reference
section as input, the fold-matched `checkpoint_best`, one dedicated inference run. Table 2
and the Table 3 segmentation-only column are built from it, so that every published number
comes from the same weights and the same inference route.

## Tables and figures

```bash
python results/generate_final_results.py     # every published table from the per-case CSVs

python figures/bland_altman/make_bland_altman_inferior_end2end.py     # Figure 2
python figures/segmentation_map/make_inferior_segmentation_maps.py \
       --cases figure3:FOLD:CASE_ID figure4:FOLD:CASE_ID              # Figures 3, 4
python figures/figure_candidates/make_figure_candidates.py            # candidate cases
```

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
```

`verify_sex_and_l3.py` also checks that the 159:141 weighted averages reproduce the pooled
values, which catches a mis-joined sex column.

## Runtime

On one H100, per fold: L3 training several hours, each body-composition model several
hours, detectors under an hour. End-to-end inference over 60 validation cases is minutes.
Two folds run concurrently, one per GPU — provided `ENROOT_RESTRICT_DEV=yes` is set.
