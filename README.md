# Fully automated L3-level body composition analysis on abdominal CT

Code for *Fully Automated Deep Learning for Third Lumbar-Level Body Composition Analysis
on Abdominal Computed Tomography in Autosomal Dominant Polycystic Kidney Disease*
(Clinical Kidney Journal, 2026).

Give it a three-dimensional abdominal CT volume; it returns cross-sectional areas of
skeletal muscle (SMA), visceral fat (VFAT), subcutaneous fat (SFAT) and intramuscular
adipose tissue (IMAT) at two L3 reference sections. No manual step anywhere.

```
CT volume ──▶ L3 vertebra segmentation ──▶ reference-section detection ──▶ body composition ──▶ CSA
              3D nnU-Net (shared)          YOLOv8n (one per level)         2D nnU-Net (one per level)
```

Five models per cross-validation fold: one L3 segmenter shared by both levels, plus a
detector and a body-composition segmenter for each of the **L3 inferior endplate**
(primary) and **L3 middle** (secondary) levels.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/USERNAME/adpkd-l3-body-composition.git
cd adpkd-l3-body-composition

conda env create -f environment/nnunet.yml      # segmentation  -> adpkd-nnunet
conda env create -f environment/analysis.yml    # detector, figures -> adpkd-analysis
```

Two environments, because nnU-Net and Ultralytics need incompatible PyTorch builds.
Details and pip alternatives in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

### 2. Download the trained weights

The weights are **not** in this repository — 25 checkpoints, 4.66 GiB. They are packaged
in nnU-Net's own `nnUNet_results` layout, so unpacking them is the whole installation.

> **Download:** `<MODEL_DOWNLOAD_URL>`
> *(filled in on publication — see [docs/WEIGHTS.md](docs/WEIGHTS.md))*

```bash
python tools/download_weights.py --source <MODEL_DOWNLOAD_URL> --dest weights/
python tools/download_weights.py --dest weights/ --verify      # checks SHA256SUMS
```

Or download the archive and unpack it so the tree looks exactly like this:

```
adpkd-l3-body-composition/
└── weights/
    ├── SHA256SUMS
    ├── nnUNet_results/
    │   ├── Dataset920_L3Vertebra0917/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/
    │   │   ├── plans.json
    │   │   ├── dataset.json
    │   │   └── fold_0/checkpoint_best.pth          (… fold_1 … fold_4)
    │   ├── Dataset917_InferiorSMA300Reann0917/nnUNetTrainer_500epochs__nnUNetPlans__2d/
    │   └── Dataset918_MiddleSMA300Reann0917/nnUNetTrainer_500epochs__nnUNetPlans__2d/
    └── yolo/
        ├── inferior/fold_0/best.pt                 (… fold_1 … fold_4)
        └── middle/fold_0/best.pt
```

`plans.json` and `dataset.json` are part of the download and must stay where they are:
nnU-Net reads the network architecture from them, and inference fails without them.

Use `checkpoint_best.pth`, never `checkpoint_final.pth` —
[why](docs/WEIGHTS.md#which-checkpoint-and-why-it-matters).

### 3. Point the code at the weights

```bash
python tools/download_weights.py --dest weights/ --write-registry config/model_registry.json
```

This writes the registry mapping each fold to its five checkpoints, then prints the
environment variables to export. They are:

```bash
export ADPKD_ROOT="$PWD"
export nnUNet_results="$PWD/weights/nnUNet_results"
export nnUNet_raw="$PWD/weights/nnUNet_results"          # unused at inference, but nnU-Net
export nnUNet_preprocessed="$PWD/weights/nnUNet_results" # refuses to start without them
export NNUNETV2_PREDICT_BIN="$(command -v nnUNetv2_predict)"
```

Also copy the config template once: `cp config/config.example.json config.json`.

### 4. Run

`code/16_run_staged_pipeline.py` imports Ultralytics and shells out to `nnUNetv2_predict`,
so run it from the **analysis** environment with `NNUNETV2_PREDICT_BIN` pointing into the
nnU-Net environment:

```bash
conda activate adpkd-analysis
export NNUNETV2_PREDICT_BIN="$HOME/miniconda3/envs/adpkd-nnunet/bin/nnUNetv2_predict"

python code/16_run_staged_pipeline.py --folds 0 --branches inferior middle
```

Output per case, under `results/`:

| File | Contents |
|---|---|
| `<case>_l3_mask.nii.gz` | L3 vertebra segmentation |
| `<case>_<branch>_slice_idx<N>.nii.gz` | the selected reference section |
| `<case>_<branch>_sma_seg.nii.gz` | four-class body composition mask |
| `per_case_*.csv` | CSA per compartment, in cm² |

**Which fold should I use?** For new scans, any of them — the five are independently
trained instances of the same pipeline. Fold 0 is a fine default; running all five and
averaging gives you a spread. Fold matching only matters when reproducing the article's
numbers on the original cohort, where each case must be processed by the fold that held
it out.

---

## What is in this repository

| | |
|---|---|
| ✅ Training code, end-to-end inference, evaluation, figures | here |
| ⬇️ Trained weights (4.7 GB) | separate download — [docs/WEIGHTS.md](docs/WEIGHTS.md) |
| ❌ CT images, segmentation masks, patient tables | **not distributed** — [docs/DATA.md](docs/DATA.md) |

No patient data of any kind is in this repository, and no hospital record numbers appear
anywhere in the code. Scripts that need identifiers read them from environment variables.

| Path | Contents |
|---|---|
| `code/` | the pipeline, stages 01–17, plus `pipeline_core.py` |
| `scripts/` | training and inference launchers |
| `environment/` | conda environment files |
| `config/` | templates for `config.json` and the model registry |
| `analysis/` | section offset, 5 mm grid, public-model comparison, external reference, sex stratification, verification |
| `figures/` | figure generation for the article |
| `results/` | `generate_final_results.py`, which rebuilds every published table |
| `tools/` | weight download and registry helper |
| `docs/` | environment, reproduction, weights, data |

---

## Training from scratch

Only needed if you are retraining on your own cohort. Everything above works with the
released weights alone.

```bash
# 1. dataset and nnU-Net raw trees  (needs your CT + manual masks; see config.json)
python code/01_build_dataset.py
python code/02_prepare_l3_nnunet.py
python code/04_build_yolo_datasets.py      # detector sets are built from the MANUAL L3 masks
python code/10_prepare_branch_nnunet.py
python code/11_build_manifests.py

# 2. train, five folds each
bash scripts/run_l3_5fold_enroot.sh        # 3d_fullres, nnUNetTrainer_500epochs
bash scripts/run_yolo_5fold_enroot.sh      # YOLOv8n, imgsz 512
bash scripts/run_bodycomp_5fold_enroot.sh  # 2d, nnUNetTrainer_500epochs, both levels

# 3. collect the checkpoints into a registry
python code/12_prepare_model_registry.py

# 4. inference and evaluation
python code/16_run_staged_pipeline.py --folds 0 1 2 3 4 --branches inferior middle
python code/14_evaluate_end_to_end.py
```

Step 3 produces the same `config/model_registry.json` that `tools/download_weights.py`
writes, so step 4 is identical either way. Full walk-through, including the supplementary
analyses and the figures: [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Two things that will cost you time if you miss them

**`export ENROOT_RESTRICT_DEV=yes`** before launching anything in a container. Without it
enroot bind-mounts every `/dev/nvidia*` node, `NVIDIA_VISIBLE_DEVICES` is ignored, and
every fold lands on physical GPU 0. All launchers set it; keep it if you adapt them.

**Use `checkpoint_best.pth`.** `nnUNetv2_train` without `--val_best` runs its end-of-training
validation with the *final* epoch weights, so the predictions left in
`nnUNet_results/.../fold_N/validation/` are not the model this repository points at. Every
number in the article comes from `checkpoint_best.pth`.
`analysis/verification/run_ckpt_test.sh` demonstrates the difference.

---

## Citation

See [CITATION.cff](CITATION.cff). Please cite the article and, if you use the weights,
their archived record.

## License

MIT — see [LICENSE](LICENSE).
