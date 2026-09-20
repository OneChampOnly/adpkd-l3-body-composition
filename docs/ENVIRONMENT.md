# Environment

Two environments were used. Keeping them apart matters: nnU-Net and Ultralytics pull
incompatible PyTorch builds.

## Hardware

| | |
|---|---|
| GPU | 2 × NVIDIA H100 PCIe, ~80 GB each |
| CPU | 2 × Intel Xeon Silver 4410Y — 24 physical cores, 48 threads |
| Memory | 128 GB |
| OS | Ubuntu 24.04.4 LTS |

One H100 is enough to train or run any single fold. Five folds were run two at a time,
one per GPU.

## Segmentation environment (nnU-Net)

Ran inside an [enroot](https://github.com/NVIDIA/enroot) container image. Any container or
conda environment with these versions will do; the launcher scripts assume enroot and take
the image path from `$IMAGE`.

| Package | Version |
|---|---|
| Python | 3.10.19 |
| nnunetv2 | 2.6.2 |
| torch | 2.5.1+cu121 (CUDA 12.1) |
| SimpleITK | 2.5.3 |
| nibabel | 5.3.3 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| batchgenerators | 0.25.1 |

```bash
conda env create -f environment/nnunet.yml   # -> adpkd-nnunet
# or: pip install -r requirements-nnunet.txt
```

## Analysis, detector and figure environment

A conda environment on the host. The launcher scripts take its prefix from `$YOLO_ENV`.

| Package | Version |
|---|---|
| Python | 3.10.19 |
| ultralytics | **8.4.10** when the detectors were trained |
| torch | 2.10.0+cu128 (CUDA 12.8) |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| nibabel | 5.3.2 |
| Pillow | 12.1.0 |
| openpyxl | 3.1.5 |

```bash
conda env create -f environment/analysis.yml  # -> adpkd-analysis
# or: pip install -r requirements.txt
```

> The ultralytics version is worth pinning. All ten released detector checkpoints record
> **8.4.10** in their metadata, which is the version the published numbers come from. The
> environment was later upgraded to 8.4.37 for unrelated work, so a fresh install may
> report a newer version than the one that produced the weights. `requirements.txt` pins
> 8.4.10.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `ADPKD_ROOT` | all scripts | repository root; defaults to the path resolved from the script's own location |
| `nnUNet_results` | inference | the unpacked `weights/nnUNet_results` |
| `nnUNet_raw`, `nnUNet_preprocessed` | inference | unused at inference, but nnU-Net refuses to start unless both point at an existing directory |
| `NNUNETV2_PREDICT_BIN` | inference | `nnUNetv2_predict` in the nnU-Net environment, when driving it from the analysis environment |
| `IMAGE` | launcher scripts | container image for the nnU-Net environment |
| `YOLO_ENV` | launcher scripts | conda prefix for the analysis environment |
| `ENROOT_RESTRICT_DEV` | launcher scripts | **must be `yes`** — see below |
| `ADPKD_CLINICAL_CSV` | sex-stratified and external-reference analyses | patient-level clinical table, not distributed |
| `ADPKD_REPLACEMENT_IDS`, `ADPKD_RETIRED_IDS` | public-model manifest audit | case identifiers, not distributed |
| `PUBLIC_MODEL_REPO` | public-model comparison | local checkout of the comparison model |
| `PRIOR_NNUNET_RESULTS` | re-annotation step only | previous round's nnU-Net results |

### ENROOT_RESTRICT_DEV

```bash
export ENROOT_RESTRICT_DEV=yes
```

Without it, enroot bind-mounts every `/dev/nvidia*` node into the container, so
`NVIDIA_VISIBLE_DEVICES` has no effect and `CUDA_VISIBLE_DEVICES=0` resolves to physical
GPU 0 for every job. Folds launched in parallel then serialise on one GPU. This affects
wall-clock time only, never results, but it is easy to miss.

## Training configuration

| | L3 vertebra | Body composition |
|---|---|---|
| Framework | nnU-Net v2 | nnU-Net v2 |
| Configuration | `3d_fullres` | `2d` |
| Trainer | `nnUNetTrainer_500epochs` | `nnUNetTrainer_500epochs` |
| Plans | `nnUNetPlans` | `nnUNetPlans` |
| Dataset | `Dataset920_L3Vertebra0917` | `Dataset917_InferiorSMA300Reann0917`, `Dataset918_MiddleSMA300Reann0917` |

Detector: YOLOv8n, image size 512, confidence threshold 0.001, trained on PNG renderings
of the **manual** L3 masks. Data augmentation followed each framework's defaults.

Cross-validation: 5 folds over 300 scans, split at the scan level, 240 training and 60
validation per fold, with the same assignment reused for every model component.
