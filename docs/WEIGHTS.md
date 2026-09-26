# Trained weights

25 checkpoints, **4.66 GiB** — five cross-validation folds × five models. Too large for
GitHub, so they are released separately.

> **Download:** Hugging Face [`OneChampOnly/adpkd-l3-body-composition`](https://huggingface.co/OneChampOnly/adpkd-l3-body-composition) · archived on Zenodo, DOI [10.5281/zenodo.22976872](https://doi.org/10.5281/zenodo.22976872)

## What is released

| Model | Per fold | Framework | File |
|---|---|---|---|
| L3 vertebra | 235 MB | nnU-Net v2, `3d_fullres` | `checkpoint_best.pth` |
| Body composition, inferior | 354 MB | nnU-Net v2, `2d` | `checkpoint_best.pth` |
| Body composition, middle | 354 MB | nnU-Net v2, `2d` | `checkpoint_best.pth` |
| Detector, inferior | 6.0 MB | Ultralytics YOLOv8n | `best.pt` |
| Detector, middle | 6.0 MB | Ultralytics YOLOv8n | `best.pt` |
| | **≈955 MB/fold** | | |

Plus `plans.json` and `dataset.json` for each of the three nnU-Net datasets. These are
small but **required**: nnU-Net reads the network architecture from them, and inference
fails without them.

## Layout

The package uses nnU-Net's own `nnUNet_results` layout, so unpacking is the whole install:

```
weights/
├── SHA256SUMS
├── nnUNet_results/
│   ├── Dataset920_L3Vertebra0917/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/
│   │   ├── plans.json
│   │   ├── dataset.json
│   │   └── fold_0 … fold_4/checkpoint_best.pth
│   ├── Dataset917_InferiorSMA300Reann0917/nnUNetTrainer_500epochs__nnUNetPlans__2d/
│   └── Dataset918_MiddleSMA300Reann0917/nnUNetTrainer_500epochs__nnUNetPlans__2d/
└── yolo/
    ├── inferior/fold_0 … fold_4/best.pt
    └── middle/fold_0 … fold_4/best.pt
```

## Setup

```bash
python tools/download_weights.py --source OneChampOnly/adpkd-l3-body-composition --dest weights/
python tools/download_weights.py --dest weights/ --verify
python tools/download_weights.py --dest weights/ --write-registry config/model_registry.json
```

The last command prints the environment variables to export. Verification matters here:
a truncated 350 MB checkpoint fails in confusing ways much later.

Fold matching is not optional when reproducing the article: each case must be processed by
the models of the fold in which it was a *validation* case. The registry encodes this.
`code/12_prepare_model_registry.py` regenerates it if you retrain.

## Where the weights are published

| Host | Location | Use |
|---|---|---|
| Hugging Face | [`OneChampOnly/adpkd-l3-body-composition`](https://huggingface.co/OneChampOnly/adpkd-l3-body-composition) | `download_weights.py --source OneChampOnly/adpkd-l3-body-composition` fetches the tree directly |
| Zenodo | DOI [10.5281/zenodo.22976872](https://doi.org/10.5281/zenodo.22976872) | archival record to cite; one `.tar.gz` per fold (≈890 MB each) plus `SHA256SUMS` |

Both hold the same files. Each Zenodo archive holds one fold's five checkpoints plus the
`plans.json`/`dataset.json`/`dataset_fingerprint.json` sidecars, so a single fold is
usable on its own. Unpack whichever you need into the same directory:

```bash
mkdir -p weights
for f in adpkd-l3-weights-v1.0.0-fold*.tar.gz; do tar -xzf "$f" -C weights; done
python tools/download_weights.py --dest weights/ --verify   # expects all five folds
```

## Which checkpoint, and why it matters

`nnUNetv2_train` run **without** `--val_best` performs its end-of-training validation with
the weights in memory at that moment, which are the *final* epoch, not the best. The
predictions left in `nnUNet_results/<dataset>/<trainer>/fold_N/validation/` therefore come
from `checkpoint_final.pth`.

For fold 0 the two differ: best is epoch 412 (EMA foreground Dice 0.925992), final is epoch
500 (0.925395). Re-running inference on five fold-0 validation cases with each checkpoint,
the stored validation export differs from `checkpoint_final` by 15 voxels out of 1.31
million, and from `checkpoint_best` by 2,536 — a factor of 169.

Every number in the article comes from `checkpoint_best.pth`, which is what is released.
`analysis/verification/run_ckpt_test.sh` reproduces the comparison.
