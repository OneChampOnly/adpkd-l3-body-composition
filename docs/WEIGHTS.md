# Trained weights

25 checkpoints, **4.7 GB** in total — five cross-validation folds × five models. Too large
for GitHub, so they are released separately and fetched with `tools/download_weights.py`.

## What is released

| Model | Per fold | Framework | File |
|---|---|---|---|
| `l3` | 235 MB | nnU-Net v2, `3d_fullres` | `checkpoint_best.pth` |
| `sma_inferior` | 354 MB | nnU-Net v2, `2d` | `checkpoint_best.pth` |
| `sma_middle` | 354 MB | nnU-Net v2, `2d` | `checkpoint_best.pth` |
| `yolo_inferior` | 6.0 MB | Ultralytics YOLOv8n | `best.pt` |
| `yolo_middle` | 6.0 MB | Ultralytics YOLOv8n | `best.pt` |
| | **955 MB/fold** | | |

`checkpoint_best.pth`, not `checkpoint_final.pth`. See "Which checkpoint" below.

## Expected layout

```
weights/
├── fold_0/
│   ├── l3/checkpoint_best.pth
│   ├── sma_inferior/checkpoint_best.pth
│   ├── sma_middle/checkpoint_best.pth
│   ├── yolo_inferior/best.pt
│   └── yolo_middle/best.pt
├── fold_1/ … fold_4/
└── SHA256SUMS
```

Then point the registry at it:

```bash
python tools/download_weights.py --dest weights/
python tools/download_weights.py --dest weights/ --write-registry config/model_registry.json
```

`config/model_registry.template.json` shows the structure. Fold matching is not optional:
each case must be processed by the models of the fold in which it was a *validation* case.
`code/12_prepare_model_registry.py` regenerates the registry if you retrain.

## Publishing the weights

Not yet uploaded. Either host works; the article's Data availability statement should cite
whichever you pick, with its DOI.

**Hugging Face** — convenient for reuse, no size ceiling in practice:

```bash
huggingface-cli repo create adpkd-l3-body-composition --type model
huggingface-cli upload <user>/adpkd-l3-body-composition weights/ . --repo-type model
```

**Zenodo** — issues a DOI, which journals prefer for archival citation. 50 GB per record
by default. Upload the tree as a single archive plus `SHA256SUMS`.

A workable arrangement is both: Zenodo for the citable DOI, Hugging Face for day-to-day
download. Record the DOI in `CITATION.cff` and in the manuscript once minted.

Ship `SHA256SUMS` alongside the weights. `tools/download_weights.py --verify` checks
against it, which catches truncated downloads — a real risk at ~1 GB per fold.

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
