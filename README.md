# Fully automated L3-level body composition analysis on abdominal CT in ADPKD

Code for the article *Fully Automated Deep Learning for Third Lumbar-Level Body
Composition Analysis on Abdominal Computed Tomography in Autosomal Dominant Polycystic
Kidney Disease* (Clinical Kidney Journal, 2026).

The pipeline takes a three-dimensional abdominal CT volume and returns cross-sectional
areas of skeletal muscle (SMA), visceral fat (VFAT), subcutaneous fat (SFAT) and
intramuscular adipose tissue (IMAT) at two L3 reference sections, with no manual step.

```
CT volume ──▶ L3 vertebra segmentation ──▶ reference-section detection ──▶ body composition ──▶ CSA
              (3D nnU-Net, shared)          (YOLOv8n, one per level)       (2D nnU-Net, one per level)
```

Five trained models per cross-validation fold: one L3 segmenter shared by both levels,
plus a detector and a body-composition segmenter for each of the **L3 inferior endplate**
(primary) and **L3 middle** (secondary) levels. Every case is processed by the models of
its own fold, so no model ever sees the case it is evaluated on.

## What is and is not in this repository

| | |
|---|---|
| ✅ Data preparation, training drivers, end-to-end inference, evaluation, figures | this repository |
| ⬇️ Trained weights (4.7 GB, 25 checkpoints) | released separately — see [docs/WEIGHTS.md](docs/WEIGHTS.md) |
| ❌ CT images, segmentation masks, patient-level tables | not released — see [docs/DATA.md](docs/DATA.md) |

Hospital record numbers appear nowhere in this repository. Scripts that need them read
them from environment variables; see [docs/DATA.md](docs/DATA.md).

## Layout

| Path | Contents |
|---|---|
| `code/` | the numbered pipeline, stages 01–17, plus `pipeline_core.py` |
| `scripts/` | cluster and single-host launchers for training and end-to-end inference |
| `analysis/` | the supplementary analyses: section offset, 5 mm grid, public-model comparison, external reference, sex stratification, verification |
| `figures/` | figure generation for the article |
| `results/` | `generate_final_results.py`, which builds every published table from the per-case outputs |
| `config/` | templates for `config.json` and the model registry |
| `docs/` | environment, reproduction, weights, data |
| `tools/` | helper for fetching the released weights |

### Pipeline stages

| Stage | Purpose |
|---|---|
| `01_build_dataset.py` | assemble one clean per-case dataset from the staging folders |
| `02_prepare_l3_nnunet.py` | stage it into nnU-Net raw layout for L3 vertebra segmentation |
| `03_refresh_label_stats.py` | recompute mask-derived label columns against the clean L3 masks |
| `04_build_yolo_datasets.py` | build per-fold YOLO detection datasets **from the manual L3 masks** |
| `05_render_bbox_qc.py`, `08`, `09` | visual QC of boxes, overlays and the re-annotation set |
| `06_make_reannotation_set.py`, `07_apply_reannotation.py` | prepare and fold in the manual re-annotation round |
| `10_prepare_branch_nnunet.py` | build one nnU-Net raw dataset per level for body composition |
| `11_build_manifests.py` | emit the case manifests the runner and evaluator consume |
| `12_prepare_model_registry.py` | stage fold-matched checkpoints and write `model_registry.json` |
| `13_run_end_to_end_5fold.py` | end-to-end inference, one fold at a time, case by case |
| `16_run_staged_pipeline.py` | the same inference staged by step, which is faster for a whole fold |
| `14_evaluate_end_to_end.py` | score the end-to-end predictions against the manual reference |
| `15_evaluate_detector_only.py` | score the detectors alone |
| `17_evaluate_detector_on_gt_mask.py` | detectors on the manual L3 mask, isolating their own contribution |

## Quick start

This code cannot be run end to end without the CT data, which is not distributable. What
you *can* do with the released weights is run the pipeline on your own abdominal CT.

```bash
git clone https://github.com/USERNAME/adpkd-l3-body-composition.git
cd adpkd-l3-body-composition

cp config/config.example.json config.json           # then edit the paths
cp config/model_registry.template.json config/model_registry.json

python tools/download_weights.py --dest weights/     # see docs/WEIGHTS.md
export ADPKD_ROOT="$PWD"

python code/16_run_staged_pipeline.py --folds 0 --branches inferior middle
```

Full walk-through: [docs/REPRODUCE.md](docs/REPRODUCE.md).
Environment and exact versions: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Two things that will cost you time if you miss them

**`ENROOT_RESTRICT_DEV=yes`.** Without it enroot bind-mounts every `/dev/nvidia*` node,
`NVIDIA_VISIBLE_DEVICES` is ignored, and every fold lands on physical GPU 0. All launcher
scripts set it; if you adapt them, keep it.

**`checkpoint_best.pth`, not the validation export.** `nnUNetv2_train` without
`--val_best` runs its own validation with the *final* epoch weights, so the predictions in
`fold_N/validation/` are not the model the registry points at. Every number in the article
comes from `checkpoint_best.pth`. `analysis/verification/run_ckpt_test.sh` demonstrates the
difference.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE). Confirm this is consistent with your institution's policy
before publishing the repository.
