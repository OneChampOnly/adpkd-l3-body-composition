#!/usr/bin/env python3
"""Build per-fold YOLO detection datasets from the clean GT L3 masks.

Every non-empty slice of a case's GT L3 mask becomes one binary-mask image. The reviewed
index carries a box and every other slice gets an empty label, so the detector learns to pick
one slice out of the stack rather than merely to find a vertebra. Fold k trains on the 240
cases outside fold k and validates on its 60, matching the segmentation folds exactly.
Both branches share one image pool and differ only in their label files.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

CLASS_NAME = {"inferior": "l3_inferior", "middle": "l3_middle"}


def yolo_box(mask_slice: np.ndarray) -> str:
    """Native axial orientation, matching how the inference pipeline writes candidate slices."""
    yy, xx = np.where(mask_slice > 0)
    h, w = mask_slice.shape
    x1, x2, y1, y2 = xx.min(), xx.max(), yy.min(), yy.max()
    return (f"0 {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
            f"{(x2 - x1 + 1) / w:.6f} {(y2 - y1 + 1) / h:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--positive-radius", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    out_root = root / "yolo" / "datasets"
    pool_dir = out_root / "mask_slices"
    pool_dir.mkdir(parents=True, exist_ok=True)

    tables = {b: list(csv.DictReader((root / "labels" / f"{b}_labels.csv").open()))
              for b in args.branches}
    reference = tables[args.branches[0]]

    slices: dict[str, list[int]] = {}
    for index, row in enumerate(reference, start=1):
        case_id = row["case_id"]
        mask = np.asanyarray(nib.load(str(dataset_dir / case_id / "l3_mask.nii.gz")).dataobj) > 0
        nz = np.flatnonzero(mask.any(axis=(0, 1))).tolist()
        slices[case_id] = nz
        for z in nz:
            out = pool_dir / f"{case_id}_z{z:04d}.png"
            if not out.exists():
                Image.fromarray(mask[:, :, z].astype(np.uint8) * 255, mode="L").save(out)
        print(f"[pool {index}/{len(reference)}] {case_id}: {len(nz)} slices", flush=True)

    summary = {"positive_radius": args.positive_radius, "mask_source": "clean GT L3 mask",
               "image_pool": str(pool_dir), "branches": {}}

    for branch in args.branches:
        rows = tables[branch]
        usable = [r for r in rows if int(r["trainable"])]
        branch_root = out_root / branch
        if branch_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{branch_root} exists; pass --overwrite")
            shutil.rmtree(branch_root)

        fold_stats = {}
        for fold in range(5):
            fold_root = branch_root / f"fold_{fold}"
            stats = {}
            for part in ("train", "val"):
                subset = [r for r in usable if (int(r["fold"]) == fold) == (part == "val")]
                image_dir, txt_dir = fold_root / "images" / part, fold_root / "labels" / part
                image_dir.mkdir(parents=True)
                txt_dir.mkdir(parents=True)
                positives = images = 0
                for row in subset:
                    case_id = row["case_id"]
                    target = int(row["z_index"])
                    mask = None
                    for z in slices[case_id]:
                        stem = f"{case_id}_z{z:04d}"
                        (image_dir / f"{stem}.png").symlink_to(pool_dir / f"{stem}.png")
                        if abs(z - target) <= args.positive_radius:
                            if mask is None:
                                mask = np.asanyarray(
                                    nib.load(str(dataset_dir / case_id / "l3_mask.nii.gz")).dataobj) > 0
                            (txt_dir / f"{stem}.txt").write_text(yolo_box(mask[:, :, z]))
                            positives += 1
                        else:
                            (txt_dir / f"{stem}.txt").write_text("")
                        images += 1
                stats[part] = {"cases": len(subset), "images": images,
                               "positive_images": positives, "negative_images": images - positives}
            (fold_root / "data.yaml").write_text(
                f"path: {fold_root}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: {CLASS_NAME[branch]}\n")
            fold_stats[str(fold)] = stats
            print(f"[{branch} fold {fold}] {stats}", flush=True)

        summary["branches"][branch] = {
            "folds": fold_stats,
            "skipped_cases": [r["case_id"] for r in rows if not int(r["trainable"])],
            "reviewer_edited_cases": [r["case_id"] for r in rows if r["z_index"] != r["z_index_original"]],
        }

    (out_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] {out_root}")


if __name__ == "__main__":
    main()
