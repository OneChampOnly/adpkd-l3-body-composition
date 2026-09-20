#!/usr/bin/env python3
"""Draw the generated YOLO boxes back onto the images the detector will train on.

Reads the positive label files straight out of the built dataset, so this checks the files on
disk rather than recomputing what they should have been.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from PIL import Image

BONE_WINDOW = (-200.0, 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    yolo_root = root / "yolo" / "datasets"
    low, high = BONE_WINDOW

    for branch in args.branches:
        rows = [r for r in csv.DictReader((root / "labels" / f"{branch}_labels.csv").open())
                if int(r["trainable"])]
        if args.limit:
            rows = rows[:args.limit]
        out_dir = root / "qc" / "bbox" / branch
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(rows, start=1):
            case_id, fold, z = row["case_id"], int(row["fold"]), int(row["z_index"])
            stem = f"{case_id}_z{z:04d}"
            # each case sits in the val split of its own fold
            split_root = yolo_root / branch / f"fold_{fold}"
            text = (split_root / "labels" / "val" / f"{stem}.txt").read_text().strip()
            if not text:
                raise ValueError(f"{case_id}: positive label file is empty ({stem}.txt)")
            _, cx, cy, bw, bh = (float(v) for v in text.split())

            mask_png = np.array(Image.open(split_root / "images" / "val" / f"{stem}.png"))
            h, w = mask_png.shape
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            ct = np.asanyarray(nib.load(str(dataset_dir / case_id / "ct.nii.gz")).dataobj[:, :, z],
                               dtype=np.float32)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
            axes[0].imshow(mask_png, cmap="gray")
            axes[0].set_title(f"YOLO input (GT L3 mask slice)  {stem}", fontsize=10)
            axes[1].imshow(np.clip((ct - low) / (high - low), 0, 1), cmap="gray")
            axes[1].set_title(f"CT z={z}", fontsize=10)
            for ax in axes:
                ax.add_patch(patches.Rectangle((x0, y0), bw * w, bh * h, fill=False,
                                               edgecolor="red", linewidth=1.8))
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{case_id}  fold {fold}  {branch} z={z}   "
                         f"box cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"f{fold}_{case_id}.png", dpi=95)
            plt.close(fig)
            print(f"[{branch} {index}/{len(rows)}] {case_id}", flush=True)
    print(f"[DONE] {root / 'qc' / 'bbox'}")


if __name__ == "__main__":
    main()
