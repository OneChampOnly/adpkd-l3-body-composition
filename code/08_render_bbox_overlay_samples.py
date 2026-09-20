#!/usr/bin/env python3
"""Render sample overlays: CT, the L3 mask the detector sees, and the generated box.

Images are shown in radiological orientation rather than raw array order, so the box
coordinates are mapped accordingly: a YOLO cx runs along the volume's second axis and cy
along the first.
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

SOFT_WINDOW = (-150.0, 250.0)


def box_patch(cx: float, cy: float, bw: float, bh: float, nx: int, ny: int, **kwargs):
    """Map a YOLO box onto the transposed display: cy spans axis 0, cx spans axis 1."""
    return patches.Rectangle(((cy - bh / 2) * nx, (cx - bw / 2) * ny),
                             bh * nx, bw * ny, fill=False, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--cases", nargs="+", default=[])
    ap.add_argument("--per-fold", type=int, default=2)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    yolo_root = root / "yolo" / "datasets"
    out_root = root / "qc" / "bbox_overlay"
    low, high = SOFT_WINDOW

    for branch in args.branches:
        rows = list(csv.DictReader((root / "labels" / f"{branch}_labels.csv").open()))
        if args.cases:
            picked = [r for r in rows if r["case_id"] in set(args.cases)]
        else:
            picked = []
            for fold in range(5):
                picked.extend([r for r in rows if int(r["fold"]) == fold][:args.per_fold])
        out_dir = out_root / branch
        out_dir.mkdir(parents=True, exist_ok=True)

        panels = []
        for row in picked:
            case_id, fold, z = row["case_id"], int(row["fold"]), int(row["z_index"])
            stem = f"{case_id}_z{z:04d}"
            label = (yolo_root / branch / f"fold_{fold}" / "labels" / "val" / f"{stem}.txt").read_text().split()
            _, cx, cy, bw, bh = (float(v) for v in label)

            ct_nii = nib.load(str(dataset_dir / case_id / "ct.nii.gz"))
            ct = np.asanyarray(ct_nii.dataobj[:, :, z], dtype=np.float32)
            mask = np.asanyarray(nib.load(str(dataset_dir / case_id / "l3_mask.nii.gz")).dataobj[:, :, z]) > 0
            nx, ny = ct.shape

            fig, ax = plt.subplots(figsize=(5.6, 5.6))
            ax.imshow(np.clip((ct.T - low) / (high - low), 0, 1), cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(~mask.T, mask.T.astype(float)), cmap="autumn",
                      alpha=0.45, origin="lower")
            ax.add_patch(box_patch(cx, cy, bw, bh, nx, ny, edgecolor="red", linewidth=2.0))
            ax.set_title(f"{case_id}  fold {fold}  {branch}  z={z}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            path = out_dir / f"f{fold}_{case_id}_z{z}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            panels.append((row, path, cx, cy, bw, bh))
            print(f"[{branch}] {case_id} fold {fold} z={z} -> {path.name}", flush=True)

        columns = 5
        rowcount = (len(panels) + columns - 1) // columns
        fig, axes = plt.subplots(rowcount, columns, figsize=(3.4 * columns, 3.9 * rowcount),
                                 squeeze=False)
        for ax, (row, _, cx, cy, bw, bh) in zip(np.ravel(axes), panels):
            case_id, fold, z = row["case_id"], int(row["fold"]), int(row["z_index"])
            ct_nii = nib.load(str(dataset_dir / case_id / "ct.nii.gz"))
            ct = np.asanyarray(ct_nii.dataobj[:, :, z], dtype=np.float32)
            mask = np.asanyarray(nib.load(str(dataset_dir / case_id / "l3_mask.nii.gz")).dataobj[:, :, z]) > 0
            nx, ny = ct.shape
            ax.imshow(np.clip((ct.T - low) / (high - low), 0, 1), cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(~mask.T, mask.T.astype(float)), cmap="autumn",
                      alpha=0.45, origin="lower")
            ax.add_patch(box_patch(cx, cy, bw, bh, nx, ny, edgecolor="red", linewidth=1.6))
            ax.set_title(f"{case_id}  f{fold}  z={z}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in np.ravel(axes)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{branch} — YOLO training boxes on the GT L3 mask", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.subplots_adjust(hspace=0.22)
        sheet = out_root / f"{branch}_sheet.png"
        fig.savefig(sheet, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[{branch}] sheet -> {sheet}")
    print(f"[DONE] {out_root}")


if __name__ == "__main__":
    main()
