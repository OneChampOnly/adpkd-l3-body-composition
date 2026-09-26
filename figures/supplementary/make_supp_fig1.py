#!/usr/bin/env python3
"""Supplementary Figure 1: representative scans with vertebral-level misidentification.

Left: coronal locator with the manual and predicted L3 masks drawn as OUTLINES, so the
vertebra underneath stays visible - filled overlays hid exactly the anatomy the reader
needs to judge. Middle/right: the manual reference section and the section the pipeline
selected, framed in the matching colour.

Dice similarity coefficients are deliberately not printed on the panels; they belong in
the figure legend.
"""
from pathlib import Path
import argparse
import csv
import os
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("ADPKD_FIGDIR", ROOT / "figures_out")) / "Supplementary_Figure_1.pdf"
# ---------------------------------------------------------------- hyperparameters
# Which coronal plane each scan's locator panel is cut at, in voxels, relative to the
# centroid of the manual L3 mask. Negative moves the plane anteriorly (towards the belly),
# positive posteriorly (towards the back). Use sweep_coronal.py to choose a value.
# Case identifiers are hospital record numbers and are not distributed. Supply them with
#   --cases CASE_ID:FOLD:CORONAL_OFFSET  (one argument per scan, in figure order)
# The offsets used for the published figure were +20 for scan 1 and +25 for scan 2.
CORONAL_OFFSET: dict[str, int] = {}
CASES: list[tuple[str, int]] = []
WIN = (-200.0, 300.0)
GT_C, PR_C = "#22c55e", "#f25a1a"        # manual = green, predicted = orange
FS = {"title": 21, "panel": 20, "legend": 14, "lw": 2.0}

man = {r["case_id"]: r for r in csv.DictReader(open(ROOT / "results_final/manifest_300_all_folds_used.csv"))}
loc = {r["case_id"]: r for r in csv.DictReader(open(ROOT / "analysis/ckj_verification/localisation_original_mm_per_case.csv"))
       if r["condition"] == "native_e2e" and r["branch"] == "inferior"}
dsc = {r["case_id"]: float(r["dice"]) for r in csv.DictReader(open(ROOT / "results_final/raw_csv/oof_l3_vertebra.csv"))}


def win(a): return np.clip((a - WIN[0]) / (WIN[1] - WIN[0]), 0, 1)


def outline(ax, mask2d, colour, aspect):
    """Draw the mask silhouette as a contour instead of filling it."""
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[colour],
                   linewidths=FS["lw"], zorder=5)


def main() -> None:
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15.0, 5.4 * len(CASES)))
    for row, (cid, fold) in enumerate(CASES):
        dy = CORONAL_OFFSET[cid]
        ct = nib.load(ROOT / f"dataset/{cid}/ct.nii.gz")
        arr = np.asanyarray(ct.dataobj).astype(np.float32)
        gtm = np.asanyarray(nib.load(ROOT / f"dataset/{cid}/l3_mask.nii.gz").dataobj) > 0
        prm = np.asanyarray(nib.load(
            ROOT / f"results/fold_{fold}/pipeline_output/{cid}/image/{cid}_l3_mask.nii.gz").dataobj) > 0

        gt_z = int(man[cid]["inferior_z_index"])
        off_mm = float(loc[cid]["signed_mm"]); sp = float(loc[cid]["native_spacing"])
        pred_z = int(round(gt_z + off_mm / sp))

        yy = np.flatnonzero(gtm.any(axis=(0, 2)))
        cor = int(round(yy.mean())) + dy if yy.size else arr.shape[1] // 2
        cor = int(np.clip(cor, 0, arr.shape[1] - 1))
        sl = np.rot90(win(arr[:, cor, :]))
        g2, p2 = np.rot90(gtm.any(axis=1)), np.rot90(prm.any(axis=1))
        sx, _, sz = ct.header.get_zooms()[:3]
        aspect = float(sz) / float(sx)

        ax = axes[row, 0]
        ax.imshow(sl, cmap="gray", vmin=0, vmax=1, aspect=aspect)
        outline(ax, g2, GT_C, aspect)
        outline(ax, p2, PR_C, aspect)
        nz = sl.shape[0]
        ax.axhline(nz - 1 - gt_z, color=GT_C, lw=FS["lw"], ls=":")
        ax.axhline(nz - 1 - pred_z, color=PR_C, lw=FS["lw"], ls="--")
        ax.set_title(f"Scan {row + 1}: coronal view", fontsize=FS["title"], pad=8)
        ax.set_xticks([]); ax.set_yticks([])

        for col, (z, lab, c) in enumerate(
                [(gt_z, "Manual reference section", GT_C),
                 (pred_z, f"Pipeline-selected section\n({abs(off_mm):.0f} mm away)", PR_C)], start=1):
            a2 = axes[row, col]
            a2.imshow(np.rot90(win(arr[:, :, z])), cmap="gray", vmin=0, vmax=1)
            for s in a2.spines.values():
                s.set_color(c); s.set_linewidth(3.0)
            # black caption: the coloured frame already identifies which section this is
            a2.set_title(lab, fontsize=FS["panel"], color="black", pad=8)
            a2.set_xticks([]); a2.set_yticks([])

    fig.legend(handles=[Line2D([0], [0], color=GT_C, lw=FS["lw"], ls=":", label="Manual reference section"),
                        Line2D([0], [0], color=PR_C, lw=FS["lw"], ls="--", label="Selected section"),
                        Line2D([0], [0], color=GT_C, lw=FS["lw"], label="Manual L3 mask (contour)"),
                        Line2D([0], [0], color=PR_C, lw=FS["lw"], label="Predicted L3 mask (contour)")],
               loc="lower center", ncol=4, frameon=False, fontsize=FS["legend"],
               handlelength=2.0, columnspacing=1.4, handletextpad=0.6)
    # h_pad opens up the gap between the Scan 1 and Scan 2 rows, which ran too close together
    fig.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=3.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print("wrote", OUT)
    for i, (cid, _f) in enumerate(CASES, start=1):
        print(f"  Scan {i}: {cid}  L3 DSC {dsc[cid]:.3f}  coronal offset {CORONAL_OFFSET[cid]:+d}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", required=True, metavar="CASE:FOLD:OFFSET",
                    help="one per scan, in figure order, e.g. --cases CASE_A:3:20 CASE_B:3:25")
    a = ap.parse_args()
    for spec in a.cases:
        cid, fold, off = spec.split(":")
        CASES.append((cid, int(fold)))
        CORONAL_OFFSET[cid] = int(off)
    main()


if __name__ == "__main__":
    _cli()
