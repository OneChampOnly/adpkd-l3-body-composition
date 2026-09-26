#!/usr/bin/env python3
"""Contact sheet of candidate coronal planes for one scan of Supplementary Figure 1.

Renders the locator panel at a range of CORONAL_OFFSET values so the offset can be
chosen by eye, then set in make_supp_fig1.py. Usage:
    python3 sweep_coronal.py [case_id] [fold] [offsets...]
"""
import sys
from pathlib import Path
import csv
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
CASE = sys.argv[1]   # hospital record number, supplied by the caller
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OFFS = [int(x) for x in sys.argv[3:]] or [-40, -30, -20, -10, 0, 10, 20, 30, 40]
OUT = Path(os.environ.get("ADPKD_FIGDIR", ROOT / "figures_out")) / f"coronal_sweep_{CASE}.png"
WIN = (-200.0, 300.0); GT_C, PR_C = "#22c55e", "#f25a1a"

man = {r["case_id"]: r for r in csv.DictReader(open(ROOT / "results_final/manifest_300_all_folds_used.csv"))}
loc = {r["case_id"]: r for r in csv.DictReader(open(ROOT / "analysis/verification/localisation_original_mm_per_case.csv"))
       if r["condition"] == "native_e2e" and r["branch"] == "inferior"}

ct = nib.load(ROOT / f"dataset/{CASE}/ct.nii.gz")
arr = np.asanyarray(ct.dataobj).astype(np.float32)
gtm = np.asanyarray(nib.load(ROOT / f"dataset/{CASE}/l3_mask.nii.gz").dataobj) > 0
prm = np.asanyarray(nib.load(ROOT / f"results/fold_{FOLD}/pipeline_output/{CASE}/image/{CASE}_l3_mask.nii.gz").dataobj) > 0
gt_z = int(man[CASE]["inferior_z_index"])
off_mm = float(loc[CASE]["signed_mm"]); sp = float(loc[CASE]["native_spacing"])
pred_z = int(round(gt_z + off_mm / sp))
base = int(round(np.flatnonzero(gtm.any(axis=(0, 2))).mean()))
g2, p2 = np.rot90(gtm.any(axis=1)), np.rot90(prm.any(axis=1))
sx, _, sz = ct.header.get_zooms()[:3]; aspect = float(sz) / float(sx)

fig, axes = plt.subplots(1, len(OFFS), figsize=(3.0 * len(OFFS), 7.0))
for ax, dy in zip(np.atleast_1d(axes), OFFS):
    cor = int(np.clip(base + dy, 0, arr.shape[1] - 1))
    sl = np.rot90(np.clip((arr[:, cor, :] - WIN[0]) / (WIN[1] - WIN[0]), 0, 1))
    ax.imshow(sl, cmap="gray", vmin=0, vmax=1, aspect=aspect)
    for m, c in ((g2, GT_C), (p2, PR_C)):
        if m.any(): ax.contour(m.astype(float), levels=[0.5], colors=[c], linewidths=2.0)
    nz = sl.shape[0]
    ax.axhline(nz - 1 - gt_z, color=GT_C, lw=2.0, ls=":")
    ax.axhline(nz - 1 - pred_z, color=PR_C, lw=2.0, ls="--")
    ax.set_title(f"offset {dy:+d}\n(plane {cor})", fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{CASE}: candidate coronal planes  (centroid plane = {base})", fontsize=18)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("wrote", OUT, "| centroid plane", base, "| volume y-size", arr.shape[1])
