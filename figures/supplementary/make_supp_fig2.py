#!/usr/bin/env python3
"""Supplementary Figure 2: a stage-1 vertebral-level failure, rendered like Figure 4.

Both rows are output of the SAME body composition model on the SAME scan. Only the
section it was given differs, so the figure isolates the contribution of stage 1:

Rows carry only the letters A and B; what each row is belongs in the figure legend.

  Row A  segmentation-only: the manual reference section is supplied as input
         (analysis/offset/, offset 0, fold-matched checkpoint_best)
  Row B  fully automated:   the section the pipeline selected after the L3 vertebral
         segmentation failed (results/fold_<k>/pipeline_output/)

Panel colours, CT window and rendering match make_inferior_segmentation_maps.py, which
produced the published Figures 3 and 4.
"""
import os
from pathlib import Path
import csv, sys
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Embed TrueType rather than Type 3, so the labels stay selectable and sharp. The CT
# panels are necessarily raster; everything drawn on top of them is vector text.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
from matplotlib.patches import Patch

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation_map"))
from make_inferior_segmentation_maps import LABELS, load_single_slice, make_rgb_overlay  # noqa: E402

ALPHA, WMIN, WMAX = 1.0, -200.0, 300.0
PANELS = [(1, "SMA"), (3, "VFAT"), (2, "SFAT"), (4, "IMAT")]   # figure order, as in Fig. 3/4
# Hospital record number: supply with --case / --fold, not distributed here.
CASE, FOLD = "", -1
OUT = Path(os.environ.get("ADPKD_FIGDIR", ROOT / "figures_out")) / "Supplementary_Figure_2.pdf"


def csa(mask_path: Path) -> dict:
    img = nib.load(str(mask_path))
    arr = np.rint(np.asanyarray(img.dataobj)).astype(int)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    zx, zy = img.header.get_zooms()[:2]
    px = float(zx) * float(zy) / 100.0
    return {info["name"]: float((arr == lab).sum()) * px for lab, info in LABELS.items()}


def row_inputs():
    img_dir = ROOT / f"results/fold_{FOLD}/pipeline_output/{CASE}/image"
    sel = sorted(img_dir.glob(f"{CASE}_inferior_slice_idx*.nii.gz"))
    assert len(sel) == 1, sel
    off = ROOT / f"analysis/offset"
    return [
        ("A", "",
         off / f"work/fold_{FOLD}/inferior/imagesTs/{CASE}_z0_0000.nii.gz",
         off / f"predictions/fold_{FOLD}/inferior/{CASE}_z0.nii.gz"),
        ("B", "",
         sel[0], img_dir / f"{CASE}_inferior_sma_seg.nii.gz"),
    ]


def main() -> None:
    rows = row_inputs()
    ncol = len(PANELS) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.05 * ncol, 3.25 * len(rows)))
    for r, (letter, title, ct_path, mask_path) in enumerate(rows):
        ct = load_single_slice(ct_path)
        mask = np.rint(load_single_slice(mask_path)).astype(np.int16)
        assert ct.shape == mask.shape, (ct.shape, mask.shape)
        for c, (lab, name) in enumerate(PANELS + [(None, "Combined")]):
            ids = tuple(LABELS) if lab is None else (lab,)
            ax = axes[r, c]
            ax.imshow(np.transpose(make_rgb_overlay(ct, mask, ids, ALPHA, WMIN, WMAX), (1, 0, 2)),
                      origin="lower", interpolation="nearest")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(name, fontsize=21, pad=9)
    fig.legend(handles=[Patch(facecolor=np.asarray(LABELS[l]["rgb"]) / 255.0,
                              label=LABELS[l]["name"]) for l, _ in PANELS],
               loc="lower center", ncol=4, frameon=False, fontsize=17, bbox_to_anchor=(0.5, 0.005))
    # h_pad separates the A and B rows, which sat flush against each other
    fig.tight_layout(rect=(0.035, 0.105, 1, 1), h_pad=3.0)
    # Row labels go in the reserved left margin, in figure coordinates, so the letter and
    # the caption cannot collide with each other or with the panels.
    fig.canvas.draw()
    for r, (letter, _title, _, _) in enumerate(rows):
        box = axes[r, 0].get_position()
        fig.text(0.008, box.y0 + box.height / 2, letter, ha="left", va="center",
                 fontsize=30, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)

    gt = csa(ROOT / f"analysis/offset/work/fold_{FOLD}/inferior/labelsTs_gt/{CASE}.nii.gz") \
        if (ROOT / f"analysis/offset/work/fold_{FOLD}/inferior/labelsTs_gt/{CASE}.nii.gz").exists() else None
    print("wrote", OUT)
    for letter, title, _, mask_path in rows:
        print(f"  row {letter}: " + "  ".join(f"{k} {v:.2f}" for k, v in csa(mask_path).items()))


def _cli() -> None:
    global CASE, FOLD
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, help="case identifier")
    ap.add_argument("--fold", required=True, type=int)
    a = ap.parse_args()
    CASE, FOLD = a.case, a.fold
    main()


if __name__ == "__main__":
    _cli()
