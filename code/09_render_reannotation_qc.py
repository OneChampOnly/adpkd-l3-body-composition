#!/usr/bin/env python3
"""Render CT + mask overlays for the hand-corrected re-annotation set, before it becomes GT.

Three panels per case: the CT alone, the mask filled over it, and the mask as outlines. The
outlines are the panel that actually answers "is this drawn on the right boundary" - a filled
overlay hides the very edge it is being judged against.

Alongside the images this writes a QC table. The cheap automatic checks are geometry (the mask
must sit on the CT grid), label presence, and the HU each label covers: muscle belongs in
-29..150 HU and the three fat compartments in -190..-30 HU, so a label whose median HU falls
outside its window is usually a compartment painted over the wrong tissue.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

SOFT_WINDOW = (-150.0, 250.0)
LABEL_NAMES = {1: "muscle", 2: "sfat", 3: "vfat", 4: "mfat"}
LABEL_COLORS = {1: (0.2, 0.8, 1.0), 2: (0.4, 1.0, 0.4), 3: (1.0, 0.65, 0.0), 4: (1.0, 0.0, 0.0)}
OVERLAY = ListedColormap([(0, 0, 0, 0)] + [(*LABEL_COLORS[i], 1) for i in sorted(LABEL_COLORS)])
# HU window each compartment should live in; muscle and fat are separated at -29 HU
HU_WINDOW = {1: (-29.0, 150.0), 2: (-190.0, -30.0), 3: (-190.0, -30.0), 4: (-190.0, -30.0)}
AIR_HU = -300.0


def load_slice(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)[0], image


def same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3))


def components(mask: np.ndarray) -> int:
    """Connected components of a binary slice, ignoring specks below 20 px."""
    image = sitk.GetImageFromArray(mask.astype(np.uint8)[None])
    labelled = sitk.RelabelComponent(sitk.ConnectedComponent(image), minimumObjectSize=20)
    return int(sitk.GetArrayFromImage(labelled).max())


def inspect(ct: np.ndarray, seg: np.ndarray, spacing: tuple[float, float]) -> dict:
    px_cm2 = spacing[0] * spacing[1] / 100.0
    stats, flags = {}, []
    for label, name in LABEL_NAMES.items():
        mask = seg == label
        count = int(mask.sum())
        stats[f"{name}_cm2"] = round(count * px_cm2, 2)
        if count == 0:
            stats[f"{name}_hu_median"] = ""
            stats[f"{name}_components"] = 0
            flags.append(f"{name}:missing")
            continue
        values = ct[mask]
        median = float(np.median(values))
        stats[f"{name}_hu_median"] = round(median, 1)
        stats[f"{name}_components"] = components(mask)
        low, high = HU_WINDOW[label]
        if not low <= median <= high:
            flags.append(f"{name}:hu_median={median:.0f}")
        air = float((values < AIR_HU).mean())
        if air > 0.05:
            flags.append(f"{name}:air={air:.0%}")
    stats["total_cm2"] = round(sum(stats[f"{n}_cm2"] for n in LABEL_NAMES.values()), 2)
    return stats, flags


def render(case_id: str, branch: str, info: dict, ct: np.ndarray, seg: np.ndarray,
           stats: dict, flags: list[str], out_path: Path) -> None:
    low, high = SOFT_WINDOW
    window = np.clip((ct - low) / (high - low), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    captions = ["CT only", "re-annotation (filled)", "re-annotation (outline)"]
    for ax, caption in zip(axes, captions):
        ax.imshow(window, cmap="gray", vmin=0, vmax=1)
        if caption.endswith("(filled)"):
            ax.imshow(np.ma.masked_where(seg == 0, seg.astype(float)), cmap=OVERLAY,
                      alpha=0.40, vmin=0, vmax=4, interpolation="nearest")
        elif caption.endswith("(outline)"):
            for label, color in LABEL_COLORS.items():
                if (seg == label).any():
                    ax.contour(seg == label, levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(caption, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    areas = "  ".join(f"{n} {stats[f'{n}_cm2']:.1f}" for n in LABEL_NAMES.values())
    title = (f"{case_id}   {branch}   fold {info['fold']}   "
             f"z {info['z_index_previous']} -> {info['z_index_new']}\n"
             f"cm²:  {areas}   |   total {stats['total_cm2']:.1f}")
    if flags:
        title += "\nFLAG  " + "   ".join(flags)
    fig.suptitle(title, fontsize=11, color="firebrick" if flags else "black")
    axes[2].legend(handles=[Line2D([], [], color=LABEL_COLORS[l], lw=2, label=LABEL_NAMES[l])
                            for l in sorted(LABEL_NAMES)],
                   loc="lower right", fontsize=7, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def contact_sheet(panels: list[tuple[str, np.ndarray, np.ndarray, bool]], out_path: Path,
                  title: str) -> None:
    columns = 8
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.1, rows * 2.25))
    low, high = SOFT_WINDOW
    for ax, panel in zip(np.ravel(axes), panels):
        case_id, ct, seg, flagged = panel
        ax.imshow(np.clip((ct - low) / (high - low), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(seg == 0, seg.astype(float)), cmap=OVERLAY,
                  alpha=0.40, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(case_id, fontsize=7, color="firebrick" if flagged else "black")
    for ax in np.ravel(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        if not ax.has_data():
            ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reannotation-dir",
                    default="${ADPKD_ROOT}/reannotation/0917_reannotation")
    ap.add_argument("--out-dir",
                    default="${ADPKD_ROOT}/qc/reannotation")
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    args = ap.parse_args()

    source = Path(args.reannotation_dir)
    out_root = Path(args.out_dir)
    table = []

    for branch in args.branches:
        branch_dir = source / branch
        if not branch_dir.is_dir():
            continue
        out_dir = out_root / branch
        out_dir.mkdir(parents=True, exist_ok=True)
        panels = []

        for case_dir in sorted(branch_dir.iterdir()):
            case_id = case_dir.name
            info = json.loads((case_dir / "info.json").read_text())
            ct, ct_image = load_slice(case_dir / f"{case_id}.nii.gz")
            seg, seg_image = load_slice(case_dir / f"{case_id}_mask.seg.nrrd")

            stats, flags = inspect(ct, seg, ct_image.GetSpacing()[:2])
            if not same_grid(ct_image, seg_image):
                flags.insert(0, "grid_mismatch")
            extra = sorted(set(np.unique(seg)) - {0, *LABEL_NAMES})
            if extra:
                flags.insert(0, f"unexpected_labels={extra}")

            render(case_id, branch, info, ct, seg, stats, flags, out_dir / f"{case_id}.png")
            panels.append((case_id, ct, seg, bool(flags)))
            table.append({"case_id": case_id, "branch": branch, "fold": info["fold"],
                          "z_index_new": info["z_index_new"],
                          "z_index_previous": info["z_index_previous"],
                          **stats, "flags": ";".join(flags)})
            print(f"   [{branch}] {case_id}{'  FLAG ' + ';'.join(flags) if flags else ''}",
                  flush=True)

        contact_sheet(panels, out_root / f"{branch}_contact.png",
                      f"{branch}  re-annotation  {len(panels)} cases  (red title = flagged)")

    with (out_root / "reannotation_qc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    flagged = [r for r in table if r["flags"]]
    print(f"[DONE] {out_root}  cases={len(table)}  flagged={len(flagged)}")
    for row in flagged:
        print(f"   {row['branch']} {row['case_id']}  {row['flags']}")


if __name__ == "__main__":
    main()
