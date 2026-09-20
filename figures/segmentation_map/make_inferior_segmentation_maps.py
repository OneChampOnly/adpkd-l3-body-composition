#!/usr/bin/env python3
"""Create paper-ready inferior inference segmentation maps for Figures 3 and 4."""

from __future__ import annotations

import os
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_OUT_DIR = ROOT / "paper_data/segmentation_map"
DEFAULT_CASES = (
    # Case identifiers are hospital record numbers and are not distributed.
    # Pass the cases you want with --cases NAME:FOLD:CASE_ID.
)

# Label IDs follow the segmentation model convention.
LABELS = {
    1: {
        "key": "sma",
        "name": "SMA",
        "full_name": "Skeletal Muscle Area",
        "rgb": (255, 242, 0),  # Yellow
    },
    2: {
        "key": "sfat",
        "name": "SFAT",
        "full_name": "Subcutaneous Fat Area",
        "rgb": (91, 133, 207),  # Blue
    },
    3: {
        "key": "vfat",
        "name": "VFAT",
        "full_name": "Visceral Fat Area",
        "rgb": (121, 180, 111),  # Green
    },
    4: {
        "key": "imat",
        "name": "IMAT",
        "full_name": "Intermuscular Adipose Tissue",
        "rgb": (239, 68, 68),  # Red
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--window-min", type=float, default=-200.0)
    parser.add_argument("--window-max", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=300)
    # Render alternative cases without touching DEFAULT_CASES, so figure candidates come out
    # under exactly the parameters the published figures use.
    parser.add_argument(
        "--cases", nargs="+", metavar="NAME:FOLD:CASE_ID", default=None,
        help="Cases to render, e.g. --cases figure3:0:CASE_ID figure4:2:CASE_ID")
    return parser.parse_args()


def load_single_slice(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 2:
        return data
    if data.ndim != 3 or data.shape[2] != 1:
        raise ValueError(f"Expected a 2D/single-slice image, got {data.shape}: {path}")
    return data[:, :, 0]


def normalize_ct(ct: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    clipped = np.clip(ct.astype(np.float32), window_min, window_max)
    return (clipped - window_min) / (window_max - window_min)


def make_rgb_overlay(
    ct: np.ndarray,
    mask: np.ndarray,
    label_ids: tuple[int, ...],
    alpha: float,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    gray = normalize_ct(ct, window_min, window_max)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for label_id in label_ids:
        color = np.asarray(LABELS[label_id]["rgb"], dtype=np.float32) / 255.0
        selected = mask == label_id
        rgb[selected] = (1.0 - alpha) * rgb[selected] + alpha * color
    return np.clip(rgb, 0.0, 1.0)


def save_png(image_rgb: np.ndarray, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.imshow(np.transpose(image_rgb, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def create_case(
    figure_name: str,
    fold: int,
    case_id: str,
    out_root: Path,
    alpha: float,
    window_min: float,
    window_max: float,
    dpi: int,
) -> list[dict[str, object]]:
    image_dir = ROOT / f"results/fold_{fold}/pipeline_output/{case_id}/image"
    slice_candidates = sorted(image_dir.glob(f"{case_id}_inferior_slice_idx*.nii.gz"))
    if len(slice_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one inferior slice for {case_id}, found {len(slice_candidates)}"
        )
    ct_path = slice_candidates[0]
    seg_path = image_dir / f"{case_id}_inferior_sma_seg.nii.gz"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Missing prediction mask: {seg_path}")

    ct = load_single_slice(ct_path)
    mask = np.rint(load_single_slice(seg_path)).astype(np.int16)
    if ct.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {case_id}: CT={ct.shape}, mask={mask.shape}")
    unexpected = sorted(set(np.unique(mask).tolist()) - {0, 1, 2, 3, 4})
    if unexpected:
        raise ValueError(f"Unexpected labels for {case_id}: {unexpected}")

    out_dir = out_root / f"{figure_name}_{case_id}"
    pixel_area_cm2 = float(nib.load(str(seg_path)).header.get_zooms()[0]) * float(
        nib.load(str(seg_path)).header.get_zooms()[1]
    ) / 100.0
    rows: list[dict[str, object]] = []

    for label_id, info in LABELS.items():
        key = str(info["key"])
        out_path = out_dir / f"{case_id}_{key}_overlay_alpha{alpha:.1f}.png"
        save_png(
            make_rgb_overlay(ct, mask, (label_id,), alpha, window_min, window_max),
            out_path,
            dpi,
        )
        pixel_count = int(np.count_nonzero(mask == label_id))
        rows.append(
            {
                "figure": figure_name,
                "fold": fold,
                "case_id": case_id,
                "source": "inferior_end_to_end_prediction",
                "output_type": key,
                "label_id": label_id,
                "label_name": info["name"],
                "color_rgb": ";".join(str(value) for value in info["rgb"]),
                "pixel_count": pixel_count,
                "area_cm2": f"{pixel_count * pixel_area_cm2:.6f}",
                "ct_path": str(ct_path),
                "seg_path": str(seg_path),
                "png_path": str(out_path),
            }
        )
        print(f"[OK] {figure_name} {case_id} {info['name']}: {out_path}")

    combined_path = out_dir / f"{case_id}_combined_overlay_alpha{alpha:.1f}.png"
    save_png(
        make_rgb_overlay(ct, mask, tuple(LABELS), alpha, window_min, window_max),
        combined_path,
        dpi,
    )
    rows.append(
        {
            "figure": figure_name,
            "fold": fold,
            "case_id": case_id,
            "source": "inferior_end_to_end_prediction",
            "output_type": "combined",
            "label_id": "1;2;3;4",
            "label_name": "SMA;SFAT;VFAT;IMAT",
            "color_rgb": "255,242,0;91,133,207;121,180,111;239,68,68",
            "pixel_count": int(np.count_nonzero(mask > 0)),
            "area_cm2": f"{np.count_nonzero(mask > 0) * pixel_area_cm2:.6f}",
            "ct_path": str(ct_path),
            "seg_path": str(seg_path),
            "png_path": str(combined_path),
        }
    )
    print(f"[OK] {figure_name} {case_id} combined: {combined_path}")
    return rows


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.window_min >= args.window_max:
        raise ValueError("--window-min must be smaller than --window-max")

    if args.cases:
        cases = []
        for spec in args.cases:
            name, fold, case_id = spec.split(":")
            cases.append((name, int(fold), case_id))
    else:
        cases = list(DEFAULT_CASES)

    rows: list[dict[str, object]] = []
    for figure_name, fold, case_id in cases:
        rows.extend(
            create_case(
                figure_name,
                fold,
                case_id,
                args.out_dir,
                args.alpha,
                args.window_min,
                args.window_max,
                args.dpi,
            )
        )

    summary_path = args.out_dir / "figure3_figure4_segmentation_map_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
