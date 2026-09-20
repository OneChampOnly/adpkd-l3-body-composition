#!/usr/bin/env python3
"""Detect inferior and middle L3 slices from fold-matched L3 masks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_slices(mask_path: Path, output: Path, case_id: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(output.glob(f"{case_id}_*.png"))
    if existing:
        return existing
    volume = np.asanyarray(nib.load(str(mask_path)).dataobj)
    paths: list[Path] = []
    for z in range(volume.shape[2]):
        if np.any(volume[:, :, z] > 0):
            path = output / f"{case_id}_{z}.png"
            Image.fromarray((volume[:, :, z] > 0).astype(np.uint8) * 255, mode="L").save(path)
            paths.append(path)
    return paths


def z_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.png$", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def predict_best(model: YOLO, paths: list[Path], device: str, imgsz: int, conf: float) -> tuple[int, float, Path]:
    if not paths:
        raise RuntimeError("L3 prediction has no non-empty slices")
    results = model.predict([str(path) for path in paths], device=device, imgsz=imgsz, conf=conf, verbose=False)
    scored: list[tuple[float, Path]] = []
    for path, result in zip(paths, results):
        score = 0.0
        if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf):
            score = float(result.boxes.conf.max().item())
        scored.append((score, path))
    score, path = max(scored, key=lambda item: item[0])
    return z_index(path), score, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    registry = json.loads((PARENT / "config/model_registry.json").read_text(encoding="utf-8"))
    # Fold-matched detectors: the 0820 registry kept one shared weight for every fold.
    shared = registry["folds"][str(args.fold)]
    manifest = [row for row in read_rows(ROOT / "manifests/z5mm_manifest_300.csv") if int(row["fold"]) == args.fold]
    models = {
        branch: YOLO(shared[f"yolo_{branch}"]["weights"])
        for branch in ("inferior", "middle")
    }
    rows: list[dict[str, object]] = []
    for number, item in enumerate(manifest, start=1):
        case_id = item["case_id"]
        mask_path = ROOT / f"results/fold_{args.fold}/l3_masks/{case_id}.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        slice_dir = ROOT / f"results/fold_{args.fold}/l3_slices/{case_id}"
        paths = extract_slices(mask_path, slice_dir, case_id)
        for branch, model in models.items():
            spec = shared[f"yolo_{branch}"]
            pred_idx, score, best_path = predict_best(model, paths, args.device, int(spec["imgsz"]), float(spec["conf"]))
            gt_idx = int(item[f"{branch}_gt_idx_z5mm"])
            signed = pred_idx - gt_idx
            rows.append({
                "fold": args.fold,
                "case_id": case_id,
                "branch": branch,
                "gt_slice_index_z5mm": gt_idx,
                "pred_slice_index_z5mm": pred_idx,
                "signed_slice_error": signed,
                "absolute_slice_error": abs(signed),
                "signed_error_mm": signed * 5.0,
                "absolute_error_mm": abs(signed) * 5.0,
                "exact_slice": int(signed == 0),
                "within_1_slice": int(abs(signed) <= 1),
                "within_2_slices": int(abs(signed) <= 2),
                "within_5mm": int(abs(signed) * 5.0 <= 5.0),
                "within_10mm": int(abs(signed) * 5.0 <= 10.0),
                "yolo_conf": score,
                "num_l3_mask_slices": len(paths),
                "best_slice_png": best_path,
                "l3_mask_path": mask_path,
                "ct_z5mm_path": item["ct_z5mm_path"],
            })
        if number == 1 or number % 10 == 0:
            print(f"[YOLO fold={args.fold}] {number}/{len(manifest)}", flush=True)
    write_rows(ROOT / f"results/fold_{args.fold}/detection_per_case.csv", rows)
    print(f"[OK] fold={args.fold} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
