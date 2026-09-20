#!/usr/bin/env python3
"""Standalone localisation: fold-matched YOLO detectors run on the GT L3 mask.

This is the upper-bound counterpart to the end-to-end localisation number. The pipeline
feeds the detector the *predicted* L3 mask, so its slice error carries whatever the L3
segmenter got wrong. Here the detector receives the reference L3 mask instead, which
isolates the detector's own contribution and leaves the difference between the two
attributable to upstream segmentation error.

The detectors were trained on the manual L3 masks (verified: the YOLO training PNGs match
dataset/<case>/l3_mask.nii.gz at Dice 1.0000), so this evaluation is in-distribution and is a
clean "perfect upstream" measurement. It is the end-to-end condition that is shifted, because
there the detector sees a predicted mask with rougher boundaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline_core as pc  # noqa: E402

from ultralytics import YOLO  # noqa: E402

BRANCHES = ("inferior", "middle")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def world_z(affine: np.ndarray, shape: tuple[int, ...], z: int) -> float:
    """Same convention as the end-to-end localisation table: centre voxel, world z."""
    cx, cy = (np.asarray(shape[:2], dtype=float) - 1.0) / 2.0
    return float((affine @ np.array([cx, cy, float(z), 1.0]))[2])


def summarise(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-device", default="0")
    ap.add_argument("--out-dir", default=str(ROOT / "results/detector_on_gt_mask"))
    args = ap.parse_args()

    registry = json.loads((ROOT / "config/model_registry.json").read_text())
    manifest = read_csv(ROOT / "manifests/manifest_300_all_folds.csv")
    e2e = {(r["case_id"], r["branch"]): r for r in read_csv(ROOT / "results/evaluation/per_case_end_to_end.csv")}

    work = ROOT / "work/detector_on_gt_mask"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_fold: dict[int, list[dict[str, str]]] = {}
    for row in manifest:
        by_fold.setdefault(int(row["fold"]), []).append(row)

    rows: list[dict[str, object]] = []
    for fold in sorted(by_fold):
        cases = by_fold[fold]
        for branch in BRANCHES:
            spec = registry["folds"][str(fold)][f"yolo_{branch}"]
            detector = YOLO(spec["weights"])
            print(f"[FOLD {fold}] {branch}: {len(cases)} cases on the GT L3 mask", flush=True)

            for index, case in enumerate(cases, start=1):
                case_id = case["case_id"]
                gt_mask = case["l3_gt_path"]
                slice_dir = work / f"fold_{fold}" / branch / case_id
                candidates = pc.extract_nonempty_mask_slices(gt_mask, str(slice_dir), case_id)
                pred_z, conf, _ = pc.pick_best_slice(
                    detector, candidates, int(spec["imgsz"]), args.yolo_device, float(spec["conf"])
                )
                if pred_z is None:
                    raise RuntimeError(f"{case_id} {branch}: detector returned no slice")

                reference = e2e[(case_id, branch)]
                gt_z = int(reference["gt_slice_index"])
                e2e_z = int(reference["pred_slice_index"])

                image = nib.load(case["ct_path"])
                affine, shape = image.affine, image.shape
                gt_mm = world_z(affine, shape, gt_z)

                rows.append({
                    "fold": fold, "case_id": case_id, "branch": branch,
                    "gt_slice_index": gt_z,
                    "gtmask_pred_slice_index": pred_z,
                    "e2e_pred_slice_index": e2e_z,
                    "gtmask_slice_error": pred_z - gt_z,
                    "e2e_slice_error": e2e_z - gt_z,
                    "gtmask_abs_error_mm": abs(world_z(affine, shape, pred_z) - gt_mm),
                    "e2e_abs_error_mm": abs(world_z(affine, shape, e2e_z) - gt_mm),
                    "gtmask_yolo_conf": conf,
                    "l3_dsc": float(reference["l3_dsc"]),
                    "n_candidate_slices": len(candidates),
                })
                if index % 20 == 0 or index == len(cases):
                    print(f"  [{index}/{len(cases)}] {case_id}", flush=True)

    per_case = out_dir / "per_case_detector_on_gt_mask.csv"
    with per_case.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"n_cases": len(manifest), "by_branch": {}}
    for branch in BRANCHES:
        sel = [r for r in rows if r["branch"] == branch]
        gt_mm = [float(r["gtmask_abs_error_mm"]) for r in sel]
        e2e_mm = [float(r["e2e_abs_error_mm"]) for r in sel]
        paired = [a - b for a, b in zip(e2e_mm, gt_mm)]
        summary["by_branch"][branch] = {
            "n": len(sel),
            "gt_mask": {
                **summarise(gt_mm),
                "exact_rate": statistics.fmean(1.0 if r["gtmask_slice_error"] == 0 else 0.0 for r in sel),
                "within_1_rate": statistics.fmean(1.0 if abs(int(r["gtmask_slice_error"])) <= 1 else 0.0 for r in sel),
            },
            "end_to_end": {
                **summarise(e2e_mm),
                "exact_rate": statistics.fmean(1.0 if r["e2e_slice_error"] == 0 else 0.0 for r in sel),
                "within_1_rate": statistics.fmean(1.0 if abs(int(r["e2e_slice_error"])) <= 1 else 0.0 for r in sel),
            },
            "paired_increase_mm": summarise(paired),
        }
    (out_dir / "detector_on_gt_mask_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
