#!/usr/bin/env python3
"""Score the reference-section detectors alone, before running the full pipeline.

Input slices come from the fold-matched L3 out-of-fold validation predictions that nnU-Net
already wrote, so no segmentation inference is repeated. Each case is scored with the detector
of its own fold. Slice error is converted to millimetres with the case's own z-spacing.
"""
from __future__ import annotations

import os
import argparse
import csv
import json
import shutil
import statistics as st
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[1]))
L3_VALIDATION = (ROOT / "workspace/nnunet/nnUNet_results/Dataset920_L3Vertebra0917"
                 / "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres")


def load_registry() -> dict:
    return json.loads((ROOT / "config/model_registry.json").read_text())


MATCH_TOLERANCE_HU = 0.5


def verify_reference(ct: np.ndarray, reference_path: Path, z: int, case_id: str, branch: str) -> None:
    """The manifest z must be the slice the ground-truth reference CT was cut from.

    Stored reference slices carry small rounding differences, so match on mean absolute
    difference; the runner-up slice sits tens of HU away, which makes the winner unambiguous.
    """
    reference = np.asanyarray(nib.load(str(reference_path)).dataobj, dtype=np.float32)
    if reference.ndim == 3:
        reference = reference[:, :, 0]
    if float(np.abs(ct[:, :, z] - reference).mean()) <= MATCH_TOLERANCE_HU:
        return
    errors = [(float(np.abs(ct[:, :, k] - reference).mean()), k) for k in range(ct.shape[2])]
    best_error, best_z = min(errors)
    raise ValueError(f"{case_id} {branch}: reference CT matches z={best_z} "
                     f"(mean |diff|={best_error:.4f}), not the manifest z={z}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default=str(ROOT / "results/detector_only"))
    ap.add_argument("--keep-slices", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work/detector_only"

    with (ROOT / "manifests/manifest_300_all_folds.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))

    rows = []
    for fold in args.folds:
        cases = [r for r in manifest if int(r["fold"]) == fold]
        models = {b: YOLO(registry["folds"][str(fold)][f"yolo_{b}"]["weights"]) for b in args.branches}
        conf = float(registry["folds"][str(fold)][f"yolo_{args.branches[0]}"]["conf"])

        for index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            ct_nii = nib.load(case["ct_path"])
            ct = np.asanyarray(ct_nii.dataobj, dtype=np.float32)
            spacing_z = float(ct_nii.header.get_zooms()[2])
            mask = np.asanyarray(
                nib.load(str(L3_VALIDATION / f"fold_{fold}" / "validation" / f"{case_id}.nii.gz")).dataobj) > 0
            slices = np.flatnonzero(mask.any(axis=(0, 1))).tolist()

            slice_dir = work / f"fold_{fold}" / case_id
            if slice_dir.exists():
                shutil.rmtree(slice_dir)
            slice_dir.mkdir(parents=True)
            paths = []
            for z in slices:
                path = slice_dir / f"{case_id}_{z}.png"
                Image.fromarray(mask[:, :, z].astype(np.uint8) * 255, mode="L").save(path)
                paths.append(path)

            for branch in args.branches:
                gt_z = int(case[f"{branch}_z_index"])
                verify_reference(ct, Path(case[f"{branch}_reference_ct_path"]), gt_z, case_id, branch)
                results = models[branch].predict(source=[str(p) for p in paths], imgsz=512,
                                                 device=args.device, conf=conf, verbose=False)
                best_z, best_conf = None, -1.0
                for z, result in zip(slices, results):
                    score = 0.0
                    if result.boxes is not None and len(result.boxes.conf) > 0:
                        score = float(result.boxes.conf.max().item())
                    if score > best_conf:
                        best_conf, best_z = score, z
                rows.append({
                    "fold": fold, "case_id": case_id, "branch": branch,
                    "gt_z": gt_z, "pred_z": best_z, "yolo_conf": round(best_conf, 6),
                    "slice_error": best_z - gt_z, "abs_slice_error": abs(best_z - gt_z),
                    "spacing_z_mm": spacing_z,
                    "error_mm": (best_z - gt_z) * spacing_z,
                    "abs_error_mm": abs(best_z - gt_z) * spacing_z,
                    "candidate_slices": len(slices),
                    "gt_z_in_candidates": int(gt_z in slices),
                })
            if not args.keep_slices:
                shutil.rmtree(slice_dir)
            print(f"[fold {fold} {index}/{len(cases)}] {case_id}", flush=True)

    per_case = out_dir / "per_case_detector_only.csv"
    with per_case.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for branch in args.branches:
        subset = [r for r in rows if r["branch"] == branch]
        errors = [r["abs_slice_error"] for r in subset]
        mm = [r["abs_error_mm"] for r in subset]
        signed_mm = [r["error_mm"] for r in subset]
        summary[branch] = {
            "n": len(subset),
            "exact_rate": st.fmean(float(e == 0) for e in errors),
            "within_1_rate": st.fmean(float(e <= 1) for e in errors),
            "within_2_rate": st.fmean(float(e <= 2) for e in errors),
            "mean_abs_slice_error": st.fmean(errors),
            "median_abs_error_mm": st.median(mm),
            "mean_abs_error_mm": st.fmean(mm),
            "sd_abs_error_mm": st.stdev(mm),
            "signed_error_mm_mean": st.fmean(signed_mm),
            "signed_error_mm_sd": st.stdev(signed_mm),
            "max_abs_error_mm": max(mm),
            "gt_outside_candidates": sum(1 for r in subset if not r["gt_z_in_candidates"]),
        }
        by_fold = {}
        for fold in args.folds:
            fold_rows = [r for r in subset if r["fold"] == fold]
            if fold_rows:
                by_fold[str(fold)] = {
                    "n": len(fold_rows),
                    "exact_rate": st.fmean(float(r["abs_slice_error"] == 0) for r in fold_rows),
                    "mean_abs_error_mm": st.fmean(r["abs_error_mm"] for r in fold_rows),
                }
        summary[branch]["by_fold"] = by_fold

    (out_dir / "detector_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] {per_case}")


if __name__ == "__main__":
    main()
