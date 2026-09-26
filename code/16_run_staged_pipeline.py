#!/usr/bin/env python3
"""Run the end-to-end pipeline stage by stage instead of case by case.

Same models, same flags, same outputs as 13_run_end_to_end_5fold.py - the per-case helpers are
imported from pipeline_core so nothing about the numbers changes. The difference is batching:
each nnU-Net checkpoint and each detector is loaded once per fold rather than once per case,
which is where almost all of the wall time was going (25 s of inference behind ~6 min of startup).

Stage order still respects the data flow: every case needs its L3 mask before detection, and its
detected slice before body-composition segmentation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_core as pc  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YOLO_IMGSZ = 512


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--yolo-device", default="0")
    ap.add_argument("--case-limit", type=int, default=0)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo cases that already have metrics")
    return ap.parse_args()


def predict_batch(spec: dict, input_dir: Path, output_dir: Path, log_path: Path, step: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        pc.NNUNETV2_PREDICT_BIN,
        "-d", spec["dataset"],
        "-c", spec["config"],
        "-f", str(spec["fold"]),
        "-tr", spec["trainer"],
        "-p", spec["plans"],
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-chk", Path(spec["checkpoint"]).name,
    ]
    pc.run_subprocess(cmd, str(log_path), step_name=step)


def stage_inputs(pairs: list[tuple[str, Path]], directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for case_id, source in pairs:
        (directory / f"{case_id}_0000.nii.gz").symlink_to(Path(source).resolve())


def main() -> None:
    args = parse_args()
    registry = json.loads((ROOT / "config/model_registry.json").read_text())

    # nnUNet_results must hold <dataset>/<trainer>__<plans>__<config>/fold_k/. With the
    # released weights that is the download directory, so NNUNET_RESULTS overrides the
    # in-tree default instead of being silently discarded.
    results_dir = Path(os.environ.get("NNUNET_RESULTS", ROOT / "work/nnUNet_results"))
    raw_dir = ROOT / "work/nnUNet_raw"
    preprocessed_dir = ROOT / "work/nnUNet_preprocessed"
    # nnU-Net refuses to start unless these exist, though inference never reads them.
    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"nnUNet_results not found: {results_dir}")
    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["nnUNet_results"] = str(results_dir)

    for fold in args.folds:
        started = time.time()
        with (ROOT / f"manifests/manifest_fold_{fold}.csv").open(newline="") as handle:
            cases = list(csv.DictReader(handle))
        if args.case_limit:
            cases = cases[:args.case_limit]
        models = registry["folds"][str(fold)]
        out_root = ROOT / f"results/fold_{fold}/pipeline_output"
        work = ROOT / f"work/staged/fold_{fold}"
        log_root = ROOT / f"logs/staged/fold_{fold}"
        log_root.mkdir(parents=True, exist_ok=True)

        pending = []
        for case in cases:
            done = all((out_root / case["case_id"] / "csv" /
                        f"{case['case_id']}_{branch}_metrics.csv").is_file() for branch in args.branches)
            if args.force or not done:
                pending.append(case)
        print(f"\n===== FOLD {fold}: {len(pending)}/{len(cases)} cases to process =====", flush=True)
        if not pending:
            continue
        for case in pending:
            for sub in ("image", "csv", "logs"):
                (out_root / case["case_id"] / sub).mkdir(parents=True, exist_ok=True)

        # Stage 1 - one L3 segmentation call for the whole fold.
        print(f"[FOLD {fold}] stage 1/4  L3 segmentation on {len(pending)} cases", flush=True)
        stage_inputs([(c["case_id"], Path(c["ct_path"])) for c in pending], work / "l3_input")
        predict_batch(models["l3"], work / "l3_input", work / "l3_pred",
                      log_root / "l3_inference.log", f"L3_FOLD{fold}")
        for case in pending:
            case_id = case["case_id"]
            source = work / "l3_pred" / f"{case_id}.nii.gz"
            if not source.is_file():
                raise FileNotFoundError(f"L3 prediction missing: {source}")
            shutil.copy2(source, out_root / case_id / "image" / f"{case_id}_l3_mask.nii.gz")

        # Stage 2 - one detector load per branch for the whole fold.
        chosen: dict[tuple[str, str], dict] = {}
        for branch in args.branches:
            spec = models[f"yolo_{branch}"]
            print(f"[FOLD {fold}] stage 2/4  {branch} detection", flush=True)
            detector = YOLO(spec["weights"])
            for case in pending:
                case_id = case["case_id"]
                case_root = out_root / case_id
                mask = str(case_root / "image" / f"{case_id}_l3_mask.nii.gz")
                slice_dir = str(work / "yolo_slices" / branch / case_id)
                candidates = pc.extract_nonempty_mask_slices(mask, slice_dir, case_id)
                z, conf, best_png = pc.pick_best_slice(
                    detector, candidates, YOLO_IMGSZ, args.yolo_device, float(spec["conf"]))
                if z is None:
                    raise RuntimeError(f"{case_id} {branch}: detector returned no slice")
                slice_nifti = case_root / "image" / f"{case_id}_{branch}_slice_idx{z}.nii.gz"
                pc.save_single_slice_nifti_from_ct(case["ct_path"], z, str(slice_nifti))
                chosen[(case_id, branch)] = {
                    "slice_index": z, "yolo_best_conf": conf, "yolo_best_img": best_png,
                    "num_l3_mask_slices": len(candidates), "slice_ct_path": str(slice_nifti),
                    "l3_mask_path": mask,
                }
            del detector

        # Stage 3 - one body-composition call per branch for the whole fold.
        for branch in args.branches:
            print(f"[FOLD {fold}] stage 3/4  {branch} body-composition segmentation", flush=True)
            stage_inputs([(c["case_id"], Path(chosen[(c["case_id"], branch)]["slice_ct_path"]))
                          for c in pending], work / f"sma_input_{branch}")
            predict_batch(models[f"sma_{branch}"], work / f"sma_input_{branch}",
                          work / f"sma_pred_{branch}", log_root / f"{branch}_sma_inference.log",
                          f"SMA_{branch.upper()}_FOLD{fold}")
            for case in pending:
                case_id = case["case_id"]
                source = work / f"sma_pred_{branch}" / f"{case_id}.nii.gz"
                if not source.is_file():
                    raise FileNotFoundError(f"{branch} prediction missing: {source}")
                shutil.copy2(source, out_root / case_id / "image" / f"{case_id}_{branch}_sma_seg.nii.gz")

        # Stage 4 - visuals and per-case tables, identical to the per-case pipeline.
        print(f"[FOLD {fold}] stage 4/4  visuals and metrics", flush=True)
        for branch in args.branches:
            for case in pending:
                case_id = case["case_id"]
                case_root = out_root / case_id
                image_dir = str(case_root / "image")
                slice_nifti = chosen[(case_id, branch)]["slice_ct_path"]
                seg = str(case_root / "image" / f"{case_id}_{branch}_sma_seg.nii.gz")
                visuals = pc.save_visual_outputs(slice_nifti, seg, image_dir, case_id, branch)
                metrics = pc.compute_area_metrics_from_ct_and_seg(slice_nifti, seg)
                result = {
                    "case_id": case_id, "branch": branch, "ct_path": case["ct_path"],
                    "l3_mask_path": chosen[(case_id, branch)]["l3_mask_path"],
                    "slice_index": chosen[(case_id, branch)]["slice_index"],
                    "yolo_best_conf": chosen[(case_id, branch)]["yolo_best_conf"],
                    "yolo_best_img": chosen[(case_id, branch)]["yolo_best_img"],
                    "num_l3_mask_slices": chosen[(case_id, branch)]["num_l3_mask_slices"],
                    "slice_ct_path": slice_nifti, "sma_seg_path": seg,
                    "mask_png": visuals["mask_png"], "overlay_png": visuals["overlay_png"],
                    "pred_sma": metrics["sma"], "pred_sma_hu": metrics["sma_hu"],
                    "pred_sfat": metrics["sfat"], "pred_vfat": metrics["vfat"],
                    "pred_imat": metrics["imat"],
                }
                csv_dir = case_root / "csv"
                pc.save_csv(str(csv_dir / f"{case_id}_{branch}_metrics.csv"), [result])
                pc.write_json(str(csv_dir / f"{case_id}_{branch}_result.json"), result)

        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        minutes = (time.time() - started) / 60
        print(f"[FOLD {fold}] done: {len(pending)} cases in {minutes:.1f} min "
              f"({minutes * 60 / max(len(pending), 1):.1f} s/case)", flush=True)


if __name__ == "__main__":
    main()
