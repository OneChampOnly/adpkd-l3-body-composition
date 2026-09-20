#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
E2E_ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_slice(image, volume: np.ndarray, z_index: int, output_path: Path) -> None:
    if not 0 <= z_index < image.shape[2]:
        raise ValueError(f"z={z_index} outside 0..{image.shape[2] - 1}")
    array = np.asarray(volume[:, :, z_index], dtype=np.float32)[:, :, None]
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, image.affine.copy(), header=header), str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    rows = read_csv(ROOT / f"manifests/gt_offset_manifest_fold_{args.fold}.csv")
    if len(rows) != 600:
        raise ValueError(f"fold {args.fold}: expected 600 rows, got {len(rows)}")

    nnunet_results = E2E_ROOT / "work/nnUNet_results"
    if not nnunet_results.is_dir():
        raise FileNotFoundError(nnunet_results)
    os.environ["nnUNet_raw"] = str(E2E_ROOT / "work/nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(E2E_ROOT / "work/nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_results)

    # Load each compressed full CT once, then write all 10 requested slices.
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    if len(by_case) != 60 or any(len(case_rows) != 10 for case_rows in by_case.values()):
        raise ValueError(f"fold {args.fold}: expected 60 cases with 10 rows each")
    print(f"[PREPARE] fold={args.fold} cases={len(by_case)} slices={len(rows)}", flush=True)
    for index, (case_id, case_rows) in enumerate(sorted(by_case.items()), start=1):
        image = nib.load(case_rows[0]["ct_path"])
        volume = np.asanyarray(image.dataobj, dtype=np.float32)
        for row in case_rows:
            extract_slice(image, volume, int(row["target_slice_index"]), Path(row["input_slice_path"]))
        del volume
        if index % 10 == 0 or index == len(by_case):
            print(f"[PREPARE] fold={args.fold} cases={index}/{len(by_case)}", flush=True)

    for branch in ("inferior", "middle"):
        selected = [row for row in rows if row["branch"] == branch]
        if len(selected) != 300:
            raise ValueError(f"fold {args.fold} {branch}: expected 300 rows, got {len(selected)}")
        input_dir = ROOT / f"work/fold_{args.fold}/{branch}/imagesTs"
        output_dir = ROOT / f"predictions/fold_{args.fold}/{branch}"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.prepare_only:
            continue

        first = selected[0]
        checkpoint_names = {Path(row["checkpoint"]).name for row in selected}
        datasets = {row["dataset"] for row in selected}
        if len(checkpoint_names) != 1 or len(datasets) != 1:
            raise ValueError(f"fold {args.fold} {branch}: inconsistent model settings")

        command = [
            "/opt/conda/envs/nnunetv2/bin/nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", first["dataset"],
            "-c", first["configuration"],
            "-f", str(args.fold),
            "-tr", first["trainer"],
            "-p", first["plans"],
            "-chk", next(iter(checkpoint_names)),
            "--continue_prediction",
            "--disable_progress_bar",
            "-npp", "4",
            "-nps", "4",
            "-device", "cuda",
        ]
        print("[INFERENCE] " + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())

        missing = [row["prediction_path"] for row in selected if not Path(row["prediction_path"]).is_file()]
        if missing:
            raise RuntimeError(f"fold {args.fold} {branch}: missing {len(missing)} predictions: {missing[:5]}")
        print(f"[DONE] fold={args.fold} branch={branch} predictions={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
