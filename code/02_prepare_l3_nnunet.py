#!/usr/bin/env python3
"""Stage the clean dataset into nnU-Net raw layout for L3 vertebra segmentation.

The clean dataset already holds a binarised L3 mask carrying the CT geometry, so this only
lays out imagesTr/labelsTr and writes dataset.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--nnunet-raw", required=True)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    manifest = {r["case_id"]: r for r in csv.DictReader((root / "dataset_manifest.csv").open())}

    raw = Path(args.nnunet_raw) / f"Dataset{cfg['l3_dataset_id']}_{cfg['l3_dataset_name']}"
    images, labels = raw / "imagesTr", raw / "labelsTr"
    for directory in (images, labels):
        directory.mkdir(parents=True, exist_ok=True)

    for index, case_id in enumerate(sorted(manifest), start=1):
        ct = dataset_dir / case_id / "ct.nii.gz"
        mask = dataset_dir / case_id / "l3_mask.nii.gz"
        if not ct.is_file() or not mask.is_file():
            raise FileNotFoundError(f"{case_id}: clean dataset is incomplete")
        image_out, label_out = images / f"{case_id}_0000.nii.gz", labels / f"{case_id}.nii.gz"
        for path in (image_out, label_out):
            if path.exists() or path.is_symlink():
                path.unlink()
        if args.copy_images:
            shutil.copy2(ct, image_out)
        else:
            os.symlink(ct.resolve(), image_out)
        shutil.copy2(mask, label_out)
        print(f"[{index}/{len(manifest)}] {case_id}", flush=True)

    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "L3_vertebra": 1},
        "numTraining": len(manifest),
        "file_ending": ".nii.gz",
    }, indent=2) + "\n")
    print(f"[DONE] {raw}  cases={len(manifest)}")


if __name__ == "__main__":
    main()
