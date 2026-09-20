#!/usr/bin/env python3
"""Turn the corrected body-composition GT into an nnU-Net raw dataset, one per branch.

The label images in ``dataset/<case>/gt/`` come from Slicer and are not all on the same grid:
a mask the annotator redrew covers the whole slice, while the older ones are cropped to their
bounding box and carry the crop origin in ``Segmentation_ReferenceImageExtentOffset``. Both are
placed back on the reference CT grid here, by integer offset rather than by resampling, so no
voxel is interpolated. Per-label voxel counts are compared before and after and a case that
changes is an error, not a warning.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

LABELS = {"muscle": 1, "sfat": 2, "vfat": 3, "mfat": 4}
OFFSET_KEY = "Segmentation_ReferenceImageExtentOffset"


def counts(array: np.ndarray) -> dict[int, int]:
    return {int(v): int((array == v).sum()) for v in LABELS.values()}


def on_ct_grid(seg_path: Path, ct: sitk.Image) -> np.ndarray:
    """Return the mask as an array on the CT grid, without interpolating."""
    seg = sitk.ReadImage(str(seg_path))
    array = sitk.GetArrayFromImage(seg).astype(np.uint8)
    target = tuple(reversed(ct.GetSize()))  # sitk is x,y,z; arrays are z,y,x
    if array.shape == target:
        return array
    if not seg.HasMetaDataKey(OFFSET_KEY):
        raise RuntimeError(f"{seg_path}: shape {array.shape} != CT {target} and no {OFFSET_KEY}")
    x0, y0, z0 = (int(v) for v in seg.GetMetaData(OFFSET_KEY).split())
    z1, y1, x1 = z0 + array.shape[0], y0 + array.shape[1], x0 + array.shape[2]
    if min(x0, y0, z0) < 0 or (z1, y1, x1) > target:
        raise RuntimeError(f"{seg_path}: crop {array.shape} at offset {(x0, y0, z0)} leaves CT {target}")
    full = np.zeros(target, dtype=np.uint8)
    full[z0:z1, y0:y1, x0:x1] = array
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--nnunet-raw", required=True)
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--expected-cases", type=int, default=300)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    raw_root = Path(args.nnunet_raw)
    case_ids = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if len(case_ids) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} cases, found {len(case_ids)}")

    for branch in args.branches:
        dataset_id = cfg[f"{branch}_dataset_id"]
        name = cfg[f"{branch}_dataset_name"]
        out_dir = raw_root / f"Dataset{dataset_id:03d}_{name}"
        images, labels = out_dir / "imagesTr", out_dir / "labelsTr"
        for directory in (images, labels):
            directory.mkdir(parents=True, exist_ok=True)

        for index, case_id in enumerate(case_ids, 1):
            ct_path = dataset_dir / case_id / "reference" / f"{branch}_ct.nii.gz"
            seg_path = dataset_dir / case_id / "gt" / f"{branch}_mask.seg.nrrd"
            ct = sitk.ReadImage(str(ct_path))

            array = on_ct_grid(seg_path, ct)
            before = counts(sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))))
            after = counts(array)
            if before != after:
                raise RuntimeError(f"{case_id} {branch}: voxel counts changed {before} -> {after}")
            missing = [n for n, v in LABELS.items() if after[v] == 0]
            if missing:
                raise RuntimeError(f"{case_id} {branch}: labels absent {missing}")

            label_image = sitk.GetImageFromArray(array)
            label_image.CopyInformation(ct)
            shutil.copy2(ct_path, images / f"{case_id}_0000.nii.gz")
            sitk.WriteImage(label_image, str(labels / f"{case_id}.nii.gz"), True)
            if index % 50 == 0:
                print(f"   [{branch}] {index}/{len(case_ids)}", flush=True)

        (out_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, **LABELS},
            "numTraining": len(case_ids),
            "numTest": 0,
            "file_ending": ".nii.gz",
            "name": name,
        }, indent=2) + "\n")
        print(f"[{branch}] {len(case_ids)} cases -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
