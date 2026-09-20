#!/usr/bin/env python3
"""Fold retouched slices and masks back into the clean dataset.

Run this once the annotator has corrected the drafts under ``reannotation/``. The previous
reference slice and GT are kept under ``dataset/<case>/superseded/`` so the change stays
reversible. Nothing is copied unless the mask passes the geometry and label checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

EXPECTED_LABELS = {1, 2, 3, 4}


def check(slice_path: Path, mask_path: Path) -> tuple[dict, list[str]]:
    ct = sitk.ReadImage(str(slice_path))
    mask = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayFromImage(mask)
    present = set(int(v) for v in np.unique(array)) - {0}
    problems = []
    if ct.GetSize()[2] != 1 or mask.GetSize()[2] != 1:
        problems.append(f"expected single-slice images, got ct={ct.GetSize()} mask={mask.GetSize()}")
    if not present:
        problems.append("mask is empty")
    if present - EXPECTED_LABELS:
        problems.append(f"unexpected label values {sorted(present - EXPECTED_LABELS)}")
    missing = EXPECTED_LABELS - present
    if missing:
        problems.append(f"missing labels {sorted(missing)}")
    if not np.allclose(ct.GetSpacing()[:2], mask.GetSpacing()[:2], atol=1e-4):
        problems.append(f"in-plane spacing differs: ct={ct.GetSpacing()} mask={mask.GetSpacing()}")
    return {"labels_present": sorted(present), "mask_size": "x".join(map(str, mask.GetSize())),
            "voxels": {str(v): int((array == v).sum()) for v in sorted(present)}}, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    ap.add_argument("--reannotation-dir", default=None,
                    help="directory holding the corrected cases; defaults to reannotation/ itself, "
                         "which still holds the untouched model drafts")
    ap.add_argument("--apply", action="store_true", help="without this the run only reports")
    ap.add_argument("--allow-missing-labels", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"
    records = []

    source_root = Path(args.reannotation_dir) if args.reannotation_dir else root / "reannotation"
    print(f"[source] {source_root}")
    for branch in args.branches:
        branch_root = source_root / branch
        if not branch_root.is_dir():
            print(f"[{branch}] nothing to apply")
            continue
        for case_dir in sorted(p for p in branch_root.iterdir() if p.is_dir()):
            case_id = case_dir.name
            slice_path = case_dir / f"{case_id}.nii.gz"
            mask_path = case_dir / f"{case_id}_mask.seg.nrrd"
            info = json.loads((case_dir / "info.json").read_text())
            stats, problems = check(slice_path, mask_path)
            blocking = [p for p in problems
                        if not (args.allow_missing_labels and p.startswith("missing labels"))]
            status = "ok" if not blocking else "blocked"

            if status == "ok" and args.apply:
                target = dataset_dir / case_id
                superseded = target / "superseded"
                superseded.mkdir(exist_ok=True)
                for source, name in ((target / "reference" / f"{branch}_ct.nii.gz", f"{branch}_ct.nii.gz"),
                                     (target / "gt" / f"{branch}_mask.seg.nrrd", f"{branch}_mask.seg.nrrd")):
                    if source.exists() and not (superseded / name).exists():
                        shutil.copy2(source, superseded / name)
                shutil.copy2(slice_path, target / "reference" / f"{branch}_ct.nii.gz")
                shutil.copy2(mask_path, target / "gt" / f"{branch}_mask.seg.nrrd")

            records.append({"case_id": case_id, "branch": branch, "fold": info["fold"],
                            "z_index_previous": info["z_index_previous"], "z_index_new": info["z_index_new"],
                            "status": status, "problems": "; ".join(problems),
                            "labels_present": ",".join(map(str, stats["labels_present"])),
                            "mask_size": stats["mask_size"]})
            flag = "" if status == "ok" else f"  <-- {'; '.join(blocking)}"
            print(f"[{branch}] {case_id} z {info['z_index_previous']}->{info['z_index_new']} {status}{flag}")

    out = source_root / "apply_report.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    blocked = [r for r in records if r["status"] != "ok"]
    mode = "applied" if args.apply else "dry run"
    print(f"[DONE] {mode}: {len(records) - len(blocked)} ok, {len(blocked)} blocked -> {out}")
    if not args.apply:
        print("       re-run with --apply to write into dataset/")


if __name__ == "__main__":
    main()
