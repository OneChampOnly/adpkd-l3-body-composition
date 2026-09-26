#!/usr/bin/env python3
"""Emit the case manifests the end-to-end runner and evaluator consume.

Column names match the 0820 pipeline so the ported runner and evaluator work unchanged.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
from pathlib import Path

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[1]))
DATASET = ROOT / "dataset"

FIELDS = [
    "fold", "case_id", "ct_path", "l3_gt_path",
    "inferior_reference_ct_path", "inferior_gt_path",
    "middle_reference_ct_path", "middle_gt_path",
    "l3_group_used", "inferior_z_index", "middle_z_index",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "manifests"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    for branch in ("inferior", "middle"):
        with (ROOT / "labels" / f"{branch}_labels.csv").open(newline="") as handle:
            labels[branch] = {r["case_id"]: r for r in csv.DictReader(handle)}

    rows = []
    with (ROOT / "dataset_manifest.csv").open(newline="") as handle:
        source = list(csv.DictReader(handle))
    for item in sorted(source, key=lambda r: (int(r["fold"]), r["case_id"])):
        case_id = item["case_id"]
        case_dir = DATASET / case_id
        rows.append({
            "fold": int(item["fold"]), "case_id": case_id,
            "ct_path": str(case_dir / "ct.nii.gz"),
            "l3_gt_path": str(case_dir / "l3_mask.nii.gz"),
            "inferior_reference_ct_path": str(case_dir / "reference" / "inferior_ct.nii.gz"),
            "inferior_gt_path": str(case_dir / "gt" / "inferior_mask.seg.nrrd"),
            "middle_reference_ct_path": str(case_dir / "reference" / "middle_ct.nii.gz"),
            "middle_gt_path": str(case_dir / "gt" / "middle_mask.seg.nrrd"),
            "l3_group_used": item["group_used"],
            "inferior_z_index": labels["inferior"][case_id]["z_index"],
            "middle_z_index": labels["middle"][case_id]["z_index"],
        })

    for row in rows:
        for key, value in row.items():
            if key.endswith("_path") and not Path(value).is_file():
                raise FileNotFoundError(f"{row['case_id']}: {key} -> {value}")

    def write(path: Path, subset: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)

    write(out_dir / "manifest_300_all_folds.csv", rows)
    counts = {}
    for fold in range(5):
        subset = [r for r in rows if r["fold"] == fold]
        write(out_dir / f"manifest_fold_{fold}.csv", subset)
        counts[str(fold)] = len(subset)

    (out_dir / "data_summary.json").write_text(json.dumps(
        {"cases": len(rows), "per_fold": counts, "dataset_root": str(DATASET)}, indent=2) + "\n")
    print(f"[OK] {out_dir}  cases={len(rows)}  per_fold={counts}")


if __name__ == "__main__":
    main()
