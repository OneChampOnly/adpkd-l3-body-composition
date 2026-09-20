#!/usr/bin/env python3
"""Recompute the mask-derived columns of the label tables against the clean GT L3 masks.

The tables were first built against the old pipeline's predicted masks. Training now uses the
GT mask, so the range and trainability columns have to be re-derived. Reviewer edits in
``z_index`` are preserved untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DERIVED = ("mask_zmin", "mask_zmax", "mask_slices", "mask_components", "z_in_mask", "trainable", "note")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--branches", nargs="+", default=["inferior", "middle"])
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    dataset_dir = root / "dataset"

    profiles = {}
    for case_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        mask = np.asanyarray(nib.load(str(case_dir / "l3_mask.nii.gz")).dataobj) > 0
        nz = np.flatnonzero(mask.any(axis=(0, 1))).tolist()
        components = 1 + sum(1 for a, b in zip(nz, nz[1:]) if b - a > 1) if nz else 0
        profiles[case_dir.name] = {"slices": set(nz), "zmin": nz[0], "zmax": nz[-1],
                                   "count": len(nz), "components": components}

    for branch in args.branches:
        path = root / "labels" / f"{branch}_labels.csv"
        rows = list(csv.DictReader(path.open()))
        changed = []
        for row in rows:
            profile = profiles[row["case_id"]]
            was_trainable = row["trainable"]
            z = int(row["z_index"])
            inside = z in profile["slices"]
            row.update({
                "mask_zmin": profile["zmin"], "mask_zmax": profile["zmax"],
                "mask_slices": profile["count"], "mask_components": profile["components"],
                "z_in_mask": int(inside), "trainable": int(inside),
                "note": "" if inside else "label outside GT L3 mask",
            })
            if was_trainable != str(row["trainable"]):
                changed.append((row["case_id"], was_trainable, row["trainable"], z,
                                profile["zmin"], profile["zmax"]))

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        skipped = [r["case_id"] for r in rows if not int(r["trainable"])]
        edited = [r["case_id"] for r in rows if r["z_index"] != r["z_index_original"]]
        print(f"[{branch}] rows={len(rows)} trainable={len(rows) - len(skipped)} "
              f"skipped={skipped} reviewer_edits={len(edited)}")
        for item in changed:
            print(f"   trainable {item[1]}->{item[2]}: {item[0]} z={item[3]} gt_mask {item[4]}-{item[5]}")


if __name__ == "__main__":
    main()
