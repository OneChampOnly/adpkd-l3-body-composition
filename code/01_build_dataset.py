#!/usr/bin/env python3
"""Assemble one clean per-case dataset from the original staging folders.

The L3 staging tree keeps a `normal` copy of all 300 cases and an `anormaly` copy of the 16
that were re-annotated. This applies that precedence explicitly: `anormaly` wins, and inside
a case folder the corrected NRRD wins over the original NIfTI. Every choice is recorded in
the per-case meta.json and in dataset_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

FIELDS = [
    "case_id", "fold", "group_used", "l3_mask_source_kind",
    "ct_source", "l3_mask_source",
    "inferior_reference_source", "inferior_gt_source",
    "middle_reference_source", "middle_gt_source",
    "shape", "spacing_mm", "l3_foreground_voxels", "l3_zmin", "l3_zmax",
]


def geometry_matches(a: sitk.Image, b: sitk.Image) -> tuple[bool, str]:
    checks = {
        "size": a.GetSize() == b.GetSize(),
        "spacing": np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4),
        "origin": np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3),
        "direction": np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4),
    }
    return all(checks.values()), json.dumps(checks)


def resolve_case_dir(root: Path, cid: str) -> Path:
    """Some staging folders carry an annotator note after the case id, e.g. `<case_id>_여기부터`."""
    exact = root / cid
    if exact.is_dir():
        return exact
    matches = [p for p in root.iterdir()
               if p.is_dir() and p.name.startswith(cid) and not p.name[len(cid):len(cid) + 1].isdigit()]
    if len(matches) != 1:
        raise FileNotFoundError(f"{cid}: expected one directory under {root}, found {sorted(m.name for m in matches)}")
    return matches[0]


def resolve_cases(root: Path, groups: list[str], mask_priority: list[str]) -> dict[str, dict]:
    """Later groups overwrite earlier ones for the same case id."""
    cases: dict[str, dict] = {}
    for group in groups:
        group_dir = root / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"missing group directory: {group_dir}")
        for case_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            cid = case_dir.name
            mask = kind = None
            for pattern in mask_priority:
                candidate = case_dir / pattern.format(cid=cid)
                if candidate.exists():
                    mask, kind = candidate, pattern.format(cid="{cid}")
                    break
            if mask is None:
                raise FileNotFoundError(f"{cid}: no L3 mask under {case_dir}")
            cases[cid] = {"group": group, "ct": case_dir / f"{cid}.nii.gz",
                          "mask": mask, "mask_kind": kind}
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["project_root"])
    out_root = root / "dataset"
    splits = json.loads(Path(cfg["splits_json"]).read_text())
    fold_of = {str(cid): fold for fold, split in enumerate(splits) for cid in split["val"]}

    cases = resolve_cases(Path(cfg["l3_source_root"]), cfg["group_priority"], cfg["l3_mask_priority"])
    if set(cases) != set(fold_of):
        only_source = sorted(set(cases) - set(fold_of))
        only_split = sorted(set(fold_of) - set(cases))
        raise ValueError(f"case sets differ: source-only={only_source}, split-only={only_split}")

    inferior_root, middle_root = Path(cfg["inferior_source_root"]), Path(cfg["middle_source_root"])
    manifest = []
    for index, (cid, item) in enumerate(sorted(cases.items()), start=1):
        case_out = out_root / cid
        if case_out.exists():
            if not args.overwrite:
                raise FileExistsError(f"{case_out} exists; pass --overwrite")
            shutil.rmtree(case_out)
        (case_out / "reference").mkdir(parents=True)
        (case_out / "gt").mkdir(parents=True)

        ct = sitk.ReadImage(str(item["ct"]))
        mask = sitk.ReadImage(str(item["mask"]))
        ok, detail = geometry_matches(ct, mask)
        if not ok:
            raise ValueError(f"{cid}: CT/L3 mask geometry mismatch: {detail}")
        array = sitk.GetArrayFromImage(mask)
        if array.ndim != 3 or not np.any(array > 0):
            raise ValueError(f"{cid}: empty or non-3D L3 mask, shape={array.shape}")

        shutil.copy2(item["ct"], case_out / "ct.nii.gz")
        binary = sitk.GetImageFromArray((array > 0).astype(np.uint8))
        binary.CopyInformation(ct)
        sitk.WriteImage(binary, str(case_out / "l3_mask.nii.gz"), True)

        sources = {}
        for branch, source_root in (("inferior", inferior_root), ("middle", middle_root)):
            source_dir = resolve_case_dir(source_root, cid)
            reference = source_dir / f"{cid}.nii.gz"
            gt = source_dir / f"{cid}_mask.seg.nrrd"
            for path in (reference, gt):
                if not path.is_file():
                    raise FileNotFoundError(f"{cid}: missing {path}")
            shutil.copy2(reference, case_out / "reference" / f"{branch}_ct.nii.gz")
            shutil.copy2(gt, case_out / "gt" / f"{branch}_mask.seg.nrrd")
            sources[f"{branch}_reference_source"] = str(reference)
            sources[f"{branch}_gt_source"] = str(gt)

        z = np.flatnonzero((array > 0).any(axis=(1, 2)))
        record = {
            "case_id": cid, "fold": fold_of[cid], "group_used": item["group"],
            "l3_mask_source_kind": item["mask_kind"],
            "ct_source": str(item["ct"]), "l3_mask_source": str(item["mask"]),
            **sources,
            "shape": "x".join(map(str, ct.GetSize())),
            "spacing_mm": "x".join(f"{v:.6g}" for v in ct.GetSpacing()),
            "l3_foreground_voxels": int((array > 0).sum()),
            "l3_zmin": int(z.min()), "l3_zmax": int(z.max()),
        }
        (case_out / "meta.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest.append(record)
        print(f"[{index}/{len(cases)}] {cid} group={item['group']} mask={item['mask_kind']}", flush=True)

    with (root / "dataset_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)

    groups = {}
    for record in manifest:
        groups[record["group_used"]] = groups.get(record["group_used"], 0) + 1
    print(f"[DONE] {out_root}  cases={len(manifest)}  groups={groups}")


if __name__ == "__main__":
    main()
