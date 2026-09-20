#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
E2E_ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
OFFSETS = (-2, -1, 0, 1, 2)
OFFSET_TAG = {-2: "m2", -1: "m1", 0: "z0", 1: "p1", 2: "p2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_manifest = read_csv(E2E_ROOT / "manifests/manifest_300_all_folds.csv")
    evaluated = read_csv(E2E_ROOT / "results/evaluation/per_case_end_to_end.csv")
    registry = json.loads((E2E_ROOT / "config/model_registry.json").read_text(encoding="utf-8"))

    gt_indices: dict[tuple[str, str], int] = {}
    for row in evaluated:
        key = (row["case_id"], row["branch"])
        if key in gt_indices:
            raise ValueError(f"Duplicate GT index row: {key}")
        gt_indices[key] = int(row["gt_slice_index"])

    expected = {(row["case_id"], branch) for row in source_manifest for branch in ("inferior", "middle")}
    if set(gt_indices) != expected:
        raise ValueError(
            f"GT index coverage mismatch: expected={len(expected)} actual={len(gt_indices)} "
            f"missing={sorted(expected - set(gt_indices))[:10]} extra={sorted(set(gt_indices) - expected)[:10]}"
        )

    rows: list[dict[str, object]] = []
    spacing_counts: dict[str, int] = {}
    for item in source_manifest:
        case_id = item["case_id"]
        fold = int(item["fold"])
        ct_path = Path(item["ct_path"])
        if not ct_path.is_file():
            raise FileNotFoundError(ct_path)
        image = nib.load(str(ct_path))
        if len(image.shape) != 3:
            raise ValueError(f"{case_id}: expected 3D CT, got {image.shape}")
        zdim = int(image.shape[2])
        slice_spacing_mm = float(np.linalg.norm(np.asarray(image.affine[:3, 2], dtype=float)))
        spacing_key = f"{slice_spacing_mm:.4f}"
        spacing_counts[spacing_key] = spacing_counts.get(spacing_key, 0) + 1

        vector = np.asarray(image.affine[:3, 2], dtype=float)
        dominant_axis = int(np.argmax(np.abs(vector)))
        if dominant_axis != 2 or vector[2] <= 0:
            raise ValueError(f"{case_id}: unexpected slice direction vector {vector.tolist()}")

        for branch in ("inferior", "middle"):
            gt_index = gt_indices[(case_id, branch)]
            model = registry["folds"][str(fold)][f"sma_{branch}"]
            checkpoint = Path(model["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for offset in OFFSETS:
                target_index = gt_index + offset
                if not 0 <= target_index < zdim:
                    raise ValueError(
                        f"{case_id} {branch}: GT={gt_index}, offset={offset}, target={target_index}, zdim={zdim}"
                    )
                tag = OFFSET_TAG[offset]
                case_key = f"{case_id}_{tag}"
                input_path = ROOT / f"work/fold_{fold}/{branch}/imagesTs/{case_key}_0000.nii.gz"
                pred_path = ROOT / f"predictions/fold_{fold}/{branch}/{case_key}.nii.gz"
                rows.append(
                    {
                        "fold": fold,
                        "case_id": case_id,
                        "branch": branch,
                        "offset_slice": offset,
                        "offset_tag": tag,
                        "gt_slice_index": gt_index,
                        "target_slice_index": target_index,
                        "zdim": zdim,
                        "slice_spacing_mm": f"{slice_spacing_mm:.6f}",
                        "offset_distance_mm": f"{offset * slice_spacing_mm:.6f}",
                        "ct_path": str(ct_path),
                        "gt_mask_path": item[f"{branch}_gt_path"],
                        "reference_ct_path": item[f"{branch}_reference_ct_path"],
                        "dataset": model["dataset"],
                        "trainer": model["trainer"],
                        "plans": model["plans"],
                        "configuration": model["config"],
                        "checkpoint": str(checkpoint),
                        "input_slice_path": str(input_path),
                        "prediction_path": str(pred_path),
                    }
                )

    if len(rows) != 3000:
        raise ValueError(f"Expected 3000 offset rows, got {len(rows)}")
    write_csv(ROOT / "manifests/gt_offset_manifest_3000.csv", rows)
    for fold in range(5):
        selected = [row for row in rows if int(row["fold"]) == fold]
        if len(selected) != 600:
            raise ValueError(f"fold {fold}: expected 600 rows, got {len(selected)}")
        write_csv(ROOT / f"manifests/gt_offset_manifest_fold_{fold}.csv", selected)

    provenance = {
        "purpose": "GT-anchored slice-offset robustness experiment",
        "cases": len(source_manifest),
        "branches": ["inferior", "middle"],
        "offsets": list(OFFSETS),
        "rows": len(rows),
        "gt_index_source": str(E2E_ROOT / "results/evaluation/per_case_end_to_end.csv"),
        "case_manifest_source": str(E2E_ROOT / "manifests/manifest_300_all_folds.csv"),
        "model_registry_source": str(E2E_ROOT / "config/model_registry.json"),
        "explicitly_ignored": ["pred_slice_index", "yolo_conf", "inferior_review selected_z_idx"],
        "direction": "All CTs validated as increasing voxel z toward positive world Z; negative offsets are inferior and positive offsets are superior.",
        "slice_spacing_case_counts": dict(sorted(spacing_counts.items(), key=lambda pair: float(pair[0]))),
    }
    (ROOT / "manifests/provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
