#!/usr/bin/env python3
"""Resample the preserved 300-case cohort to 5 mm and map reference z-indices."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
SOURCE_MANIFEST = PARENT / "manifests/manifest_300_all_folds.csv"
TARGET_Z_MM = 3.0
Z_TOLERANCE_MM = 0.05


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_index(full_ct: sitk.Image, reference_ct: sitk.Image) -> tuple[int, float]:
    """Recover source z-index by pixel matching, returning index and full-image MAE."""
    full = sitk.GetArrayViewFromImage(full_ct)
    reference = sitk.GetArrayFromImage(reference_ct)
    if reference.shape[0] != 1:
        raise ValueError(f"Reference CT must contain one slice, got {reference.shape}")
    if full.shape[1:] != reference.shape[1:]:
        raise ValueError(f"In-plane mismatch: full={full.shape}, reference={reference.shape}")

    ref2d = reference[0].astype("float32", copy=False)
    step_y = max(1, ref2d.shape[0] // 64)
    step_x = max(1, ref2d.shape[1] // 64)
    sampled_full = full[:, ::step_y, ::step_x].astype("float32", copy=False)
    sampled_ref = ref2d[::step_y, ::step_x]
    import numpy as np

    sampled_mae = np.mean(np.abs(sampled_full - sampled_ref), axis=(1, 2))
    candidate_count = min(5, full.shape[0])
    candidates = np.argpartition(sampled_mae, candidate_count - 1)[:candidate_count]
    errors = {
        int(z): float(np.mean(np.abs(full[int(z)].astype("float32", copy=False) - ref2d)))
        for z in candidates
    }
    best = min(errors, key=errors.get)
    return best, errors[best]


def resample_z5(source: sitk.Image) -> sitk.Image:
    spacing = source.GetSpacing()
    size = source.GetSize()
    target_spacing = (spacing[0], spacing[1], TARGET_Z_MM)
    target_size = [
        max(1, int(round(size[i] * spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(source.GetDirection())
    resampler.SetOutputOrigin(source.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetInterpolator(sitk.sitkLanczosWindowedSinc)
    return resampler.Execute(source)


def map_index(source: sitk.Image, target: sitk.Image, source_z: int) -> tuple[int, float]:
    """Map a source slice center to target continuous index using physical geometry."""
    center = (source.GetSize()[0] // 2, source.GetSize()[1] // 2, int(source_z))
    point = source.TransformIndexToPhysicalPoint(center)
    continuous = target.TransformPhysicalPointToContinuousIndex(point)
    target_z = int(round(continuous[2]))
    target_z = min(max(target_z, 0), target.GetSize()[2] - 1)
    residual_mm = abs(continuous[2] - target_z) * TARGET_Z_MM
    return target_z, residual_mm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_rows = read_rows(SOURCE_MANIFEST)
    if args.limit:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise RuntimeError("No manifest rows")

    resampled_root = ROOT / "data/resampled_ct_z3mm"
    input_root = ROOT / "data/nnunet_inputs"
    manifest_rows: list[dict[str, object]] = []

    for number, row in enumerate(source_rows, start=1):
        case_id = row["case_id"]
        fold = int(row["fold"])
        source_path = Path(row["ct_path"])
        output_path = resampled_root / f"{case_id}.nii.gz"
        fold_input = input_root / f"fold_{fold}/{case_id}_0000.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold_input.parent.mkdir(parents=True, exist_ok=True)

        source = sitk.ReadImage(str(source_path))
        original_spacing = source.GetSpacing()
        needs_resample = abs(original_spacing[2] - TARGET_Z_MM) > Z_TOLERANCE_MM
        if not output_path.is_file():
            target = resample_z5(source) if needs_resample else source
            sitk.WriteImage(target, str(output_path), True)
        target = sitk.ReadImage(str(output_path))
        if abs(target.GetSpacing()[2] - TARGET_Z_MM) > 1e-4:
            raise RuntimeError(f"{case_id}: final z spacing is {target.GetSpacing()[2]}")

        if fold_input.is_symlink() or fold_input.exists():
            if fold_input.resolve() != output_path.resolve():
                fold_input.unlink()
        if not fold_input.exists():
            fold_input.symlink_to(output_path)

        mapped: dict[str, object] = {}
        for branch in ("inferior", "middle"):
            reference = sitk.ReadImage(row[f"{branch}_reference_ct_path"])
            original_idx, match_mae = reference_index(source, reference)
            z5_idx, residual_mm = map_index(source, target, original_idx)
            mapped[f"original_{branch}_gt_idx"] = original_idx
            mapped[f"{branch}_gt_idx_z3mm"] = z5_idx
            mapped[f"{branch}_reference_match_mae_hu"] = f"{match_mae:.8g}"
            mapped[f"{branch}_mapping_rounding_residual_mm"] = f"{residual_mm:.8g}"

        manifest_rows.append(
            {
                "fold": fold,
                "case_id": case_id,
                "original_ct_path": source_path,
                "ct_z3mm_path": output_path,
                "nnunet_input_path": fold_input,
                "was_resampled": int(needs_resample),
                "original_spacing_x_mm": f"{original_spacing[0]:.8g}",
                "original_spacing_y_mm": f"{original_spacing[1]:.8g}",
                "original_spacing_z_mm": f"{original_spacing[2]:.8g}",
                "final_spacing_z_mm": f"{target.GetSpacing()[2]:.8g}",
                "original_size_z": source.GetSize()[2],
                "final_size_z": target.GetSize()[2],
                **mapped,
            }
        )
        if number == 1 or number % 20 == 0:
            print(f"[PREPARE] {number}/{len(source_rows)}", flush=True)

    if len({str(row["case_id"]) for row in manifest_rows}) != len(manifest_rows):
        raise RuntimeError("Duplicate case IDs in output manifest")
    write_rows(ROOT / "manifests/z3mm_manifest_300.csv", manifest_rows)
    summary = [
        {"metric": "total_cases", "value": len(manifest_rows)},
        {"metric": "kept_original_z3mm", "value": sum(not int(row["was_resampled"]) for row in manifest_rows)},
        {"metric": "resampled_to_z3mm", "value": sum(int(row["was_resampled"]) for row in manifest_rows)},
        {"metric": "target_z_spacing_mm", "value": TARGET_Z_MM},
    ]
    write_rows(ROOT / "manifests/resampling_summary.csv", summary)
    print(f"[OK] prepared {len(manifest_rows)} cases", flush=True)


if __name__ == "__main__":
    main()
