#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
LABELS = {"sma": 1, "sfat": 2, "vfat": 3, "imat": 4}
OFFSETS = (-2, -1, 0, 1, 2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denominator


def align_single_slice_gt(gt_img: sitk.Image, pred_img: sitk.Image) -> sitk.Image:
    """Pin a single-slice GT mask onto the prediction's slice frame.

    The two share an export convention only by accident. The pipeline writes each
    extracted slice with the parent volume's affine, so its origin z is slice 0's world
    position; the masks re-exported during reannotation carry the slice's own world z.
    Resampling those against each other in world space yields an empty GT, which is what
    zeroed the 56 reannotated inferior cases and the 1 reannotated middle case.

    Both images are single-slice and share in-plane direction and spacing, so adopting the
    prediction's slice axis leaves the identity resample doing only the in-plane crop/pad
    that Segment Editor exports require.
    """
    if gt_img.GetSize()[2] != 1 or pred_img.GetSize()[2] != 1:
        raise ValueError(f'Expected single-slice masks: gt={gt_img.GetSize()} pred={pred_img.GetSize()}')
    gt_direction, pred_direction = gt_img.GetDirection(), pred_img.GetDirection()
    if not np.allclose(gt_direction[0:2], pred_direction[0:2], atol=1e-5) or \
       not np.allclose(gt_direction[3:5], pred_direction[3:5], atol=1e-5):
        raise ValueError(f'In-plane direction mismatch: gt={gt_direction} pred={pred_direction}')
    if not np.allclose(gt_img.GetSpacing()[:2], pred_img.GetSpacing()[:2], atol=1e-5):
        raise ValueError(f'In-plane spacing mismatch: gt={gt_img.GetSpacing()} pred={pred_img.GetSpacing()}')

    direction = list(gt_direction)
    direction[2], direction[5], direction[8] = pred_direction[2], pred_direction[5], pred_direction[8]
    origin = list(gt_img.GetOrigin()); origin[2] = pred_img.GetOrigin()[2]
    spacing = list(gt_img.GetSpacing()); spacing[2] = pred_img.GetSpacing()[2]

    aligned = sitk.Image(gt_img)
    aligned.SetDirection(direction)
    aligned.SetOrigin(origin)
    aligned.SetSpacing(spacing)
    return aligned

def tissue_metrics(gt_path: Path, pred_path: Path) -> dict[str, float]:
    gt_image = sitk.ReadImage(str(gt_path))
    pred_image = sitk.ReadImage(str(pred_path))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    gt_on_pred = sitk.Resample(
        align_single_slice_gt(gt_image, pred_image),
        pred_image,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        gt_image.GetPixelID(),
    )
    gt = sitk.GetArrayFromImage(gt_on_pred)
    pred = sitk.GetArrayFromImage(pred_image)
    if gt.shape != pred.shape:
        raise ValueError(f"Resampled GT/pred shape mismatch: {gt.shape} != {pred.shape}")
    pixel_area_cm2 = float(pred_image.GetSpacing()[0] * pred_image.GetSpacing()[1]) / 100.0
    result: dict[str, float] = {}
    for tissue, label in LABELS.items():
        gt_mask = gt == label
        pred_mask = pred == label
        gt_area = float(gt_mask.sum()) * pixel_area_cm2
        pred_area = float(pred_mask.sum()) * pixel_area_cm2
        bias = pred_area - gt_area
        result[f"central_gt_dsc_{tissue}"] = dice(gt_mask, pred_mask)
        result[f"gt_{tissue}_cm2"] = gt_area
        result[f"pred_{tissue}_cm2"] = pred_area
        result[f"bias_{tissue}_cm2"] = bias
        result[f"abs_error_{tissue}_cm2"] = abs(bias)
        result[f"percent_error_{tissue}"] = 100.0 * bias / gt_area if gt_area > 0 else float("nan")
    return result


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 2 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def main() -> None:
    manifest = read_csv(ROOT / "manifests/gt_offset_manifest_3000.csv")
    if len(manifest) != 3000:
        raise ValueError(f"Expected 3000 manifest rows, got {len(manifest)}")
    missing = [row["prediction_path"] for row in manifest if not Path(row["prediction_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} predictions; examples={missing[:10]}")

    rows: list[dict[str, object]] = []
    for index, item in enumerate(manifest, start=1):
        row: dict[str, object] = {
            "fold": int(item["fold"]),
            "case_id": item["case_id"],
            "branch": item["branch"],
            "offset_slice": int(item["offset_slice"]),
            "offset_direction": (
                "inferior" if int(item["offset_slice"]) < 0
                else "superior" if int(item["offset_slice"]) > 0
                else "GT"
            ),
            "gt_slice_index": int(item["gt_slice_index"]),
            "target_slice_index": int(item["target_slice_index"]),
            "slice_spacing_mm": float(item["slice_spacing_mm"]),
            "offset_distance_mm": float(item["offset_distance_mm"]),
            "prediction_path": item["prediction_path"],
            "gt_mask_path": item["gt_mask_path"],
        }
        row.update(tissue_metrics(Path(item["gt_mask_path"]), Path(item["prediction_path"])))
        rows.append(row)
        if index % 100 == 0 or index == len(manifest):
            print(f"[EVALUATE] {index}/{len(manifest)}", flush=True)

    baseline: dict[tuple[str, str], dict[str, object]] = {
        (str(row["case_id"]), str(row["branch"])): row
        for row in rows
        if int(row["offset_slice"]) == 0
    }
    if len(baseline) != 600:
        raise ValueError(f"Expected 600 baseline rows, got {len(baseline)}")
    for row in rows:
        base = baseline[(str(row["case_id"]), str(row["branch"]))]
        for tissue in LABELS:
            delta = float(row[f"pred_{tissue}_cm2"]) - float(base[f"pred_{tissue}_cm2"])
            row[f"delta_vs_offset0_{tissue}_cm2"] = delta
            base_area = float(base[f"pred_{tissue}_cm2"])
            row[f"percent_delta_vs_offset0_{tissue}"] = (
                100.0 * delta / base_area if base_area > 0 else float("nan")
            )

    output = ROOT / "statistics"
    write_csv(output / "per_case_gt_offset_metrics.csv", rows)

    summaries: list[dict[str, object]] = []
    nested: dict[str, object] = {}
    for branch in ("inferior", "middle"):
        nested[branch] = {}
        for offset in OFFSETS:
            selected = [
                row for row in rows
                if row["branch"] == branch and int(row["offset_slice"]) == offset
            ]
            if len(selected) != 300:
                raise ValueError(f"{branch} offset {offset}: expected 300 rows, got {len(selected)}")
            nested[branch][str(offset)] = {}
            for tissue in LABELS:
                gt = [float(row[f"gt_{tissue}_cm2"]) for row in selected]
                pred = [float(row[f"pred_{tissue}_cm2"]) for row in selected]
                bias = [float(row[f"bias_{tissue}_cm2"]) for row in selected]
                absolute = [abs(value) for value in bias]
                overlap = [float(row[f"central_gt_dsc_{tissue}"]) for row in selected]
                delta0 = [float(row[f"delta_vs_offset0_{tissue}_cm2"]) for row in selected]
                pct_error = finite([float(row[f"percent_error_{tissue}"]) for row in selected])
                pct_delta0 = finite([float(row[f"percent_delta_vs_offset0_{tissue}"]) for row in selected])
                values: dict[str, object] = {
                    "branch": branch,
                    "offset_slice": offset,
                    "direction": "inferior" if offset < 0 else "superior" if offset > 0 else "GT",
                    "tissue": tissue,
                    "n": len(selected),
                    "gt_area_mean_cm2": statistics.fmean(gt),
                    "gt_area_sd_cm2": sample_sd(gt),
                    "pred_area_mean_cm2": statistics.fmean(pred),
                    "pred_area_sd_cm2": sample_sd(pred),
                    "bias_mean_cm2": statistics.fmean(bias),
                    "bias_sd_cm2": sample_sd(bias),
                    "mae_cm2": statistics.fmean(absolute),
                    "rmse_cm2": math.sqrt(statistics.fmean(value * value for value in bias)),
                    "mean_percent_error": statistics.fmean(pct_error) if pct_error else float("nan"),
                    "pearson_r_gt_vs_pred": pearson(gt, pred),
                    "central_gt_overlap_dsc_mean": statistics.fmean(overlap),
                    "central_gt_overlap_dsc_sd": sample_sd(overlap),
                    "delta_vs_offset0_mean_cm2": statistics.fmean(delta0),
                    "delta_vs_offset0_sd_cm2": sample_sd(delta0),
                    "abs_delta_vs_offset0_mean_cm2": statistics.fmean(abs(value) for value in delta0),
                    "percent_delta_vs_offset0_mean": statistics.fmean(pct_delta0) if pct_delta0 else float("nan"),
                }
                summaries.append(values)
                nested[branch][str(offset)][tissue] = {
                    key: value for key, value in values.items()
                    if key not in {"branch", "offset_slice", "direction", "tissue"}
                }

    write_csv(output / "summary_by_branch_offset_tissue.csv", summaries)
    result = {
        "design": {
            "cases": 300,
            "branches": ["inferior", "middle"],
            "offsets": list(OFFSETS),
            "predictions": len(rows),
            "anchor": "manual/reference GT z-index only",
            "yolo_or_predicted_z_used": False,
            "important_note": (
                "GT tissue masks exist only at offset 0. DSC at nonzero offsets is central-GT overlap, "
                "not independently annotated adjacent-slice segmentation accuracy."
            ),
            "standard_deviation": "sample SD (n-1)",
        },
        "by_branch_offset_tissue": nested,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"[DONE] rows={len(rows)} summaries={len(summaries)} output={output}", flush=True)


if __name__ == "__main__":
    main()
