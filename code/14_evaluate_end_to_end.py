#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'sma': 1, 'sfat': 2, 'vfat': 3, 'imat': 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--branches', choices=('inferior', 'middle', 'both'), default='both')
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denom


def reference_index(full_ct: sitk.Image, reference_ct: sitk.Image) -> int:
    """Recover the source z-index by matching reference pixels to the full CT."""
    full = sitk.GetArrayViewFromImage(full_ct)
    reference = sitk.GetArrayFromImage(reference_ct)
    if reference.shape[0] != 1:
        raise ValueError(f'Reference CT must contain one slice, got {reference.shape}')
    if full.shape[1:] != reference.shape[1:]:
        raise ValueError(f'Reference/full CT in-plane mismatch: full={full.shape}, reference={reference.shape}')

    ref2d = reference[0].astype(np.float32, copy=False)
    step_y = max(1, ref2d.shape[0] // 64)
    step_x = max(1, ref2d.shape[1] // 64)
    sampled_full = full[:, ::step_y, ::step_x].astype(np.float32, copy=False)
    sampled_ref = ref2d[::step_y, ::step_x]
    sampled_mae = np.mean(np.abs(sampled_full - sampled_ref), axis=(1, 2))

    candidate_count = min(5, full.shape[0])
    candidates = np.argpartition(sampled_mae, candidate_count - 1)[:candidate_count]
    full_mae = {
        int(z): float(np.mean(np.abs(full[int(z)].astype(np.float32, copy=False) - ref2d)))
        for z in candidates
    }
    return min(full_mae, key=full_mae.get)


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

def segmentation_metrics(gt_path: Path, pred_path: Path) -> dict[str, float]:
    gt_img, pred_img = sitk.ReadImage(str(gt_path)), sitk.ReadImage(str(pred_path))
    if gt_img.GetDimension() != 3 or pred_img.GetDimension() != 3:
        raise ValueError(f'Expected 3D single-slice masks: gt={gt_img.GetDimension()}D pred={pred_img.GetDimension()}D')

    # Segment Editor GT exports may be cropped. Expand the GT onto the full
    # prediction grid so prediction pixels outside the GT crop remain false
    # positives instead of being discarded.
    identity = sitk.Transform(3, sitk.sitkIdentity)
    gt_on_pred = sitk.Resample(
        align_single_slice_gt(gt_img, pred_img), pred_img, identity,
        sitk.sitkNearestNeighbor, 0, gt_img.GetPixelID()
    )
    gt = sitk.GetArrayFromImage(gt_on_pred)
    pred = sitk.GetArrayFromImage(pred_img)
    pixel_area_cm2 = float(pred_img.GetSpacing()[0] * pred_img.GetSpacing()[1]) / 100.0
    result: dict[str, float] = {}
    for tissue, label in LABELS.items():
        gt_mask, pred_mask = gt == label, pred == label
        gt_area = float(gt_mask.sum()) * pixel_area_cm2
        pred_area = float(pred_mask.sum()) * pixel_area_cm2
        result[f'dsc_{tissue}'] = dice(gt_mask, pred_mask)
        result[f'gt_{tissue}_cm2'] = gt_area
        result[f'pred_{tissue}_cm2'] = pred_area
        result[f'abs_error_{tissue}_cm2'] = abs(pred_area - gt_area)
    return result


def main() -> None:
    args = parse_args()
    branches = ('inferior', 'middle') if args.branches == 'both' else (args.branches,)
    manifest = read_csv(ROOT / 'manifests/manifest_300_all_folds.csv')
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    l3_by_case: dict[str, float] = {}

    for item in manifest:
        case_id, fold = item['case_id'], int(item['fold'])
        case_output = ROOT / f'results/fold_{fold}/pipeline_output/{case_id}'
        l3_pred = case_output / f'image/{case_id}_l3_mask.nii.gz'
        if l3_pred.is_file():
            gt = sitk.GetArrayFromImage(sitk.ReadImage(item['l3_gt_path'])) > 0
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(l3_pred))) > 0
            if gt.shape != pred.shape:
                raise ValueError(f'{case_id}: L3 shape mismatch gt={gt.shape}, pred={pred.shape}')
            l3_by_case[case_id] = dice(gt, pred)
        else:
            missing.append({'fold': fold, 'case_id': case_id, 'stage': 'l3', 'expected': str(l3_pred)})

        full_ct = sitk.ReadImage(item['ct_path'])
        for branch in branches:
            metrics_csv = case_output / f'csv/{case_id}_{branch}_metrics.csv'
            pred_seg = case_output / f'image/{case_id}_{branch}_sma_seg.nii.gz'
            if not metrics_csv.is_file() or not pred_seg.is_file():
                missing.append({'fold': fold, 'case_id': case_id, 'stage': branch, 'expected': str(metrics_csv)})
                continue
            pipeline = read_csv(metrics_csv)[0]
            ref_ct = sitk.ReadImage(item[f'{branch}_reference_ct_path'])
            gt_index = reference_index(full_ct, ref_ct)
            pred_index = int(float(pipeline['slice_index']))
            row: dict[str, object] = {
                'fold': fold, 'case_id': case_id, 'branch': branch,
                'l3_dsc': l3_by_case.get(case_id, float('nan')),
                'gt_slice_index': gt_index, 'pred_slice_index': pred_index,
                'slice_error': pred_index - gt_index, 'abs_slice_error': abs(pred_index - gt_index),
                'exact_slice': int(pred_index == gt_index),
                'within_1_slice': int(abs(pred_index - gt_index) <= 1),
                'within_2_slices': int(abs(pred_index - gt_index) <= 2),
                'yolo_conf': float(pipeline['yolo_best_conf']),
            }
            row.update(segmentation_metrics(Path(item[f'{branch}_gt_path']), pred_seg))
            rows.append(row)

    output = ROOT / 'results/evaluation'
    write_csv(output / 'per_case_end_to_end.csv', rows)
    write_csv(output / 'missing_cases.csv', missing)
    summary: dict[str, object] = {
        'manifest_cases': len(manifest), 'branches': list(branches),
        'evaluated_branch_rows': len(rows), 'missing_outputs': len(missing),
        'l3_evaluated_cases': len(l3_by_case),
        'l3_mean_dsc': statistics.fmean(l3_by_case.values()) if l3_by_case else None,
        'by_branch': {},
    }
    for branch in branches:
        selected = [row for row in rows if row['branch'] == branch]
        branch_summary: dict[str, object] = {
            'n': len(selected),
            'exact_slice_rate': statistics.fmean(float(row['exact_slice']) for row in selected) if selected else None,
            'within_1_slice_rate': statistics.fmean(float(row['within_1_slice']) for row in selected) if selected else None,
            'within_2_slices_rate': statistics.fmean(float(row['within_2_slices']) for row in selected) if selected else None,
            'slice_mae': statistics.fmean(float(row['abs_slice_error']) for row in selected) if selected else None,
            'tissues': {},
        }
        for tissue in LABELS:
            branch_summary['tissues'][tissue] = {
                'mean_dsc': statistics.fmean(float(row[f'dsc_{tissue}']) for row in selected) if selected else None,
                'area_mae_cm2': statistics.fmean(float(row[f'abs_error_{tissue}_cm2']) for row in selected) if selected else None,
            }
        summary['by_branch'][branch] = branch_summary
    (output / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
