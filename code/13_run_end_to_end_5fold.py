#!/usr/bin/env python3
"""Drive the end-to-end pipeline one fold at a time with fold-matched models throughout.

Every model for fold k - L3 segmentation, the branch detector, and the branch body-composition
segmenter - comes from registry['folds'][k], so the detector no longer sees its own test cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Where nnU-Net looks up <dataset>/<trainer>__<plans>__<config>/fold_k/. With the released
# weights this is the download directory; pipeline_core.py reads the same variable.
NNUNET_RESULTS = Path(os.environ.get('NNUNET_RESULTS', ROOT / 'work/nnUNet_results'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--branches', choices=('inferior', 'middle', 'both'), default='both')
    parser.add_argument('--case-limit', type=int, default=0)
    parser.add_argument('--yolo-device', default='0')
    parser.add_argument('--keep-work', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def stage(rows: list[dict[str, str]], fold: int) -> Path:
    destination = ROOT / f'work/fold_inputs/fold_{fold}'
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob('*.nii.gz'):
        path.unlink()
    for row in rows:
        (destination / f"{row['case_id']}.nii.gz").symlink_to(Path(row['ct_path']).resolve())
    return destination


def require_file(value: str | None, label: str) -> str:
    if not value:
        raise RuntimeError(f'{label} is not configured')
    if not Path(value).is_file():
        raise FileNotFoundError(f'{label} is missing: {value}')
    return value


def main() -> None:
    args = parse_args()
    registry = json.loads((ROOT / 'config/model_registry.json').read_text(encoding='utf-8'))
    requested = ('inferior', 'middle') if args.branches == 'both' else (args.branches,)

    for fold in args.folds:
        if fold not in range(5):
            raise ValueError(f'Invalid fold {fold}')
        rows = read_rows(ROOT / f'manifests/manifest_fold_{fold}.csv')
        selected = rows[:args.case_limit] if args.case_limit else rows
        model = registry['folds'][str(fold)]
        for name in ('l3', *(f'sma_{branch}' for branch in requested)):
            if name not in model:
                raise RuntimeError(f'fold {fold}: {name} is not in the registry yet')
            require_file(model[name]['checkpoint'], f'fold {fold} {name} checkpoint')
        for branch in requested:
            name = f'yolo_{branch}'
            if name not in model:
                raise RuntimeError(f'fold {fold}: {name} is not in the registry yet')
            require_file(model[name]['weights'], f'fold {fold} {name} weight')

        l3 = model['l3']
        input_dir = stage(selected, fold)
        output_dir = ROOT / f'results/fold_{fold}/pipeline_output'
        command = [
            sys.executable, str(ROOT / 'code/pipeline_core.py'), '--resume-skip-complete',
            '--input-dir', str(input_dir), '--pattern', '*.nii.gz', '--out-root', str(output_dir),
            '--slice-mode', args.branches,
            '--nnunet-raw', str(ROOT / 'work/nnUNet_raw'),
            '--nnunet-preprocessed', str(ROOT / 'work/nnUNet_preprocessed'),
            '--nnunet-results', str(NNUNET_RESULTS),
            '--l3-ckpt-path', l3['checkpoint'], '--l3-dataset-name', l3['dataset'],
            '--l3-config', l3['config'], '--l3-fold', str(fold),
            '--l3-trainer', l3['trainer'], '--l3-plans', l3['plans'],
            '--yolo-device', args.yolo_device,
            '--sma-fold', str(fold), '--sma-config', '2d',
            '--sma-trainer', 'nnUNetTrainer_500epochs', '--sma-plans', 'nnUNetPlans',
        ]
        for branch in requested:
            yolo = model[f'yolo_{branch}']
            sma = model[f'sma_{branch}']
            command.extend([
                f'--yolo-{branch}-weights', yolo['weights'], '--yolo-conf', str(yolo['conf']),
                f'--sma-{branch}-ckpt-path', sma['checkpoint'],
                f'--sma-{branch}-dataset-name', sma['dataset'],
            ])
        if args.keep_work:
            command.append('--keep-work')
        detectors = ', '.join(f'{branch}=fold {model[f"yolo_{branch}"]["fold"]}' for branch in requested)
        print(f'[FOLD {fold}] cases={len(selected)} branches={args.branches}; '
              f'segmentation fold={fold}, detection {detectors}', flush=True)
        print('[CMD]', ' '.join(shlex.quote(value) for value in command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
