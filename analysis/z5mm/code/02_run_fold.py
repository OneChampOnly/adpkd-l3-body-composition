#!/usr/bin/env python3
"""Run fold-matched L3 nnUNet inference for one prepared 5 mm fold."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--predict-bin", default="/opt/conda/envs/nnunetv2/bin/nnUNetv2_predict")
    args = parser.parse_args()

    registry = json.loads((PARENT / "config/model_registry.json").read_text(encoding="utf-8"))
    model = registry["folds"][str(args.fold)]["l3"]
    input_dir = ROOT / f"data/nnunet_inputs/fold_{args.fold}"
    output_dir = ROOT / f"results/fold_{args.fold}/l3_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(input_dir.glob("*_0000.nii.gz"))
    if not inputs:
        raise RuntimeError(f"No inputs in {input_dir}")

    os.environ["nnUNet_raw"] = str(PARENT / "work/nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(PARENT / "work/nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(PARENT / "work/nnUNet_results")
    command = [
        args.predict_bin,
        "-d", model["dataset"],
        "-c", model["config"],
        "-f", str(args.fold),
        "-tr", model["trainer"],
        "-p", model["plans"],
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-chk", Path(model["checkpoint"]).name,
        "--disable_tta",
    ]
    print("[COMMAND]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    expected = {path.name.replace("_0000.nii.gz", ".nii.gz") for path in inputs}
    actual = {path.name for path in output_dir.glob("*.nii.gz")}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Fold {args.fold}: missing {len(missing)} predictions: {missing[:5]}")
    print(f"[OK] fold={args.fold} predictions={len(expected)}", flush=True)


if __name__ == "__main__":
    main()
