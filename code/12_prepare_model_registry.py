#!/usr/bin/env python3
"""Stage fold-matched model links and write the end-to-end model registry.

Unlike the 0820 registry, the detectors are fold-matched too: evaluation fold k is scored
with the detector that never saw fold k during training.
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[1]))
NNUNET_RESULTS = ROOT / "workspace/nnunet/nnUNet_results"
YOLO_RUNS = ROOT / "yolo/runs"

SEGMENTERS = {
    "l3": {"dataset": "Dataset920_L3Vertebra0917", "config": "3d_fullres"},
    "sma_inferior": {"dataset": "Dataset917_InferiorSMA300Reann0917", "config": "2d"},
    "sma_middle": {"dataset": "Dataset918_MiddleSMA300Reann0917", "config": "2d"},
}
TRAINER = "nnUNetTrainer_500epochs"
PLANS = "nnUNetPlans"
DETECTORS = {"yolo_inferior": "inferior", "yolo_middle": "middle"}
YOLO_IMGSZ = 512
YOLO_CONF = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def segmenter_done(checkpoint: Path) -> bool:
    """nnU-Net writes checkpoint_best.pth throughout training; only final marks a finished run."""
    return checkpoint.is_file() and (checkpoint.parent / "checkpoint_final.pth").is_file()


def detector_done(weights: Path, epochs: int) -> bool:
    """Ultralytics also rewrites best.pt each time validation improves."""
    results = weights.parent.parent / "results.csv"
    if not (weights.is_file() and results.is_file()):
        return False
    rows = [line for line in results.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        return False
    return int(float(rows[-1].split(",")[0])) >= epochs


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace a real file: {destination}")
    destination.symlink_to(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-complete", action="store_true",
                    help="fail instead of reporting when a model is still training")
    ap.add_argument("--yolo-epochs", type=int, default=100,
                    help="epoch count a finished detector run must reach")
    args = ap.parse_args()

    registry = {
        "schema_version": 4,
        "fold_policy": ("L3 segmentation, both reference-section detectors, and both body-composition "
                        "segmenters all use the checkpoint of the case's own fold, so no model saw the "
                        "case it is evaluated on."),
        "splits": str(ROOT / "splits/splits_final.json"),
        "folds": {},
    }
    pending = []

    for fold in range(5):
        entry = {}
        for name, spec in SEGMENTERS.items():
            source = (NNUNET_RESULTS / spec["dataset"] /
                      f"{TRAINER}__{PLANS}__{spec['config']}" / f"fold_{fold}" / "checkpoint_best.pth")
            destination = ROOT / "models" / f"fold_{fold}" / name / "checkpoint_best.pth"
            if segmenter_done(source):
                link(source, destination)
            else:
                state = "still training" if source.is_file() else "not started"
                pending.append(f"fold {fold} {name}: {state} ({source})")
                continue
            entry[name] = {
                "dataset": spec["dataset"], "config": spec["config"],
                "trainer": TRAINER, "plans": PLANS, "fold": fold,
                "checkpoint": str(destination), "source": str(source),
            }

        for name, branch in DETECTORS.items():
            source = YOLO_RUNS / branch / f"fold_{fold}" / "weights" / "best.pt"
            destination = ROOT / "models" / f"fold_{fold}" / name / "best.pt"
            if detector_done(source, args.yolo_epochs):
                link(source, destination)
            else:
                state = "still training" if source.is_file() else "not started"
                pending.append(f"fold {fold} {name}: {state} ({source})")
                continue
            entry[name] = {
                "class": f"l3_{branch}", "fold": fold,
                "weights": str(destination), "source": str(source),
                "sha256": sha256(source), "imgsz": YOLO_IMGSZ, "conf": YOLO_CONF,
            }
        registry["folds"][str(fold)] = entry

    # nnUNetv2_predict resolves plans.json and dataset.json through nnUNet_results.
    work_results = ROOT / "work/nnUNet_results"
    for spec in SEGMENTERS.values():
        source = NNUNET_RESULTS / spec["dataset"]
        if source.is_dir():
            link(source, work_results / spec["dataset"])
    for name in ("nnUNet_raw", "nnUNet_preprocessed"):
        (ROOT / "work" / name).mkdir(parents=True, exist_ok=True)

    complete = not pending
    registry["complete"] = complete
    out_path = ROOT / "config/model_registry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2) + "\n")

    for fold in range(5):
        have = sorted(registry["folds"][str(fold)])
        print(f"[fold {fold}] {len(have)}/5  {have}")
    if pending:
        print(f"\n[WAITING] {len(pending)} checkpoint(s) not written yet:")
        for item in pending:
            print("   ", item)
        if args.require_complete:
            raise SystemExit(1)
    else:
        print("\n[OK] all 25 fold-matched models are staged")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
