#!/usr/bin/env python3
"""Fetch the released weights, verify them, and write a matching model registry.

The weights are not in this repository: 4.66 GiB across 25 checkpoints. They are
distributed in nnU-Net's own nnUNet_results layout, so that setting one environment
variable is enough for nnUNetv2_predict to find the models, their plans.json and their
dataset.json.

    python tools/download_weights.py --source <URL_OR_HF_REPO> --dest weights/
    python tools/download_weights.py --dest weights/ --verify
    python tools/download_weights.py --dest weights/ --write-registry config/model_registry.json

The last command also prints the environment variables to export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Filled in on publication; see docs/WEIGHTS.md.
DEFAULT_SOURCE = ""

TRAINER, PLANS = "nnUNetTrainer_500epochs", "nnUNetPlans"
SEGMENTERS = {
    "l3": ("Dataset920_L3Vertebra0917", "3d_fullres"),
    "sma_inferior": ("Dataset917_InferiorSMA300Reann0917", "2d"),
    "sma_middle": ("Dataset918_MiddleSMA300Reann0917", "2d"),
}
DETECTORS = {"yolo_inferior": ("inferior", "l3_inferior"),
             "yolo_middle": ("middle", "l3_middle")}
SIDECARS = ("plans.json", "dataset.json")
FOLDS = range(5)
YOLO_IMGSZ, YOLO_CONF = 512, 0.001


def trainer_dir(config: str) -> str:
    return f"{TRAINER}__{PLANS}__{config}"


def checkpoint_rel(dataset: str, config: str, fold: int) -> str:
    return f"nnUNet_results/{dataset}/{trainer_dir(config)}/fold_{fold}/checkpoint_best.pth"


def detector_rel(branch: str, fold: int) -> str:
    return f"yolo/{branch}/fold_{fold}/best.pt"


def required_files() -> list[str]:
    out = []
    for dataset, config in SEGMENTERS.values():
        for name in SIDECARS:
            out.append(f"nnUNet_results/{dataset}/{trainer_dir(config)}/{name}")
        out += [checkpoint_rel(dataset, config, f) for f in FOLDS]
    for branch, _ in DETECTORS.values():
        out += [detector_rel(branch, f) for f in FOLDS]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_present(dest: Path) -> list[str]:
    return [r for r in required_files() if not (dest / r).is_file()]


def verify(dest: Path) -> int:
    sums = dest / "SHA256SUMS"
    if not sums.is_file():
        print(f"[!] {sums} not found; cannot verify", file=sys.stderr)
        return 1
    want = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            want[name.strip().lstrip("*").lstrip("./")] = digest
    bad = 0
    for rel, digest in sorted(want.items()):
        p = dest / rel
        if not p.is_file():
            print(f"  MISSING  {rel}"); bad += 1
        elif sha256(p) != digest:
            print(f"  MISMATCH {rel}"); bad += 1
    print(f"{len(want) - bad}/{len(want)} files verified")
    if bad:
        print("\nRe-download the failing files; a truncated 350 MB checkpoint is the usual cause.")
    return 1 if bad else 0


def write_registry(dest: Path, out: Path) -> None:
    missing = check_present(dest)
    if missing:
        print(f"[!] {len(missing)} files missing under {dest}", file=sys.stderr)
        for m in missing[:5]:
            print(f"      {m}", file=sys.stderr)
        sys.exit(1)
    dest = dest.resolve()
    reg = {
        "schema_version": 4,
        "fold_policy": ("L3 segmentation, both reference-section detectors, and both "
                        "body-composition segmenters all use the checkpoint of the case's "
                        "own fold, so no model saw the case it is evaluated on."),
        "splits": "splits/splits_final.json",
        "folds": {},
    }
    for f in FOLDS:
        entry = {}
        for key, (dataset, config) in SEGMENTERS.items():
            entry[key] = {"dataset": dataset, "config": config, "trainer": TRAINER,
                          "plans": PLANS, "fold": f,
                          "checkpoint": str(dest / checkpoint_rel(dataset, config, f))}
        for key, (branch, cls) in DETECTORS.items():
            entry[key] = {"class": cls, "fold": f, "imgsz": YOLO_IMGSZ, "conf": YOLO_CONF,
                          "weights": str(dest / detector_rel(branch, f))}
        reg["folds"][str(f)] = entry
    reg["complete"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}\n")
    print("Export these before running the pipeline:\n")
    print(f'    export nnUNet_results="{dest / "nnUNet_results"}"')
    print(f'    export nnUNet_raw="{dest / "nnUNet_results"}"')
    print(f'    export nnUNet_preprocessed="{dest / "nnUNet_results"}"')
    print('    export NNUNETV2_PREDICT_BIN="$(command -v nnUNetv2_predict)"')
    print("\nnnUNet_raw and nnUNet_preprocessed are unused at inference time, but nnU-Net")
    print("refuses to start unless they are set to an existing directory.")


def download(source: str, dest: Path) -> None:
    if not source:
        print(__doc__)
        print("No release location is configured.\n"
              "Set DEFAULT_SOURCE in this file, or pass --source, once the weights are\n"
              "published. Expected layout under --dest:\n")
        print("    weights/")
        print("    ├── SHA256SUMS")
        print("    ├── nnUNet_results/")
        print("    │   ├── Dataset920_L3Vertebra0917/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/")
        print("    │   │   ├── plans.json, dataset.json")
        print("    │   │   └── fold_0 … fold_4/checkpoint_best.pth")
        print("    │   ├── Dataset917_InferiorSMA300Reann0917/…__2d/")
        print("    │   └── Dataset918_MiddleSMA300Reann0917/…__2d/")
        print("    └── yolo/{inferior,middle}/fold_0 … fold_4/best.pt")
        print(f"\n    {len(required_files())} files, 4.66 GiB")
        sys.exit(2)
    dest.mkdir(parents=True, exist_ok=True)
    if "/" in source and not source.startswith("http"):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("[!] pip install huggingface_hub")
        snapshot_download(repo_id=source, repo_type="model", local_dir=str(dest))
    else:
        import urllib.request
        for rel in required_files() + ["SHA256SUMS"]:
            target = dest / rel
            if target.is_file():
                print(f"  have {rel}"); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"  get  {rel}", flush=True)
            urllib.request.urlretrieve(f"{source.rstrip('/')}/{rel}", target)
    missing = check_present(dest)
    print(f"[OK] {dest} — {len(required_files()) - len(missing)}/{len(required_files())} files present")
    if missing:
        sys.exit(f"[!] still missing {len(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("weights"))
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="Hugging Face repo id, or a base URL for direct download")
    ap.add_argument("--verify", action="store_true", help="check SHA256SUMS and exit")
    ap.add_argument("--write-registry", type=Path, metavar="PATH",
                    help="write a model registry pointing at --dest, then exit")
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify(args.dest))
    if args.write_registry:
        write_registry(args.dest, args.write_registry); return
    download(args.source, args.dest)


if __name__ == "__main__":
    main()
