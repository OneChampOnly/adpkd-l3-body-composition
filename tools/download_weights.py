#!/usr/bin/env python3
"""Fetch the released weights and, optionally, write a matching model registry.

The weights are not in this repository: 25 checkpoints, 4.7 GB. Point --source at the
release (a Hugging Face repo id or a base URL) once it is published; see docs/WEIGHTS.md.

    python tools/download_weights.py --dest weights/
    python tools/download_weights.py --dest weights/ --verify
    python tools/download_weights.py --dest weights/ --write-registry config/model_registry.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Set once the weights are published. Leaving this empty makes the script explain
# itself rather than fail with a confusing network error.
DEFAULT_SOURCE = ""      # e.g. "<user>/adpkd-l3-body-composition" or "https://zenodo.org/records/<id>/files"

FOLDS = range(5)
NNUNET = {"l3": "checkpoint_best.pth",
          "sma_inferior": "checkpoint_best.pth",
          "sma_middle": "checkpoint_best.pth"}
YOLO = {"yolo_inferior": "best.pt", "yolo_middle": "best.pt"}
REGISTRY_META = {
    "l3": {"dataset": "Dataset920_L3Vertebra0917", "config": "3d_fullres"},
    "sma_inferior": {"dataset": "Dataset917_InferiorSMA300Reann0917", "config": "2d"},
    "sma_middle": {"dataset": "Dataset918_MiddleSMA300Reann0917", "config": "2d"},
}
TRAINER, PLANS = "nnUNetTrainer_500epochs", "nnUNetPlans"


def expected_files() -> list[str]:
    out = []
    for f in FOLDS:
        for key, name in {**NNUNET, **YOLO}.items():
            out.append(f"fold_{f}/{key}/{name}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(dest: Path) -> int:
    sums = dest / "SHA256SUMS"
    if not sums.is_file():
        print(f"[!] {sums} not found; cannot verify", file=sys.stderr)
        return 1
    want = {}
    for line in sums.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            want[name.strip().lstrip("*")] = digest
    bad = 0
    for rel, digest in sorted(want.items()):
        p = dest / rel
        if not p.is_file():
            print(f"  MISSING  {rel}"); bad += 1; continue
        got = sha256(p)
        if got != digest:
            print(f"  MISMATCH {rel}\n           expected {digest}\n           got      {got}")
            bad += 1
        else:
            print(f"  ok       {rel}")
    print(f"\n{len(want) - bad}/{len(want)} files verified")
    return 1 if bad else 0


def write_registry(dest: Path, out: Path) -> None:
    missing = [f for f in expected_files() if not (dest / f).is_file()]
    if missing:
        print(f"[!] {len(missing)} checkpoints missing under {dest}; "
              f"first: {missing[:3]}", file=sys.stderr)
        sys.exit(1)
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
        for key, meta in REGISTRY_META.items():
            entry[key] = {**meta, "trainer": TRAINER, "plans": PLANS, "fold": f,
                          "checkpoint": str((dest / f"fold_{f}/{key}/checkpoint_best.pth").resolve())}
        for key, branch in (("yolo_inferior", "l3_inferior"), ("yolo_middle", "l3_middle")):
            entry[key] = {"class": branch, "fold": f, "imgsz": 512, "conf": 0.001,
                          "weights": str((dest / f"fold_{f}/{key}/best.pt").resolve())}
        reg["folds"][str(f)] = entry
    reg["complete"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")


def download(source: str, dest: Path) -> None:
    if not source:
        print(__doc__)
        print("No release location is configured yet.\n"
              "Set DEFAULT_SOURCE in this file, or pass --source, once the weights are\n"
              "published. Expected layout under --dest:\n")
        for f in expected_files()[:5]:
            print("   ", f)
        print("    ... 25 files, 4.7 GB total, plus SHA256SUMS")
        sys.exit(2)
    if "/" in source and not source.startswith("http"):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("[!] pip install huggingface_hub", file=sys.stderr); sys.exit(1)
        snapshot_download(repo_id=source, repo_type="model", local_dir=str(dest))
        print(f"[OK] downloaded {source} -> {dest}")
        return
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    for rel in expected_files() + ["SHA256SUMS"]:
        target = dest / rel
        if target.is_file():
            print(f"  have {rel}"); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{source.rstrip('/')}/{rel}"
        print(f"  get  {rel}")
        urllib.request.urlretrieve(url, target)
    print(f"[OK] downloaded -> {dest}")


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
