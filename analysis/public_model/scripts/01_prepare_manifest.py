#!/usr/bin/env python3
"""Build the 300-case inferior public-model comparison manifest for the reannotated cohort.

Reannotation moved the inferior GT slice for 56 cases, so every reference slice CT in
Dataset917 imagesTr is potentially a different image than the one the public model saw in
the original submission. Nothing is reused: all 300 cases are marked for fresh inference.
"""

from __future__ import annotations

import os
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[3]))
BASE = ROOT / "analysis/public_model"
MANIFESTS = ROOT / "manifests"
RAW = ROOT / "workspace/nnunet/nnUNet_raw/Dataset917_InferiorSMA300Reann0917"
RESULT = (ROOT / "workspace/nnunet/nnUNet_results/Dataset917_InferiorSMA300Reann0917"
          / "nnUNetTrainer_500epochs__nnUNetPlans__2d")
# Two cases were replaced at reannotation while keeping their cross-validation
# slots. The identifiers are hospital record numbers and are not distributed;
# supply them as comma-separated lists if you need to reproduce that audit.
REPLACEMENT_IDS = {s for s in os.environ.get("ADPKD_REPLACEMENT_IDS", "").split(",") if s}
RETIRED_IDS = {s for s in os.environ.get("ADPKD_RETIRED_IDS", "").split(",") if s}


def replace_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {dst}")
    dst.symlink_to(src)


def main() -> None:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for fold in range(5):
        path = MANIFESTS / f"manifest_fold_{fold}.csv"
        with path.open(newline="", encoding="utf-8-sig") as f:
            source_rows = list(csv.DictReader(f))
        if len(source_rows) != 60:
            raise ValueError(f"Expected 60 cases in fold {fold}, got {len(source_rows)}")

        for source in source_rows:
            case_id = source["case_id"]
            if case_id in seen:
                raise ValueError(f"Duplicate case_id: {case_id}")
            seen.add(case_id)

            ct = RAW / "imagesTr" / f"{case_id}_0000.nii.gz"
            gt = RAW / "labelsTr" / f"{case_id}.nii.gz"
            proposed = RESULT / f"fold_{fold}" / "validation" / f"{case_id}.nii.gz"
            for required in (ct, gt, proposed):
                if not required.exists():
                    raise FileNotFoundError(required)

            public_input = BASE / "public_input" / f"{case_id}_0000.nii.gz"
            replace_symlink(ct, public_input)

            rows.append({
                "fold": str(fold),
                "case_id": case_id,
                "ct_path": str(ct),
                "gt_path": str(gt),
                "proposed_pred_path": str(proposed),
                "public_input_path": str(public_input),
                "public_pred_path": str(BASE / "public_pred" / f"{case_id}_0000.nii.gz"),
                "public_prediction_source": "new_inference",
            })

    if len(rows) != 300 or len(seen) != 300:
        raise ValueError(f"Expected exactly 300 unique cases, got rows={len(rows)}, unique={len(seen)}")
    if seen & RETIRED_IDS:
        raise ValueError(f"Retired IDs remain: {sorted(seen & RETIRED_IDS)}")
    if not REPLACEMENT_IDS <= seen:
        raise ValueError(f"Replacement IDs missing: {sorted(REPLACEMENT_IDS - seen)}")

    out = BASE / "manifest" / "inferior_public_model_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out}")
    print(f"Cases: {len(rows)}; fold counts: {dict(sorted(Counter(r['fold'] for r in rows).items()))}")
    print(f"Public inputs staged in: {BASE / 'public_input'}")


if __name__ == "__main__":
    main()
