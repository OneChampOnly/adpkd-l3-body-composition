#!/usr/bin/env python3
"""Compute revised Supplementary Tables 5A and 5B from the preserved cohort."""

from __future__ import annotations

import os

# Supplied via the environment; see analysis/public_model/scripts/01_prepare_manifest.py
REPLACEMENT_IDS = [s for s in os.environ.get("ADPKD_REPLACEMENT_IDS", "").split(",") if s]
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


BASE = Path("${ADPKD_ROOT}/analysis/public_model")
OLD_PER_CASE = Path("${ADPKD_ROOT}/external_inputs/compare_ours_vs_github_vs_gt_per_case.csv")
MANIFEST = BASE / "manifest" / "inferior_public_model_manifest.csv"
LABELS = {"SMA": 1, "SFAT": 2, "VFAT": 3, "IMAT": 4}
ORDER = ["SMA", "VFAT", "SFAT"]


def areas(path: Path) -> dict[str, float]:
    image = nib.load(str(path))
    array = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    spacing = image.header.get_zooms()
    pixel_area_cm2 = float(spacing[0] * spacing[1] / 100.0)
    return {tissue: float(np.count_nonzero(array == label) * pixel_area_cm2) for tissue, label in LABELS.items()}


def mean_sd(values: pd.Series, digits: int = 1) -> str:
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.astype(str).to_dict("records"):
        lines.append("| " + " | ".join(row[c] for c in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if len(manifest) != 300 or len({r["case_id"] for r in manifest}) != 300:
        raise ValueError("Manifest must contain exactly 300 unique cases")

    rows: list[dict[str, object]] = []
    for row in manifest:
        paths = {
            "gt": Path(row["gt_path"]),
            "proposed": Path(row["proposed_pred_path"]),
            "public": Path(row["public_pred_path"]),
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing masks: " + ", ".join(missing))
        values = {method: areas(path) for method, path in paths.items()}
        out: dict[str, object] = {
            "fold": int(row["fold"]),
            "case_id": row["case_id"],
            "public_prediction_source": row["public_prediction_source"],
            **{f"{name}_path": str(path) for name, path in paths.items()},
        }
        for tissue in LABELS:
            for method in values:
                out[f"{method}_{tissue.lower()}_cm2"] = values[method][tissue]
            out[f"proposed_error_{tissue.lower()}_cm2"] = values["proposed"][tissue] - values["gt"][tissue]
            out[f"public_error_{tissue.lower()}_cm2"] = values["public"][tissue] - values["gt"][tissue]
        rows.append(out)

    per_case = pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)
    table_dir = BASE / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(table_dir / "inferior_comparison_per_case.csv", index=False)

    rows_5a = []
    rows_5b = []
    raw_rows = []
    for tissue in ORDER:
        key = tissue.lower()
        gt = per_case[f"gt_{key}_cm2"]
        proposed = per_case[f"proposed_{key}_cm2"]
        public = per_case[f"public_{key}_cm2"]
        proposed_error = proposed - gt
        public_error = public - gt
        proposed_abs = proposed_error.abs()
        public_abs = public_error.abs()
        proposed_r = float(gt.corr(proposed, method="pearson"))
        public_r = float(gt.corr(public, method="pearson"))

        rows_5a.append({
            "Tissue": tissue,
            "N": "300/300",
            "Manual GT, cm²": mean_sd(gt),
            "Proposed method, cm²": mean_sd(proposed),
            "Public model, cm²": mean_sd(public),
        })
        rows_5b.append({
            "Tissue": tissue,
            "Proposed error, cm²": mean_sd(proposed_error),
            "Public model error, cm²": mean_sd(public_error),
            "Proposed MAE, cm²": mean_sd(proposed_abs),
            "Public model MAE, cm²": mean_sd(public_abs),
            "Proposed r": f"{proposed_r:.3f}",
            "Public model r": f"{public_r:.3f}",
        })
        raw_rows.append({
            "Tissue": tissue,
            "N": 300,
            "GT_mean": gt.mean(), "GT_SD": gt.std(ddof=1),
            "Proposed_mean": proposed.mean(), "Proposed_SD": proposed.std(ddof=1),
            "Public_mean": public.mean(), "Public_SD": public.std(ddof=1),
            "Proposed_error_mean": proposed_error.mean(), "Proposed_error_SD": proposed_error.std(ddof=1),
            "Public_error_mean": public_error.mean(), "Public_error_SD": public_error.std(ddof=1),
            "Proposed_MAE_mean": proposed_abs.mean(), "Proposed_MAE_SD": proposed_abs.std(ddof=1),
            "Public_MAE_mean": public_abs.mean(), "Public_MAE_SD": public_abs.std(ddof=1),
            "Proposed_r": proposed_r, "Public_r": public_r,
        })

    table_5a = pd.DataFrame(rows_5a)
    table_5b = pd.DataFrame(rows_5b)
    table_5a.to_csv(table_dir / "supplementary_table_5A_revised.csv", index=False)
    table_5b.to_csv(table_dir / "supplementary_table_5B_revised.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(table_dir / "supplementary_tables_5A_5B_revised_raw.csv", index=False)
    (table_dir / "supplementary_table_5A_revised.md").write_text(markdown(table_5a), encoding="utf-8")
    (table_dir / "supplementary_table_5B_revised.md").write_text(markdown(table_5b), encoding="utf-8")
    per_case[per_case["case_id"].isin(REPLACEMENT_IDS)].to_csv(
        table_dir / "replacement_cases_metrics.csv", index=False
    )

    try:
        old = pd.read_csv(OLD_PER_CASE, dtype={"case_id": str})
        shared = per_case.merge(old, on=["fold", "case_id"], suffixes=("_current", "_old"))
    except (FileNotFoundError, KeyError) as exc:
        print(f"[SKIP] pre-reannotation GT audit unavailable: {exc}")
        shared = None
    audit_rows = []
    for tissue, current_col, old_col in () if shared is None else (
        ("SMA", "gt_sma_cm2", "gt_muscle_cm2"),
        ("VFAT", "gt_vfat_cm2_current", "gt_vfat_cm2_old"),
        ("SFAT", "gt_sfat_cm2_current", "gt_sfat_cm2_old"),
    ):
        delta = shared[current_col] - shared[old_col]
        audit_rows.append({
            "Tissue": tissue,
            "N shared": len(shared),
            "N GT area changed (>1e-6 cm²)": int((delta.abs() > 1e-6).sum()),
            "Current minus old GT mean, cm²": float(delta.mean()),
            "Maximum absolute GT difference, cm²": float(delta.abs().max()),
        })
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(table_dir / "current_vs_original_gt_audit.csv", index=False)

    print("Supplementary Table 5A (revised)")
    print(table_5a.to_string(index=False))
    print("\nSupplementary Table 5B (revised)")
    print(table_5b.to_string(index=False))


if __name__ == "__main__":
    main()
