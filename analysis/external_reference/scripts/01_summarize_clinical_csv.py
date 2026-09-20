#!/usr/bin/env python3
"""Summarize inferior body-composition values for the current preserved 300 cases."""

from __future__ import annotations

import os
import math
from pathlib import Path

import pandas as pd


BASE = Path("${ADPKD_ROOT}/analysis/external_reference")
# Fold/case membership only: these are externally measured clinical values, unaffected by
# the reannotation, so the cohort manifest is the whole dependency.
MANIFEST = Path("${ADPKD_ROOT}/manifests/manifest_300_all_folds.csv")
SOURCE = Path(os.environ.get("ADPKD_CLINICAL_CSV",
    str(Path(__file__).resolve().parents[3] / "external_inputs/clinical_table.csv")))
AREA_COLUMNS = ["sfat", "vfat", "sma", "imat", "lama", "nama"]
NORMALIZED_COLUMNS = ["sma_bmi", "lama_bmi", "nama_bmi", "sma_ht", "lama_ht", "nama_ht"]


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2f}"


def markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.astype(str).to_dict("records"):
        lines.append("| " + " | ".join(row[c] for c in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype={"case_id": str})[["fold", "case_id"]]
    source = pd.read_csv(SOURCE, encoding="cp949", dtype={"id_n": str}, na_values=["NA", ""])
    selected = manifest.merge(source, left_on="case_id", right_on="id_n", how="left", validate="one_to_one")
    if len(selected) != 300 or selected["id_n"].isna().any():
        missing = selected.loc[selected["id_n"].isna(), "case_id"].tolist()
        raise ValueError(f"Expected 300/300 matched cases; missing={missing}")

    columns = AREA_COLUMNS + NORMALIZED_COLUMNS
    rows = []
    for column in columns:
        values = pd.to_numeric(selected[column], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else math.nan
        sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append({
            "Variable": column.upper(),
            "Group": "area_cm2" if column in AREA_COLUMNS else "normalized",
            "N": len(values),
            "Missing": 300 - len(values),
            "Mean": mean,
            "SD": sd,
            "Mean ± SD": f"{fmt(mean)} ± {fmt(sd)}" if len(values) else "NA",
            "Median": float(values.median()) if len(values) else math.nan,
            "Min": float(values.min()) if len(values) else math.nan,
            "Max": float(values.max()) if len(values) else math.nan,
        })

    out = BASE / "tables"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["fold", "case_id", "id_n", *columns]
    selected[keep].to_csv(out / "clinical_csv_inferior_300_per_case.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "clinical_csv_inferior_300_summary.csv", index=False)
    display = summary[["Variable", "Group", "N", "Missing", "Mean ± SD", "Median", "Min", "Max"]].copy()
    for column in ["Median", "Min", "Max"]:
        display[column] = display[column].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")
    (out / "clinical_csv_inferior_300_summary.md").write_text(markdown(display), encoding="utf-8")

    identity = (
        pd.to_numeric(selected["sma"], errors="coerce")
        - pd.to_numeric(selected["lama"], errors="coerce")
        - pd.to_numeric(selected["nama"], errors="coerce")
    )
    print("Matched cases: 300/300")
    print(display.to_string(index=False))
    print(f"Max |SMA-(LAMA+NAMA)|: {identity.abs().max():.6f} cm2")


if __name__ == "__main__":
    main()
