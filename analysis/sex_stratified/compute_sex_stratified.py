#!/usr/bin/env python3
"""Sex-stratified body composition performance, for CKJ Supplementary Table 7.

CKJ asks authors to report sex-stratified results where appropriate and to say so when a
study is underpowered to detect sex differences. Nothing is retrained here: the per-case
out-of-fold results are simply split by the sex recorded in the clinical table.

Between-sex tests are reported for transparency, but the study was not designed or powered
to detect sex differences in segmentation accuracy, and the p values are not adjusted for
the number of comparisons.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLINICAL = Path(os.environ.get("ADPKD_CLINICAL_CSV",
    str(Path(__file__).resolve().parents[2] / "external_inputs/clinical_table.csv")))
TISSUE_ORDER = ["SMA", "VFAT", "SFAT", "IMAT"]
SEXES = [("F", "Women"), ("M", "Men")]


def mean_sd(values: pd.Series, digits: int = 2) -> str:
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def markdown(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.astype(str).to_dict("records"):
        lines.append("| " + " | ".join(row[c] for c in cols) + " |")
    return "\n".join(lines) + "\n"


def load_sex() -> pd.DataFrame:
    clinical = pd.read_csv(CLINICAL, encoding="cp949", dtype={"id_n": str})
    manifest = pd.read_csv(ROOT / "manifests/manifest_300_all_folds.csv", dtype={"case_id": str})
    merged = manifest[["case_id", "fold"]].merge(
        clinical[["id_n", "sex", "age", "ht", "wt", "bmi"]],
        left_on="case_id", right_on="id_n", how="left", validate="one_to_one")
    if merged.sex.isna().any():
        raise ValueError(f"Unmatched cases: {merged.loc[merged.sex.isna(), 'case_id'].tolist()}")
    counts = merged.sex.value_counts().to_dict()
    print(f"cohort: {counts}")
    return merged.drop(columns=["id_n"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sex = load_sex()

    body = pd.read_csv(ROOT / "results_final/raw_csv/oof_body_composition.csv", dtype={"case_id": str})
    body = body.merge(sex[["case_id", "sex"]], on="case_id", how="left", validate="many_to_one")
    loc = pd.read_csv(ROOT / "results_final/raw_csv/oof_localization.csv", dtype={"case_id": str})
    loc = loc.merge(sex[["case_id", "sex"]], on="case_id", how="left", validate="one_to_one")
    l3 = pd.read_csv(ROOT / "results_final/raw_csv/oof_l3_vertebra.csv", dtype={"case_id": str})
    l3 = l3.merge(sex[["case_id", "sex"]], on="case_id", how="left", validate="one_to_one")

    body.to_csv(OUT / "per_case_body_composition_with_sex.csv", index=False)

    # ---- Supplementary Table 7: DSC and MAE in women and in men ----------------
    rows, raw = [], []
    for level in ("inferior", "middle"):
        for tissue in TISSUE_ORDER:
            sel = body[(body.level == level) & (body.tissue == tissue)]
            entry = {"Level": level.capitalize(), "Tissue": tissue}
            stat: dict[str, object] = {"level": level, "tissue": tissue}
            groups = {}
            for code, label in SEXES:
                g = sel[sel.sex == code]
                groups[code] = g
                entry[f"{label} (n={len(g)}), DSC"] = mean_sd(g.dice, 3)
                entry[f"{label} (n={len(g)}), MAE, cm²"] = mean_sd(g.seg_abs_error_cm2)
                entry[f"{label} (n={len(g)}), E2E MAE, cm²"] = mean_sd(g.e2e_abs_error_cm2)
                stat.update({
                    f"{code}_n": len(g),
                    f"{code}_dsc_mean": g.dice.mean(), f"{code}_dsc_sd": g.dice.std(ddof=1),
                    f"{code}_mae_mean": g.seg_abs_error_cm2.mean(), f"{code}_mae_sd": g.seg_abs_error_cm2.std(ddof=1),
                    f"{code}_e2e_mae_mean": g.e2e_abs_error_cm2.mean(), f"{code}_e2e_mae_sd": g.e2e_abs_error_cm2.std(ddof=1),
                    f"{code}_gt_csa_mean": g.gt_csa_cm2.mean(), f"{code}_gt_csa_sd": g.gt_csa_cm2.std(ddof=1),
                })
            # Mann-Whitney U: DSC and absolute error are bounded and skewed, so a rank test
            # is the honest default here rather than a t test.
            for key, column in (("dsc", "dice"), ("mae", "seg_abs_error_cm2"), ("e2e_mae", "e2e_abs_error_cm2")):
                p = stats.mannwhitneyu(groups["F"][column], groups["M"][column], alternative="two-sided").pvalue
                stat[f"{key}_p"] = float(p)
                if key in ("dsc", "mae"):
                    entry[f"P ({'DSC' if key == 'dsc' else 'MAE'})"] = f"{p:.3f}" if p >= 0.001 else "<0.001"
            rows.append(entry)
            raw.append(stat)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "supplementary_table_7_sex_stratified.csv", index=False)
    pd.DataFrame(raw).to_csv(OUT / "supplementary_table_7_raw.csv", index=False)
    (OUT / "supplementary_table_7_sex_stratified.md").write_text(markdown(table), encoding="utf-8")

    # ---- upstream stages, for the Results sentence -----------------------------
    stages = []
    for label, frame, column, digits in (
        ("L3 vertebral segmentation DSC", l3, "dice", 3),
        ("Inferior localization error, mm", loc, "inferior_abs_error_mm", 2),
        ("Middle localization error, mm", loc, "middle_abs_error_mm", 2),
    ):
        entry = {"Metric": label}
        for code, name in SEXES:
            g = frame[frame.sex == code][column]
            entry[f"{name} (n={len(g)})"] = mean_sd(g, digits)
        p = stats.mannwhitneyu(frame[frame.sex == "F"][column], frame[frame.sex == "M"][column],
                               alternative="two-sided").pvalue
        entry["P"] = f"{p:.3f}" if p >= 0.001 else "<0.001"
        stages.append(entry)
    stage_table = pd.DataFrame(stages)
    stage_table.to_csv(OUT / "sex_stratified_upstream_stages.csv", index=False)
    (OUT / "sex_stratified_upstream_stages.md").write_text(markdown(stage_table), encoding="utf-8")

    summary = {
        "n_women": int((sex.sex == "F").sum()),
        "n_men": int((sex.sex == "M").sum()),
        "test": "Mann-Whitney U, two-sided, unadjusted for multiple comparisons",
        "note": ("The study was not designed or powered to detect sex differences in "
                 "segmentation accuracy; these comparisons are descriptive."),
        "significant_at_0.05_unadjusted": [
            {"level": r["level"], "tissue": r["tissue"],
             "dsc_p": r["dsc_p"], "mae_p": r["mae_p"], "e2e_mae_p": r["e2e_mae_p"]}
            for r in raw if min(r["dsc_p"], r["mae_p"], r["e2e_mae_p"]) < 0.05
        ],
    }
    (OUT / "sex_stratified_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== Upstream stages ===")
    print(stage_table.to_string(index=False))
    print("\n=== Supplementary Table 7 (inferior) ===")
    print(table[table.Level == "Inferior"].to_string(index=False))
    print("\n=== Supplementary Table 7 (middle) ===")
    print(table[table.Level == "Middle"].to_string(index=False))
    print(f"\nunadjusted p<0.05 comparisons: {len(summary['significant_at_0.05_unadjusted'])}")


if __name__ == "__main__":
    main()
