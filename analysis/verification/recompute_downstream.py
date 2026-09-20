#!/usr/bin/env python3
"""Downstream tables that also used the validation export: Supp Tables 3, 5, 6, 7."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TISSUES = ["SMA", "VFAT", "SFAT", "IMAT"]
def sd(v): return float(np.std(v, ddof=1))
def msd(v, d=2): return f"{v.mean():.{d}f} ± {sd(v):.{d}f}"

best = pd.read_csv(OUT / "best_weights_per_case_segonly.csv", dtype={"case_id": str})
inf = best[best.level == "inferior"].set_index(["case_id", "tissue"])

# ---------- Supp Table 3: external reference vs proposed ----------
ext = pd.read_csv(ROOT / "analysis/external_reference/tables/clinical_csv_inferior_300_per_case.csv",
                  dtype={"case_id": str})
old3 = pd.read_csv(ROOT / "results_final/summary_tables/external_reference.csv")
print("=== Supplementary Table 3 (external reference) ===")
rows = []
for t in ["SMA", "VFAT", "SFAT"]:
    e = ext.set_index("case_id")[t.lower()]
    p = inf.xs(t, level="tissue").seg_pred_csa_cm2.reindex(e.index)
    err = p - e
    r, _ = pearsonr(e, p)
    o = old3[old3.tissue == t].iloc[0]
    rows.append({"tissue": t, "n": len(e), "external_mean": e.mean(), "external_sd": sd(e),
                 "proposed_mean": p.mean(), "proposed_sd": sd(p),
                 "error_mean": err.mean(), "error_sd": sd(err), "pearson_r": r})
    print(f"  {t:5} proposed {o.proposed_mean:7.2f}±{o.proposed_sd:5.2f} -> {p.mean():7.2f}±{sd(p):5.2f}"
          f" | error {o.error_mean:+.2f}±{o.error_sd:.2f} -> {err.mean():+.2f}±{sd(err):.2f}"
          f" | r {o.pearson_r:.3f} -> {r:.3f}")
pd.DataFrame(rows).to_csv(OUT / "best_weights_external_reference.csv", index=False)

# ---------- Supp Tables 5 and 6: public model ----------
pub = pd.read_csv(ROOT / "analysis/public_model/tables/inferior_comparison_per_case.csv",
                  dtype={"case_id": str}).set_index("case_id")
old5 = pd.read_csv(ROOT / "results_final/summary_tables/public_model.csv")
print("\n=== Supplementary Tables 5/6 (public model comparison) ===")
rows = []
for t in ["SMA", "VFAT", "SFAT"]:
    k = t.lower()
    gt = pub[f"gt_{k}_cm2"]
    prop = inf.xs(t, level="tissue").seg_pred_csa_cm2.reindex(gt.index)
    pe = pub[f"public_{k}_cm2"]
    perr, pubrr = prop - gt, pe - gt
    rp, _ = pearsonr(gt, prop); rq, _ = pearsonr(gt, pe)
    o = old5[old5.tissue == t].iloc[0]
    rows.append({"tissue": t, "n": len(gt), "gt_mean": gt.mean(), "gt_sd": sd(gt),
                 "proposed_mean": prop.mean(), "proposed_sd": sd(prop),
                 "public_mean": pe.mean(), "public_sd": sd(pe),
                 "proposed_error_mean": perr.mean(), "proposed_error_sd": sd(perr),
                 "public_error_mean": pubrr.mean(), "public_error_sd": sd(pubrr),
                 "proposed_mae_mean": perr.abs().mean(), "proposed_mae_sd": sd(perr.abs()),
                 "public_mae_mean": pubrr.abs().mean(), "public_mae_sd": sd(pubrr.abs()),
                 "proposed_r": rp, "public_r": rq})
    print(f"  {t:5} proposed CSA {o.proposed_mean:7.2f} -> {prop.mean():7.2f}"
          f" | proposed err {o.proposed_error_mean:+.2f}±{o.proposed_error_sd:.2f} -> {perr.mean():+.2f}±{sd(perr):.2f}"
          f" | proposed MAE {o.proposed_mae_mean:.2f}±{o.proposed_mae_sd:.2f} -> {perr.abs().mean():.2f}±{sd(perr.abs()):.2f}"
          f" | r {o.proposed_r:.3f} -> {rp:.3f}   [public columns unchanged]")
pd.DataFrame(rows).to_csv(OUT / "best_weights_public_model.csv", index=False)

# ---------- Supp Table 7: sex-stratified ----------
clin = pd.read_csv(ROOT / "external_inputs/251212_ADPKD_total.csv", encoding="cp949", dtype={"id_n": str})
man = pd.read_csv(ROOT / "manifests/manifest_300_all_folds.csv", dtype={"case_id": str})
sex = man[["case_id"]].merge(clin[["id_n", "sex"]], left_on="case_id", right_on="id_n")[["case_id", "sex"]]
e2e = pd.read_csv(ROOT / "results_final/raw_csv/oof_body_composition.csv", dtype={"case_id": str})
e2e = e2e[["case_id", "level", "tissue", "e2e_abs_error_cm2"]]
b = best.merge(sex, on="case_id").merge(e2e, on=["case_id", "level", "tissue"])
print("\n=== Supplementary Table 7 (sex-stratified, seg-only columns) ===")
rows = []
for level in ("inferior", "middle"):
    for t in TISSUES:
        s = b[(b.level == level) & (b.tissue == t)]
        F, M = s[s.sex == "F"], s[s.sex == "M"]
        pd_ = mannwhitneyu(F.dsc, M.dsc, alternative="two-sided").pvalue
        pm = mannwhitneyu(F.seg_abs_error_cm2, M.seg_abs_error_cm2, alternative="two-sided").pvalue
        rows.append({"level": level, "tissue": t,
                     "women_gt_csa": msd(F.gt_csa_cm2), "men_gt_csa": msd(M.gt_csa_cm2),
                     "women_dsc": msd(F.dsc, 3), "men_dsc": msd(M.dsc, 3), "p_dsc": pd_,
                     "women_mae": msd(F.seg_abs_error_cm2), "men_mae": msd(M.seg_abs_error_cm2),
                     "p_mae": pm})
        print(f"  {level:8} {t:5} DSC {msd(F.dsc,3)} / {msd(M.dsc,3)} P={pd_:.3e}"
              f" | MAE {msd(F.seg_abs_error_cm2)} / {msd(M.seg_abs_error_cm2)} P={pm:.4f}")
df7 = pd.DataFrame(rows); df7.to_csv(OUT / "best_weights_sex_table7.csv", index=False)
print(f"\n  smallest MAE P = {df7.p_mae.min():.6f} ({df7.loc[df7.p_mae.idxmin()].level} "
      f"{df7.loc[df7.p_mae.idxmin()].tissue});  all > 0.10: {bool((df7.p_mae>0.10).all())}")
print(f"  all DSC P < 0.001: {bool((df7.p_dsc<0.001).all())}")
print(f"\n[OK] wrote {OUT}")
