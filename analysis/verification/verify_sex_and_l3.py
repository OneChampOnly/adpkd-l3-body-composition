#!/usr/bin/env python3
"""CKJ verification A-3 / A-4 / A-5 / A-6: sex-stratified L3, GT CSA, exact P values.

Run: python verify_sex_and_l3.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLINICAL = ROOT / "external_inputs/251212_ADPKD_total.csv"
TISSUES = ["SMA", "VFAT", "SFAT", "IMAT"]


def load_sex() -> pd.DataFrame:
    clin = pd.read_csv(CLINICAL, encoding="cp949", dtype={"id_n": str})
    man = pd.read_csv(ROOT / "manifests/manifest_300_all_folds.csv", dtype={"case_id": str})
    m = man[["case_id", "fold"]].merge(clin[["id_n", "sex"]], left_on="case_id",
                                       right_on="id_n", how="left", validate="one_to_one")
    assert not m.sex.isna().any()
    return m.drop(columns=["id_n"])


def msd(v, d=3):
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def main() -> None:
    sex = load_sex()
    nF = int((sex.sex == "F").sum()); nM = int((sex.sex == "M").sum())
    print(f"cohort: women={nF} men={nM} total={nF+nM}\n")

    # ---------------- A-3: L3 vertebra DSC + localization error ----------------
    l3 = pd.read_csv(ROOT / "results_final/raw_csv/oof_l3_vertebra.csv", dtype={"case_id": str})
    l3 = l3.merge(sex[["case_id", "sex"]], on="case_id", validate="one_to_one")
    loc = pd.read_csv(ROOT / "results_final/raw_csv/oof_localization.csv", dtype={"case_id": str})
    loc = loc.merge(sex[["case_id", "sex"]], on="case_id", validate="one_to_one")

    print("=== A-3  L3 vertebra rows for Supplementary Table 7 ===")
    a3 = []
    for label, frame, col, d in (
        ("L3 vertebral segmentation DSC", l3, "dice", 3),
        ("Inferior localization error, mm", loc, "inferior_abs_error_mm", 2),
        ("Middle localization error, mm", loc, "middle_abs_error_mm", 2),
    ):
        F = frame[frame.sex == "F"][col]; M = frame[frame.sex == "M"][col]
        p = stats.mannwhitneyu(F, M, alternative="two-sided").pvalue
        overall = frame[col]
        wavg = (nF * F.mean() + nM * M.mean()) / (nF + nM)
        a3.append({"metric": label, f"women_n{nF}": msd(F, d), f"men_n{nM}": msd(M, d),
                   "p_exact": p, "overall": msd(overall, d),
                   "weighted_avg_check": round(float(wavg), 6),
                   "overall_mean": round(float(overall.mean()), 6),
                   "wavg_matches": bool(abs(wavg - overall.mean()) < 5e-7)})
        print(f"  {label:34s} F={msd(F,d):18s} M={msd(M,d):18s} P={p:.6g}")
        print(f"     weighted avg {wavg:.6f}  vs overall {overall.mean():.6f}  "
              f"match={abs(wavg-overall.mean())<5e-7}")
    pd.DataFrame(a3).to_csv(OUT / "a3_l3_sex_rows.csv", index=False)

    # ---------------- A-4: GT CSA by sex ----------------
    bc = pd.read_csv(ROOT / "results_final/raw_csv/oof_body_composition.csv", dtype={"case_id": str})
    bc = bc.merge(sex[["case_id", "sex"]], on="case_id", validate="many_to_one")
    print("\n=== A-4  Manual ground-truth CSA by sex (cm2) ===")
    a4 = []
    for level in ("inferior", "middle"):
        for t in TISSUES:
            s = bc[(bc.level == level) & (bc.tissue == t)]
            F = s[s.sex == "F"].gt_csa_cm2; M = s[s.sex == "M"].gt_csa_cm2
            larger = "M" if M.mean() > F.mean() else "F"
            expect = "M" if t in ("SMA", "VFAT") else "F"
            a4.append({"level": level, "tissue": t, f"women_n{nF}": msd(F, 2),
                       f"men_n{nM}": msd(M, 2), "women_mean": F.mean(), "men_mean": M.mean(),
                       "larger_sex": larger, "manuscript_claims": expect,
                       "claim_holds": larger == expect,
                       "p_exact": stats.mannwhitneyu(F, M, alternative="two-sided").pvalue})
            flag = "OK " if larger == expect else "!! "
            print(f"  {flag}{level:8s} {t:5s} F={msd(F,2):18s} M={msd(M,2):18s} larger={larger} expected={expect}")
    pd.DataFrame(a4).to_csv(OUT / "a4_gt_csa_by_sex.csv", index=False)

    # ---------------- A-5: exact P values for all 8 DSC + 8 MAE comparisons ----------------
    print("\n=== A-5  Exact P values (Mann-Whitney U, two-sided) ===")
    a5 = []
    for level in ("inferior", "middle"):
        for t in TISSUES:
            s = bc[(bc.level == level) & (bc.tissue == t)]
            F = s[s.sex == "F"]; M = s[s.sex == "M"]
            row = {"level": level, "tissue": t,
                   "women_dsc": msd(F.dice, 3), "men_dsc": msd(M.dice, 3),
                   "women_mae": msd(F.seg_abs_error_cm2, 2), "men_mae": msd(M.seg_abs_error_cm2, 2)}
            for key, col in (("dsc", "dice"), ("mae", "seg_abs_error_cm2"),
                             ("e2e_mae", "e2e_abs_error_cm2")):
                a, b = F[col].to_numpy(), M[col].to_numpy()
                row[f"p_{key}_asymptotic"] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                row[f"p_{key}_exact_method"] = stats.mannwhitneyu(
                    a, b, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
                # Cohen's d on the reported means, for comparison with the rank test
                sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
                row[f"cohens_d_{key}"] = (a.mean() - b.mean()) / sp if sp else np.nan
                row[f"p_{key}_ttest"] = stats.ttest_ind(a, b, equal_var=True).pvalue
            row["n_ties_dsc"] = int(len(F.dice) + len(M.dice) - len(np.unique(np.concatenate([F.dice, M.dice]))))
            a5.append(row)
            print(f"  {level:8s} {t:5s} DSC P={row['p_dsc_asymptotic']:.3e} (t-test {row['p_dsc_ttest']:.3f}, "
                  f"d={row['cohens_d_dsc']:+.3f})   MAE P={row['p_mae_asymptotic']:.4f}")
    df5 = pd.DataFrame(a5)
    df5.to_csv(OUT / "a5_exact_p_values.csv", index=False)

    # ---------------- A-6: smallest MAE P ----------------
    pmin = df5.p_mae_asymptotic.min()
    arg = df5.loc[df5.p_mae_asymptotic.idxmin()]
    print(f"\n=== A-6  smallest MAE P = {pmin:.6f}  ({arg.level} {arg.tissue}) ===")
    print(f"  all MAE P > 0.10 ? {bool((df5.p_mae_asymptotic > 0.10).all())}")

    json.dump({"n_women": nF, "n_men": nM,
               "a4_all_claims_hold": bool(pd.DataFrame(a4).claim_holds.all()),
               "a6_smallest_mae_p": float(pmin),
               "a6_smallest_mae_p_where": f"{arg.level} {arg.tissue}",
               "a6_all_gt_0.10": bool((df5.p_mae_asymptotic > 0.10).all())},
              (OUT / "verify_summary.json").open("w"), indent=2)
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
