#!/usr/bin/env python3
"""Recompute every segmentation-only quantity from checkpoint_best.

Background. Table 2 and everything derived from "the proposed method applied to the
manual reference section" was read from nnU-Net's training-time validation export,
which is produced with the FINAL (epoch 500) weights because nnUNetv2_train was run
without --val_best. Every other analysis in the paper uses checkpoint_best (epoch 412
for fold 0). Verified by re-running inference with both checkpoints on five fold-0
validation cases: the validation export differs from checkpoint_final by 15 voxels in
1.31 M and from checkpoint_best by 2536.

The offset experiment's offset-0 condition is exactly the right replacement: the GT
reference section as input, the fold-matched checkpoint_best, one dedicated
nnUNetv2_predict run. Verified fold-matched (checkpoint fold == row fold for all 3000
rows, every case in its own fold's validation split).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TISSUES = ["SMA", "VFAT", "SFAT", "IMAT"]


def sd(v): return float(np.std(v, ddof=1))


def load_offset0() -> pd.DataFrame:
    off = pd.read_csv(ROOT / "analysis/offset/statistics/per_case_gt_offset_metrics.csv",
                      dtype={"case_id": str})
    o0 = off[off.offset_slice == 0].copy()
    assert len(o0) == 600, len(o0)
    rows = []
    for r in o0.itertuples(index=False):
        for t in TISSUES:
            k = t.lower()
            rows.append({"case_id": r.case_id, "fold": r.fold, "level": r.branch, "tissue": t,
                         "dsc": getattr(r, f"central_gt_dsc_{k}"),
                         "gt_csa_cm2": getattr(r, f"gt_{k}_cm2"),
                         "seg_pred_csa_cm2": getattr(r, f"pred_{k}_cm2"),
                         "seg_error_cm2": getattr(r, f"bias_{k}_cm2"),
                         "seg_abs_error_cm2": getattr(r, f"abs_error_{k}_cm2")})
    return pd.DataFrame(rows)


def table2(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for level in ("inferior", "middle"):
        for t in TISSUES:
            s = df[(df.level == level) & (df.tissue == t)]
            r, _ = pearsonr(s.gt_csa_cm2, s.seg_pred_csa_cm2)
            out.append({"level": level, "tissue": t, "n": len(s),
                        "dsc_mean": s.dsc.mean(), "dsc_sd": sd(s.dsc),
                        "gt_mean_cm2": s.gt_csa_cm2.mean(), "gt_sd_cm2": sd(s.gt_csa_cm2),
                        "proposed_mean_cm2": s.seg_pred_csa_cm2.mean(),
                        "proposed_sd_cm2": sd(s.seg_pred_csa_cm2),
                        "error_mean_cm2": s.seg_error_cm2.mean(), "error_sd_cm2": sd(s.seg_error_cm2),
                        "mae_mean_cm2": s.seg_abs_error_cm2.mean(),
                        "mae_sd_cm2": sd(s.seg_abs_error_cm2), "pearson_r": r})
    return pd.DataFrame(out)


def table3(df: pd.DataFrame) -> pd.DataFrame:
    e2e = pd.read_csv(ROOT / "results_final/raw_csv/oof_body_composition.csv", dtype={"case_id": str})
    e2e = e2e[["case_id", "level", "tissue", "e2e_abs_error_cm2"]]
    m = df.merge(e2e, on=["case_id", "level", "tissue"], validate="one_to_one")
    assert len(m) == len(df), (len(m), len(df))
    m["paired_ae_change_cm2"] = m.e2e_abs_error_cm2 - m.seg_abs_error_cm2
    out = []
    for level in ("inferior", "middle"):
        for t in TISSUES:
            s = m[(m.level == level) & (m.tissue == t)]
            out.append({"level": level, "tissue": t, "n": len(s),
                        "segonly_mae_mean_cm2": s.seg_abs_error_cm2.mean(),
                        "segonly_mae_sd_cm2": sd(s.seg_abs_error_cm2),
                        "e2e_mae_mean_cm2": s.e2e_abs_error_cm2.mean(),
                        "e2e_mae_sd_cm2": sd(s.e2e_abs_error_cm2),
                        "paired_ae_change_mean_cm2": s.paired_ae_change_cm2.mean(),
                        "paired_ae_change_sd_cm2": sd(s.paired_ae_change_cm2),
                        "relative_increase_pct": 100 * (s.e2e_abs_error_cm2.mean() /
                                                        s.seg_abs_error_cm2.mean() - 1)})
    return pd.DataFrame(out), m


def bland_altman(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t in TISSUES:
        s = df[(df.level == "inferior") & (df.tissue == t)]
        b, s_ = s.seg_error_cm2.mean(), sd(s.seg_error_cm2)
        out.append({"tissue": t, "n": len(s), "bias_cm2": b, "sd_difference_cm2": s_,
                    "lower_loa_cm2": b - 1.96 * s_, "upper_loa_cm2": b + 1.96 * s_})
    return pd.DataFrame(out)


def main() -> None:
    df = load_offset0()
    df.to_csv(OUT / "best_weights_per_case_segonly.csv", index=False)
    t2, (t3, merged) = table2(df), table3(df)
    t2.to_csv(OUT / "best_weights_table2.csv", index=False)
    t3.to_csv(OUT / "best_weights_table3.csv", index=False)
    ba = bland_altman(df); ba.to_csv(OUT / "best_weights_bland_altman_inferior_segonly.csv", index=False)
    merged.to_csv(OUT / "best_weights_per_case_with_e2e.csv", index=False)

    old2 = pd.read_csv(ROOT / "results_final/summary_tables/table2.csv")
    old3 = pd.read_csv(ROOT / "results_final/summary_tables/table3.csv")
    print("=== TABLE 2  old (validation export / checkpoint_final) -> new (checkpoint_best) ===")
    for level in ("inferior", "middle"):
        print(f"-- {level}")
        for t in TISSUES:
            o = old2[(old2.level == level) & (old2.tissue == t)].iloc[0]
            n = t2[(t2.level == level) & (t2.tissue == t)].iloc[0]
            print(f"  {t:5} DSC {o.dsc_mean:.3f}±{o.dsc_sd:.3f} -> {n.dsc_mean:.3f}±{n.dsc_sd:.3f}"
                  f" | CSA {o.proposed_mean_cm2:7.2f}±{o.proposed_sd_cm2:5.2f} -> {n.proposed_mean_cm2:7.2f}±{n.proposed_sd_cm2:5.2f}"
                  f" | err {o.error_mean_cm2:+.2f}±{o.error_sd_cm2:.2f} -> {n.error_mean_cm2:+.2f}±{n.error_sd_cm2:.2f}"
                  f" | MAE {o.mae_mean_cm2:.2f}±{o.mae_sd_cm2:.2f} -> {n.mae_mean_cm2:.2f}±{n.mae_sd_cm2:.2f}"
                  f" | r {o.pearson_r:.3f} -> {n.pearson_r:.3f}")
    print("\n=== TABLE 3  seg-only and paired change ===")
    for level in ("inferior", "middle"):
        print(f"-- {level}")
        for t in TISSUES:
            o = old3[(old3.level == level) & (old3.tissue == t)].iloc[0]
            n = t3[(t3.level == level) & (t3.tissue == t)].iloc[0]
            print(f"  {t:5} seg {o.segonly_mae_mean_cm2:.2f}±{o.segonly_mae_sd_cm2:.2f} -> "
                  f"{n.segonly_mae_mean_cm2:.2f}±{n.segonly_mae_sd_cm2:.2f} | e2e {n.e2e_mae_mean_cm2:.2f}±{n.e2e_mae_sd_cm2:.2f}"
                  f" | change {o.paired_ae_change_mean_cm2:+.2f}±{o.paired_ae_change_sd_cm2:.2f} -> "
                  f"{n.paired_ae_change_mean_cm2:+.2f}±{n.paired_ae_change_sd_cm2:.2f}"
                  f" | +{o.relative_increase_pct:.1f}% -> +{n.relative_increase_pct:.1f}%")
    print("\n=== GT CSA unchanged? ===")
    for level in ("inferior", "middle"):
        for t in TISSUES:
            o = old2[(old2.level == level) & (old2.tissue == t)].iloc[0]
            n = t2[(t2.level == level) & (t2.tissue == t)].iloc[0]
            d = abs(o.gt_mean_cm2 - n.gt_mean_cm2)
            assert d < 1e-4, (level, t, d)
    print("  yes: max |delta| < 1e-4 cm2 for all 8 level/tissue pairs")
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
