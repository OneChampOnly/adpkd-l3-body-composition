#!/usr/bin/env python3
"""Compare localisation error on the native grid, the 5 mm grid and the 3 mm grid.

3 mm is nnU-Net's planned target z-spacing for the L3 segmenter; 5 mm was the cohort's
modal spacing. Both normalised runs resample the volume, re-run L3 segmentation and
re-run detection, so each is a measurement on its own grid rather than a rescaling.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent


def load_summary(tree: str) -> dict:
    p = ROOT / tree / "results/summary/overall_summary.json"
    if not p.is_file():
        return {}
    return {r["branch"]: r for r in json.loads(p.read_text())["overall"]}


def quantisation(target: float) -> dict[str, dict[str, float]]:
    m = pd.read_csv(ROOT / "z5mm/manifests/z5mm_manifest_300.csv", dtype={"case_id": str})
    sp = m.original_spacing_z_mm.to_numpy()
    res = ~np.isclose(sp, target, atol=0.05)
    out = {}
    for br in ("inferior", "middle"):
        z = m[f"original_{br}_gt_idx"].to_numpy() * sp
        r = np.where(res, np.abs(z - np.round(z / target) * target), 0.0)
        out[br] = {"n_resampled": int(res.sum()), "mean": float(r.mean()),
                   "max": float(r.max()), "n_over_1mm": int((r > 1).sum())}
    return out


def main() -> None:
    native = pd.read_csv(ROOT / "results_final/raw_csv/oof_localization.csv", dtype={"case_id": str})
    nat = {"inferior": (native.inferior_abs_error_mm.mean(), native.inferior_abs_error_mm.std(ddof=1)),
           "middle": (native.middle_abs_error_mm.mean(), native.middle_abs_error_mm.std(ddof=1))}
    s5, s3 = load_summary("z5mm"), load_summary("z3mm")
    q5, q3 = quantisation(5.0), quantisation(3.0)

    print("=" * 86)
    print("ABSOLUTE LOCALISATION ERROR (mm), 300 cases")
    print("=" * 86)
    print(f"{'':10} {'native (per-scan)':>22} {'5 mm grid':>20} {'3 mm grid':>20}")
    for br in ("inferior", "middle"):
        n = f"{nat[br][0]:.2f} ± {nat[br][1]:.2f}"
        a = f"{s5[br]['absolute_error_mm_mean']:.2f} ± {s5[br]['absolute_error_mm_sd']:.2f}" if s5 else "—"
        b = f"{s3[br]['absolute_error_mm_mean']:.2f} ± {s3[br]['absolute_error_mm_sd']:.2f}" if s3 else "—"
        print(f"{br:10} {n:>22} {a:>20} {b:>20}")

    print()
    print(f"{'':10} {'':22} {'5 mm grid':>20} {'3 mm grid':>20}")
    for label, key in (("exact section", "exact_slice_percent"),
                       ("within 1 section", "within_1_slice_percent"),
                       ("within 5 mm", "within_5mm_percent"),
                       ("within 10 mm", "within_10mm_percent")):
        for br in ("inferior", "middle"):
            a = f"{s5[br][key]:.1f}%" if s5 else "—"
            b = f"{s3[br][key]:.1f}%" if s3 else "—"
            print(f"{label:10} {br:>22} {a:>20} {b:>20}")

    print()
    print("=" * 86)
    print("QUANTISATION COST OF THE GRID (before any model is applied)")
    print("=" * 86)
    print(f"{'':10} {'resampled':>12} {'mean |disp|':>13} {'max |disp|':>12} {'cases >1 mm':>13}")
    for name, q in (("5 mm", q5), ("3 mm", q3)):
        for br in ("inferior", "middle"):
            d = q[br]
            print(f"{name:4} {br:6} {d['n_resampled']:>11d}/300 {d['mean']:>12.3f} "
                  f"{d['max']:>12.3f} {d['n_over_1mm']:>13d}")

    rows = []
    for br in ("inferior", "middle"):
        rows.append({"branch": br,
                     "native_mean": nat[br][0], "native_sd": nat[br][1],
                     "z5_mean": s5[br]["absolute_error_mm_mean"] if s5 else None,
                     "z5_sd": s5[br]["absolute_error_mm_sd"] if s5 else None,
                     "z5_exact_pct": s5[br]["exact_slice_percent"] if s5 else None,
                     "z3_mean": s3[br]["absolute_error_mm_mean"] if s3 else None,
                     "z3_sd": s3[br]["absolute_error_mm_sd"] if s3 else None,
                     "z3_exact_pct": s3[br]["exact_slice_percent"] if s3 else None,
                     "z5_resampled": q5[br]["n_resampled"], "z5_max_disp": q5[br]["max"],
                     "z5_over_1mm": q5[br]["n_over_1mm"],
                     "z3_resampled": q3[br]["n_resampled"], "z3_max_disp": q3[br]["max"],
                     "z3_over_1mm": q3[br]["n_over_1mm"]})
    pd.DataFrame(rows).to_csv(OUT / "grid_comparison.csv", index=False)
    print(f"\n[OK] {OUT / 'grid_comparison.csv'}")


if __name__ == "__main__":
    main()
