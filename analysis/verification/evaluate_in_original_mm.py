#!/usr/bin/env python3
"""Evaluate every localisation condition on one common scale: original-CT physical mm.

Reviewer's point, adopted here. Previously the resampled runs compared a predicted index
on the resampled grid against the ground-truth index *mapped onto that grid*, so the
rounding of the ground truth (up to 2.5 mm on the 5 mm grid) was folded into the reported
error. Here the ground truth stays where it was annotated - the original volume - and each
prediction is converted to world z through the affine of the volume it came from. Nothing
is rounded, and the four conditions become directly comparable.

Conditions
  native_e2e   CT at original spacing -> L3 nnU-Net -> YOLO            (the deployed pipeline)
  gtmask       manual L3 mask         -> YOLO                          (ablation, no nnU-Net)
  z5mm         CT resampled to 5 mm   -> L3 nnU-Net -> YOLO            (prior-protocol comparison)
  z3mm         CT resampled to 3 mm   -> L3 nnU-Net -> YOLO            (finer-grid comparison)
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import nibabel as nib, numpy as np, pandas as pd

ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent
BRANCHES = ("inferior", "middle")


def world_z(affine: np.ndarray, shape, z) -> np.ndarray:
    """World z of the slice centre, same convention as results_final/generate_final_results.py."""
    cx, cy = (np.asarray(shape[:2], dtype=float) - 1.0) / 2.0
    z = np.asarray(z, dtype=float)
    return affine[2, 0] * cx + affine[2, 1] * cy + affine[2, 2] * z + affine[2, 3]


def geometry(paths: pd.Series) -> dict[str, tuple[np.ndarray, tuple]]:
    out = {}
    for cid, p in paths.items():
        img = nib.load(str(p))
        out[cid] = (img.affine, img.shape)
    return out


def main() -> None:
    man = pd.read_csv(ROOT / "manifests/manifest_300_all_folds.csv", dtype={"case_id": str})
    man = man.set_index("case_id")
    orig = geometry(man.ct_path)
    gt_idx = {b: man[f"{b}_z_index"].astype(int).to_dict() for b in BRANCHES}
    gt_mm = {b: {c: world_z(*orig[c], gt_idx[b][c]) for c in orig} for b in BRANCHES}
    spacing = {c: float(abs(orig[c][0][2, 2])) for c in orig}

    rows = []

    # ---- native end-to-end ----
    e2e = pd.read_csv(ROOT / "results/evaluation/per_case_end_to_end.csv", dtype={"case_id": str})
    for r in e2e.itertuples():
        c, b = r.case_id, r.branch
        rows.append({"condition": "native_e2e", "case_id": c, "branch": b,
                     "pred_mm": world_z(*orig[c], r.pred_slice_index), "gt_mm": gt_mm[b][c],
                     "native_spacing": spacing[c]})

    # ---- ablation: manual L3 mask -> YOLO (no nnU-Net) ----
    gtm = pd.read_csv(ROOT / "results/detector_on_gt_mask/per_case_detector_on_gt_mask.csv",
                      dtype={"case_id": str})
    for r in gtm.itertuples():
        c, b = r.case_id, r.branch
        rows.append({"condition": "gtmask", "case_id": c, "branch": b,
                     "pred_mm": world_z(*orig[c], r.gtmask_pred_slice_index), "gt_mm": gt_mm[b][c],
                     "native_spacing": spacing[c]})

    # ---- resampled grids ----
    for tree, tag in (("z5mm", "z5mm"), ("z3mm", "z3mm")):
        mpath = ROOT / tree / f"manifests/{tree}_manifest_300.csv"
        if not mpath.is_file():
            continue
        rm = pd.read_csv(mpath, dtype={"case_id": str}).set_index("case_id")
        geo = geometry(rm[f"ct_{tree}_path"])
        det = pd.concat([pd.read_csv(ROOT / tree / f"results/fold_{k}/detection_per_case.csv",
                                     dtype={"case_id": str}) for k in range(5)], ignore_index=True)
        col = f"pred_slice_index_{tree}"
        for r in det.itertuples():
            c, b = r.case_id, r.branch
            rows.append({"condition": tag, "case_id": c, "branch": b,
                         "pred_mm": world_z(*geo[c], getattr(r, col)), "gt_mm": gt_mm[b][c],
                         "native_spacing": spacing[c]})

    df = pd.DataFrame(rows)
    df["signed_mm"] = df.pred_mm - df.gt_mm
    df["abs_mm"] = df.signed_mm.abs()
    # section tolerance expressed in ORIGINAL slices, so the conditions stay comparable
    df["within_1_native_slice"] = df.abs_mm <= df.native_spacing + 1e-6
    df["within_2_native_slices"] = df.abs_mm <= 2 * df.native_spacing + 1e-6
    df.to_csv(OUT / "localisation_original_mm_per_case.csv", index=False)

    def block(sub: pd.DataFrame) -> dict:
        return {"n": len(sub),
                "signed_mean": sub.signed_mm.mean(), "signed_sd": sub.signed_mm.std(ddof=1),
                "abs_mean": sub.abs_mm.mean(), "abs_sd": sub.abs_mm.std(ddof=1),
                "abs_median": sub.abs_mm.median(),
                "abs_q1": sub.abs_mm.quantile(.25), "abs_q3": sub.abs_mm.quantile(.75),
                "exact_native": float((sub.abs_mm < 1e-6).mean() * 100),
                "within_1_slice_pct": float(sub.within_1_native_slice.mean() * 100),
                "within_2_slices_pct": float(sub.within_2_native_slices.mean() * 100),
                "within_5mm_pct": float((sub.abs_mm <= 5).mean() * 100),
                "within_10mm_pct": float((sub.abs_mm <= 10).mean() * 100)}

    summary = []
    for cond in df.condition.unique():
        for b in BRANCHES:
            summary.append({"condition": cond, "branch": b,
                            **block(df[(df.condition == cond) & (df.branch == b)])})
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "localisation_original_mm_summary.csv", index=False)

    strat = []
    for cond in df.condition.unique():
        for b in BRANCHES:
            sub = df[(df.condition == cond) & (df.branch == b)]
            for sp, g in sub.groupby("native_spacing"):
                strat.append({"condition": cond, "branch": b, "native_spacing_mm": sp,
                              **block(g)})
    pd.DataFrame(strat).to_csv(OUT / "localisation_original_mm_by_thickness.csv", index=False)
    print(f"[OK] {len(df)} rows, {df.condition.nunique()} conditions -> {OUT}")


if __name__ == "__main__":
    main()
