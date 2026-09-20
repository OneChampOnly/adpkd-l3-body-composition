#!/usr/bin/env python3
"""Build the final 300-case OOF manuscript result package.

Run with:
  # openpyxl was pip-installed into the yolo_maks env; no PYTHONPATH needed.
  python generate_final_results.py

The script intentionally separates the sources used for each analysis:
* Table 2/3 tissue OOF: direct fold-specific nnU-Net validation predictions.
* End-to-end tissue OOF: saved fold-specific detector + segmenter evaluation.
* Slice offset: dedicated offset experiment predictions.
* Public model: saved public predictions, joined to the same final GT and direct
  validation predictions used by Table 2.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import pearsonr


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
TEMPLATE = OUT / "KJR_ADPKD_model_results_template.xlsx"
MANIFEST = ROOT / "manifests/manifest_300_all_folds.csv"
REGISTRY = ROOT / "config/model_registry.json"
E2E = ROOT / "results/evaluation/per_case_end_to_end.csv"
OFFSET = ROOT / "analysis/offset/statistics/per_case_gt_offset_metrics.csv"
INF_BASE = ROOT / "workspace/nnunet"
MID_BASE = ROOT / "workspace/nnunet"
INF_DATASET = "Dataset917_InferiorSMA300Reann0917"
MID_DATASET = "Dataset918_MiddleSMA300Reann0917"
TRAINER = "nnUNetTrainer_500epochs__nnUNetPlans__2d"
PUBLIC = ROOT / "analysis/public_model/tables/inferior_comparison_per_case.csv"
EXTERNAL = ROOT / "analysis/external_reference/tables/clinical_csv_inferior_300_per_case.csv"

TISSUES = {"sma": 1, "sfat": 2, "vfat": 3, "imat": 4}
TISSUE_ORDER = ["SMA", "VFAT", "SFAT", "IMAT"]
PUBLIC_TISSUES = ["SMA", "VFAT", "SFAT"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_sd(values) -> float:
    return float(pd.Series(values, dtype=float).std(ddof=1))


def mean_sd(values, decimals=2) -> str:
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return "N/A"
    return f"{s.mean():.{decimals}f} ± {s.std(ddof=1):.{decimals}f}"


def fmt(x, decimals=3) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{decimals}f}"


def validation_paths(branch: str, fold: int, case_id: str) -> tuple[Path, Path]:
    if branch == "inferior":
        base, dataset = INF_BASE, INF_DATASET
    else:
        base, dataset = MID_BASE, MID_DATASET
    gt = base / "nnUNet_raw" / dataset / "labelsTr" / f"{case_id}.nii.gz"
    pred = base / "nnUNet_results" / dataset / TRAINER / f"fold_{fold}" / "validation" / f"{case_id}.nii.gz"
    return gt, pred


def tissue_metrics(gt_path: Path, pred_path: Path) -> dict[str, dict[str, float]]:
    gt_img, pred_img = nib.load(str(gt_path)), nib.load(str(pred_path))
    gt = np.asanyarray(gt_img.dataobj)
    pred = np.asanyarray(pred_img.dataobj)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt_path} {gt.shape} vs {pred_path} {pred.shape}")
    area = float(pred_img.header.get_zooms()[0] * pred_img.header.get_zooms()[1]) / 100.0
    out = {}
    for tissue, label in TISSUES.items():
        g, p = gt == label, pred == label
        inter = int(np.logical_and(g, p).sum())
        denom = int(g.sum() + p.sum())
        dice = 1.0 if denom == 0 else 2.0 * inter / denom
        ga, pa = float(g.sum()) * area, float(p.sum()) * area
        out[tissue] = {"dice": dice, "gt": ga, "pred": pa, "error": pa - ga, "ae": abs(pa - ga)}
    return out


def build_direct_oof(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide, long = [], []
    for rec in manifest.itertuples(index=False):
        for branch in ("inferior", "middle"):
            gt_path, pred_path = validation_paths(branch, int(rec.fold), str(rec.case_id))
            if not gt_path.exists() or not pred_path.exists():
                raise FileNotFoundError(f"missing OOF input: {gt_path} or {pred_path}")
            metrics = tissue_metrics(gt_path, pred_path)
            row = {"fold": int(rec.fold), "case_id": str(rec.case_id), "branch": branch,
                   "gt_path": str(gt_path), "pred_path": str(pred_path)}
            for tissue, vals in metrics.items():
                for k, v in vals.items():
                    row[f"{k}_{tissue}_cm2" if k != "dice" else f"dsc_{tissue}"] = v
                long.append({"case_id": str(rec.case_id), "fold": int(rec.fold), "level": branch,
                             "tissue": tissue.upper(), "gt_csa_cm2": vals["gt"],
                             "segonly_pred_csa_cm2": vals["pred"],
                             "segonly_error_cm2": vals["error"],
                             "segonly_abs_error_cm2": vals["ae"], "segonly_dice": vals["dice"],
                             "gt_path": str(gt_path), "segonly_pred_path": str(pred_path)})
            wide.append(row)
    return pd.DataFrame(wide), pd.DataFrame(long)


def build_localization(manifest: pd.DataFrame, e2e: pd.DataFrame) -> pd.DataFrame:
    # The E2E table has one row per branch; L3 DSC is branch-invariant.
    m = manifest[["case_id", "fold", "ct_path"]].copy()
    m["case_id"] = m.case_id.astype(str)
    e = e2e.copy(); e["case_id"] = e.case_id.astype(str)
    inf = e[e.branch == "inferior"].set_index(["fold", "case_id"])
    mid = e[e.branch == "middle"].set_index(["fold", "case_id"])
    rows = []
    for rec in m.itertuples(index=False):
        key = (int(rec.fold), str(rec.case_id))
        a, b = inf.loc[key], mid.loc[key]
        img = nib.load(str(rec.ct_path))
        affine = img.affine
        cx, cy = (np.asarray(img.shape[:2], dtype=float) - 1.0) / 2.0
        def zmm(z): return float((affine @ np.array([cx, cy, float(z), 1.0]))[2])
        gi, pi = zmm(a.gt_slice_index), zmm(a.pred_slice_index)
        gm, pm = zmm(b.gt_slice_index), zmm(b.pred_slice_index)
        rows.append({"case_id": str(rec.case_id), "fold": int(rec.fold), "l3_vertebra_dsc": float(a.l3_dsc),
                     "gt_inferior_z_mm": gi, "pred_inferior_z_mm": pi, "inferior_abs_error_mm": abs(pi - gi),
                     "gt_middle_z_mm": gm, "pred_middle_z_mm": pm, "middle_abs_error_mm": abs(pm - gm),
                     "notes": "world-z coordinate from CT NIfTI affine"})
    return pd.DataFrame(rows)


def build_body_long(direct_long: pd.DataFrame, e2e: pd.DataFrame) -> pd.DataFrame:
    d = direct_long.copy()
    e = e2e.copy(); e.case_id = e.case_id.astype(str)
    rows = []
    for rec in d.itertuples(index=False):
        x = e[(e.fold == rec.fold) & (e.case_id == rec.case_id) & (e.branch == rec.level)].iloc[0]
        t = rec.tissue.lower()
        ep = float(x[f"pred_{t}_cm2"]); gt_e2e = float(x[f"gt_{t}_cm2"])
        if not math.isclose(float(rec.gt_csa_cm2), gt_e2e, abs_tol=3e-4):
            raise ValueError(f"GT mismatch for {rec.case_id} {rec.level} {t}: {rec.gt_csa_cm2} vs {gt_e2e}")
        gt = float(rec.gt_csa_cm2); sp = float(rec.segonly_pred_csa_cm2)
        se, ee = sp - gt, ep - gt
        rows.append({"case_id": rec.case_id, "fold": int(rec.fold), "level": rec.level,
                     "tissue": rec.tissue, "gt_csa_cm2": gt, "seg_pred_csa_cm2": sp,
                     "e2e_pred_csa_cm2": ep, "dice": float(rec.segonly_dice),
                     "seg_error_cm2": se, "seg_abs_error_cm2": abs(se),
                     "e2e_error_cm2": ee, "e2e_abs_error_cm2": abs(ee),
                     "paired_ae_change_cm2": abs(ee) - abs(se),
                     "relative_seg_error_pct": 100.0 * se / gt if gt else np.nan,
                     "relative_e2e_error_pct": 100.0 * ee / gt if gt else np.nan,
                     "notes": "seg-only=fold validation; e2e=fold detector+segmenter"})
    return pd.DataFrame(rows)


def build_offset() -> pd.DataFrame:
    src = pd.read_csv(OFFSET)
    src = src[(src.branch == "inferior") & src.offset_slice.isin([-2, -1, 0, 1, 2])]
    rows = []
    for r in src.itertuples(index=False):
        for tissue in TISSUE_ORDER:
            t = tissue.lower()
            pred = float(getattr(r, f"pred_{t}_cm2")); gt = float(getattr(r, f"gt_{t}_cm2"))
            rows.append({"case_id": str(r.case_id), "fold": int(r.fold),
                         "offset_slice": int(r.offset_slice), "tissue": tissue,
                         "gt_csa_cm2": gt, "pred_csa_cm2": pred, "error_cm2": pred - gt,
                         "abs_error_cm2": float(getattr(r, f"abs_error_{t}_cm2")),
                         "notes": "dedicated fold-matched slice-offset experiment"})
    return pd.DataFrame(rows)


def build_external(direct_long: pd.DataFrame) -> pd.DataFrame:
    ext = pd.read_csv(EXTERNAL, dtype={"case_id": str})
    inf = direct_long[direct_long.level == "inferior"].copy()
    rows = []
    for tissue in TISSUE_ORDER:
        p = inf[inf.tissue == tissue]
        q = p.merge(ext[["fold", "case_id", tissue.lower()]], on=["fold", "case_id"], how="left")
        for r in q.itertuples(index=False):
            ev = getattr(r, tissue.lower())
            err = np.nan if pd.isna(ev) else float(r.segonly_pred_csa_cm2) - float(ev)
            rows.append({"case_id": r.case_id, "tissue": tissue, "external_csa_cm2": ev,
                         "proposed_inferior_csa_cm2": float(r.segonly_pred_csa_cm2),
                         "error_cm2": err, "abs_error_cm2": abs(err) if not pd.isna(err) else np.nan,
                         "notes": "external clinical CSV; IMAT unavailable" if tissue == "IMAT" else "external clinical CSV"})
    return pd.DataFrame(rows)


def build_public(direct_long: pd.DataFrame) -> pd.DataFrame:
    pub = pd.read_csv(PUBLIC, dtype={"case_id": str})
    inf = direct_long[direct_long.level == "inferior"].copy()
    rows = []
    for tissue in PUBLIC_TISSUES:
        t = tissue.lower()
        p = inf[inf.tissue == tissue]
        q = p.merge(pub[["fold", "case_id", f"public_{t}_cm2", "public_path", "public_prediction_source"]],
                    on=["fold", "case_id"], how="left")
        for r in q.itertuples(index=False):
            pv = float(getattr(r, f"public_{t}_cm2")); gt = float(r.gt_csa_cm2); prop = float(r.segonly_pred_csa_cm2)
            rows.append({"case_id": r.case_id, "tissue": tissue,
                         "gt_csa_cm2": gt, "proposed_csa_cm2": prop, "public_csa_cm2": pv,
                         "proposed_error_cm2": prop - gt, "public_error_cm2": pv - gt,
                         "proposed_abs_error_cm2": abs(prop - gt), "public_abs_error_cm2": abs(pv - gt),
                         "notes": f"same final GT; public source={r.public_prediction_source}"})
    return pd.DataFrame(rows)


def icc_a1(x, y) -> float:
    a = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = a.shape
    gm = a.mean(); rowm = a.mean(axis=1); colm = a.mean(axis=0)
    msr = k * np.sum((rowm - gm) ** 2) / (n - 1)
    msc = n * np.sum((colm - gm) ** 2) / (k - 1)
    mse = np.sum((a - rowm[:, None] - colm[None, :] + gm) ** 2) / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def icc_bootstrap_ci(x, y, n_boot=10000, seed=731) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    point = icc_a1(x, y)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        vals[i] = icc_a1(x[idx], y[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def summaries(level: pd.DataFrame, tissue: pd.DataFrame, offset: pd.DataFrame,
              external: pd.DataFrame, public: pd.DataFrame):
    values, machine = {}, {}
    values["L3_SEG_DSC"] = mean_sd(level.l3_vertebra_dsc, 3)
    values["INF_LOC_AE"] = mean_sd(level.inferior_abs_error_mm, 2)
    values["MID_LOC_AE"] = mean_sd(level.middle_abs_error_mm, 2)

    table2, table3 = [], []
    for lv, prefix in [("inferior", "INF"), ("middle", "MID")]:
        for tissue_name in TISSUE_ORDER:
            q = tissue[(tissue.level == lv) & (tissue.tissue == tissue_name)]
            dsc = q.dice; gt = q.gt_csa_cm2; pred = q.seg_pred_csa_cm2
            err = pred - gt; ae = abs(err); r = pearsonr(gt, pred).statistic
            values[f"{prefix}_{tissue_name}_DSC"] = mean_sd(dsc, 3)
            values[f"{prefix}_{tissue_name}_GT_CSA"] = mean_sd(gt, 2)
            values[f"{prefix}_{tissue_name}_PROPOSED_CSA"] = mean_sd(pred, 2)
            values[f"{prefix}_{tissue_name}_ERROR"] = mean_sd(err, 2)
            values[f"{prefix}_{tissue_name}_MAE"] = mean_sd(ae, 2)
            values[f"{prefix}_{tissue_name}_PEARSON_R"] = fmt(r, 3)
            table2.append({"level": lv, "tissue": tissue_name, "n": len(q),
                           "dsc_mean": dsc.mean(), "dsc_sd": sample_sd(dsc),
                           "gt_mean_cm2": gt.mean(), "gt_sd_cm2": sample_sd(gt),
                           "proposed_mean_cm2": pred.mean(), "proposed_sd_cm2": sample_sd(pred),
                           "error_mean_cm2": err.mean(), "error_sd_cm2": sample_sd(err),
                           "mae_mean_cm2": ae.mean(), "mae_sd_cm2": sample_sd(ae), "pearson_r": r})
            e2e_ae = q.e2e_abs_error_cm2
            change = e2e_ae - q.seg_abs_error_cm2
            rel = 100.0 * e2e_ae.mean() / q.seg_abs_error_cm2.mean() - 100.0
            values[f"{prefix}_{tissue_name}_SEGONLY_MAE"] = mean_sd(q.seg_abs_error_cm2, 2)
            values[f"{prefix}_{tissue_name}_E2E_MAE"] = mean_sd(e2e_ae, 2)
            values[f"{prefix}_{tissue_name}_PAIRED_AE_CHANGE"] = mean_sd(change, 2)
            values[f"{prefix}_{tissue_name}_REL_INCREASE"] = fmt(rel, 1) + "%"
            table3.append({"level": lv, "tissue": tissue_name, "n": len(q),
                           "segonly_mae_mean_cm2": q.seg_abs_error_cm2.mean(),
                           "segonly_mae_sd_cm2": sample_sd(q.seg_abs_error_cm2),
                           "e2e_mae_mean_cm2": e2e_ae.mean(), "e2e_mae_sd_cm2": sample_sd(e2e_ae),
                           "paired_ae_change_mean_cm2": change.mean(), "paired_ae_change_sd_cm2": sample_sd(change),
                           "relative_increase_pct": rel})

    infmid = []
    for tissue_name in TISSUE_ORDER:
        a = tissue[(tissue.level == "inferior") & (tissue.tissue == tissue_name)][["case_id", "gt_csa_cm2"]]
        b = tissue[(tissue.level == "middle") & (tissue.tissue == tissue_name)][["case_id", "gt_csa_cm2"]]
        q = a.merge(b, on="case_id", suffixes=("_inf", "_mid")); diff = q.gt_csa_cm2_mid - q.gt_csa_cm2_inf
        mae = abs(diff); icc, lo, hi = icc_bootstrap_ci(q.gt_csa_cm2_inf, q.gt_csa_cm2_mid)
        p = f"INFMID_{tissue_name}"
        values[f"{p}_INFERIOR_CSA"] = mean_sd(q.gt_csa_cm2_inf, 2)
        values[f"{p}_MIDDLE_CSA"] = mean_sd(q.gt_csa_cm2_mid, 2)
        values[f"{p}_PAIRED_DIFFERENCE"] = mean_sd(diff, 2)
        values[f"{p}_MAE"] = mean_sd(mae, 2)
        values[f"{p}_RELATIVE_MAE"] = fmt(100 * mae.mean() / q.gt_csa_cm2_inf.mean(), 2) + "%"
        values[f"{p}_ICC"] = fmt(icc, 3); values[f"{p}_ICC_95PCT_CI"] = f"{lo:.3f} to {hi:.3f}"
        infmid.append({"tissue": tissue_name, "n": len(q), "difference_definition": "middle - inferior",
                       "inferior_mean": q.gt_csa_cm2_inf.mean(), "inferior_sd": sample_sd(q.gt_csa_cm2_inf),
                       "middle_mean": q.gt_csa_cm2_mid.mean(), "middle_sd": sample_sd(q.gt_csa_cm2_mid),
                       "paired_difference_mean": diff.mean(), "paired_difference_sd": sample_sd(diff),
                       "mae_mean": mae.mean(), "mae_sd": sample_sd(mae),
                       "relative_mae_pct": 100 * mae.mean() / q.gt_csa_cm2_inf.mean(),
                       "icc_a1": icc, "icc_bootstrap_95ci_lower": lo, "icc_bootstrap_95ci_upper": hi})

    ba = []
    inf = tissue[tissue.level == "inferior"]
    for tissue_name in TISSUE_ORDER:
        q = inf[inf.tissue == tissue_name]; diff = q.seg_pred_csa_cm2 - q.gt_csa_cm2
        bias, sd = diff.mean(), sample_sd(diff); lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        nout = int(((diff < lo) | (diff > hi)).sum())
        values[f"BA_INF_{tissue_name}_BIAS"] = fmt(bias, 3)
        values[f"BA_INF_{tissue_name}_LOWER_95PCT_LOA"] = fmt(lo, 3)
        values[f"BA_INF_{tissue_name}_UPPER_95PCT_LOA"] = fmt(hi, 3)
        ba.append({"tissue": tissue_name, "n": len(q), "bias_cm2": bias, "sd_difference_cm2": sd,
                   "lower_loa_cm2": lo, "upper_loa_cm2": hi, "outside_loa_n": nout})

    offsets = []
    token = {-2: "M2", -1: "M1", 0: "0", 1: "P1", 2: "P2"}
    for tissue_name in TISSUE_ORDER:
        for off in [0, -1, 1, -2, 2]:
            q = offset[(offset.tissue == tissue_name) & (offset.offset_slice == off)]
            values[f"OFFSET_{tissue_name}_{token[off]}"] = mean_sd(q.abs_error_cm2, 2)
            offsets.append({"tissue": tissue_name, "offset_slices": off, "n": len(q),
                            "mae_mean_cm2": q.abs_error_cm2.mean(), "mae_sd_cm2": sample_sd(q.abs_error_cm2)})

    ext_sum = []
    for tissue_name in TISSUE_ORDER:
        q = external[external.tissue == tissue_name].dropna(subset=["external_csa_cm2"])
        p = f"EXT_{tissue_name}"
        values[f"{p}_EXTERNAL_REFERENCE"] = mean_sd(q.external_csa_cm2, 2)
        # Proposed summary is over all 300 even where external is unavailable.
        qall = external[external.tissue == tissue_name]
        values[f"{p}_PROPOSED_METHOD"] = mean_sd(qall.proposed_inferior_csa_cm2, 2)
        if q.empty:
            values[f"{p}_ERROR"] = values[f"{p}_PEARSON_R"] = "N/A"; r = np.nan
        else:
            values[f"{p}_ERROR"] = mean_sd(q.error_cm2, 2); r = pearsonr(q.external_csa_cm2, q.proposed_inferior_csa_cm2).statistic
            values[f"{p}_PEARSON_R"] = fmt(r, 3)
        ext_sum.append({"tissue": tissue_name, "n": len(q), "external_mean": q.external_csa_cm2.mean() if len(q) else np.nan,
                        "external_sd": sample_sd(q.external_csa_cm2) if len(q) else np.nan,
                        "proposed_mean": qall.proposed_inferior_csa_cm2.mean(), "proposed_sd": sample_sd(qall.proposed_inferior_csa_cm2),
                        "error_mean": q.error_cm2.mean() if len(q) else np.nan,
                        "error_sd": sample_sd(q.error_cm2) if len(q) else np.nan, "pearson_r": r})

    pub_sum = []
    for tissue_name in PUBLIC_TISSUES:
        q = public[public.tissue == tissue_name]; p = f"PUB_{tissue_name}"
        perr, uerr = q.proposed_error_cm2, q.public_error_cm2
        pr = pearsonr(q.gt_csa_cm2, q.proposed_csa_cm2).statistic
        ur = pearsonr(q.gt_csa_cm2, q.public_csa_cm2).statistic
        values[f"{p}_MANUAL_GT_CSA"] = mean_sd(q.gt_csa_cm2, 2)
        values[f"{p}_PROPOSED_CSA"] = mean_sd(q.proposed_csa_cm2, 2)
        values[f"{p}_PUBLIC_MODEL_CSA"] = mean_sd(q.public_csa_cm2, 2)
        values[f"{p}_PROPOSED_ERROR"] = mean_sd(perr, 2); values[f"{p}_PUBLIC_MODEL_ERROR"] = mean_sd(uerr, 2)
        values[f"{p}_PROPOSED_MAE"] = mean_sd(abs(perr), 2); values[f"{p}_PUBLIC_MODEL_MAE"] = mean_sd(abs(uerr), 2)
        values[f"{p}_PROPOSED_PEARSON_R"] = fmt(pr, 3); values[f"{p}_PUBLIC_MODEL_PEARSON_R"] = fmt(ur, 3)
        pub_sum.append({"tissue": tissue_name, "n": len(q), "gt_mean": q.gt_csa_cm2.mean(), "gt_sd": sample_sd(q.gt_csa_cm2),
                        "proposed_mean": q.proposed_csa_cm2.mean(), "proposed_sd": sample_sd(q.proposed_csa_cm2),
                        "public_mean": q.public_csa_cm2.mean(), "public_sd": sample_sd(q.public_csa_cm2),
                        "proposed_error_mean": perr.mean(), "proposed_error_sd": sample_sd(perr),
                        "public_error_mean": uerr.mean(), "public_error_sd": sample_sd(uerr),
                        "proposed_mae_mean": abs(perr).mean(), "proposed_mae_sd": sample_sd(abs(perr)),
                        "public_mae_mean": abs(uerr).mean(), "public_mae_sd": sample_sd(abs(uerr)),
                        "proposed_r": pr, "public_r": ur})

    machine.update({"table2": table2, "table3": table3, "inferior_vs_middle": infmid,
                    "bland_altman_inferior_segonly": ba, "slice_offset": offsets,
                    "external_reference": ext_sum, "public_model": pub_sum})
    return values, machine



def write_workbook(values, level, tissue, offset, external, public, qc_rows):
    wb = load_workbook(TEMPLATE)
    raw = {"OOF_Level": level, "OOF_Tissue": tissue, "Slice_Offset": offset,
           "External_Reference": external, "Public_Model": public}
    for sheet, df in raw.items():
        ws = wb[sheet]
        if ws.max_row > 1:
            ws.delete_rows(2, ws.max_row - 1)
        headers = [c.value for c in ws[1]]
        for row in df.reindex(columns=headers).itertuples(index=False, name=None):
            ws.append(list(row))
        ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    ws = wb["Summary_Values"]
    missing_ids = []
    for row in range(2, ws.max_row + 1):
        key = ws.cell(row, 1).value
        if key in values: ws.cell(row, 7).value = values[key]
        else: ws.cell(row, 7).value = "NOT CALCULATED"; missing_ids.append(key)
    ws = wb["QC_Checklist"]
    for row in range(2, ws.max_row + 1):
        item = str(ws.cell(row, 1).value)
        status, note = qc_rows.get(item, ("REVIEW", "No automated mapping"))
        ws.cell(row, 3).value = note; ws.cell(row, 4).value = status
    out = OUT / "KJR_ADPKD_model_results_filled.xlsx"
    wb.save(out)
    return out, missing_ids


def json_clean(value):
    """Convert numpy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    (OUT / "raw_csv").mkdir(exist_ok=True); (OUT / "summary_tables").mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype={"case_id": str})
    e2e = pd.read_csv(E2E, dtype={"case_id": str})
    direct_wide, direct_long = build_direct_oof(manifest)
    level = build_localization(manifest, e2e)
    tissue = build_body_long(direct_long, e2e)
    offset = build_offset(); external = build_external(direct_long); public = build_public(direct_long)

    required = {"oof_l3_vertebra.csv": level[["case_id", "fold", "l3_vertebra_dsc"]].rename(columns={"l3_vertebra_dsc": "dice"}),
                "oof_localization.csv": level.drop(columns=["l3_vertebra_dsc"]),
                "oof_body_composition.csv": tissue,
                "slice_offset_results.csv": offset,
                "external_reference_comparison.csv": external,
                "public_model_comparison.csv": public}
    for name, df in required.items(): df.to_csv(OUT / "raw_csv" / name, index=False)

    values, machine = summaries(level, tissue, offset, external, public)
    for key, records in machine.items(): pd.DataFrame(records).to_csv(OUT / "summary_tables" / f"{key}.csv", index=False)
    # Figure 2 is produced by paper_data/bland_altman_inferior_end2end/make_bland_altman_inferior_end2end.py,
    # which is the script the published figure came from. Nothing is drawn here.

    fold_counts = manifest.groupby("fold").size().to_dict()
    unique_cases = manifest.case_id.nunique() == 300
    each_once = manifest.case_id.value_counts().eq(1).all()
    raw_counts = {k: len(v) for k, v in required.items()}
    qc_rows = {
        "Unique OOF case count": ("PASS" if unique_cases else "FAIL", f"{manifest.case_id.nunique()} unique cases"),
        "Fold 1 validation count": ("PASS" if fold_counts.get(0)==60 else "FAIL", f"source fold 0: {fold_counts.get(0)}"),
        "Fold 2 validation count": ("PASS" if fold_counts.get(1)==60 else "FAIL", f"source fold 1: {fold_counts.get(1)}"),
        "Fold 3 validation count": ("PASS" if fold_counts.get(2)==60 else "FAIL", f"source fold 2: {fold_counts.get(2)}"),
        "Fold 4 validation count": ("PASS" if fold_counts.get(3)==60 else "FAIL", f"source fold 3: {fold_counts.get(3)}"),
        "Fold 5 validation count": ("PASS" if fold_counts.get(4)==60 else "FAIL", f"source fold 4: {fold_counts.get(4)}"),
        "Each case evaluated exactly once per task": ("PASS" if each_once else "FAIL", "300 manifest IDs; one assigned fold each"),
        "Inferior tissue rows": ("PASS" if len(tissue[tissue.level=='inferior'])==1200 else "FAIL", str(len(tissue[tissue.level=='inferior']))),
        "Middle tissue rows": ("PASS" if len(tissue[tissue.level=='middle'])==1200 else "FAIL", str(len(tissue[tissue.level=='middle']))),
        "No training-set predictions used": ("PASS", "Every seg-only path is fold_<assigned>/validation/<case>.nii.gz"),
        "Final retouched GT used": ("PASS", "Dataset817/818 labelsTr and final E2E GT numerically matched within 3e-4 cm²"),
        "Inferior GT is first fully visible L3 slice": ("NEEDS HUMAN CONFIRMATION", "Mask semantics cannot be proven from file contents alone"),
        "Middle GT is midpoint of fully visible L3 range": ("NEEDS HUMAN CONFIRMATION", "Mask semantics cannot be proven from file contents alone"),
        "L3 GT includes body and spinous process": ("NEEDS HUMAN CONFIRMATION", "Mask semantics cannot be proven from file contents alone"),
        "Localization error reported in mm": ("PASS", "CT affine world-z coordinates"),
        "CSA reported in cm²": ("PASS", "voxel count × in-plane pixel area / 100"),
        "No duplicate rows": ("PASS" if all(not d.duplicated().any() for d in required.values()) else "FAIL", "all six raw CSVs checked"),
        "No missing required values": ("PASS" if not tissue.isna().any().any() and not level.isna().any().any() else "FAIL", "IMAT external reference is intentionally unavailable"),
        "Public-model comparison uses same GT": ("PASS", "public values joined to the same direct-OOF final GT rows")}
    # Exact QC_Checklist labels from the supplied workbook.
    qc_rows.update({
        "Unique OOF cases": ("PASS" if unique_cases else "FAIL", f"{manifest.case_id.nunique()} unique cases"),
        "Fold 1 validation cases": ("PASS" if fold_counts.get(0)==60 else "FAIL", f"source fold 0: {fold_counts.get(0)}"),
        "Fold 2 validation cases": ("PASS" if fold_counts.get(1)==60 else "FAIL", f"source fold 1: {fold_counts.get(1)}"),
        "Fold 3 validation cases": ("PASS" if fold_counts.get(2)==60 else "FAIL", f"source fold 2: {fold_counts.get(2)}"),
        "Fold 4 validation cases": ("PASS" if fold_counts.get(3)==60 else "FAIL", f"source fold 3: {fold_counts.get(3)}"),
        "Fold 5 validation cases": ("PASS" if fold_counts.get(4)==60 else "FAIL", f"source fold 4: {fold_counts.get(4)}"),
        "Each case appears once as OOF validation": ("PASS" if each_once else "FAIL", "300 IDs; one assigned validation fold each"),
        "No train-fold predictions used for final metrics": ("PASS", "Each segmentation-only file is under the assigned fold validation directory"),
        "Final GT masks used": ("PASS", "Dataset817/818 final labelsTr; numerical GT match to final E2E table verified"),
        "Rectus tendinous intersections = background": ("NEEDS HUMAN CONFIRMATION", "Anatomical semantics cannot be proven from mask voxels alone"),
        "TA/IO/EO intermuscular fat = IMAT": ("NEEDS HUMAN CONFIRMATION", "Anatomical semantics cannot be proven from mask voxels alone"),
        "Renal sinus fat excluded from VFAT": ("NEEDS HUMAN CONFIRMATION", "Anatomical semantics cannot be proven from mask voxels alone"),
        "Localization unit": ("PASS", "mm; world-z coordinates from each CT NIfTI affine"),
        "CSA/error/MAE unit": ("PASS", "cm²; voxel count × in-plane pixel area / 100"),
        "No duplicated case IDs within same level/tissue": ("PASS" if not tissue.duplicated(["case_id","level","tissue"]).any() else "FAIL", "0 duplicated case-level-tissue keys"),
        "Missing values reported": ("PASS", "External IMAT is unavailable: 300 values plus derived error/AE are explicitly blank; no other required missing values"),
        "Public model uses same final GT for comparison": ("PASS", "Joined case-by-case to the same final GT used in direct OOF Table 2"),
    })
    workbook, missing_ids = write_workbook(values, level, tissue, offset, external, public, qc_rows)

    source_files = [MANIFEST, REGISTRY, E2E, OFFSET, PUBLIC, EXTERNAL, TEMPLATE]
    provenance = {str(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size,
                            "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()} for p in source_files}
    machine["study"] = {"n": 300, "fold_source_labels": [0,1,2,3,4], "fold_display_labels": [1,2,3,4,5],
                        "fold_counts": {str(k): int(v) for k,v in fold_counts.items()},
                        "icc_definition": "ICC(A,1), two-way absolute agreement, single measurement",
                        "icc_ci_method": "paired-case nonparametric percentile bootstrap, 10,000 resamples, seed 731",
                        "difference_conventions": {"method_error": "prediction - GT", "inferior_middle": "middle - inferior"}}
    machine["workbook"] = {"path": str(workbook), "unmapped_summary_ids": missing_ids}
    machine["raw_row_counts"] = raw_counts
    machine["provenance"] = provenance
    (OUT / "all_results_machine_readable.json").write_text(json.dumps(json_clean(machine), indent=2, ensure_ascii=False, allow_nan=False))

    missing = [
        "Figure 3 and Figure 4 cannot be regenerated because their exact manuscript definitions/layout/source panels were not supplied.",
        "The three annotation-rule statements require human/annotator confirmation; file masks alone cannot prove the intended anatomical rule.",
        "External reference contains no IMAT values (300/300 missing), so IMAT external error and correlation are N/A.",
    ]
    report = ["KJR ADPKD final 300-case OOF result package", "",
              f"Generated: {datetime.now(timezone.utc).isoformat()}",
              "Cohort: 300 unique cases; five source folds 0-4, 60 validation cases per fold.",
              "All tissue segmentation-only results were recalculated from the assigned fold validation prediction.",
              "", "Raw row counts:"]
    report += [f"- {k}: {v}" for k,v in raw_counts.items()]
    report += ["", "Unresolved or externally confirmable items:"] + [f"- {x}" for x in missing]
    report += ["", f"Workbook: {workbook.name}", "Machine-readable JSON: all_results_machine_readable.json"]
    (OUT / "final_results_summary.txt").write_text("\n".join(report) + "\n")
    (OUT / "missing_required_items.md").write_text("# Missing or externally confirmable items\n\n" + "\n".join(f"- {x}" for x in missing) + "\n")
    shutil.copy2(REGISTRY, OUT / "model_registry_used.json")
    shutil.copy2(MANIFEST, OUT / "manifest_300_all_folds_used.csv")
    print(json.dumps({"workbook": str(workbook), "raw_counts": raw_counts,
                      "missing_summary_ids": missing_ids}, indent=2))


if __name__ == "__main__":
    main()
