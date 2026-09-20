#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = str(PROJECT_ROOT / "output")
DEFAULT_INPUT_CT_NIFTI = ""  # fill this before running if you want no CLI args

NNUNET_RAW_DEFAULT = os.environ.get("NNUNET_RAW", str(PROJECT_ROOT / "nnUNet_raw"))
NNUNET_PREPROCESSED_DEFAULT = os.environ.get("NNUNET_PREPROCESSED", str(PROJECT_ROOT / "nnUNet_preprocessed"))
NNUNET_RESULTS_DEFAULT = os.environ.get("NNUNET_RESULTS", str(PROJECT_ROOT / "nnUNet_results"))
NNUNETV2_PREDICT_BIN = os.environ.get("NNUNETV2_PREDICT_BIN") or shutil.which("nnUNetv2_predict") or "nnUNetv2_predict"

L3_CKPT_DEFAULT = str(PROJECT_ROOT / "nnUNet_results" / "Dataset878_L3Vertebra_full_878cases" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0" / "checkpoint_best.pth")
YOLO_MIDDLE_WEIGHTS_DEFAULT = str(PROJECT_ROOT / "weight" / "detection" / "middle" / "best.pt")
YOLO_INFERIOR_WEIGHTS_DEFAULT = str(PROJECT_ROOT / "weight" / "detection" / "inferior" / "best.pt")
SMA_MIDDLE_CKPT_DEFAULT = str(PROJECT_ROOT / "nnUNet_results" / "Dataset318_SMA_4label_middle_seg" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth")
SMA_INFERIOR_CKPT_DEFAULT = str(PROJECT_ROOT / "nnUNet_results" / "Dataset331_SMA_4label_inf_seg" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth")

L3_CONFIG_DEFAULT = "3d_fullres"
L3_FOLD_DEFAULT = "0"
SMA_CONFIG_DEFAULT = "2d"
SMA_FOLD_DEFAULT = "0"
SMA_TRAINER_DEFAULT = "nnUNetTrainer"
SMA_PLANS_DEFAULT = "nnUNetPlans"

HU_SMA_MIN = -29
HU_SMA_MAX = 150
DISP_HU_MIN = -200
DISP_HU_MAX = 200
OVERLAY_ALPHA = 0.3

LABEL_MAP = {
    "sma": 1,
    "sfat": 2,
    "vfat": 3,
    "imat": 4,
}

BRANCHES = ("middle", "inferior")


@dataclass
class BranchModelConfig:
    branch: str
    yolo_weights: str
    sma_ckpt_path: str
    sma_dataset_name: str
    sma_checkpoint_name: str
    sma_config: str
    sma_fold: str
    sma_trainer: str
    sma_plans: str


@dataclass
class PipelineConfig:
    out_root: str
    input_cts: List[str]
    slice_mode: str
    nnunet_raw: str
    nnunet_preprocessed: str
    nnunet_results: str
    l3_ckpt_path: str
    l3_dataset_name: str
    l3_config: str
    l3_fold: str
    l3_trainer: str
    l3_plans: str
    l3_checkpoint_name: str
    yolo_imgsz: int
    yolo_device: str
    yolo_conf: float
    branch_configs: Dict[str, BranchModelConfig]
    resume_skip_complete: bool
    keep_work: bool


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="3D CT -> L3 segmentation -> YOLO middle/inferior slice -> SMA pipeline"
    )
    parser.add_argument("--input-ct", default=DEFAULT_INPUT_CT_NIFTI, help="Single input CT nifti path (*.nii or *.nii.gz)")
    parser.add_argument("--input-dir", help="Directory containing CT nifti files")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern for --input-dir")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Root output directory")
    parser.add_argument(
        "--slice-mode",
        choices=("middle", "inferior", "both"),
        default="inferior",
        help="Which detection/SMA branch to run",
    )

    parser.add_argument("--nnunet-raw", default=NNUNET_RAW_DEFAULT)
    parser.add_argument("--nnunet-preprocessed", default=NNUNET_PREPROCESSED_DEFAULT)
    parser.add_argument("--nnunet-results", default=NNUNET_RESULTS_DEFAULT)

    parser.add_argument("--l3-ckpt-path", default=L3_CKPT_DEFAULT)
    parser.add_argument("--l3-dataset-name", default=None)
    parser.add_argument("--l3-config", default=L3_CONFIG_DEFAULT)
    parser.add_argument("--l3-fold", default=L3_FOLD_DEFAULT)
    parser.add_argument("--l3-trainer", default="nnUNetTrainer")
    parser.add_argument("--l3-plans", default="nnUNetPlans")

    parser.add_argument("--yolo-middle-weights", default=YOLO_MIDDLE_WEIGHTS_DEFAULT)
    parser.add_argument("--yolo-inferior-weights", default=YOLO_INFERIOR_WEIGHTS_DEFAULT)
    parser.add_argument("--yolo-imgsz", type=int, default=512)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-conf", type=float, default=0.001)

    parser.add_argument("--sma-middle-ckpt-path", default=SMA_MIDDLE_CKPT_DEFAULT)
    parser.add_argument("--sma-inferior-ckpt-path", default=SMA_INFERIOR_CKPT_DEFAULT)
    parser.add_argument("--sma-middle-dataset-name", default=None)
    parser.add_argument("--sma-inferior-dataset-name", default=None)
    parser.add_argument("--sma-config", default=SMA_CONFIG_DEFAULT)
    parser.add_argument("--sma-fold", default=SMA_FOLD_DEFAULT)
    parser.add_argument("--sma-trainer", default=SMA_TRAINER_DEFAULT)
    parser.add_argument("--sma-plans", default=SMA_PLANS_DEFAULT)

    parser.add_argument("--resume-skip-complete", action="store_true", help="Skip cases whose target branch metrics CSV files already exist")
    parser.add_argument("--keep-work", action="store_true", help="Keep per-case work directories")

    args = parser.parse_args()

    input_ct_value = (args.input_ct or "").strip()
    input_dir_value = (args.input_dir or "").strip()

    if not input_ct_value and not input_dir_value:
        parser.error("Either --input-ct, --input-dir, or DEFAULT_INPUT_CT_NIFTI must be provided.")
    if input_ct_value and input_dir_value:
        parser.error("Use only one of --input-ct or --input-dir.")

    if input_ct_value:
        input_cts = [os.path.abspath(input_ct_value)]
    else:
        search_pattern = os.path.join(os.path.abspath(args.input_dir), args.pattern)
        input_cts = sorted(
            p for p in glob.glob(search_pattern) if os.path.isfile(p) and is_nifti_path(p)
        )
    if not input_cts:
        parser.error("No input CT files were found.")

    l3_ckpt_path = os.path.abspath(args.l3_ckpt_path)
    l3_dataset_name = args.l3_dataset_name or infer_dataset_name_from_checkpoint(l3_ckpt_path)
    l3_checkpoint_name = os.path.basename(l3_ckpt_path)

    branch_configs = build_branch_configs(args)

    return PipelineConfig(
        out_root=os.path.abspath(args.out_root),
        input_cts=input_cts,
        slice_mode=args.slice_mode,
        nnunet_raw=args.nnunet_raw,
        nnunet_preprocessed=args.nnunet_preprocessed,
        nnunet_results=args.nnunet_results,
        l3_ckpt_path=l3_ckpt_path,
        l3_dataset_name=l3_dataset_name,
        l3_config=args.l3_config,
        l3_fold=args.l3_fold,
        l3_trainer=args.l3_trainer,
        l3_plans=args.l3_plans,
        l3_checkpoint_name=l3_checkpoint_name,
        yolo_imgsz=args.yolo_imgsz,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        branch_configs=branch_configs,
        resume_skip_complete=args.resume_skip_complete,
        keep_work=args.keep_work,
    )


def build_branch_configs(args: argparse.Namespace) -> Dict[str, BranchModelConfig]:
    branch_specs = {}
    for branch in BRANCHES:
        yolo_weights = os.path.abspath(getattr(args, f"yolo_{branch}_weights"))
        sma_ckpt_path = os.path.abspath(getattr(args, f"sma_{branch}_ckpt_path"))
        sma_dataset_name = getattr(args, f"sma_{branch}_dataset_name") or infer_dataset_name_from_checkpoint(
            sma_ckpt_path
        )
        branch_specs[branch] = BranchModelConfig(
            branch=branch,
            yolo_weights=yolo_weights,
            sma_ckpt_path=sma_ckpt_path,
            sma_dataset_name=sma_dataset_name,
            sma_checkpoint_name=os.path.basename(sma_ckpt_path),
            sma_config=args.sma_config,
            sma_fold=args.sma_fold,
            sma_trainer=args.sma_trainer,
            sma_plans=args.sma_plans,
        )
    return branch_specs


def is_nifti_path(path: str) -> bool:
    return path.endswith(".nii.gz") or path.endswith(".nii")


def infer_dataset_name_from_checkpoint(ckpt_path: str) -> str:
    parts = os.path.normpath(ckpt_path).split(os.sep)
    for part in parts:
        if part.startswith("Dataset"):
            return part
    raise ValueError(f"Could not infer dataset name from checkpoint path: {ckpt_path}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def safe_case_id(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def setup_nnunet_env(cfg: PipelineConfig):
    os.environ["nnUNet_raw"] = cfg.nnunet_raw
    os.environ["nnUNet_preprocessed"] = cfg.nnunet_preprocessed
    os.environ["nnUNet_results"] = cfg.nnunet_results
    ensure_dir(cfg.nnunet_raw)


def get_target_branches(cfg: PipelineConfig) -> List[str]:
    if cfg.slice_mode == "both":
        return list(BRANCHES)
    return [cfg.slice_mode]


def validate_paths(cfg: PipelineConfig):
    if not os.path.exists(cfg.nnunet_results):
        raise FileNotFoundError(f"nnUNet_results not found: {cfg.nnunet_results}")
    if not os.path.exists(cfg.l3_ckpt_path):
        raise FileNotFoundError(f"L3 checkpoint not found: {cfg.l3_ckpt_path}")

    for branch in get_target_branches(cfg):
        branch_cfg = cfg.branch_configs[branch]
        if not os.path.exists(branch_cfg.yolo_weights):
            raise FileNotFoundError(f"[{branch}] YOLO weights not found: {branch_cfg.yolo_weights}")
        if not os.path.exists(branch_cfg.sma_ckpt_path):
            raise FileNotFoundError(f"[{branch}] SMA checkpoint not found: {branch_cfg.sma_ckpt_path}")

    for ct_path in cfg.input_cts:
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"Input CT not found: {ct_path}")


def run_subprocess(cmd: Sequence[str], log_path: str, step_name: str):
    ensure_dir(os.path.dirname(log_path))
    print(f"[RUN] {step_name}: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[CMD]\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{step_name}] {line.rstrip()}")
            f.write(line)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed. See log: {log_path}")


def copy_ct_for_l3_inference(ct_path: str, case_id: str, l3_input_dir: str) -> str:
    ensure_dir(l3_input_dir)
    dst = os.path.join(l3_input_dir, f"{case_id}_0000.nii.gz")
    shutil.copy2(ct_path, dst)
    return dst


def run_l3_nnunet(case_id: str, l3_input_dir: str, l3_pred_dir: str, case_log_dir: str, cfg: PipelineConfig) -> str:
    ensure_dir(l3_pred_dir)
    cmd = [
        NNUNETV2_PREDICT_BIN,
        "-d", cfg.l3_dataset_name,
        "-c", cfg.l3_config,
        "-f", cfg.l3_fold,
        "-tr", cfg.l3_trainer,
        "-p", cfg.l3_plans,
        "-i", l3_input_dir,
        "-o", l3_pred_dir,
        "-chk", cfg.l3_checkpoint_name,
    ]
    log_path = os.path.join(case_log_dir, f"{case_id}_l3_inference.log")
    run_subprocess(cmd, log_path, step_name="L3_NNUNET")

    pred_path = os.path.join(l3_pred_dir, f"{case_id}.nii.gz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"L3 prediction not found: {pred_path}")
    return pred_path


def save_slice_png(mask_slice: np.ndarray, out_path: str):
    arr = (mask_slice > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(out_path)


def extract_nonempty_mask_slices(mask_path: str, out_dir: str, case_id: str) -> List[str]:
    ensure_dir(out_dir)
    nii = nib.load(mask_path)
    vol = nii.get_fdata()
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {vol.shape} for {mask_path}")

    img_paths = []
    for z in range(vol.shape[2]):
        sl = vol[:, :, z]
        if np.any(sl > 0):
            png_path = os.path.join(out_dir, f"{case_id}_{z}.png")
            save_slice_png(sl, png_path)
            img_paths.append(png_path)
    return sorted(img_paths)


def parse_z_from_png(path: str) -> Optional[int]:
    match = re.search(r"_(\d+)\.png$", os.path.basename(path))
    return int(match.group(1)) if match else None


def pick_best_slice(
    model: YOLO,
    img_paths: Sequence[str],
    yolo_imgsz: int,
    yolo_device: str,
    yolo_conf: float,
) -> Tuple[Optional[int], float, Optional[str]]:
    if not img_paths:
        return None, 0.0, None

    print(f"[YOLO] evaluating {len(img_paths)} candidate slices")
    results = model.predict(
        source=list(img_paths),
        imgsz=yolo_imgsz,
        device=yolo_device,
        conf=yolo_conf,
        verbose=True,
    )

    best_img = None
    best_conf = -1.0
    scored = []
    for path, res in zip(img_paths, results):
        conf = 0.0
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            conf = float(res.boxes.conf.max().item())
        scored.append((path, conf))
        if conf > best_conf:
            best_conf = conf
            best_img = path

    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (path, conf) in enumerate(scored[:10], start=1):
        print(f"[YOLO] top{rank:02d} z={parse_z_from_png(path)} conf={conf:.4f} path={os.path.basename(path)}")

    if best_img is None:
        return None, 0.0, None
    print(f"[YOLO] selected z={parse_z_from_png(best_img)} conf={max(best_conf, 0.0):.4f}")
    return parse_z_from_png(best_img), max(best_conf, 0.0), best_img


def save_single_slice_nifti_from_ct(ct_path: str, slice_idx: int, out_path: str) -> None:
    nii = nib.load(ct_path)
    if len(nii.shape) != 3:
        raise ValueError(f"CT must be 3D, got {nii.shape}")
    zdim = nii.shape[2]
    if slice_idx < 0 or slice_idx >= zdim:
        raise ValueError(f"slice_idx out of range: idx={slice_idx}, zdim={zdim}")
    sl = np.asanyarray(nii.dataobj[:, :, slice_idx], dtype=np.float32)
    sl3 = sl[:, :, None]
    out_img = nib.Nifti1Image(sl3, nii.affine.copy(), header=nii.header.copy())
    nib.save(out_img, out_path)


def run_sma_nnunet(
    case_id: str,
    input_nifti_path: str,
    sma_input_dir: str,
    sma_pred_dir: str,
    case_log_dir: str,
    branch_cfg: BranchModelConfig,
) -> str:
    reset_dir(sma_input_dir)
    ensure_dir(sma_pred_dir)

    input_0000_path = os.path.join(sma_input_dir, f"{case_id}_0000.nii.gz")
    shutil.copy2(input_nifti_path, input_0000_path)

    cmd = [
        NNUNETV2_PREDICT_BIN,
        "-i", sma_input_dir,
        "-o", sma_pred_dir,
        "-d", branch_cfg.sma_dataset_name,
        "-c", branch_cfg.sma_config,
        "-f", branch_cfg.sma_fold,
        "-tr", branch_cfg.sma_trainer,
        "-p", branch_cfg.sma_plans,
        "-chk", branch_cfg.sma_checkpoint_name,
    ]
    log_path = os.path.join(case_log_dir, f"{case_id}_{branch_cfg.branch}_sma_inference.log")
    run_subprocess(cmd, log_path, step_name=f"SMA_NNUNET_{branch_cfg.branch.upper()}")

    pred_path = os.path.join(sma_pred_dir, f"{case_id}.nii.gz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"SMA prediction not found: {pred_path}")
    return pred_path


def clip_for_display(img2d: np.ndarray) -> np.ndarray:
    x = np.clip(img2d, DISP_HU_MIN, DISP_HU_MAX)
    return (x - DISP_HU_MIN) / (DISP_HU_MAX - DISP_HU_MIN + 1e-8)


def build_overlay_cmap():
    colors = [
        (0.0, 0.0, 0.0, 0.0),
        (0.2, 0.8, 1.0, 1.0),
        (0.4, 1.0, 0.4, 1.0),
        (1.0, 0.65, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
    ]
    return ListedColormap(colors)


def compute_area_metrics_from_ct_and_seg(ct_slice_nifti: str, pred_seg_nifti: str) -> Dict[str, float]:
    ct_nii = nib.load(ct_slice_nifti)
    seg_nii = nib.load(pred_seg_nifti)

    ct = ct_nii.get_fdata()
    seg = seg_nii.get_fdata()
    if ct.ndim != 3 or seg.ndim != 3:
        raise ValueError(f"Expected single-slice 3D nifti, got ct={ct.shape}, seg={seg.shape}")

    ct2d = ct[:, :, 0]
    seg2d = seg[:, :, 0]

    zooms = ct_nii.header.get_zooms()
    if len(zooms) < 2:
        raise ValueError(f"Invalid zooms in nifti header: {zooms}")

    pixel_area_cm2 = (float(zooms[0]) * float(zooms[1])) / 100.0

    out: Dict[str, float] = {}
    for name, label in LABEL_MAP.items():
        out[name] = float((seg2d == label).sum()) * pixel_area_cm2

    sma_mask = seg2d == LABEL_MAP["sma"]
    hu_mask = (ct2d >= HU_SMA_MIN) & (ct2d <= HU_SMA_MAX)
    out["sma_hu"] = float((sma_mask & hu_mask).sum()) * pixel_area_cm2
    return out


def save_png(arr: np.ndarray, out_path: str, cmap: str = "gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visual_outputs(ct_slice_nifti: str, pred_seg_nifti: str, image_dir: str, case_id: str, branch: str):
    ensure_dir(image_dir)
    ct = nib.load(ct_slice_nifti).get_fdata()[:, :, 0]
    seg = nib.load(pred_seg_nifti).get_fdata()[:, :, 0]

    ct_disp = clip_for_display(ct)
    seg_masked = np.ma.masked_where(seg == 0, seg)
    cmap = build_overlay_cmap()

    ct_png = os.path.join(image_dir, f"{case_id}_{branch}_ct.png")
    mask_png = os.path.join(image_dir, f"{case_id}_{branch}_mask.png")
    overlay_png = os.path.join(image_dir, f"{case_id}_{branch}_overlay_alpha03.png")

    save_png(ct_disp, ct_png, cmap="gray")
    save_png(seg, mask_png, cmap=cmap, vmin=0, vmax=4)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(ct_disp, cmap="gray")
    ax.imshow(seg_masked, cmap=cmap, alpha=OVERLAY_ALPHA, vmin=0, vmax=4, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(overlay_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "ct_png": ct_png,
        "mask_png": mask_png,
        "overlay_png": overlay_png,
    }


def write_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[Dict[str, object]]):
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def case_branch_metrics_csv(out_root: str, case_id: str, branch: str) -> str:
    return os.path.join(out_root, case_id, "csv", f"{case_id}_{branch}_metrics.csv")


def load_completed_case_results(case_id: str, cfg: PipelineConfig) -> Optional[List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for branch in get_target_branches(cfg):
        metrics_csv = case_branch_metrics_csv(cfg.out_root, case_id, branch)
        branch_rows = load_csv_rows(metrics_csv)
        if not branch_rows:
            return None
        rows.extend(branch_rows)
    return rows


def process_branch(
    ct_path: str,
    case_id: str,
    branch: str,
    model: YOLO,
    l3_mask_path: str,
    yolo_slice_dir: str,
    case_root: str,
    log_dir: str,
    branch_cfg: BranchModelConfig,
    cfg: PipelineConfig,
) -> Dict[str, object]:
    image_dir = os.path.join(case_root, "image")
    csv_dir = os.path.join(case_root, "csv")
    work_dir = os.path.join(case_root, "work", branch)
    sma_input_dir = os.path.join(work_dir, "sma_input")
    sma_pred_dir = os.path.join(work_dir, "sma_pred")

    print(f"[STEP 3/6] branch={branch} extract non-empty L3 mask slices")
    img_paths = extract_nonempty_mask_slices(l3_mask_path, yolo_slice_dir, case_id)
    print(f"[STEP 3/6] branch={branch} candidate_slices={len(img_paths)}")
    print(f"[STEP 4/6] branch={branch} run YOLO detection")
    pred_idx, best_conf, best_img = pick_best_slice(
        model, img_paths, cfg.yolo_imgsz, cfg.yolo_device, cfg.yolo_conf
    )
    if pred_idx is None:
        raise RuntimeError(f"[{branch}] YOLO could not determine a slice index.")

    print(f"[STEP 4/6] branch={branch} selected_z={pred_idx} conf={best_conf:.4f}")
    slice_nifti = os.path.join(image_dir, f"{case_id}_{branch}_slice_idx{pred_idx}.nii.gz")
    print(f"[STEP 5/6] branch={branch} extract CT slice from original volume")
    save_single_slice_nifti_from_ct(ct_path, pred_idx, slice_nifti)

    print(f"[STEP 6/6] branch={branch} run SMA segmentation")
    sma_seg_work = run_sma_nnunet(
        case_id=case_id,
        input_nifti_path=slice_nifti,
        sma_input_dir=sma_input_dir,
        sma_pred_dir=sma_pred_dir,
        case_log_dir=log_dir,
        branch_cfg=branch_cfg,
    )
    sma_seg_final = os.path.join(image_dir, f"{case_id}_{branch}_sma_seg.nii.gz")
    shutil.copy2(sma_seg_work, sma_seg_final)

    visuals = save_visual_outputs(slice_nifti, sma_seg_final, image_dir, case_id, branch)
    print(f"[STEP 6/6] branch={branch} saved mask={os.path.basename(visuals['mask_png'])} overlay={os.path.basename(visuals['overlay_png'])}")
    metrics = compute_area_metrics_from_ct_and_seg(slice_nifti, sma_seg_final)
    print(f"[STEP 6/6] branch={branch} metrics sma={metrics['sma']:.2f} sma_hu={metrics['sma_hu']:.2f} sfat={metrics['sfat']:.2f} vfat={metrics['vfat']:.2f} imat={metrics['imat']:.2f}")

    result = {
        "case_id": case_id,
        "branch": branch,
        "ct_path": ct_path,
        "l3_mask_path": l3_mask_path,
        "slice_index": pred_idx,
        "yolo_best_conf": best_conf,
        "yolo_best_img": best_img,
        "num_l3_mask_slices": len(img_paths),
        "slice_ct_path": slice_nifti,
        "sma_seg_path": sma_seg_final,
        "mask_png": visuals["mask_png"],
        "overlay_png": visuals["overlay_png"],
        "pred_sma": metrics["sma"],
        "pred_sma_hu": metrics["sma_hu"],
        "pred_sfat": metrics["sfat"],
        "pred_vfat": metrics["vfat"],
        "pred_imat": metrics["imat"],
    }

    per_branch_csv = os.path.join(csv_dir, f"{case_id}_{branch}_metrics.csv")
    save_csv(per_branch_csv, [result])
    write_json(os.path.join(csv_dir, f"{case_id}_{branch}_result.json"), result)
    result["metrics_csv"] = per_branch_csv
    return result


def process_case(ct_path: str, models: Dict[str, YOLO], cfg: PipelineConfig) -> List[Dict[str, object]]:
    case_id = safe_case_id(ct_path)
    case_root = os.path.join(cfg.out_root, case_id)
    image_dir = os.path.join(case_root, "image")
    csv_dir = os.path.join(case_root, "csv")
    work_dir = os.path.join(case_root, "work")
    log_dir = os.path.join(case_root, "logs")

    ensure_dir(image_dir)
    ensure_dir(csv_dir)
    reset_dir(work_dir)
    ensure_dir(log_dir)

    l3_input_dir = os.path.join(work_dir, "l3_input")
    l3_pred_dir = os.path.join(work_dir, "l3_pred")
    yolo_slice_root = os.path.join(work_dir, "yolo_l3_mask_slices")

    print(f"[STEP 1/6] case={case_id} copy input CT for L3 inference")
    copy_ct_for_l3_inference(ct_path, case_id, l3_input_dir)
    print(f"[STEP 2/6] case={case_id} run L3 vertebra segmentation")
    l3_mask_path = run_l3_nnunet(case_id, l3_input_dir, l3_pred_dir, log_dir, cfg)

    l3_mask_final = os.path.join(image_dir, f"{case_id}_l3_mask.nii.gz")
    shutil.copy2(l3_mask_path, l3_mask_final)
    print(f"[STEP 2/6] case={case_id} saved L3 mask -> {l3_mask_final}")

    branch_results: List[Dict[str, object]] = []
    for branch in get_target_branches(cfg):
        branch_result = process_branch(
            ct_path=ct_path,
            case_id=case_id,
            branch=branch,
            model=models[branch],
            l3_mask_path=l3_mask_final,
            yolo_slice_dir=os.path.join(yolo_slice_root, branch),
            case_root=case_root,
            log_dir=log_dir,
            branch_cfg=cfg.branch_configs[branch],
            cfg=cfg,
        )
        branch_results.append(branch_result)

    if not cfg.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    return branch_results


def main():
    cfg = parse_args()
    setup_nnunet_env(cfg)
    validate_paths(cfg)

    if cfg.resume_skip_complete:
        ensure_dir(cfg.out_root)
    else:
        reset_dir(cfg.out_root)

    target_branches = get_target_branches(cfg)
    print("[INFO] verbose_copy       = True")
    print("[INFO] n_cases            =", len(cfg.input_cts))
    print("[INFO] slice_mode         =", cfg.slice_mode)
    print("[INFO] out_root           =", cfg.out_root)
    print("[INFO] l3_dataset_name    =", cfg.l3_dataset_name)
    for branch in target_branches:
        branch_cfg = cfg.branch_configs[branch]
        print(f"[INFO] {branch}_yolo       =", branch_cfg.yolo_weights)
        print(f"[INFO] {branch}_sma_dataset=", branch_cfg.sma_dataset_name)

    models = {branch: YOLO(cfg.branch_configs[branch].yolo_weights) for branch in target_branches}

    success_rows: List[Dict[str, object]] = []
    skipped_complete_cases = 0
    failed_rows: List[Dict[str, object]] = []

    t0 = time.time()

    for idx, ct_path in enumerate(cfg.input_cts, start=1):
        case_id = safe_case_id(ct_path)
        print(f"\n[CASE {idx}/{len(cfg.input_cts)}] {case_id}")
        if cfg.resume_skip_complete:
            completed_rows = load_completed_case_results(case_id, cfg)
            if completed_rows is not None:
                success_rows.extend(completed_rows)
                skipped_complete_cases += 1
                print(f"[SKIP] {case_id}: existing metrics found for all target branches")
                continue
        try:
            case_start = time.time()
            branch_results = process_case(ct_path, models, cfg)
            success_rows.extend(branch_results)
            print(f"[CASE DONE] {case_id} elapsed={time.time() - case_start:.1f}s")
            for result in branch_results:
                print(
                    f"[OK] branch={result['branch']} "
                    f"idx={result['slice_index']} "
                    f"conf={result['yolo_best_conf']:.4f} "
                    f"sma={result['pred_sma']:.2f} "
                    f"sfat={result['pred_sfat']:.2f} "
                    f"vfat={result['pred_vfat']:.2f}"
                )
        except Exception as e:
            failed_rows.append({
                "case_id": case_id,
                "ct_path": ct_path,
                "error": str(e),
            })
            print(f"[FAIL] {case_id}: {e}")

    success_csv = os.path.join(cfg.out_root, "summary_success.csv")
    failed_csv = os.path.join(cfg.out_root, "summary_failed.csv")
    save_csv(success_csv, success_rows)
    save_csv(failed_csv, failed_rows)

    run_summary = {
        "n_input": len(cfg.input_cts),
        "n_success_rows": len(success_rows),
        "n_skipped_complete_cases": skipped_complete_cases,
        "n_failed_cases": len(failed_rows),
        "summary_success_csv": success_csv,
        "summary_failed_csv": failed_csv,
        "out_root": cfg.out_root,
        "slice_mode": cfg.slice_mode,
        "l3_dataset_name": cfg.l3_dataset_name,
        "branches": {
            branch: {
                "yolo_weights": cfg.branch_configs[branch].yolo_weights,
                "sma_ckpt_path": cfg.branch_configs[branch].sma_ckpt_path,
                "sma_dataset_name": cfg.branch_configs[branch].sma_dataset_name,
            }
            for branch in target_branches
        },
    }
    write_json(os.path.join(cfg.out_root, "run_summary.json"), run_summary)

    print("\n[DONE]")
    print(" - success:", success_csv)
    print(" - failed :", failed_csv)
    print(" - summary:", os.path.join(cfg.out_root, "run_summary.json"))


if __name__ == "__main__":
    main()
