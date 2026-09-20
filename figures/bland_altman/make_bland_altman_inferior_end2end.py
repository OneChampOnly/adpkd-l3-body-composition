#!/usr/bin/env python3
"""Bland-Altman analysis of inferior GT versus full end-to-end predictions."""

from __future__ import annotations

import os
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_INPUT = ROOT / "results/evaluation/per_case_end_to_end.csv"
DEFAULT_OUTPUT = ROOT / "paper_data/bland_altman_inferior_end2end"

TISSUES = {
    "sma": {"name": "SMA", "color": "#4056A1", "x_label": "Mean of SMA area (cm²)", "x_limits": (0.0, 250.0), "x_tick_step": 50.0, "y_tick_step": 5.0},
    "sfat": {"name": "SFAT", "color": "#4056A1", "x_label": "Mean of SFAT area (cm²)", "x_limits": (0.0, 400.0), "x_tick_step": 100.0, "y_tick_step": 5.0},
    "vfat": {"name": "VFAT", "color": "#4056A1", "x_label": "Mean of VFAT area (cm²)", "x_limits": (0.0, 400.0), "x_tick_step": 100.0, "y_tick_step": 5.0},
    "imat": {"name": "IMAT", "color": "#4056A1", "x_label": "Mean of IMAT area (cm²)", "x_limits": (0.0, 20.0), "x_tick_step": 2.5, "y_tick_step": 1.0},
}


def finite_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_inferior_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"fold", "case_id", "branch"}
        for tissue in TISSUES:
            required.update({f"gt_{tissue}_cm2", f"pred_{tissue}_cm2"})
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        rows = [row for row in reader if row["branch"].strip().lower() == "inferior"]

    ids = [row["case_id"].strip() for row in rows]
    if len(rows) != 300 or len(set(ids)) != 300:
        raise RuntimeError(f"Expected 300 unique inferior cases, found rows={len(rows)}, unique={len(set(ids))}")
    return rows


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    pad = max((high - low) * fraction, 0.5) if low != high else max(abs(low) * fraction, 1.0)
    return low - pad, high + pad


def tissue_values(rows: list[dict[str, str]], tissue: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (finite_float(row[f"gt_{tissue}_cm2"]), finite_float(row[f"pred_{tissue}_cm2"]))
        for row in rows
    ]
    if any(not math.isfinite(gt) or not math.isfinite(pred) for gt, pred in pairs):
        raise RuntimeError(f"Non-finite {tissue.upper()} GT/prediction value detected")
    return np.asarray([x[0] for x in pairs]), np.asarray([x[1] for x in pairs])


def calculate_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    differences = pred - gt
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    lower = bias - 1.96 * sd
    upper = bias + 1.96 * sd
    return {
        "n": float(len(differences)),
        "bias": bias,
        "sd": sd,
        "loa_lower": lower,
        "loa_upper": upper,
        "outside_loa": float(np.sum((differences < lower) | (differences > upper))),
    }


def draw_panel(ax: plt.Axes, gt: np.ndarray, pred: np.ndarray, tissue: str, panel_title: str | None = None) -> dict[str, float]:
    info = TISSUES[tissue]
    means = (gt + pred) / 2.0
    differences = pred - gt
    stats = calculate_stats(gt, pred)

    ax.scatter(
        means,
        differences,
        s=14,
        facecolors="white",
        edgecolors=info["color"],
        linewidths=0.6,
        alpha=0.82,
    )

    x_min, x_max = info["x_limits"]
    y_values = np.concatenate((differences, [stats["loa_lower"], stats["loa_upper"]]))
    y_step = info["y_tick_step"]
    y_min = math.floor(float(np.min(y_values)) / y_step) * y_step
    y_max = math.ceil(float(np.max(y_values)) / y_step) * y_step
    x_pad = 0.025 * (x_max - x_min)
    y_pad = 0.4 * y_step
    x_plot_min, x_plot_max = x_min - x_pad, x_max + x_pad
    line_style = {"color": "#202020", "clip_on": True, "solid_capstyle": "butt", "dash_capstyle": "butt"}
    ax.plot([x_min, x_max], [stats["bias"], stats["bias"]], linewidth=1.05, **line_style)
    ax.plot([x_min, x_max], [stats["loa_upper"], stats["loa_upper"]], linewidth=0.9, linestyle="--", **line_style)
    ax.plot([x_min, x_max], [stats["loa_lower"], stats["loa_lower"]], linewidth=0.9, linestyle="--", **line_style)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_xticks(np.arange(x_min, x_max + info["x_tick_step"] / 2.0, info["x_tick_step"]))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_yticks(np.arange(y_min, y_max + y_step / 2.0, y_step))
    ax.set_title(panel_title or f"Bland–Altman Plot for {info['name']}", fontsize=12, pad=9)
    ax.set_xlabel(info["x_label"], fontsize=9.5)
    ax.set_ylabel("Difference (cm²)", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.5, direction="in")

    label_x = x_max
    label_style = {
        "ha": "right",
        "va": "center",
        "fontsize": 7.8,
        "linespacing": 1.25,
    }
    ax.text(label_x, stats["loa_upper"], f"+1.96 SD\n{stats['loa_upper']:.2f}", **label_style)
    ax.text(label_x, stats["bias"], f"Mean\n{stats['bias']:.2f}", **label_style)
    ax.text(label_x, stats["loa_lower"], f"−1.96 SD\n{stats['loa_lower']:.2f}", **label_style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#222222")
    return stats


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_inferior_rows(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    per_case_rows: list[dict[str, object]] = []
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for tissue, info in TISSUES.items():
        gt, pred = tissue_values(rows, tissue)
        values[tissue] = (gt, pred)
        fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=200)
        stats = draw_panel(ax, gt, pred, tissue)
        fig.tight_layout()
        png = args.output_dir / f"bland_altman_{tissue}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        ax.set_title("")
        fig.tight_layout()
        no_title_png = args.output_dir / f"bland_altman_{tissue}_no_title.png"
        fig.savefig(no_title_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append({
            "tissue": info["name"],
            "n": int(stats["n"]),
            "mean_difference_cm2": f"{stats['bias']:.6f}",
            "sd_cm2": f"{stats['sd']:.6f}",
            "loa_lower_cm2": f"{stats['loa_lower']:.6f}",
            "loa_upper_cm2": f"{stats['loa_upper']:.6f}",
            "outside_loa_n": int(stats["outside_loa"]),
            "png_path": str(png),
        })

        for row, gt_value, pred_value in zip(rows, gt, pred):
            difference = float(pred_value - gt_value)
            per_case_rows.append({
                "fold": row["fold"],
                "case_id": row["case_id"],
                "tissue": info["name"],
                "gt_cm2": f"{gt_value:.6f}",
                "end_to_end_cm2": f"{pred_value:.6f}",
                "mean_cm2": f"{(gt_value + pred_value) / 2:.6f}",
                "difference_pred_minus_gt_cm2": f"{difference:.6f}",
                "outside_loa": int(difference < stats["loa_lower"] or difference > stats["loa_upper"]),
            })

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), dpi=200)
    for label, (ax, tissue) in zip(("A", "B", "C", "D"), zip(axes.flat, TISSUES)):
        gt, pred = values[tissue]
        draw_panel(ax, gt, pred, tissue, f"{label}. {TISSUES[tissue]['name']}")
    main_title = fig.suptitle("Inferior GT vs End-to-End Prediction: Bland–Altman Analysis", fontsize=15, y=1.01)
    fig.tight_layout()
    combined = args.output_dir / "bland_altman_combined.png"
    fig.savefig(combined, dpi=300, bbox_inches="tight")
    main_title.remove()
    fig._suptitle = None
    for ax in axes.flat:
        ax.set_title("")
    fig.tight_layout()
    combined_no_title = args.output_dir / "bland_altman_combined_no_title.png"
    fig.savefig(combined_no_title, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_csv(
        args.output_dir / "bland_altman_summary.csv",
        ["tissue", "n", "mean_difference_cm2", "sd_cm2", "loa_lower_cm2", "loa_upper_cm2", "outside_loa_n", "png_path"],
        summary_rows,
    )
    write_csv(
        args.output_dir / "bland_altman_per_case.csv",
        ["fold", "case_id", "tissue", "gt_cm2", "end_to_end_cm2", "mean_cm2", "difference_pred_minus_gt_cm2", "outside_loa"],
        per_case_rows,
    )
    print(f"[OK] inferior cases: {len(rows)}")
    print(f"[OK] combined figure: {combined}")
    print(f"[OK] combined figure without titles: {combined_no_title}")
    print(f"[OK] summary: {args.output_dir / 'bland_altman_summary.csv'}")


if __name__ == "__main__":
    main()
